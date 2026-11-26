"""3D LSA-ResUNet architecture for dual-channel PSMA PET/CT tumor segmentation."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Dict, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from monai.networks.blocks import ADN, ResidualUnit
from monai.networks.layers.convutils import same_padding, stride_minus_kernel_padding
from monai.networks.layers.factories import Act, Conv, Norm
from monai.networks.layers.simplelayers import SkipConnection


def remap_legacy_state_dict(state_dict: dict) -> dict:
    """Remap legacy '.la.' submodule keys to '.lsa.' for backward checkpoint compatibility."""
    return {k.replace(".la.", ".lsa."): v for k, v in state_dict.items()}


def build_norm_layer_3d(norm_layer_cfg: Union[str, Dict], num_features: int) -> Tuple[str, nn.Module]:
    """Build 3D normalization layer."""
    if isinstance(norm_layer_cfg, str):
        norm_type = norm_layer_cfg.lower()
    elif isinstance(norm_layer_cfg, dict):
        norm_type = norm_layer_cfg.get("type", "bn").lower()
    else:
        raise TypeError(f"Unsupported norm_layer_cfg: {norm_layer_cfg}")

    if norm_type in ["bn", "batchnorm"]:
        return "bn", nn.BatchNorm3d(num_features)
    if norm_type in ["gn", "groupnorm"]:
        return "gn", nn.GroupNorm(32, num_features)
    if norm_type in ["ln", "layernorm"]:
        return "ln", nn.LayerNorm(num_features)
    raise ValueError(f"Unsupported norm_layer type: {norm_type}")


class Convolution(nn.Sequential):
    """3D Convolution / Transposed-Convolution block with optional normalization, dropout, and activation."""

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        strides: Sequence[int] | int = 1,
        kernel_size: Sequence[int] | int = 3,
        adn_ordering: str = "NDA",
        act: tuple | str | None = "PRELU",
        norm: tuple | str | None = "INSTANCE",
        dropout: tuple | str | float | None = None,
        dropout_dim: int | None = 1,
        dilation: Sequence[int] | int = 1,
        groups: int = 1,
        bias: bool = True,
        conv_only: bool = False,
        is_transposed: bool = False,
        padding: Sequence[int] | int | None = None,
        output_padding: Sequence[int] | int | None = None,
    ) -> None:
        super().__init__()
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.is_transposed = is_transposed

        if padding is None:
            padding = same_padding(kernel_size, dilation)
        conv_type = Conv[Conv.CONVTRANS if is_transposed else Conv.CONV, self.spatial_dims]

        conv: nn.Module
        if is_transposed:
            if output_padding is None:
                output_padding = stride_minus_kernel_padding(1, strides)
            conv = conv_type(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=strides,
                padding=padding,
                output_padding=output_padding,
                groups=groups,
                bias=bias,
                dilation=dilation,
            )
        else:
            conv = conv_type(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=strides,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )

        self.add_module("conv", conv)

        if conv_only or (act is None and norm is None and dropout is None):
            return

        self.add_module(
            "adn",
            ADN(
                ordering=adn_ordering,
                in_channels=out_channels,
                act=act,
                norm=norm,
                norm_dim=self.spatial_dims,
                dropout=dropout,
                dropout_dim=dropout_dim,
            ),
        )


class LightweightSpatialAttention3D(nn.Module):
    """Lightweight Spatial Attention (LSA) module with local/dilated depthwise branches and spatial gating."""

    def __init__(self, dim: int, norm_layer: Union[str, Dict], act_layer: type[nn.Module]) -> None:
        super().__init__()
        pointwise_groups = 8 if dim % 8 == 0 else 4 if dim % 4 == 0 else 2 if dim % 2 == 0 else 1

        self.local = nn.Sequential(
            nn.Conv3d(dim, dim, 3, 1, 1, groups=dim, bias=False),
            build_norm_layer_3d(norm_layer, dim)[1],
            act_layer(),
        )
        self.context = nn.Sequential(
            nn.Conv3d(dim, dim, 3, 1, 2, dilation=2, groups=dim, bias=False),
            build_norm_layer_3d(norm_layer, dim)[1],
            act_layer(),
        )
        self.mix = nn.Sequential(
            nn.Conv3d(dim, dim, 1, groups=pointwise_groups, bias=False),
            build_norm_layer_3d(norm_layer, dim)[1],
            act_layer(),
        )

        self.spatial_gate = nn.Conv3d(4, 1, 3, 1, 1, bias=False)
        self.gamma = nn.Parameter(torch.tensor(0.05))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.mix(self.local(x) + self.context(x))

        avg_map = torch.mean(feat, dim=1, keepdim=True)
        max_map, _ = torch.max(feat, dim=1, keepdim=True)
        raw_avg_map = torch.mean(x, dim=1, keepdim=True)
        raw_max_map, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = torch.sigmoid(
            self.spatial_gate(torch.cat([avg_map, max_map, raw_avg_map, raw_max_map], dim=1))
        )
        feat = feat * spatial_att

        return x + self.gamma * feat


class DownsamplingResidualUnit(nn.Module):
    """Downsampling Residual Unit (DRU) / Bottleneck unit with embedded Lightweight Spatial Attention (LSA)."""

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        strides: Sequence[int] | int = 1,
        kernel_size: Sequence[int] | int = 3,
        subunits: int = 2,
        adn_ordering: str = "NDA",
        act: tuple | str | None = "PRELU",
        norm: tuple | str | None = "INSTANCE",
        dropout: tuple | str | float | None = None,
        dropout_dim: int | None = 1,
        dilation: Sequence[int] | int = 1,
        bias: bool = True,
        last_conv_only: bool = False,
        padding: Sequence[int] | int | None = None,
    ) -> None:
        super().__init__()
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Sequential()
        self.residual = nn.Identity()
        self.lsa = LightweightSpatialAttention3D(out_channels, "bn", nn.GELU)

        if not padding:
            padding = same_padding(kernel_size, dilation)
        schannels = in_channels
        sstrides = strides
        subunits = max(1, subunits)

        for su in range(subunits):
            conv_only = last_conv_only and su == (subunits - 1)
            unit = Convolution(
                self.spatial_dims,
                schannels,
                out_channels,
                strides=sstrides,
                kernel_size=kernel_size,
                adn_ordering=adn_ordering,
                act=act,
                norm=norm,
                dropout=dropout,
                dropout_dim=dropout_dim,
                dilation=dilation,
                bias=bias,
                conv_only=conv_only,
                padding=padding,
            )
            self.conv.add_module(f"unit{su:d}", unit)
            schannels = out_channels
            sstrides = 1

        if np.prod(strides) != 1 or in_channels != out_channels:
            rkernel_size = kernel_size
            rpadding = padding
            if np.prod(strides) == 1:
                rkernel_size = 1
                rpadding = 0
            conv_type = Conv[Conv.CONV, self.spatial_dims]
            self.residual = conv_type(in_channels, out_channels, rkernel_size, strides, rpadding, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res: torch.Tensor = self.residual(x)
        cx: torch.Tensor = self.lsa(self.conv(x))
        return cx + res


class LSAResUNet(nn.Module):
    """3D Residual U-Net integrated with Lightweight Spatial Attention (LSA-ResUNet)."""

    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 2,
        out_channels: int = 1,
        channels: Sequence[int] = (16, 32, 64, 128, 256),
        strides: Sequence[int] = (2, 2, 2, 2),
        kernel_size: Sequence[int] | int = 3,
        up_kernel_size: Sequence[int] | int = 3,
        num_res_units: int = 2,
        act: tuple | str = Act.PRELU,
        norm: tuple | str = Norm.INSTANCE,
        dropout: float = 0.0,
        bias: bool = True,
        adn_ordering: str = "NDA",
    ) -> None:
        super().__init__()

        if len(channels) < 2:
            raise ValueError("the length of `channels` should be no less than 2.")
        delta = len(strides) - (len(channels) - 1)
        if delta < 0:
            raise ValueError("the length of `strides` should equal to `len(channels) - 1`.")
        if delta > 0:
            warnings.warn(f"`len(strides) > len(channels) - 1`, the last {delta} values of strides will not be used.")
        if isinstance(kernel_size, Sequence) and len(kernel_size) != spatial_dims:
            raise ValueError("the length of `kernel_size` should equal to `dimensions`.")
        if isinstance(up_kernel_size, Sequence) and len(up_kernel_size) != spatial_dims:
            raise ValueError("the length of `up_kernel_size` should equal to `dimensions`.")

        self.dimensions = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.strides = strides
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.bias = bias
        self.adn_ordering = adn_ordering

        def _create_block(
            inc: int, outc: int, channels: Sequence[int], strides: Sequence[int], is_top: bool
        ) -> nn.Module:
            c = channels[0]
            s = strides[0]

            subblock: nn.Module
            if len(channels) > 2:
                subblock = _create_block(c, c, channels[1:], strides[1:], False)
                upc = c * 2
            else:
                subblock = self._get_bottom_layer(c, channels[1])
                upc = c + channels[1]

            down = self._get_down_layer(inc, c, s, is_top)
            up = self._get_up_layer(upc, outc, s, is_top)
            return self._get_connection_block(down, up, subblock)

        self.model = _create_block(in_channels, out_channels, self.channels, self.strides, True)

    def _get_connection_block(self, down_path: nn.Module, up_path: nn.Module, subblock: nn.Module) -> nn.Module:
        return nn.Sequential(down_path, SkipConnection(subblock), up_path)

    def _get_down_layer(self, in_channels: int, out_channels: int, strides: int, is_top: bool) -> nn.Module:
        if self.num_res_units > 0:
            return DownsamplingResidualUnit(
                self.dimensions,
                in_channels,
                out_channels,
                strides=strides,
                kernel_size=self.kernel_size,
                subunits=self.num_res_units,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                adn_ordering=self.adn_ordering,
            )
        return Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=self.kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            adn_ordering=self.adn_ordering,
        )

    def _get_bottom_layer(self, in_channels: int, out_channels: int) -> nn.Module:
        return self._get_down_layer(in_channels, out_channels, 1, False)

    def _get_up_layer(self, in_channels: int, out_channels: int, strides: int, is_top: bool) -> nn.Module:
        conv: Convolution | nn.Sequential = Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=self.up_kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            conv_only=is_top and self.num_res_units == 0,
            is_transposed=True,
            adn_ordering=self.adn_ordering,
        )
        if self.num_res_units > 0:
            ru = ResidualUnit(
                self.dimensions,
                out_channels,
                out_channels,
                strides=1,
                kernel_size=self.kernel_size,
                subunits=1,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                last_conv_only=is_top,
                adn_ordering=self.adn_ordering,
            )
            conv = nn.Sequential(conv, ru)
        return conv

    def load_state_dict(self, state_dict, strict: bool = True, **kwargs):
        return super().load_state_dict(remap_legacy_state_dict(state_dict), strict=strict, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


if __name__ == "__main__":
    net = LSAResUNet(
        spatial_dims=3,
        in_channels=2,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )
    dummy = torch.randn(1, 2, 32, 32, 32)
    out = net(dummy)
    params_m = sum(p.numel() for p in net.parameters()) / 1e6
    print(f"LSAResUNet output shape: {tuple(out.shape)} | Parameters: {params_m:.3f} M")
