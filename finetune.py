"""Fine-tuning script for LSA-ResUNet (3D dual-channel PET/CT) on FAHSU-PSMA."""

import argparse
import datetime
import logging
import os
import sys
from glob import glob

import monai
import numpy as np
import torch
from monai.data import list_data_collate, decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    CutMixd,
    EnsureChannelFirstd,
    LoadImaged,
    Rand3DElasticd,
    RandCropByPosNegLabeld,
    RandRotate90d,
    RandScaleIntensityd,
    ScaleIntensityd,
)
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from LSA import LSAResUNet, remap_legacy_state_dict


def build_file_list(split_dir):
    """Pair PET, CT, and MERGED_MASK files strictly by case ID prefix."""
    pet_files = sorted(glob(os.path.join(split_dir, "*_PET.nii.gz")))
    if not pet_files:
        pet_files = sorted(glob(os.path.join(split_dir, "*PET.nii.gz")))

    records = []
    for pet_path in pet_files:
        fname = os.path.basename(pet_path)
        prefix = fname.replace("_PET.nii.gz", "").replace("PET.nii.gz", "")
        ct_path = os.path.join(split_dir, f"{prefix}_CT.nii.gz")
        if not os.path.exists(ct_path):
            ct_path = os.path.join(split_dir, f"{prefix}CT.nii.gz")
        seg_path = os.path.join(split_dir, f"{prefix}_MERGED_MASK.nii.gz")
        if not os.path.exists(seg_path):
            seg_path = os.path.join(split_dir, f"{prefix}MERGED_MASK.nii.gz")

        if not os.path.exists(ct_path) or not os.path.exists(seg_path):
            raise FileNotFoundError(
                f"Missing paired CT or MERGED_MASK for {pet_path} (expected {ct_path}, {seg_path})"
            )
        records.append({"img": pet_path, "img_ct": ct_path, "seg": seg_path})

    if not records:
        raise FileNotFoundError(f"No *PET.nii.gz files found in {split_dir}")
    return records


def save_run_info(out_dir, info):
    os.makedirs(out_dir, exist_ok=True)
    txt_path = os.path.join(out_dir, "run_info.txt")
    lines = [
        "=" * 70,
        "Run Info",
        f"Timestamp : {datetime.datetime.now():%Y-%m-%d %H:%M:%S}",
        "-" * 70,
    ]
    for k, v in info.items():
        lines.append(f"{k:22s}: {v}")
    lines.append("=" * 70)
    with open(txt_path, "w", encoding="utf-8") as fw:
        fw.write("\n".join(lines) + "\n")
    return txt_path


def main(args):
    if args.aug_mode not in ("fixed", "random"):
        raise ValueError(f"Invalid aug_mode: {args.aug_mode}")

    seed = args.seed if args.aug_mode == "fixed" else int.from_bytes(os.urandom(4), "big")
    monai.utils.set_determinism(seed=seed)
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "train_log.txt")
    log_fp = open(log_path, "a", encoding="utf-8", buffering=1)
    sys.stdout = log_fp
    sys.stderr = log_fp

    train_files = build_file_list(os.path.join(args.data_dir, "train"))
    val_files = build_file_list(os.path.join(args.data_dir, "val"))

    save_run_info(
        args.output_dir,
        {
            "script": os.path.abspath(__file__),
            "pretrained_path": args.pretrained_path,
            "output_dir": args.output_dir,
            "aug_mode": args.aug_mode,
            "seed": seed,
            "train_count": len(train_files),
            "val_count": len(val_files),
        },
    )

    train_transforms = Compose(
        [
            LoadImaged(keys=["img", "img_ct", "seg"]),
            EnsureChannelFirstd(keys=["img", "img_ct", "seg"]),
            ScaleIntensityd(keys=["img", "img_ct"]),
            RandRotate90d(keys=["img", "img_ct", "seg"], prob=0.5, spatial_axes=[0, 2]),
            RandCropByPosNegLabeld(
                keys=["img", "img_ct", "seg"],
                label_key="seg",
                spatial_size=[32, 32, 32],
                pos=1,
                neg=1,
                num_samples=args.num_samples,
            ),
            Rand3DElasticd(
                keys=["img", "seg"],
                prob=0.3,
                sigma_range=(3, 5),
                magnitude_range=(2, 5),
                mode=("bilinear", "nearest"),
            ),
            RandScaleIntensityd(keys=["img", "img_ct"], factors=(-0.2, 0.2), prob=0.2),
        ]
    )

    val_transforms = Compose(
        [
            LoadImaged(keys=["img", "img_ct", "seg"]),
            EnsureChannelFirstd(keys=["img", "img_ct", "seg"]),
            ScaleIntensityd(keys=["img", "img_ct"]),
        ]
    )

    # Apply regional CutMix exclusively to PET ("img"), keeping CT and tumour masks ("seg") unaltered.
    cutmix_batch_size = args.batch_size * args.num_samples
    cutmix_transform = CutMixd(
        keys=["img"],
        batch_size=cutmix_batch_size,
        label_keys=None,
        alpha=0.3,
        allow_missing_keys=False,
    )
    cutmix_seeded = False

    def custom_collate_fn(batch):
        nonlocal cutmix_seeded
        if not cutmix_seeded:
            worker = torch.utils.data.get_worker_info()
            cutmix_transform.set_random_state(
                seed=(worker.seed if worker is not None else seed + 2) % (2**32)
            )
            cutmix_seeded = True

        collated_batch = list_data_collate(batch)
        for key in ["img", "img_ct", "seg"]:
            if key in collated_batch:
                collated_batch[key] = collated_batch[key].as_tensor()

        return cutmix_transform(collated_batch)

    worker_seed_fn = monai.data.worker_init_fn
    train_gen = torch.Generator().manual_seed(seed + 1)

    train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=worker_seed_fn,
        generator=train_gen,
    )

    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        num_workers=max(1, args.num_workers // 2),
        collate_fn=list_data_collate,
    )

    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LSAResUNet(
        spatial_dims=3,
        in_channels=2,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)

    if args.pretrained_path and os.path.exists(args.pretrained_path):
        model_dict = model.state_dict()
        state_dict = remap_legacy_state_dict(torch.load(args.pretrained_path, map_location=device))
        pretrained_dict = {
            k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape
        }
        filtered_keys = set(state_dict.keys()) - set(pretrained_dict.keys())
        print("Filtered mismatched keys:", filtered_keys)
        model.load_state_dict(pretrained_dict, strict=False)
    else:
        print(f"Warning: pretrained_path not found ({args.pretrained_path}); training from scratch.")

    loss_function = monai.losses.DiceFocalLoss(
        sigmoid=True, gamma=2.0, alpha=0.3, include_background=False
    )
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-7)

    best_metric = -1.0
    best_metric_epoch = -1

    for epoch in range(args.epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{args.epochs}")
        model.train()
        epoch_loss = 0.0
        step = 0

        for batch_data in train_loader:
            step += 1
            inputs = batch_data["img"].to(device)
            inputs_ct = batch_data["img_ct"].to(device)
            labels = (batch_data["seg"].to(device) == 1).float()
            combined = torch.cat((inputs, inputs_ct), dim=1)

            optimizer.zero_grad()
            outputs = model(combined)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            epoch_len = max(1, len(train_ds) // train_loader.batch_size)
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")

        epoch_loss /= max(step, 1)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % args.val_interval == 0:
            model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    val_images = val_data["img"].to(device)
                    val_images_ct = val_data["img_ct"].to(device)
                    val_labels = (val_data["seg"].to(device) == 1).float()
                    combined = torch.cat((val_images, val_images_ct), dim=1)

                    val_outputs = sliding_window_inference(combined, (32, 32, 32), 4, model)
                    val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                    dice_metric(y_pred=val_outputs, y=val_labels)

                metric = dice_metric.aggregate().item()
                dice_metric.reset()
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    save_path = os.path.join(args.output_dir, "best_metric_model_segmentation3d_dict.pth")
                    torch.save(model.state_dict(), save_path)
                    print("saved new best metric model")

                print(
                    f"current epoch: {epoch + 1} current mean dice: {metric:.4f} "
                    f"best mean dice: {best_metric:.4f} at epoch {best_metric_epoch}"
                )

    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    log_fp.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune LSA-ResUNet on FAHSU-PSMA")
    parser.add_argument("--data_dir", type=str, default="./data/FAHSU-PSMA", help="Root dir with train/ and val/")
    parser.add_argument(
        "--pretrained_path",
        type=str,
        default="./outputs/pretrain/best_model_segmentation3d.pth",
        help="Path to pretrained checkpoint",
    )
    parser.add_argument("--output_dir", type=str, default="./outputs/finetune", help="Directory to save weights/logs")
    parser.add_argument("--aug_mode", type=str, default="fixed", choices=["fixed", "random"])
    parser.add_argument("--seed", type=int, default=1076250484)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--val_interval", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_samples", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gpu", type=str, default=None, help="Optional CUDA_VISIBLE_DEVICES id")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    main(args)
