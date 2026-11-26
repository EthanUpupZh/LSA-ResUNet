"""Pretraining script for LSA-ResUNet (3D dual-channel PET/CT) on DEEP-PSMA."""

import argparse
import logging
import os
import sys
from datetime import datetime
from glob import glob

import monai
import torch
from monai.data import decollate_batch, list_data_collate
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    RandCropByPosNegLabeld,
    RandRotate90d,
    RandScaleIntensityd,
    ScaleIntensityd,
)
from torch.utils.data import DataLoader

from LSA import LSAResUNet


LOG_FH = None
LOG_PATH = None
MANIFEST_PATH = None
RUN_STAMP = None


def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg=""):
    print(msg)
    if LOG_FH is not None:
        LOG_FH.write(str(msg) + "\n")
        LOG_FH.flush()


def init_run_logs(out_dir):
    global LOG_FH, LOG_PATH, RUN_STAMP
    os.makedirs(out_dir, exist_ok=True)
    RUN_STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_PATH = os.path.join(out_dir, f"train_log_{RUN_STAMP}.txt")
    LOG_FH = open(LOG_PATH, "a", encoding="utf-8")
    log("=" * 80)
    log(f"Start time : {now_str()}")
    log(f"Output dir : {os.path.abspath(out_dir)}")
    log("=" * 80)
    return LOG_PATH


def close_log():
    global LOG_FH
    if LOG_FH is not None:
        LOG_FH.close()
        LOG_FH = None


def write_run_manifest(out_dir, extra=None):
    global MANIFEST_PATH
    lines = [
        "=" * 80,
        "Run Manifest",
        f"Created : {now_str()}",
        "=" * 80,
    ]
    if extra:
        for k, v in extra.items():
            lines.append(f"  {k:<20s}: {v}")
        lines.append("")
    path = os.path.join(out_dir, f"run_info_{RUN_STAMP}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    MANIFEST_PATH = path
    return path


def finalize_run_manifest():
    if MANIFEST_PATH is not None:
        with open(MANIFEST_PATH, "a", encoding="utf-8") as f:
            f.write(f"End time   : {now_str()}\n")


def build_file_list(split_dir):
    """Pair PET, CT, and TTB label files strictly by case ID prefix."""
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
        seg_path = os.path.join(split_dir, f"{prefix}_TTB.nii.gz")
        if not os.path.exists(seg_path):
            seg_path = os.path.join(split_dir, f"{prefix}TTB.nii.gz")

        if not os.path.exists(ct_path) or not os.path.exists(seg_path):
            raise FileNotFoundError(
                f"Missing paired CT or TTB mask for {pet_path} (expected {ct_path}, {seg_path})"
            )
        records.append({"img": pet_path, "img_ct": ct_path, "seg": seg_path})

    if not records:
        raise FileNotFoundError(f"No *PET.nii.gz files found in {split_dir}")
    return records


def main(args):
    if args.aug_mode not in ("fixed", "random"):
        raise ValueError(f"Invalid aug_mode: {args.aug_mode}")

    seed = args.seed if args.aug_mode == "fixed" else int.from_bytes(os.urandom(4), "big")
    monai.utils.set_determinism(seed=seed)

    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    init_run_logs(args.output_dir)
    train_files = build_file_list(os.path.join(args.data_dir, "train"))
    val_files = build_file_list(os.path.join(args.data_dir, "val"))
    log(f"Loaded {len(train_files)} train scans and {len(val_files)} val scans | aug_mode={args.aug_mode}, seed={seed}")

    train_transforms = Compose(
        [
            LoadImaged(keys=["img", "img_ct", "seg"]),
            EnsureChannelFirstd(keys=["img", "img_ct", "seg"]),
            ScaleIntensityd(keys=["img", "img_ct"]),
            RandScaleIntensityd(keys=["img"], factors=(-0.2, 0.2), prob=0.2),
            RandRotate90d(keys=["img", "img_ct", "seg"], prob=0.5, spatial_axes=[0, 2]),
            RandCropByPosNegLabeld(
                keys=["img", "img_ct", "seg"],
                label_key="seg",
                spatial_size=[32, 32, 32],
                pos=1,
                neg=1,
                num_samples=args.num_samples,
            ),
        ]
    )

    val_transforms = Compose(
        [
            LoadImaged(keys=["img", "img_ct", "seg"]),
            EnsureChannelFirstd(keys=["img", "img_ct", "seg"]),
            ScaleIntensityd(keys=["img", "img_ct"]),
        ]
    )

    worker_seed_fn = monai.data.worker_init_fn
    train_gen = torch.Generator().manual_seed(seed + 1)

    train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=worker_seed_fn,
        generator=train_gen,
    )

    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        num_workers=args.num_workers,
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

    loss_function = monai.losses.DiceCELoss(sigmoid=True)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    write_run_manifest(
        args.output_dir,
        extra={
            "model": "LSAResUNet(2ch->1ch, channels=(16,32,64,128,256))",
            "loss": "DiceCELoss(sigmoid=True)",
            "optimizer": f"Adam(lr={args.lr})",
            "aug_mode": args.aug_mode,
            "seed": seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "num_samples": args.num_samples,
            "train_count": len(train_files),
            "val_count": len(val_files),
        },
    )

    best_metric = -1.0
    best_metric_epoch = -1

    for epoch in range(args.epochs):
        log("-" * 40)
        log(f"[{now_str()}] Epoch {epoch + 1}/{args.epochs}")
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
            epoch_loss += loss.item()

        epoch_loss /= max(step, 1)
        log(f"[{now_str()}] Epoch {epoch + 1} average loss: {epoch_loss:.4f}")

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
                    save_path = os.path.join(args.output_dir, "best_model_segmentation3d.pth")
                    torch.save(model.state_dict(), save_path)
                    log(f"[{now_str()}] Saved new best model: {save_path} (DSC={best_metric:.4f})")

                log(
                    f"[{now_str()}] Val DSC: {metric:.4f} | Best: {best_metric:.4f} @ epoch {best_metric_epoch}"
                )

    log("=" * 80)
    log(f"[{now_str()}] Training completed. Best DSC: {best_metric:.4f} @ epoch {best_metric_epoch}")
    close_log()
    finalize_run_manifest()


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain LSA-ResUNet on DEEP-PSMA")
    parser.add_argument("--data_dir", type=str, default="./data/DEEP-PSMA", help="Root dir with train/ and val/")
    parser.add_argument("--output_dir", type=str, default="./outputs/pretrain", help="Directory to save weights/logs")
    parser.add_argument("--aug_mode", type=str, default="random", choices=["fixed", "random"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--val_interval", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_samples", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gpu", type=str, default=None, help="Optional CUDA_VISIBLE_DEVICES id")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.multiprocessing.set_sharing_strategy("file_system")
    main(args)
