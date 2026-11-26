"""Inference script for LSA-ResUNet (supports single checkpoint or multi-fold ensemble)."""

import argparse
import os
import time
from glob import glob

import monai
import nibabel as nib
import numpy as np
import torch
from monai.data import list_data_collate
from monai.inferers import sliding_window_inference
from monai.transforms import Compose, EnsureChannelFirstd, LoadImaged, ScaleIntensityd
from torch.utils.data import DataLoader

from LSA import LSAResUNet


def build_test_list(data_dir):
    """Pair PET and CT NIfTI files strictly by case ID prefix."""
    pet_files = sorted(glob(os.path.join(data_dir, "*_PET.nii.gz")))
    if not pet_files:
        pet_files = sorted(glob(os.path.join(data_dir, "*PET.nii.gz")))

    records = []
    for pet_path in pet_files:
        fname = os.path.basename(pet_path)
        prefix = fname.replace("_PET.nii.gz", "").replace("PET.nii.gz", "")
        ct_path = os.path.join(data_dir, f"{prefix}_CT.nii.gz")
        if not os.path.exists(ct_path):
            ct_path = os.path.join(data_dir, f"{prefix}CT.nii.gz")
        if not os.path.exists(ct_path):
            raise FileNotFoundError(f"Missing paired CT for {pet_path} (expected {ct_path})")
        records.append({"img": pet_path, "img_ct": ct_path, "case_id": prefix})

    if not records:
        raise FileNotFoundError(f"No *PET.nii.gz files found in {data_dir}")
    return records


def load_models(weight_paths, device):
    models = []
    for wp in weight_paths:
        if not os.path.exists(wp):
            raise FileNotFoundError(f"Checkpoint not found: {wp}")
        model = LSAResUNet(
            spatial_dims=3,
            in_channels=2,
            out_channels=1,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        ).to(device)
        state_dict = torch.load(wp, map_location=device)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        models.append(model)
    return models


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    test_files = build_test_list(args.data_dir)
    print(f"Found {len(test_files)} cases in {args.data_dir}")

    transforms = Compose(
        [
            LoadImaged(keys=["img", "img_ct"]),
            EnsureChannelFirstd(keys=["img", "img_ct"]),
            ScaleIntensityd(keys=["img", "img_ct"]),
        ]
    )

    test_ds = monai.data.Dataset(data=test_files, transform=transforms)
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=args.num_workers, collate_fn=list_data_collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = load_models(args.weights, device)
    print(f"Loaded {len(models)} checkpoint(s) on {device}.")

    roi_size = (args.roi_size, args.roi_size, args.roi_size)
    times = []

    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            case_id = test_files[idx]["case_id"]
            pet_path = test_files[idx]["img"]

            val_pet = batch["img"].to(device)
            val_ct = batch["img_ct"].to(device)
            combined = torch.cat((val_pet, val_ct), dim=1)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.time()

            prob_sum = None
            for model in models:
                logits = sliding_window_inference(combined, roi_size, args.sw_batch_size, model)
                probs = torch.sigmoid(logits)
                prob_sum = probs if prob_sum is None else (prob_sum + probs)
            mean_prob = prob_sum / len(models)
            pred_mask = (mean_prob >= args.threshold).squeeze().cpu().numpy().astype(np.uint8)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            dt = time.time() - t0
            times.append(dt)

            # Save prediction preserving original PET affine and header
            ref_nii = nib.load(pet_path)
            out_nii = nib.Nifti1Image(pred_mask, ref_nii.affine, ref_nii.header)
            out_nii.set_data_dtype(np.uint8)
            out_path = os.path.join(args.output_dir, f"{case_id}_PET_seg.nii.gz")
            nib.save(out_nii, out_path)
            print(f"[{idx + 1}/{len(test_files)}] {case_id}: saved {out_path} ({dt*1000:.1f} ms)")

    print(f"Inference completed. Mean time per scan: {np.mean(times)*1000:.1f} ms")


def parse_args():
    parser = argparse.ArgumentParser(description="Run LSA-ResUNet inference on PSMA PET/CT")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory with *_PET.nii.gz and *_CT.nii.gz")
    parser.add_argument(
        "--weights",
        type=str,
        nargs="+",
        required=True,
        help="One checkpoint path (single model) or multiple paths (ensemble)",
    )
    parser.add_argument("--output_dir", type=str, default="./outputs/preds", help="Directory for predicted masks")
    parser.add_argument("--roi_size", type=int, default=32)
    parser.add_argument("--sw_batch_size", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--gpu", type=str, default=None, help="Optional CUDA_VISIBLE_DEVICES id")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    main(args)
