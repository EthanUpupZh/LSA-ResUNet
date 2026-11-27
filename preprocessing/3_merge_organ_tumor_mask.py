"""Step 3: Exclude prostate from ORGAN_MASK, merge TTB tumour mask, and save as <case_id>_*.nii.gz.

Expected input directory (from Step 2):
    <input_dir>/<case_id>/PET.nii.gz, CT.nii.gz, ORGAN_MASK.nii.gz, [TTB.nii.gz]

Outputs in <output_dir>/:
    <case_id>_PET.nii.gz
    <case_id>_CT.nii.gz
    <case_id>_MERGED_MASK.nii.gz  (created when TTB.nii.gz is present: prostate label 22 set to 0,
                                   and TTB == 1 overwrites overlapping organs voxel-by-voxel without dilation)
"""

import argparse
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np


def process_case(case_dir: Path, out_dir: Path, prostate_label: int) -> bool:
    pet_path = case_dir / "PET.nii.gz"
    ct_path = case_dir / "CT.nii.gz"
    organ_path = case_dir / "ORGAN_MASK.nii.gz"
    ttb_path = case_dir / "TTB.nii.gz"

    if not (pet_path.is_file() and ct_path.is_file()):
        print(f"[Skip] {case_dir.name}: missing PET.nii.gz or CT.nii.gz")
        return False

    cid = case_dir.name
    shutil.copy2(pet_path, out_dir / f"{cid}_PET.nii.gz")
    shutil.copy2(ct_path, out_dir / f"{cid}_CT.nii.gz")

    if ttb_path.is_file() and organ_path.is_file():
        organ_nii = nib.load(str(organ_path))
        ttb_nii = nib.load(str(ttb_path))

        organ_data = np.rint(organ_nii.get_fdata()).astype(np.int16)
        ttb_data = np.rint(ttb_nii.get_fdata()).astype(np.int16)

        if organ_data.shape != ttb_data.shape:
            raise ValueError(f"{cid}: shape mismatch between ORGAN_MASK {organ_data.shape} and TTB {ttb_data.shape}")

        merged = organ_data.copy()
        merged[merged == prostate_label] = 0
        # Avoid colliding with tumour label 1 if any non-tumour organ happens to use 1
        merged[merged == 1] = 0
        merged[ttb_data == 1] = 1

        out_mask = nib.Nifti1Image(merged, organ_nii.affine, organ_nii.header)
        nib.save(out_mask, str(out_dir / f"{cid}_MERGED_MASK.nii.gz"))
        print(f"[OK] {cid}: exported PET, CT, and MERGED_MASK (prostate={prostate_label} excluded)")
    else:
        print(f"[OK] {cid}: exported PET and CT (inference-only mode)")

    return True


def main():
    parser = argparse.ArgumentParser(description="Exclude prostate, merge tumour mask, and export flat NIfTIs.")
    parser.add_argument("--input_dir", type=str, required=True, help="Output directory from Step 2")
    parser.add_argument("--output_dir", type=str, required=True, help="Flat output directory for training/inference")
    parser.add_argument("--prostate_label", type=int, default=22, help="Prostate label ID in ORGAN_MASK to exclude")
    args = parser.parse_args()

    in_root = Path(args.input_dir)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    case_dirs = sorted(p for p in in_root.iterdir() if p.is_dir())
    if not case_dirs:
        raise FileNotFoundError(f"No case subdirectories found in {in_root}")

    processed = sum(process_case(d, out_root, args.prostate_label) for d in case_dirs)
    print(f"Step 3 complete: {processed}/{len(case_dirs)} case(s) exported to {out_root}")


if __name__ == "__main__":
    main()
