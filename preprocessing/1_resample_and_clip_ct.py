"""Step 1: Downsample CT, ORGAN_MASK, and optional TTB to native PET resolution and clip CT HU to [-1000, 1000].

Expected input directory structure:
    <input_dir>/<case_id>/
        PET.nii.gz         (SUVbw map, defines the native PET reference grid)
        CT.nii.gz          (CT volume in Hounsfield Units)
        ORGAN_MASK.nii.gz  (TotalSegmentator --ml output)
        TTB.nii.gz         (optional ground-truth tumour mask, label == 1)

Outputs in <output_dir>/<case_id>/:
    PET.nii.gz, CT.nii.gz (clipped to [-1000, 1000] HU and downsampled to PET resolution),
    ORGAN_MASK.nii.gz (nearest-neighbour downsampled to PET resolution), and TTB.nii.gz (if provided).
"""

import argparse
import shutil
from pathlib import Path

import SimpleITK as sitk


def resample_to_reference(image: sitk.Image, reference: sitk.Image, is_label: bool) -> sitk.Image:
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetDefaultPixelValue(0)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)
    return resampler.Execute(image)


def process_case(case_dir: Path, out_case_dir: Path, min_hu: float, max_hu: float) -> bool:
    pet_path = case_dir / "PET.nii.gz"
    ct_path = case_dir / "CT.nii.gz"
    organ_path = case_dir / "ORGAN_MASK.nii.gz"
    ttb_path = case_dir / "TTB.nii.gz"

    if not (pet_path.is_file() and ct_path.is_file() and organ_path.is_file()):
        print(f"[Skip] {case_dir.name}: missing PET.nii.gz, CT.nii.gz, or ORGAN_MASK.nii.gz")
        return False

    out_case_dir.mkdir(parents=True, exist_ok=True)

    pet_sitk = sitk.ReadImage(str(pet_path))
    ct_sitk = sitk.ReadImage(str(ct_path))
    organ_sitk = sitk.ReadImage(str(organ_path))

    ct_clipped = sitk.Clamp(ct_sitk, sitk.sitkFloat32, min_hu, max_hu)
    ct_aligned = resample_to_reference(ct_clipped, pet_sitk, is_label=False)
    organ_aligned = resample_to_reference(organ_sitk, pet_sitk, is_label=True)

    shutil.copy2(pet_path, out_case_dir / "PET.nii.gz")
    sitk.WriteImage(ct_aligned, str(out_case_dir / "CT.nii.gz"))
    sitk.WriteImage(organ_aligned, str(out_case_dir / "ORGAN_MASK.nii.gz"))

    if ttb_path.is_file():
        ttb_sitk = sitk.ReadImage(str(ttb_path))
        ttb_aligned = resample_to_reference(ttb_sitk, pet_sitk, is_label=True)
        sitk.WriteImage(ttb_aligned, str(out_case_dir / "TTB.nii.gz"))

    print(f"[OK] {case_dir.name}: downsampled to PET grid {pet_sitk.GetSize()}, CT clipped to [{min_hu}, {max_hu}] HU")
    return True


def main():
    parser = argparse.ArgumentParser(description="Resample CT/masks to native PET space and clip CT HU.")
    parser.add_argument("--input_dir", type=str, required=True, help="Root directory with <case_id>/ subfolders")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for aligned volumes")
    parser.add_argument("--min_hu", type=float, default=-1000.0, help="Lower CT HU clipping bound")
    parser.add_argument("--max_hu", type=float, default=1000.0, help="Upper CT HU clipping bound")
    args = parser.parse_args()

    in_root = Path(args.input_dir)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    case_dirs = sorted(p for p in in_root.iterdir() if p.is_dir())
    if not case_dirs:
        raise FileNotFoundError(f"No case subdirectories found in {in_root}")

    processed = sum(process_case(d, out_root / d.name, args.min_hu, args.max_hu) for d in case_dirs)
    print(f"Step 1 complete: {processed}/{len(case_dirs)} case(s) saved to {out_root}")


if __name__ == "__main__":
    main()
