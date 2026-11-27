"""Step 2: Crop PET, CT, ORGAN_MASK, and optional TTB to the target region.

Computes the 3D world-coordinate bounding box from bladder/prostate labels
in ORGAN_MASK.nii.gz (default TotalSegmentator labels: 21=urinary_bladder, 22=prostate),
expands it by (50, 50) mm along X, (50, 50) mm along Y, and (50, 20) mm along Z,
and crops all volumes to the target region while preserving native spacing and updating the NIfTI affine.
"""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np


def get_world_bbox(mask_path: Path, label_values: list[int]):
    mask_img = nib.load(str(mask_path))
    mask_data = np.rint(mask_img.get_fdata()).astype(np.int32)

    coords_list = []
    for val in label_values:
        vox = np.array(np.where(mask_data == val)).T
        if len(vox) > 0:
            coords_list.append(nib.affines.apply_affine(mask_img.affine, vox))

    if not coords_list:
        return None

    all_coords = np.vstack(coords_list)
    return (
        float(all_coords[:, 0].min()),
        float(all_coords[:, 0].max()),
        float(all_coords[:, 1].min()),
        float(all_coords[:, 1].max()),
        float(all_coords[:, 2].min()),
        float(all_coords[:, 2].max()),
    )


def crop_by_world_bbox(img_path: Path, bbox, expand_x, expand_y, expand_z):
    img = nib.load(str(img_path))
    orig_dtype = img.get_data_dtype()
    data = img.get_fdata()
    affine = img.affine
    header = img.header.copy()

    x_min, x_max, y_min, y_max, z_min, z_max = bbox
    x_min -= expand_x[0]
    x_max += expand_x[1]
    y_min -= expand_y[0]
    y_max += expand_y[1]
    z_min -= expand_z[0]
    z_max += expand_z[1]

    inv_affine = np.linalg.inv(affine)
    corners = [
        (x_min, y_min, z_min), (x_min, y_min, z_max),
        (x_min, y_max, z_min), (x_min, y_max, z_max),
        (x_max, y_min, z_min), (x_max, y_min, z_max),
        (x_max, y_max, z_min), (x_max, y_max, z_max),
    ]
    vox = np.array([(inv_affine @ [c[0], c[1], c[2], 1.0])[:3] for c in corners])
    floored = np.floor(vox).astype(int)

    x0 = max(0, int(floored[:, 0].min()))
    x1 = min(data.shape[0] - 1, int(floored[:, 0].max()))
    y0 = max(0, int(floored[:, 1].min()))
    y1 = min(data.shape[1] - 1, int(floored[:, 1].max()))
    z0 = max(0, int(floored[:, 2].min()))
    z1 = min(data.shape[2] - 1, int(floored[:, 2].max()))

    if x0 >= x1 or y0 >= y1 or z0 >= z1:
        return None

    cropped = data[x0 : x1 + 1, y0 : y1 + 1, z0 : z1 + 1].astype(orig_dtype)
    new_affine = affine.copy()
    new_affine[:3, 3] = (affine @ [x0, y0, z0, 1.0])[:3]
    return nib.Nifti1Image(cropped, new_affine, header)


def process_case(case_dir: Path, out_case_dir: Path, labels, expand_x, expand_y, expand_z) -> bool:
    organ_path = case_dir / "ORGAN_MASK.nii.gz"
    pet_path = case_dir / "PET.nii.gz"
    ct_path = case_dir / "CT.nii.gz"
    ttb_path = case_dir / "TTB.nii.gz"

    if not (organ_path.is_file() and pet_path.is_file() and ct_path.is_file()):
        print(f"[Skip] {case_dir.name}: missing PET.nii.gz, CT.nii.gz, or ORGAN_MASK.nii.gz")
        return False

    bbox = get_world_bbox(organ_path, labels)
    if bbox is None:
        print(f"[Skip] {case_dir.name}: labels {labels} not found in ORGAN_MASK.nii.gz")
        return False

    targets = [("PET.nii.gz", pet_path), ("CT.nii.gz", ct_path), ("ORGAN_MASK.nii.gz", organ_path)]
    if ttb_path.is_file():
        targets.append(("TTB.nii.gz", ttb_path))

    out_case_dir.mkdir(parents=True, exist_ok=True)
    for fname, fpath in targets:
        cropped = crop_by_world_bbox(fpath, bbox, expand_x, expand_y, expand_z)
        if cropped is None:
            print(f"[Skip] {case_dir.name}: empty crop for {fname}")
            return False
        nib.save(cropped, str(out_case_dir / fname))

    print(f"[OK] {case_dir.name}: cropped shape {cropped.shape}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Crop aligned volumes around bladder/prostate ROI.")
    parser.add_argument("--input_dir", type=str, required=True, help="Output directory from Step 1")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for cropped volumes")
    parser.add_argument("--labels", type=int, nargs="+", default=[21, 22], help="Anchor labels in ORGAN_MASK")
    parser.add_argument("--expand_x", type=float, nargs=2, default=[50.0, 50.0], help="X expansion (mm)")
    parser.add_argument("--expand_y", type=float, nargs=2, default=[50.0, 50.0], help="Y expansion (mm)")
    parser.add_argument("--expand_z", type=float, nargs=2, default=[50.0, 20.0], help="Z expansion (mm)")
    args = parser.parse_args()

    in_root = Path(args.input_dir)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    case_dirs = sorted(p for p in in_root.iterdir() if p.is_dir())
    if not case_dirs:
        raise FileNotFoundError(f"No case subdirectories found in {in_root}")

    processed = sum(
        process_case(d, out_root / d.name, args.labels, tuple(args.expand_x), tuple(args.expand_y), tuple(args.expand_z))
        for d in case_dirs
    )
    print(f"Step 2 complete: {processed}/{len(case_dirs)} case(s) saved to {out_root}")


if __name__ == "__main__":
    main()
