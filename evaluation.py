"""Evaluation script for voxel-level, lesion-level, and PET quantitative metrics."""

import argparse
import csv
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import label


def compute_segmentation_metrics(pred_array, gt_array, eps=1e-7):
    """Voxel-level DSC, PPV, and Sensitivity."""
    pred_bin = (pred_array > 0).astype(np.uint8)
    gt_bin = (gt_array > 0).astype(np.uint8)

    tp = np.sum(pred_bin * gt_bin)
    fp = np.sum(pred_bin * (1 - gt_bin))
    fn = np.sum((1 - pred_bin) * gt_bin)

    dsc = (2.0 * tp) / (2.0 * tp + fp + fn + eps) if (2.0 * tp + fp + fn) > 0 else 1.0
    ppv = tp / (tp + fp + eps) if (tp + fp) > 0 else 0.0
    sensitivity = tp / (tp + fn + eps) if (tp + fn) > 0 else 1.0
    return float(dsc), float(ppv), float(sensitivity)


def compute_lesion_metrics(pred_array, gt_array, pet_array, eps=1e-7):
    """Lesion-level detection metrics (6-connected; >=10% GT lesion overlap & covers GT SUVmax voxel)."""
    pred_bin = (pred_array > 0).astype(np.uint8)
    gt_bin = (gt_array > 0).astype(np.uint8)

    pred_labeled, num_pred = label(pred_bin)
    gt_labeled, num_gt = label(gt_bin)

    if num_gt == 0 and num_pred == 0:
        return 1.0, 1.0, 0, 0, 0, 0, 0

    gt_lesions = {}
    for g_id in range(1, num_gt + 1):
        g_mask = gt_labeled == g_id
        g_size = np.sum(g_mask)
        g_pet_vals = np.where(g_mask, pet_array, -np.inf)
        max_idx = np.unravel_index(np.argmax(g_pet_vals), pet_array.shape)
        gt_lesions[g_id] = {
            "mask": g_mask,
            "size": g_size,
            "max_voxel_coord": max_idx,
        }

    matched_pred_indices = set()
    matched_gt_indices = set()

    for p_id in range(1, num_pred + 1):
        p_mask = pred_labeled == p_id
        for g_id, g_info in gt_lesions.items():
            intersection = np.sum(p_mask & g_info["mask"])
            cond1 = intersection >= (0.1 * g_info["size"])
            cond2 = bool(p_mask[g_info["max_voxel_coord"]])
            if cond1 and cond2:
                matched_pred_indices.add(p_id)
                matched_gt_indices.add(g_id)

    tp_lesion = len(matched_pred_indices)
    fp_lesion = num_pred - tp_lesion
    fn_lesion = num_gt - len(matched_gt_indices)

    lesion_ppv = tp_lesion / (num_pred + eps) if num_pred > 0 else (1.0 if num_gt == 0 else 0.0)
    lesion_sens = len(matched_gt_indices) / (num_gt + eps) if num_gt > 0 else 1.0
    return float(lesion_ppv), float(lesion_sens), int(num_pred), int(num_gt), int(tp_lesion), int(fp_lesion), int(fn_lesion)


def compute_pet_quantitative_parameters(pet_array, mask_array, spacing):
    """SUVmean, SUVmax, PSMA-TV (ml), and TL-PSMA."""
    v_voxel_ml = (spacing[0] * spacing[1] * spacing[2]) / 1000.0
    roi_voxels = pet_array[mask_array > 0]
    total_voxels = roi_voxels.size

    if total_voxels == 0:
        return 0.0, 0.0, 0.0, 0.0

    suv_mean = float(np.mean(roi_voxels))
    suv_max = float(np.max(roi_voxels))
    psma_tv = float(v_voxel_ml * total_voxels)
    tl_psma = float(psma_tv * suv_mean)
    return suv_mean, suv_max, psma_tv, tl_psma


def extract_case_id(filename: str) -> str:
    """Extract case ID by stripping known prediction NIfTI suffixes."""
    for suffix in ("_PET_seg.nii.gz", "_seg.nii.gz", "_PET.nii.gz", ".nii.gz"):
        if filename.endswith(suffix):
            return filename[: -len(suffix)]
    return Path(filename).stem


def find_matching_file(target_dir: Path, base_id: str, role: str):
    if role == "pet":
        suffixes = ("_PET.nii.gz", ".nii.gz")
    else:
        suffixes = ("_ground_truth.nii.gz", "_MERGED_MASK.nii.gz", "_TTB.nii.gz", ".nii.gz")

    for suffix in suffixes:
        candidate = target_dir / f"{base_id}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def check_geometry(pred_itk, gt_itk, pet_itk):
    for name, image in (("GT", gt_itk), ("PET", pet_itk)):
        if pred_itk.GetDimension() != image.GetDimension():
            raise ValueError(f"{name} dimension mismatch")
        if pred_itk.GetSize() != image.GetSize():
            raise ValueError(f"{name} size mismatch")
        for attr in ("GetSpacing", "GetOrigin", "GetDirection"):
            if not np.allclose(getattr(pred_itk, attr)(), getattr(image, attr)(), rtol=0, atol=1e-5):
                raise ValueError(f"{name} geometry mismatch: {attr}")


def main(args):
    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)
    pet_dir = Path(args.pet_dir) if args.pet_dir else gt_dir
    output_csv = Path(args.output_csv)

    pred_files = sorted(pred_dir.glob("*.nii.gz"))
    if not pred_files:
        raise FileNotFoundError(f"No .nii.gz predictions found in {pred_dir}")

    header = (
        f"{'Scan_ID':<12} | {'Pred#':<6} | {'GT#':<5} | {'L-TP':<5} | {'L-FP':<5} | {'L-FN':<5} | "
        f"{'DSC':<8} | {'V-PPV':<8} | {'V-Sens':<8} | {'L-PPV':<8} | {'L-Sens':<8} | "
        f"{'Pred SUVmean':<12} | {'GT SUVmean':<11} | {'Pred SUVmax':<11} | {'GT SUVmax':<10} | "
        f"{'Pred TV(ml)':<11} | {'GT TV(ml)':<10} | {'ARVM':<7} | {'SUVm SRE':<8}"
    )
    print(header)
    print("-" * len(header))

    seen_ids = set()
    results = []
    eps = 1e-7

    for pred_file in pred_files:
        patient_id = extract_case_id(pred_file.name)
        scan_id = patient_id.replace("patient", "scan")

        if patient_id in seen_ids:
            continue
        seen_ids.add(patient_id)

        gt_file = find_matching_file(gt_dir, patient_id, "gt")
        pet_file = find_matching_file(pet_dir, patient_id, "pet")
        if gt_file is None:
            raise FileNotFoundError(f"Missing GT file for {patient_id} in {gt_dir}")
        if pet_file is None:
            raise FileNotFoundError(f"Missing PET file for {patient_id} in {pet_dir}")

        pred_itk = sitk.ReadImage(str(pred_file))
        gt_itk = sitk.ReadImage(str(gt_file))
        pet_itk = sitk.ReadImage(str(pet_file))
        check_geometry(pred_itk, gt_itk, pet_itk)

        pred_img = (sitk.GetArrayFromImage(pred_itk) > 0).astype(np.uint8)
        gt_raw = sitk.GetArrayFromImage(gt_itk)
        gt_img = ((gt_raw == 1) if "MERGED_MASK" in gt_file.name else (gt_raw > 0)).astype(np.uint8)
        pet_img = sitk.GetArrayFromImage(pet_itk)
        pet_spacing = pet_itk.GetSpacing()

        dsc, v_ppv, v_sens = compute_segmentation_metrics(pred_img, gt_img, eps=eps)
        l_ppv, l_sens, num_pred, num_gt, l_tp, l_fp, l_fn = compute_lesion_metrics(pred_img, gt_img, pet_img, eps=eps)

        p_mean, p_max, p_tv, p_tl = compute_pet_quantitative_parameters(pet_img, pred_img, pet_spacing)
        g_mean, g_max, g_tv, g_tl = compute_pet_quantitative_parameters(pet_img, gt_img, pet_spacing)

        arvm = abs(p_tv - g_tv) / (g_tv + eps) if g_tv > 0 else 0.0
        suvm_sre = (p_mean - g_mean) / (g_mean + eps) if g_mean > 0 else 0.0

        results.append(
            {
                "id": scan_id,
                "num_pred": num_pred,
                "num_gt": num_gt,
                "l_tp": l_tp,
                "l_fp": l_fp,
                "l_fn": l_fn,
                "dsc": dsc,
                "v_ppv": v_ppv,
                "v_sens": v_sens,
                "l_ppv": l_ppv,
                "l_sens": l_sens,
                "p_mean": p_mean,
                "g_mean": g_mean,
                "p_max": p_max,
                "g_max": g_max,
                "p_tv": p_tv,
                "g_tv": g_tv,
                "p_tl": p_tl,
                "g_tl": g_tl,
                "arvm": arvm,
                "suvm_sre": suvm_sre,
            }
        )

        print(
            f"{scan_id:<12} | {num_pred:<6d} | {num_gt:<5d} | {l_tp:<5d} | {l_fp:<5d} | {l_fn:<5d} | "
            f"{dsc:<8.4f} | {v_ppv:<8.4f} | {v_sens:<8.4f} | {l_ppv:<8.4f} | {l_sens:<8.4f} | "
            f"{p_mean:<12.4f} | {g_mean:<11.4f} | {p_max:<11.4f} | {g_max:<10.4f} | "
            f"{p_tv:<11.4f} | {g_tv:<10.4f} | {arvm:<7.4f} | {suvm_sre:<8.4f}"
        )

    if not results:
        print("No valid cases evaluated.")
        return

    def get_stat(key):
        vals = [r[key] for r in results]
        return float(np.mean(vals)), float(np.std(vals)), float(np.min(vals)), float(np.max(vals))

    def get_median_iqr(key):
        vals = [r[key] for r in results]
        return float(np.median(vals)), float(np.percentile(vals, 25)), float(np.percentile(vals, 75))

    print("-" * len(header))
    print(f"Total Evaluated: {len(results)} cases\n")
    print("Lesion Counts & Confusion Matrix Statistics:")
    print(f"  Total Lesions        : Pred = {sum(r['num_pred'] for r in results)} | GT = {sum(r['num_gt'] for r in results)}")
    print(
        f"  Total Lesion Matrix  : TP = {sum(r['l_tp'] for r in results)} | "
        f"FP = {sum(r['l_fp'] for r in results)} | FN = {sum(r['l_fn'] for r in results)}\n"
    )

    print("Voxel-Level Segmentation Metrics:")
    for label_name, k in [("DSC", "dsc"), ("PPV", "v_ppv"), ("Sensitivity", "v_sens")]:
        m, s, mn, mx = get_stat(k)
        med, q1, q3 = get_median_iqr(k)
        print(f"  {label_name:<21}: Mean±SD = {m:.4f} ± {s:.4f} (Range = {mn:.4f} - {mx:.4f}) | Median(IQR) = {med:.4f} [{q1:.4f}, {q3:.4f}]")

    print("\nLesion-Level Detection Metrics:")
    for label_name, k in [("Lesion PPV", "l_ppv"), ("Lesion Sensitivity", "l_sens")]:
        m, s, mn, mx = get_stat(k)
        med, q1, q3 = get_median_iqr(k)
        print(f"  {label_name:<21}: Mean±SD = {m:.4f} ± {s:.4f} (Range = {mn:.4f} - {mx:.4f}) | Median(IQR) = {med:.4f} [{q1:.4f}, {q3:.4f}]")

    print("\nPET Quantitative Parameters (Pred vs GT):")
    for name, pk, gk in [
        ("SUVmean", "p_mean", "g_mean"),
        ("SUVmax", "p_max", "g_max"),
        ("PSMA-TV(ml)", "p_tv", "g_tv"),
        ("TL-PSMA", "p_tl", "g_tl"),
    ]:
        pm, ps, pmin, pmax = get_stat(pk)
        pmed, pq1, pq3 = get_median_iqr(pk)
        gm, gs, gmin, gmax = get_stat(gk)
        gmed, gq1, gq3 = get_median_iqr(gk)
        print(f"  {name:<21}: Pred = Mean±SD = {pm:.4f} ± {ps:.4f} (Range = {pmin:.4f} - {pmax:.4f}) | Median(IQR) = {pmed:.4f} [{pq1:.4f}, {pq3:.4f}]")
        print(f"  {'':<21}  GT   = Mean±SD = {gm:.4f} ± {gs:.4f} (Range = {gmin:.4f} - {gmax:.4f}) | Median(IQR) = {gmed:.4f} [{gq1:.4f}, {gq3:.4f}]")

    print("\nRelative Error Statistics:")
    for name, k in [("ARVM", "arvm"), ("SUVmean SRE", "suvm_sre")]:
        m, s, mn, mx = get_stat(k)
        med, q1, q3 = get_median_iqr(k)
        print(f"  {name:<21}: Mean±SD = {m:.4f} ± {s:.4f} (Range = {mn:.4f} - {mx:.4f}) | Median(IQR) = {med:.4f} [{q1:.4f}, {q3:.4f}]")

    csv_fieldnames = [
        "Scan_ID",
        "Pred_Lesions",
        "GT_Lesions",
        "Lesion_TP",
        "Lesion_FP",
        "Lesion_FN",
        "DSC",
        "Voxel_PPV",
        "Voxel_Sensitivity",
        "Lesion_PPV",
        "Lesion_Sensitivity",
        "Pred_SUVmean",
        "GT_SUVmean",
        "Pred_SUVmax",
        "GT_SUVmax",
        "Pred_PSMA_TV_ml",
        "GT_PSMA_TV_ml",
        "Pred_TL_PSMA",
        "GT_TL_PSMA",
        "ARVM_Absolute_Relative_Volume_Metric",
        "SUVmean_Signed_Relative_Error",
    ]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "Scan_ID": r["id"],
                    "Pred_Lesions": r["num_pred"],
                    "GT_Lesions": r["num_gt"],
                    "Lesion_TP": r["l_tp"],
                    "Lesion_FP": r["l_fp"],
                    "Lesion_FN": r["l_fn"],
                    "DSC": f"{r['dsc']:.4f}",
                    "Voxel_PPV": f"{r['v_ppv']:.4f}",
                    "Voxel_Sensitivity": f"{r['v_sens']:.4f}",
                    "Lesion_PPV": f"{r['l_ppv']:.4f}",
                    "Lesion_Sensitivity": f"{r['l_sens']:.4f}",
                    "Pred_SUVmean": f"{r['p_mean']:.4f}",
                    "GT_SUVmean": f"{r['g_mean']:.4f}",
                    "Pred_SUVmax": f"{r['p_max']:.4f}",
                    "GT_SUVmax": f"{r['g_max']:.4f}",
                    "Pred_PSMA_TV_ml": f"{r['p_tv']:.4f}",
                    "GT_PSMA_TV_ml": f"{r['g_tv']:.4f}",
                    "Pred_TL_PSMA": f"{r['p_tl']:.4f}",
                    "GT_TL_PSMA": f"{r['g_tl']:.4f}",
                    "ARVM_Absolute_Relative_Volume_Metric": f"{r['arvm']:.4f}",
                    "SUVmean_Signed_Relative_Error": f"{r['suvm_sre']:.4f}",
                }
            )
    print(f"\n[CSV Saved] {output_csv}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate voxel-level, lesion-level, and PET quantitative metrics for PSMA PET/CT segmentation"
    )
    parser.add_argument("--pred_dir", type=str, required=True, help="Directory with predicted *_PET_seg.nii.gz")
    parser.add_argument("--gt_dir", type=str, required=True, help="Directory with ground truth or MERGED_MASK NIfTIs")
    parser.add_argument(
        "--pet_dir",
        type=str,
        default=None,
        help="Directory with SUVbw *_PET.nii.gz (defaults to --gt_dir if omitted)",
    )
    parser.add_argument("--output_csv", type=str, default="./outputs/evaluation_metrics.csv")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
