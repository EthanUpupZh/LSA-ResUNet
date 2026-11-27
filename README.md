# LSA-ResUNet

Source code for **"LSA-ResUNet: Lightweight Spatial Attention Residual UNet with Transfer Learning for Multicenter PSMA PET/CT Prostate Tumor Segmentation"** (implemented with PyTorch and MONAI).

---

## 1. Repository Structure

```text
LSA-ResUNet/
├── LSA.py                    # 3D LSA-ResUNet architecture (LSAResUNet, DRU, LSA module)
├── pretrain.py               # Supervised 3D pretraining on DEEP-PSMA (PET+CT)
├── finetune.py               # Fine-tuning on FAHSU-PSMA with Organ CutMix
├── inference.py              # Sliding-window inference (supports single or multi-fold checkpoints)
├── evaluation.py             # Voxel-level, lesion-level, and PET quantitative evaluation
├── requirements.txt          # Tested Python package dependencies
├── LICENSE                   # Apache-2.0 License
├── assets/
│   └── data_mask_overview.png    # Visual overview of input PET/CT, MERGED_MASK, and predictions
├── weights/
│   └── lsa_resunet_finetuned.pth # Fine-tuned LSA-ResUNet checkpoint (~19 MB)
└── demo_data/
    └── TCIA_sample/              # 2 public sample cases from TCIA PSMA-PET-CT-Lesions
```

---

## 2. Environment Setup

Tested with Python 3.10 and CUDA-enabled PyTorch (please install a CUDA-compatible `torch` matching your GPU driver first if on Windows):

```bash
pip install -r requirements.txt
```

---

## 3. Quick-Start Demo (Out-of-the-Box)

Two public test cases (`sample1`, <sup>18</sup>F-PSMA; `sample2`, <sup>68</sup>Ga-PSMA) from **The Cancer Imaging Archive (TCIA) PSMA-PET-CT-Lesions** collection are included in `./demo_data/TCIA_sample/` for immediate verification of inference and quantitative evaluation:

```bash
# 1. Run sliding-window inference with the released checkpoint
python inference.py --data_dir ./demo_data/TCIA_sample --weights ./weights/lsa_resunet_finetuned.pth --output_dir ./outputs/demo_preds

# 2. Compute voxel-level, lesion-level, and PET quantitative biomarkers
python evaluation.py --pred_dir ./outputs/demo_preds --gt_dir ./demo_data/TCIA_sample --output_csv ./outputs/demo_metrics.csv
```

**Expected Output Files**:

`./outputs/demo_preds/sample1_PET_seg.nii.gz`, `./outputs/demo_preds/sample2_PET_seg.nii.gz` (binary 3D NIfTI masks preserving input PET affine/header).

`./outputs/demo_metrics.csv` (21-column CSV report).

**Example Console Output**:

```text
Scan_ID | Pred# | GT# | L-TP | L-FP | L-FN | DSC    | V-PPV  | V-Sens | L-PPV  | L-Sens
sample1 | 3     | 1   | 1    | 2    | 0    | 0.8511 | 0.7672 | 0.9555 | 0.3333 | 1.0000
sample2 | 1     | 1   | 1    | 0    | 0    | 0.8395 | 0.8019 | 0.8808 | 1.0000 | 1.0000

Total Evaluated: 2 cases | Mean DSC = 0.8453 ± 0.0058
[CSV Saved] ./outputs/demo_metrics.csv
```

---

## 4. Data Structure & Mask Conventions

Datasets must be **pre-split into `train/`, `val/`, and `test/` subdirectories in advance** (the training scripts read these folders directly and do not perform internal dataset splitting). Within each split, paired 3D NIfTI volumes are matched strictly by `<case_id>` prefix:

```text
data/
├── DEEP-PSMA/
│   ├── train/  (<case_id>_PET.nii.gz, <case_id>_CT.nii.gz, <case_id>_TTB.nii.gz)
│   └── val/
└── FAHSU-PSMA/
    ├── train/  (<case_id>_PET.nii.gz, <case_id>_CT.nii.gz, <case_id>_MERGED_MASK.nii.gz)
    ├── val/
    └── test/
```

### Sample Data & Mask Overview

![Sample Data and Mask Overview](./assets/data_mask_overview.png)

*Representative axial slices from the two TCIA demo cases, `sample1` with <sup>18</sup>F-PSMA and `sample2` with <sup>68</sup>Ga-PSMA. **(A)** PSMA PET; **(B)** Registered CT; **(C)** Merged organ and tumour mask; **(D)** Ground-truth vs. single-model prediction, with `3D DSC` computed over the full volume.*

### Mask & Augmentation Details

**Organ & Tumour Mask Merging (`MERGED_MASK`)**:

Organ masks (`OM`) are derived from CT using [TotalSegmentator](https://github.com/wasserth/TotalSegmentator), with the **prostate mask excluded** during preprocessing.

Ground-truth tumour masks (`TM`, `label == 1`) take priority over overlapping pelvic organs (including the urinary bladder, `label == 2`, and pelvic bones/structures, `label > 2`), overwriting overlapping voxels **directly voxel-by-voxel without morphological dilation**.

During fine-tuning, `RandCropByPosNegLabeld` samples `32×32×32` patches using `MERGED_MASK > 0` as foreground, while the loss and validation metrics evaluate strictly on `seg == 1` (tumour).

**Organ CutMix (`finetune.py`)**:

Applied via MONAI `CutMixd(keys=["img"], batch_size=64, alpha=0.3)` exclusively to the **PET** patches sampled using `MERGED_MASK`, while keeping paired **CT** volumes and **tumour masks (`seg`)** unaltered.

**Seed Control**:

Both `pretrain.py` and `finetune.py` support `--aug_mode fixed --seed <N>` (deterministic reproducibility) and `--aug_mode random` (random seed logged automatically to `run_info`).

---

## 5. Training, Inference & Evaluation

### Pretraining (DEEP-PSMA)

```bash
python pretrain.py --data_dir ./data/DEEP-PSMA --output_dir ./outputs/pretrain
```

Optional flags: `--aug_mode random` (or `fixed --seed 42`), `--epochs 400`, `--batch_size 2`, `--num_samples 128`, `--lr 1e-3`, `--gpu 0`.

**Outputs** (saved in `./outputs/pretrain/`):

`best_model_segmentation3d.pth` (best validation DSC checkpoint).

`train_log_<timestamp>.txt` (epoch loss and validation DSC every 2 epochs).

`run_info_<timestamp>.txt` (random seed, `aug_mode`, hyperparameters, and run summary).

### Fine-tuning (FAHSU-PSMA)

```bash
python finetune.py --data_dir ./data/FAHSU-PSMA --pretrained_path ./outputs/pretrain/best_model_segmentation3d.pth --output_dir ./outputs/finetune
```

Optional flags: `--aug_mode fixed --seed 1076250484` (or `random`), `--epochs 400`, `--batch_size 2`, `--num_samples 32`, `--lr 1e-3`, `--gpu 0`.

**Outputs** (saved in `./outputs/finetune/`):

`best_metric_model_segmentation3d_dict.pth` (best validation DSC checkpoint).

`train_log.txt` (step/epoch training loss and validation DSC every 2 epochs).

`run_info.txt` (random seed, `aug_mode`, pretrained checkpoint path, and dataset counts).

### Inference (Single Checkpoint or Optional Multi-Fold Ensemble)

```bash
# Single checkpoint inference
python inference.py --data_dir ./data/FAHSU-PSMA/test --weights ./weights/lsa_resunet_finetuned.pth --output_dir ./outputs/preds_local

# Optional: 5-fold ensemble inference (averages sigmoid probabilities across folds)
python inference.py --data_dir ./data/TCIA/test --weights ./outputs/fold0.pth ./outputs/fold1.pth ./outputs/fold2.pth ./outputs/fold3.pth ./outputs/fold4.pth --output_dir ./outputs/preds_tcia_ensemble
```

**Outputs** (saved in `--output_dir`, e.g., `./outputs/preds_local/`): `<case_id>_PET_seg.nii.gz` binary segmentation masks matching the input PET geometry.

### Evaluation (Voxel-Level, Lesion-Level & PET Quantitative Biomarkers)

```bash
# When *_PET.nii.gz and *_MERGED_MASK.nii.gz are in the same test directory:
python evaluation.py --pred_dir ./outputs/preds_local --gt_dir ./data/FAHSU-PSMA/test --output_csv ./outputs/evaluation_metrics_local.csv

# Or with separate --gt_dir and --pet_dir:
python evaluation.py --pred_dir ./outputs/preds_tcia_ensemble --gt_dir ./data/TCIA/ground_truth --pet_dir ./data/TCIA/pet_images --output_csv ./outputs/evaluation_metrics_tcia.csv
```

**Outputs**: Prints per-case and cohort summary statistics (Mean ± SD, Range, Median [IQR]) to the terminal and saves the 21-column CSV table to `--output_csv` (e.g., `./outputs/evaluation_metrics_local.csv`).
**Evaluation Criteria & Exported CSV (21 Columns)**:

**Lesion Counts & Matrix**: `Scan_ID`, `Pred_Lesions`, `GT_Lesions`, `Lesion_TP`, `Lesion_FP`, `Lesion_FN` (3D 6-connected components; a lesion is a true positive if overlap ≥ 10% of the GT lesion volume **and** the predicted lesion covers the GT lesion's SUV<sub>max</sub> peak voxel).

**Voxel-Level Metrics**: `DSC`, `Voxel_PPV`, `Voxel_Sensitivity`.

**Lesion-Level Metrics**: `Lesion_PPV`, `Lesion_Sensitivity`.

**PET Quantitative Biomarkers & Errors**: `Pred_SUVmean`, `GT_SUVmean`, `Pred_SUVmax`, `GT_SUVmax`, `Pred_PSMA_TV_ml`, `GT_PSMA_TV_ml`, `Pred_TL_PSMA`, `GT_TL_PSMA`, `ARVM_Absolute_Relative_Volume_Metric`, `SUVmean_Signed_Relative_Error`.

---

## 6. Data Availability & License

**Code License**: Released under the [Apache-2.0 License](./LICENSE).

**TCIA PSMA-PET-CT-Lesions** (including the 2 sample cases in `./demo_data/TCIA_sample/`): Publicly available on The Cancer Imaging Archive ([https://www.cancerimagingarchive.net/collection/psma-pet-ct-lesions/](https://www.cancerimagingarchive.net/collection/psma-pet-ct-lesions/)) under CC BY 4.0.

**DEEP-PSMA**: Challenge dataset ([https://deep-psma.grand-challenge.org/](https://deep-psma.grand-challenge.org/)).

**FAHSU-PSMA**: Institutional clinical cohort; de-identified data are available from the corresponding author upon reasonable academic request and ethics approval.
