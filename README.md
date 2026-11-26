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

Two anonymized public test cases (`sample1`, [$^{18}$F]F-PSMA; `sample2`, [$^{68}$Ga]Ga-PSMA) from **The Cancer Imaging Archive (TCIA) PSMA-PET-CT-Lesions** collection are included in `./demo_data/TCIA_sample/` for immediate verification of inference and quantitative evaluation:

```bash
# 1. Run sliding-window inference with the released checkpoint
python inference.py \
  --data_dir ./demo_data/TCIA_sample \
  --weights ./weights/lsa_resunet_finetuned.pth \
  --output_dir ./outputs/demo_preds \
  --gpu 0

# 2. Compute voxel-level, lesion-level, and PET quantitative biomarkers
python evaluation.py \
  --pred_dir ./outputs/demo_preds \
  --gt_dir ./demo_data/TCIA_sample \
  --output_csv ./outputs/demo_metrics.csv
```

---

## 4. Data Structure & Mask Conventions

Each dataset split (`train/`, `val/`, `test/`) stores paired 3D NIfTI volumes matched strictly by `<case_id>` prefix:

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

*Figure: Representative axial slices from the two included anonymized public TCIA demo cases (`sample1`, $^{18}$F-PSMA; `sample2`, $^{68}$Ga-PSMA). **(A)** Input PSMA PET (`*_PET.nii.gz`, SUV 0–5); **(B)** Registered CT (`*_CT.nii.gz`); **(C)** `*_MERGED_MASK.nii.gz` overlaid on CT, showing ground-truth tumour (`label = 1`, red, priority overwrite without dilation) and non-tumour pelvic organs/bones (`label >= 2`, including urinary bladder and pelvic bones with prostate excluded, teal); **(D)** Ground-truth tumour contour (`label == 1`, solid green) compared against single-model `lsa_resunet_finetuned.pth` prediction (`*_PET_seg.nii.gz`, dashed red).*

### Mask & Augmentation Details
* **Organ & Tumour Mask Merging (`MERGED_MASK`)**:
  * Organ masks (`OM`) are derived from CT using [TotalSegmentator](https://github.com/wasserth/TotalSegmentator), with the **prostate mask excluded** during preprocessing.
  * Ground-truth tumour masks (`TM`, `label == 1`) take priority over overlapping pelvic organs (including the urinary bladder, `label == 2`, and pelvic bones/structures, `label > 2`), overwriting overlapping voxels **directly voxel-by-voxel without morphological dilation**.
  * During fine-tuning, `RandCropByPosNegLabeld` samples $32^3$ patches using `MERGED_MASK > 0` as foreground, while the loss and validation metrics evaluate strictly on `seg == 1` (tumour).
* **Organ CutMix (`finetune.py`)**:
  * Applied via MONAI `CutMixd(keys=["img"], batch_size=64, alpha=0.3)` exclusively to the **PET** patches sampled using `MERGED_MASK`, while keeping paired **CT** volumes and **tumour masks (`seg`)** unaltered.
* **Seed Control**:
  * Both `pretrain.py` and `finetune.py` support `--aug_mode fixed --seed <N>` (deterministic reproducibility) and `--aug_mode random` (random seed logged automatically to `run_info`).

---

## 5. Training, Inference & Evaluation

### Pretraining (DEEP-PSMA)
```bash
# Default random seed mode (or pass --aug_mode fixed --seed 42 for deterministic runs)
python pretrain.py \
  --data_dir ./data/DEEP-PSMA \
  --output_dir ./outputs/pretrain \
  --aug_mode random \
  --epochs 400 --batch_size 2 --num_samples 128 --lr 1e-3 --gpu 0
```
* **Outputs**: `best_model_segmentation3d.pth`, `train_log_<timestamp>.txt`, `run_info_<timestamp>.txt`.

### Fine-tuning (FAHSU-PSMA)
```bash
python finetune.py \
  --data_dir ./data/FAHSU-PSMA \
  --pretrained_path ./outputs/pretrain/best_model_segmentation3d.pth \
  --output_dir ./outputs/finetune \
  --aug_mode fixed --seed 1076250484 \
  --epochs 400 --batch_size 2 --num_samples 32 --lr 1e-3 --gpu 0
```
* **Outputs**: `best_metric_model_segmentation3d_dict.pth`, `train_log.txt`, `run_info.txt`.

### Inference (Single Checkpoint or Optional Multi-Fold Ensemble)
```bash
# Single checkpoint inference (using the released weight)
python inference.py \
  --data_dir ./data/FAHSU-PSMA/test \
  --weights ./weights/lsa_resunet_finetuned.pth \
  --output_dir ./outputs/preds_local \
  --gpu 0

# Optional: 5-fold ensemble inference (if you train 5 cross-validation folds)
python inference.py \
  --data_dir ./data/TCIA/test \
  --weights ./outputs/fold0.pth ./outputs/fold1.pth ./outputs/fold2.pth ./outputs/fold3.pth ./outputs/fold4.pth \
  --output_dir ./outputs/preds_tcia_ensemble \
  --gpu 0
```

### Evaluation (Voxel-Level, Lesion-Level & PET Quantitative Biomarkers)
```bash
# If *_PET.nii.gz and *_MERGED_MASK.nii.gz are in the same test directory:
python evaluation.py \
  --pred_dir ./outputs/preds_local \
  --gt_dir ./data/FAHSU-PSMA/test \
  --output_csv ./outputs/evaluation_metrics_local.csv

# Or specify separate --gt_dir and --pet_dir:
python evaluation.py \
  --pred_dir ./outputs/preds_tcia_ensemble \
  --gt_dir ./data/TCIA/ground_truth \
  --pet_dir ./data/TCIA/pet_images \
  --output_csv ./outputs/evaluation_metrics_tcia.csv
```
* **Evaluation Criteria & Exported CSV (21 Columns)**:
  * **Lesion Counts & Matrix**: `Scan_ID`, `Pred_Lesions`, `GT_Lesions`, `Lesion_TP`, `Lesion_FP`, `Lesion_FN` (3D 6-connected components; a lesion is a true positive if overlap $\ge 10\%$ of the GT lesion volume **and** the predicted lesion covers the GT lesion's $\text{SUV}_{\max}$ peak voxel).
  * **Voxel-Level Metrics**: `DSC`, `Voxel_PPV`, `Voxel_Sensitivity`.
  * **Lesion-Level Metrics**: `Lesion_PPV`, `Lesion_Sensitivity`.
  * **PET Quantitative Biomarkers & Errors**: `Pred_SUVmean`, `GT_SUVmean`, `Pred_SUVmax`, `GT_SUVmax`, `Pred_PSMA_TV_ml`, `GT_PSMA_TV_ml`, `Pred_TL_PSMA`, `GT_TL_PSMA`, `ARVM_Absolute_Relative_Volume_Metric`, `SUVmean_Signed_Relative_Error`.

---

## 6. Released Checkpoint & Benchmark Results

This repository releases a representative single-model checkpoint (`./weights/lsa_resunet_finetuned.pth`, ~19 MB). Note that for fair comparison against 5-fold `nnU-Net` in Table 5 of the paper, 5-fold cross-validation probability averaging was evaluated:

| Configuration | Checkpoint | Internal FAHSU-PSMA (n=12) DSC | External TCIA (n=52) DSC |
| :--- | :--- | :---: | :---: |
| **LSA-ResUNet (Single Model, Released)** | `weights/lsa_resunet_finetuned.pth` | 0.790 | 0.681 |
| **LSA-ResUNet (5-Fold Ensemble, Paper Table 5)** | 5-fold CV ensemble | **0.795** | **0.691** |

---

## 7. Data Availability & License

* **Code License**: Released under the [Apache-2.0 License](./LICENSE).
* **Public Datasets**:
  * **TCIA PSMA-PET-CT-Lesions** (including the 2 sample cases in `./demo_data/TCIA_sample/`): Publicly available on The Cancer Imaging Archive ([https://www.cancerimagingarchive.net/collection/psma-pet-ct-lesions/](https://www.cancerimagingarchive.net/collection/psma-pet-ct-lesions/)) under CC BY 4.0.
  * **DEEP-PSMA**: Challenge dataset ([https://deep-psma.grand-challenge.org/](https://deep-psma.grand-challenge.org/)).
  * **FAHSU-PSMA**: Institutional clinical cohort; de-identified data are available from the corresponding author upon reasonable academic request and ethics approval.
