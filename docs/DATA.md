# Data sources and Drive layout

## Sources (not redistributed)

| Dataset | Source | Used as |
|---|---|---|
| BharatPotHole (7,074 images, 640 × 640, one class) | Kaggle: https://www.kaggle.com/datasets/surbhisaswatimohanty/bharatpothole — Sahoo RR, Mohanty SS, Mishra S (2025) iWatchRoadv2, arXiv:2510.16375 | Training, validation and internal test |
| RDD2022, India subset, class D40 (pothole) | Arya D, Maeda H, Ghosh SK, Toshniwal D, Sekimoto Y (2024) Geosci Data J 11(4):846–862, https://doi.org/10.1002/gdj3.260 | External test (1,530 images, 3,187 boxes) |

Check the licence of each dataset before reuse. This repository releases only file names, hashes,
split assignments, audit tables, predictions and metrics.

## Expected Google Drive layout

```
MyDrive/POTHOLE/
  archive (1).zip                       BharatPotHole as downloaded from Kaggle (never modified)
  RDD2022 India Pothole D40.zip         RDD2022 India subset (any of VOC XML / YOLO / COCO annotations)
  scripts/                              the files in scripts/ of this repository
  dataset_assessment/                   01: per_image.csv, leakage_summary.csv, near-duplicate pairs, ...
  BharatPotHole_v2/                     02: manifest_v2.csv, BharatPotHole_v2.zip, B_original_test_clean.txt
  rdd_india_assessment/                 03: RDD_India_D40_yolo.zip, domain-shift tables
  experiments/family_study_v2/          04: runs/yolo26n_s0, runs/yolo26s_s0, pretrained/
  experiments/leakage_paper/            05: runs/yolo26n_A_s0, cache/*.pkl, results/
                                        06: revision/      07: publication/figures/
  release/                              08: bharatpothole-leakage-artifacts(.zip)
```

## Release archive (`08_export_release.py`)

| File | Content |
|---|---|
| `splits/manifest_v2.csv` | Every released file: MD5, video identifier (`seq_prefix`), source frame, original split, new split or drop reason, box counts, d_H to the v2 training set |
| `splits/excluded_images.csv` | Exact duplicates and export copies removed before splitting |
| `splits/a_test_subsets.csv` | Published test set with d_H to training and the Leaked / B-clean / B-strict flags |
| `audit/` | Per-image audit (MD5, size, brightness, blur, labels, near-duplicate distances, flags) |
| `rdd_india/` | RDD2022-India audit and overlap with BharatPotHole |
| `results/` | All result tables (T0–T5 from script 05, R1–R6 from script 06) |
| `predictions/` | Cached per-image predictions (`{model}_{protocol}.pkl`: normalised xyxy boxes, confidences) |
| `SHA256SUMS.txt` | Checksums of all files |

`B_original_test_clean.txt` defines **B-strict** (337 images). **B-clean** (403 images) follows the rule
stated in the article and is recomputed from `per_image.csv` (`has_dup_in_train` or `source_in_train`).
