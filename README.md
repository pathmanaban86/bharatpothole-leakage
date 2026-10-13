# How much does data leakage inflate road-defect detection performance?

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23337738.svg)](https://doi.org/10.5281/zenodo.23337738)

Code, split definitions and analysis for the article

> Pugazhendi PP. *How much does data leakage inflate road-defect detection performance? Evidence from a
> re-split of BharatPotHole.* Submitted to *Data Science for Transportation* (Springer Nature), 2026.

The study audits the public **BharatPotHole** dataset (7,074 dashboard-video frames, released with
iWatchRoadv2), builds a deduplicated **video-disjoint** split, and measures how train–test leakage changes
the AP<sub>50</sub> of YOLO26 detectors. Everything runs in Google Colab on one T4 GPU; only three
detectors are trained.

## Main results

| Analysis | Result |
|---|---|
| Duplication in the release | 86 exact duplicates + 2,769 export copies (40.4%); 1,333 of 4,196 positive images are unique frames |
| Overlap in the published split | 38.8% of test images have a near-duplicate in training; 99.4% come from training videos |
| Same model, positive test images | AP<sub>50</sub> 0.619 on the published test set vs 0.467 on its near-duplicate-free subset: gap 0.152 (95% CI 0.090–0.234) |
| Sensitivity of that gap | 0.152–0.174 across five definitions of the near-duplicate-free subset |
| Controlled contamination test | Copies of training frames raise AP<sub>50</sub> of a video-disjoint YOLO26n from 0.133 to 0.494; copies of unseen frames leave it at 0.149 |
| External test (RDD2022 India, potholes) | AP<sub>50</sub> 0.072 (published split) vs 0.097–0.099 (video-disjoint split) |

Confidence intervals: cluster bootstrap over videos, 1,000 resamples, percentile method.

## Repository layout

```
scripts/
  01_audit_bharatpothole.py        audit: MD5, export copies, pHash + SSIM near-duplicates, videos, labels
  02_build_split_v2.py             deduplication, label cleaning, video-disjoint 70:20:10 split (Protocol C)
  03_audit_rdd_india.py            RDD2022-India audit, D40 (pothole) external test set, overlap check
  04_train_protocol_c.py           YOLO training on the video-disjoint split (resumable, fault tolerant)
  05_protocol_a_and_evaluation.py  YOLO26n on the published split (Protocol A) + evaluation of all models
  06_revision_checks.py            composition, criteria overlap, threshold sensitivity, bootstrap check,
                                   contamination control, final subset results (R1-R6)
  07_make_figures.py               Figures 1, 2, 4, 5, 6 (PDF, 600-dpi TIFF, PNG)
  08_export_release.py             collects split files and result tables (no images) into one zip
  09_visual_evidence.py            Fig. 3 (d_H examples) and supplementary Figs. S1-S6 (leaked pairs,
                                   export copies, predictions, contamination test, annotation conflicts,
                                   domain shift)
                                   and a small labelled sample dataset
  pubstyle.py                      shared figure style, colours and notation
notebooks/
  run_pipeline_colab.ipynb         runs scripts 01-08 in order
results/reference_outputs/         result summaries produced for the article (for comparison)
docs/
  DATA.md                          data sources, Drive layout, what each output file contains
  METHODS.md                       definitions used in the code (leakage relations, subsets, metrics)
```

## Data

Images are **not** included; download them from the original sources (see [docs/DATA.md](docs/DATA.md)):

* **BharatPotHole** – Kaggle: <https://www.kaggle.com/datasets/surbhisaswatimohanty/bharatpothole>
  (Sahoo, Mohanty and Mishra 2025, arXiv:2510.16375). Save the download as `MyDrive/POTHOLE/archive (1).zip`.
* **RDD2022, India subset** – Arya et al. 2024, *Geoscience Data Journal* 11(4):846–862,
  <https://doi.org/10.1002/gdj3.260>. Save as `MyDrive/POTHOLE/RDD2022 India Pothole D40.zip`.

The split itself is released as file lists with MD5 hashes (`splits/` in the release archive, built by
`08_export_release.py`), so it can be rebuilt from the public release without redistributing images.

## Reproducing the results (Google Colab, T4 GPU)

1. Clone this repository (`git clone https://github.com/pathmanaban86/bharatpothole-leakage`) and copy
   `scripts/` to `MyDrive/POTHOLE/scripts/`; place the two data zips as above.
2. Open `notebooks/run_pipeline_colab.ipynb`, or run the steps one by one:

| Step | Command (Colab cell) | GPU time | Main outputs |
|---|---|---|---|
| 1 | `%run "/content/drive/MyDrive/POTHOLE/scripts/01_audit_bharatpothole.py"` | none | `dataset_assessment/per_image.csv` |
| 2 | `%run ".../02_build_split_v2.py"` | none | `BharatPotHole_v2/manifest_v2.csv`, `BharatPotHole_v2.zip` |
| 3 | `%run ".../03_audit_rdd_india.py"` | none | `rdd_india_assessment/RDD_India_D40_yolo.zip` |
| 4 | `!MODELS=yolo26n.pt,yolo26s.pt SEEDS=0 python ".../04_train_protocol_c.py"` | several hours | `experiments/family_study_v2/runs/*_s0/` |
| 5 | `!python ".../05_protocol_a_and_evaluation.py"` | ≈ 3 h (2.7 h training) | `experiments/leakage_paper/results/`, `cache/` |
| 6 | `!python ".../06_revision_checks.py"` | minutes | `experiments/leakage_paper/revision/` |
| 7 | `!python ".../07_make_figures.py"` | none | `experiments/leakage_paper/publication/figures/` |
| 8 | `!python ".../08_export_release.py" --with-predictions` | none | `release/bharatpothole-leakage-artifacts.zip` |
| 9 | `!python ".../09_visual_evidence.py"` | none (CPU) | `publication/supplementary/` (Fig. 3, Figs. S1–S6, Online Resource 2), `release/sample_dataset.zip` |

Steps 4 and 5 resume automatically after a disconnect; re-run the same cell. Step 5 caches every
prediction, so steps 5–8 can be repeated without inference. `POTHOLE_DRIVE=/path` changes the root folder
for scripts 05–08.

### Mapping to the article

| Article | Produced by |
|---|---|
| Table 2, Fig. 1 | `05` → `results/T0_audit_numbers.json`; `07` |
| Table 3, Fig. 2 | `01` (`per_image.csv`), `02` (`manifest_v2.csv`); `07` |
| Table 4 | `05` → `T1_T3_main_results.csv`; B-clean row from `06` → `R6_subsets_all.csv` |
| Fig. 3 | `09` → `publication/supplementary/Fig3_distance_examples` |
| Table 5, Fig. 4 | `06` → `R6_subsets_positive.csv`, `R6_bins_positive.csv`; `07` |
| Table 6, Fig. 5 | `06` → `R5_substitution_controls.csv`; `07` |
| Fig. 6 | `05` → `T1_T3_main_results.csv`; `07` |
| Online Resource 1 | `06` → `REVISION_CHECKS.md` (R1–R4) |
| Online Resource 2 (Figs. S1–S6) | `09` → `publication/supplementary/Online_Resource_2.docx` |

## Supplemental materials

`supplementary/Supplemental_Materials.pdf` holds the supplemental materials of the article:
Tables S1–S4 (sensitivity checks R1–R4) and Figs. S1–S6 (visual evidence of leaked pairs).
The example frames in Figs. S1–S6 are reduced-size excerpts of BharatPotHole images, shown
for scholarly analysis; the dataset itself is not redistributed here.

## Software

Python 3.11+, Ultralytics 8.4.162 (pinned; the scripts install it), PyTorch as provided by Colab,
NumPy, pandas, OpenCV, scikit-image, Matplotlib. See `requirements.txt`.

## Citation

See `CITATION.cff`. Please also cite the BharatPotHole and RDD2022 sources above.

## License

Code: MIT (see `LICENSE`). The datasets keep their original licences and are not redistributed here.
