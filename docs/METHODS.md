# Definitions used in the code

## Leakage relations (script 01)

| Relation | Rule | Column in `per_image.csv` |
|---|---|---|
| Exact duplicate | identical MD5 of the image file | `md5` |
| Export copy | same source-frame name after removing the Roboflow suffix `.rf.<hash>` | `source_name`, `source_in_train` |
| Near-duplicate | 64-bit DCT pHash (32 × 32 grey, INTER_AREA; 8 × 8 low-frequency DCT block; bit = coefficient > median excluding DC). Candidate if d_H ≤ 8; confirmed if d_H ≤ 4, or 5 ≤ d_H ≤ 8 and SSIM ≥ 0.80 on 128 × 128 grey thumbnails | `nn_phash_to_train`, `has_dup_in_train`, `n_confirmed_dups` |
| Shared video | same video identifier (file-name prefix) | `seq_prefix`, `seq_prefix_in_train` |

Thresholds: `PHASH_CANDIDATE_THR = 8`, `PHASH_STRICT_THR = 4`, `SSIM_CONFIRM_THR = 0.80`, `THUMB = 128`.

## Split construction (script 02)

1. Remove exact duplicates; keep one export copy per source frame (brightness closest to the median of its copies).
2. Clean labels: polygons → enclosing box; clip to the image; drop boxes < 2 px; merge identical boxes.
3. Union–find groups over confirmed near-duplicates, shared source frames and shared videos.
4. Randomised greedy assignment of whole groups to 70:20:10 (3,000 orderings), balancing images,
   positive images, boxes and small boxes (√area < 32 px). Result: 2,927 / 867 / 425 images.

## Test subsets (scripts 05, 06)

| Name | Definition |
|---|---|
| A-test | published test set (662 images) |
| Leaked | A-test images with a confirmed near-duplicate or shared source frame in training (259) |
| B-clean | all other A-test images (403); still share videos with training |
| B-strict | 337-image sensitivity variant that also excludes 66 images with confirmed near-duplicates in validation or test |
| C-test | test set of the video-disjoint split (425) |

## Evaluation (script 05; identical evaluator in 06)

* Predictions: confidence 0.001, NMS IoU 0.7, ≤ 300 detections, input 640 px.
* Ground truth: same cleaning rules as above, applied to every test image.
* Matching: greedy by confidence at IoU 0.50:0.05:0.95; AP by all-point interpolation.
* R: recall at the confidence threshold that maximises F1 at IoU 0.50.
* Uncertainty: cluster bootstrap over videos (1,000 resamples; RDD images resampled individually),
  percentile intervals; paired differences use the same resample for both subsets.
* Contamination test: a share ρ of the 134 positive C-test images is replaced, without replacement,
  by export copies of training frames (726, one per source frame) or of unseen C-test frames (112);
  each copy keeps its own exported annotation; 20 draws per level.
