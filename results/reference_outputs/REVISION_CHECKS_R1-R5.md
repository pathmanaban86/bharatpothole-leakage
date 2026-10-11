# Revision checks (2026-10-11 10:23)
per_image.csv columns: ['md5', 'file_kb', 'width', 'height', 'corrupt', 'mean_luma', 'std_luma', 'blur_laplacian_var', 'rel_path', 'split', 'group', 'stem', 'source_name', 'seq_prefix', 'seq_index', 'n_boxes', 'label_status', 'n_label_issues', 'label_file', 'low_light', 'nn_phash_to_test', 'nn_phash_to_train', 'nn_phash_to_valid', 'has_dup_in_train', 'n_confirmed_dups', 'source_in_train', 'seq_prefix_in_train', 'component', 'new_split']
manifest_v2.csv columns: ['rel_path', 'original_split', 'new_split', 'drop_reason', 'is_rep', 'component', 'seq_prefix', 'source_name', 'md5', 'n_boxes', 'n_small', 'dest_name', 'n_boxes_clean', 'nn_phash_to_train_v2']

## R1 Composition of training and test sets
| set | images | positive_images | boxes | boxes_per_image | small_boxes_% | median_box_px | videos | videos_with_positives | pos_per_video_median | pos_per_video_max | unique_positive_source_frames |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Protocol A train | 5067 | 3062 | 8777 | 1.73 | 60.5 | 26.8 | 153 |  |  |  | 1157 |
| Protocol C train | 2927 | 934 | 2836 | 0.97 | 56.3 | 28.9 | 136 |  |  |  | 934 |
| A-test | 662 | 371 | 1148 | 1.73 | 51.8 | 30.9 | 93 | 53 | 4.0 | 30 |  |
| Leaked | 259 | 246 | 643 | 2.48 | 58.9 | 27.6 | 50 | 40 | 4.0 | 26 |  |
| B-clean | 337 | 59 | 201 | 0.6 | 41.8 | 35.3 | 72 | 29 | 1.0 | 7 |  |
| C-test | 425 | 134 | 381 | 0.9 | 56.4 | 29.6 | 10 | 9 | 4.0 | 57 |  |

A-test images from videos NOT in Protocol A training: 4

## R2 Leakage criteria in A-test (n = 662)
| criterion | images | % |
|---|---|---|
| confirmed near-duplicate (pHash rule: d<=4, or d 5-8 with SSIM>=0.80) | 259 | 39.1 |
| shared source frame | 242 | 36.6 |
| d_H <= 4 | 257 | 38.8 |
| d_H 5-8 | 76 | 11.5 |
| d_H 5-8 confirmed by SSIM | 2 | 0.3 |
| confirmed AND shared source | 242 | 36.6 |
| confirmed only | 17 | 2.6 |
| shared source only | 0 | 0.0 |
| leaked (either) | 259 | 39.1 |
| d_H <= 4 but not leaked | 0 | 0.0 |
| leaked but d_H > 4 | 2 | 0.3 |

A-test images in neither Leaked nor B-clean: 66
  their d_H: min 8, median 12, max 20; d_H<=4: 0, 5-8: 12, >8: 54; positive: 66
  example files: ['20250524114752_0060_frame_00059_jpg.rf.461e3f1a4cd57608a887931071a4c8ce.jpg', '20250524114752_0060_frame_00059_jpg.rf.d686b989f21a329b2f9541eb220b1a90.jpg', '20250524114752_0060_frame_00077_jpg.rf.512ac79d7ccbbc6d7ac587ead07f2341.jpg', '20250524114752_0060_frame_00087_jpg.rf.8cc04a46053f9ab1b21ef0b381d7df6d.jpg', '20250524115152_0060_frame_00019_jpg.rf.eb2ff2fd453e25719cabeb05ccf57368.jpg']
  diagnostics: component shared with a training image = 66; source frame in validation = 38; source frame in training (by name) = 0; nn_phash_to_valid <= 4 = 38; n_confirmed_dups > 0 = 66; seq_prefix_in_train = 66; label_status != ok = 66; n_label_issues > 0 = 0; low_light = 0
  label_status values: {'labelled': 66}
  same diagnostics for B-clean (n=337): component shared with train = 333; label_status != ok = 337; n_confirmed_dups > 0 = 34
B-clean images that are also flagged Leaked: 0

## R3 Sensitivity to the pHash threshold (positive A-test images, YOLO26n Protocol A)
Near-duplicate-free subset at threshold tau: d_H > tau and no shared source frame.
| tau | n_clean_pos | clean_videos | AP50_A | AP50_clean | diff | CI_lo | CI_hi | rel_% |
|---|---|---|---|---|---|---|---|---|
| 2 | 127 | 36 | 0.619 | 0.465 | 0.154 | 0.093 | 0.238 | 33.1 |
| 4 | 125 | 36 | 0.619 | 0.467 | 0.152 | 0.09 | 0.234 | 32.6 |
| 6 | 122 | 34 | 0.619 | 0.46 | 0.159 | 0.099 | 0.245 | 34.6 |
| 8 | 99 | 31 | 0.619 | 0.448 | 0.171 | 0.1 | 0.304 | 38.2 |
| manuscript rule | 59 | 29 | 0.619 | 0.445 | 0.174 | 0.098 | 0.283 | 39.0 |

## R4 Bootstrap check (positive images, manuscript rule)
Videos with positive A-test images: 53
1,000 resamples: discarded (no boxes in B-clean) = 0; CI diff = [0.098, 0.283]
5,000 resamples (seed 1): discarded = 0; CI A = [0.556, 0.704]; CI B = [0.356, 0.536]; CI diff = [0.092, 0.274]; share of resamples with diff <= 0 = 0.0002

## R5 Substitution controls (Protocol C models, C-test)
Pools: copies of training frames (seen by the model) vs copies of C-test frames (unseen).
| pool | images | boxes | boxes_per_image | small_boxes_% | median_box_px |
|---|---|---|---|---|---|
| C-test positives (original) | 134 | 381 | 2.84 | 56.4 | 29.6 |
| copies of training frames | 726 | 2235 | 3.08 | 57.4 | 28.4 |
| copies of C-test frames | 112 | 322 | 2.88 | 54.3 | 30.3 |

| model | pool | rho_% | reps | AP50_mean | AP50_sd |
|---|---|---|---|---|---|
| yolo26n | train copies (leaked) | 0 | 1 | 0.133 | 0.0 |
| yolo26n | train copies (leaked) | 10 | 20 | 0.165 | 0.015 |
| yolo26n | train copies (leaked) | 20 | 20 | 0.202 | 0.02 |
| yolo26n | train copies (leaked) | 30 | 20 | 0.246 | 0.028 |
| yolo26n | train copies (leaked) | 40 | 20 | 0.286 | 0.025 |
| yolo26n | train copies (leaked) | 50 | 20 | 0.303 | 0.031 |
| yolo26n | train copies (leaked) | 75 | 20 | 0.387 | 0.033 |
| yolo26n | train copies (leaked) | 100 | 20 | 0.494 | 0.024 |
| yolo26n | C-test copies (unseen) | 0 | 1 | 0.133 | 0.0 |
| yolo26n | C-test copies (unseen) | 10 | 20 | 0.137 | 0.009 |
| yolo26n | C-test copies (unseen) | 20 | 20 | 0.136 | 0.011 |
| yolo26n | C-test copies (unseen) | 30 | 20 | 0.142 | 0.01 |
| yolo26n | C-test copies (unseen) | 40 | 20 | 0.14 | 0.014 |
| yolo26n | C-test copies (unseen) | 50 | 20 | 0.141 | 0.009 |
| yolo26n | C-test copies (unseen) | 75 | 20 | 0.147 | 0.012 |
| yolo26n | C-test copies (unseen) | 100 | 20 | 0.149 | 0.032 |
| yolo26n | all C-test copies (no substitution) | - | 1 | 0.193 | 0 |
| yolo26s | train copies (leaked) | 0 | 1 | 0.135 | 0.0 |
| yolo26s | train copies (leaked) | 10 | 20 | 0.165 | 0.015 |
| yolo26s | train copies (leaked) | 20 | 20 | 0.2 | 0.024 |
| yolo26s | train copies (leaked) | 30 | 20 | 0.226 | 0.025 |
| yolo26s | train copies (leaked) | 40 | 20 | 0.258 | 0.031 |
| yolo26s | train copies (leaked) | 50 | 20 | 0.275 | 0.035 |
| yolo26s | train copies (leaked) | 75 | 20 | 0.37 | 0.028 |
| yolo26s | train copies (leaked) | 100 | 20 | 0.436 | 0.03 |
| yolo26s | C-test copies (unseen) | 0 | 1 | 0.135 | 0.0 |
| yolo26s | C-test copies (unseen) | 10 | 20 | 0.136 | 0.009 |
| yolo26s | C-test copies (unseen) | 20 | 20 | 0.139 | 0.013 |
| yolo26s | C-test copies (unseen) | 30 | 20 | 0.141 | 0.013 |
| yolo26s | C-test copies (unseen) | 40 | 20 | 0.145 | 0.017 |
| yolo26s | C-test copies (unseen) | 50 | 20 | 0.147 | 0.016 |
| yolo26s | C-test copies (unseen) | 75 | 20 | 0.147 | 0.01 |
| yolo26s | C-test copies (unseen) | 100 | 20 | 0.15 | 0.021 |
| yolo26s | all C-test copies (no substitution) | - | 1 | 0.189 | 0 |
