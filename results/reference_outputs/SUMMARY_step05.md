# Leakage paper - results summary (2026-10-11 09:01)
Ultralytics 8.4.162 | seed 0 | bootstrap 1000 resamples by video | AP computed with an own
all-point evaluator (conf 0.001), identical for every model and test set.

## T0 Audit (carried forward)
```
{
  "images_released": 7074,
  "exact_md5_duplicates_removed": 86,
  "augmentation_copies_removed": 2769,
  "unique_frames": 4219,
  "duplicate_share_%": 40.4,
  "positive_images_released": 4196,
  "positive_images_unique": 1333,
  "negative_images_released": 2878,
  "negative_images_unique": 2886,
  "boxes_released": 12221,
  "boxes_unique_clean": 4009,
  "v2_split_sizes": {
    "train": 2927,
    "valid": 867,
    "test": 425
  }
}
```

## T1 Inflation: same Protocol A model, original test vs leakage-free subset
| model   | metric   |   original_test |   leakage_free_B |   abs_inflation |   abs_lo |   abs_hi |   rel_inflation_% |
|:--------|:---------|----------------:|-----------------:|----------------:|---------:|---------:|------------------:|
| yolo26n | AP50     |          0.6152 |           0.4252 |          0.19   |   0.1079 |   0.2959 |           44.688  |
| yolo26n | AP50_95  |          0.2572 |           0.135  |          0.1223 |   0.0815 |   0.1866 |           90.5965 |

## T1-T3 All protocols and test sets
| model   | trained_on            | test_set                             |   n_images |   n_pos_images |   n_gt |   AP50 |   AP50_95 |      P |      R |     F1 |   AP50_lo |   AP50_hi |   AP50_95_lo |   AP50_95_hi |
|:--------|:----------------------|:-------------------------------------|-----------:|---------------:|-------:|-------:|----------:|-------:|-------:|-------:|----------:|----------:|-------------:|-------------:|
| yolo26n | A (original split)    | original test (A)                    |        662 |            371 |   1148 | 0.6152 |    0.2572 | 0.7717 | 0.5505 | 0.6426 |    0.5553 |    0.6898 |       0.223  |       0.3101 |
| yolo26n | A (original split)    | leakage-free subset (B)              |        337 |             59 |    201 | 0.4252 |    0.135  | 0.6393 | 0.3881 | 0.483  |    0.3303 |    0.517  |       0.0976 |       0.1647 |
| yolo26n | A (original split)    | leaked subset                        |        259 |            246 |    643 | 0.7391 |    0.3518 | 0.8416 | 0.661  | 0.7404 |    0.6455 |    0.8171 |       0.2917 |       0.4131 |
| yolo26n | A (original split)    | RDD2022-India (external)             |       1530 |           1530 |   3187 | 0.0722 |    0.0207 | 0.2417 | 0.1076 | 0.1489 |    0.0622 |    0.0845 |       0.0175 |       0.0251 |
| yolo26n | C (video-disjoint v2) | v2 test (C)                          |        425 |            134 |    381 | 0.1326 |    0.0416 | 0.3429 | 0.189  | 0.2437 |    0.0056 |    0.2277 |       0.0014 |       0.0734 |
| yolo26n | C (video-disjoint v2) | copies of train frames (100% leaked) |        726 |            726 |   2235 | 0.541  |    0.2195 | 0.6403 | 0.4953 | 0.5585 |  nan      |  nan      |     nan      |     nan      |
| yolo26n | C (video-disjoint v2) | RDD2022-India (external)             |       1530 |           1530 |   3187 | 0.0989 |    0.0321 | 0.2147 | 0.1572 | 0.1815 |    0.0871 |    0.1126 |       0.0279 |       0.0373 |
| yolo26s | C (video-disjoint v2) | v2 test (C)                          |        425 |            134 |    381 | 0.135  |    0.0452 | 0.3801 | 0.1706 | 0.2355 |    0.0071 |    0.2286 |       0.0021 |       0.0777 |
| yolo26s | C (video-disjoint v2) | copies of train frames (100% leaked) |        726 |            726 |   2235 | 0.471  |    0.1783 | 0.5564 | 0.4676 | 0.5081 |  nan      |  nan      |     nan      |     nan      |
| yolo26s | C (video-disjoint v2) | RDD2022-India (external)             |       1530 |           1530 |   3187 | 0.0973 |    0.0315 | 0.2185 | 0.1503 | 0.1781 |    0.0863 |    0.1104 |       0.0276 |       0.0365 |

## T4 Leakage dose-response (Protocol C models)
| model   |   leaked_fraction_of_positives |   reps |   AP50_mean |   AP50_sd |   AP50_95_mean |   AP50_95_sd |   R_mean |
|:--------|-------------------------------:|-------:|------------:|----------:|---------------:|-------------:|---------:|
| yolo26n |                           0    |      1 |      0.1326 |    0      |         0.0416 |       0      |   0.189  |
| yolo26n |                           0.1  |     20 |      0.1652 |    0.0146 |         0.0549 |       0.0066 |   0.2227 |
| yolo26n |                           0.2  |     20 |      0.2016 |    0.02   |         0.0704 |       0.0087 |   0.2563 |
| yolo26n |                           0.3  |     20 |      0.2458 |    0.0279 |         0.0915 |       0.0113 |   0.3038 |
| yolo26n |                           0.4  |     20 |      0.2861 |    0.0253 |         0.1085 |       0.0101 |   0.338  |
| yolo26n |                           0.5  |     20 |      0.3029 |    0.0314 |         0.1172 |       0.0144 |   0.353  |
| yolo26n |                           0.75 |     20 |      0.3867 |    0.0334 |         0.1535 |       0.0158 |   0.4168 |
| yolo26n |                           1    |      1 |      0.5383 |    0      |         0.2345 |       0      |   0.5041 |
| yolo26s |                           0    |      1 |      0.135  |    0      |         0.0452 |       0      |   0.1706 |
| yolo26s |                           0.1  |     20 |      0.1661 |    0.02   |         0.0574 |       0.0072 |   0.2102 |
| yolo26s |                           0.2  |     20 |      0.1931 |    0.0271 |         0.0684 |       0.0096 |   0.24   |
| yolo26s |                           0.3  |     20 |      0.2185 |    0.026  |         0.0779 |       0.012  |   0.2574 |
| yolo26s |                           0.4  |     20 |      0.2464 |    0.0227 |         0.0877 |       0.009  |   0.2829 |
| yolo26s |                           0.5  |     20 |      0.2914 |    0.0255 |         0.1085 |       0.0115 |   0.3235 |
| yolo26s |                           0.75 |     20 |      0.3586 |    0.0307 |         0.1354 |       0.0133 |   0.39   |
| yolo26s |                           1    |      1 |      0.4563 |    0      |         0.1731 |       0      |   0.4264 |

## T5 Accuracy vs similarity to training data (Protocol A)
| model   | phash_bin            |   n_images |   n_pos_images |   n_gt |   AP50 |   AP50_95 |      P |      R |     F1 |   AP50_lo |   AP50_hi |
|:--------|:---------------------|-----------:|---------------:|-------:|-------:|----------:|-------:|-------:|-------:|----------:|----------:|
| yolo26n | 0-4 (near-identical) |        257 |            246 |    643 | 0.7394 |    0.3519 | 0.8416 | 0.661  | 0.7404 |    0.6504 |    0.826  |
| yolo26n | 5-8                  |         76 |             26 |     68 | 0.5862 |    0.1787 | 0.8333 | 0.5147 | 0.6364 |    0.4642 |    0.7429 |
| yolo26n | 9-12                 |        171 |             41 |     86 | 0.4514 |    0.1502 | 0.661  | 0.4535 | 0.5379 |    0.2928 |    0.6118 |
| yolo26n | >12 (dissimilar)     |        158 |             58 |    351 | 0.446  |    0.1497 | 0.5464 | 0.4701 | 0.5054 |    0.271  |    0.5126 |

Earlier audit tables, v2 build report, RDD domain comparison, family-study validation summary and the
dataset diagnostics are copied in carried_forward/.