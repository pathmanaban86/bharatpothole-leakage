# Revision checks (2026-10-11 10:38)
per_image.csv columns: ['md5', 'file_kb', 'width', 'height', 'corrupt', 'mean_luma', 'std_luma', 'blur_laplacian_var', 'rel_path', 'split', 'group', 'stem', 'source_name', 'seq_prefix', 'seq_index', 'n_boxes', 'label_status', 'n_label_issues', 'label_file', 'low_light', 'nn_phash_to_test', 'nn_phash_to_train', 'nn_phash_to_valid', 'has_dup_in_train', 'n_confirmed_dups', 'source_in_train', 'seq_prefix_in_train', 'component', 'new_split']
manifest_v2.csv columns: ['rel_path', 'original_split', 'new_split', 'drop_reason', 'is_rep', 'component', 'seq_prefix', 'source_name', 'md5', 'n_boxes', 'n_small', 'dest_name', 'n_boxes_clean', 'nn_phash_to_train_v2']

## R6 Subset results (YOLO26n, Protocol A)
B-rule = A-test minus Leaked (no confirmed near-duplicate, no shared source frame in training); B-strict = the 337-image list used so far.
B-rule 403 images; B-strict 337; in B-rule only 66
  in B-rule only: nn_phash_to_valid<=4 = 38; nn_phash_to_test<=4 = 48; n_confirmed_dups>0 = 66; min nn_phash_to_train = 8
  B-strict: nn_phash_to_valid<=4 = 20; nn_phash_to_test<=4 = 24; n_confirmed_dups>0 = 34; min nn_phash_to_train = 6
### all images
| subset | images | pos | gt | videos | AP50 | AP50_CI | AP50_95 | AP50_95_CI | R | dAP50 vs B-rule | dAP50_95 vs B-rule | dAP50 vs B-strict | dAP50_95 vs B-strict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A-test | 662 | 371 | 1148 | 93 | 0.615 | [0.555, 0.690] | 0.257 | [0.223, 0.310] | 0.551 | +0.158 [0.095, 0.246] (35%) | +0.110 [0.070, 0.175] | +0.190 [0.108, 0.296] (45%) | +0.122 [0.082, 0.187] |
| Leaked | 259 | 246 | 643 | 50 | 0.739 | [0.646, 0.817] | 0.352 | [0.292, 0.413] | 0.661 | +0.282 [0.167, 0.375] (62%) | +0.205 [0.146, 0.272] | +0.314 [0.184, 0.435] (74%) | +0.217 [0.153, 0.282] |
| B-rule | 403 | 125 | 505 | 79 | 0.457 | [0.389, 0.519] | 0.147 | [0.108, 0.177] | 0.493 |  |  |  |  |
| B-strict | 337 | 59 | 201 | 72 | 0.425 | [0.330, 0.517] | 0.135 | [0.098, 0.165] | 0.388 |  |  |  |  |

### positive images
| subset | images | pos | gt | videos | AP50 | AP50_CI | AP50_95 | AP50_95_CI | R | dAP50 vs B-rule | dAP50_95 vs B-rule | dAP50 vs B-strict | dAP50_95 vs B-strict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A-test | 371 | 371 | 1148 | 53 | 0.619 | [0.552, 0.702] | 0.259 | [0.219, 0.316] | 0.551 | +0.152 [0.090, 0.234] (33%) | +0.109 [0.068, 0.176] | +0.174 [0.098, 0.283] (39%) | +0.118 [0.076, 0.184] |
| Leaked | 246 | 246 | 643 | 40 | 0.739 | [0.642, 0.819] | 0.352 | [0.289, 0.422] | 0.661 | +0.272 [0.163, 0.365] (58%) | +0.202 [0.141, 0.270] | +0.294 [0.168, 0.407] (66%) | +0.211 [0.147, 0.279] |
| B-rule | 125 | 125 | 505 | 36 | 0.467 | [0.392, 0.538] | 0.15 | [0.109, 0.181] | 0.493 |  |  |  |  |
| B-strict | 59 | 59 | 201 | 29 | 0.445 | [0.358, 0.545] | 0.141 | [0.106, 0.173] | 0.418 |  |  |  |  |
