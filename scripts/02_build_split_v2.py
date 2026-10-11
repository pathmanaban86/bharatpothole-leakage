"""
BharatPotHole v2 - clean, leakage-free benchmark builder  (Google Colab)
=======================================================================
Uses the outputs of bharatpothole_assessment.py (dataset_assessment/ folder) to:

  1. Remove exact duplicates (MD5) and Roboflow augmentation copies
     (one representative image per source frame; copies are never allowed in val/test)
  2. Clean labels (malformed lines, out-of-range boxes, degenerate/duplicate boxes)
  3. Build a VIDEO-DISJOINT split that is stratified on images, positive frames,
     boxes and small boxes (randomised greedy search over thousands of trials)
  4. Verify: no video / source frame / MD5 / duplicate-component shared across splits,
     and re-measure train-test pHash proximity
  5. Write a ready-to-train YOLO folder + data.yaml, and the three Step-3 protocols:
       (a) original split           (b) original test minus leaked images
       (c) v2 video-disjoint split
  6. Save manifests/reports and a zipped copy of the dataset to Drive

Run cells top to bottom. Nothing in the original zip or the assessment folder is modified.
"""

# %% [1] Setup & configuration -------------------------------------------------------
try:                                   # Colab: mount Drive once; elsewhere set POTHOLE paths below
    import os as _os
    from google.colab import drive
    if not _os.path.ismount('/content/drive'):
        drive.mount('/content/drive')
except Exception:
    pass

import os, re, json, shutil, zipfile, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

ZIP_PATH    = Path('/content/drive/MyDrive/POTHOLE/archive (1).zip')
ASSESS_DIR  = ZIP_PATH.parent / 'dataset_assessment'      # output of the audit script
EXTRACT_DIR = Path('/content/pothole_extracted')          # original data (local disk)
LOCAL_V2    = Path('/content/BharatPotHole_v2')           # training copy (local disk, fast)
DRIVE_V2    = ZIP_PATH.parent / 'BharatPotHole_v2'        # manifests, reports, zipped dataset

TARGET          = {'train': 0.70, 'valid': 0.20, 'test': 0.10}
FEATURE_WEIGHTS = {'images': 1.0, 'positives': 1.0, 'boxes': 0.5, 'small_boxes': 0.5}
N_TRIALS        = 3000     # random restarts of the greedy group assignment
MIN_COMPONENTS  = 8        # each split must contain at least this many videos/components
KEEP_AUG_COPIES_IN_TRAIN = False   # False = one image per source frame everywhere
                                   # (recommended; rely on YOLO's online augmentation)
MIN_BOX_PX      = 2        # boxes narrower/shorter than this (pixels) are dropped
DUP_PHASH_THR   = 4        # for post-split verification
SEED            = 42

DRIVE_V2.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(SEED)
print('Assessment folder :', ASSESS_DIR)
print('Local v2 dataset  :', LOCAL_V2)
print('Drive outputs     :', DRIVE_V2)


# %% [2] Make sure the original data is extracted, load audit tables ------------------
if not (EXTRACT_DIR / '.extracted_ok').exists():
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH) as zf:
        zf.extractall(EXTRACT_DIR)
    (EXTRACT_DIR / '.extracted_ok').touch()
    print('Re-extracted original dataset')

pi = pd.read_csv(ASSESS_DIR / 'per_image.csv')
pb = pd.read_csv(ASSESS_DIR / 'per_box.csv')
pi['img_idx'] = np.arange(len(pi))                      # per_box.img_idx refers to this order
pi['corrupt'] = pi['corrupt'].fillna(False).astype(bool)
for c in ('has_dup_in_train', 'source_in_train'):
    pi[c] = pi[c].fillna(False).astype(bool)
small = pb[pb['size_cat'].astype(str).str.startswith('small')].groupby('img_idx').size()
pi['n_small'] = pi.img_idx.map(small).fillna(0).astype(int)
assert pi.group.nunique() == 1, 'Script expects one dataset version; filter pi by group first.'
print(f'{len(pi)} images, {pi.component.nunique()} duplicate/video components, '
      f'{pi.seq_prefix.nunique()} video prefixes')


# %% [3] De-duplication ---------------------------------------------------------------
pi['drop_reason'] = ''
pi.loc[pi.corrupt, 'drop_reason'] = 'corrupt'

dup_md5 = pi.duplicated('md5', keep='first') & (pi.drop_reason == '')
pi.loc[dup_md5, 'drop_reason'] = 'exact_md5_duplicate'

# One representative per Roboflow source frame: the copy whose brightness is closest to
# the median of its copies (brightness/exposure jitter is the commonest offline augmentation).
alive = pi[pi.drop_reason == ''].copy()
alive['luma_dev'] = (alive.mean_luma -
                     alive.groupby('source_name').mean_luma.transform('median')).abs()
rep_idx = alive.sort_values(['luma_dev', 'img_idx']).drop_duplicates('source_name').index
pi['is_rep'] = False
pi.loc[rep_idx, 'is_rep'] = True
print(pi.drop_reason.replace('', 'kept').value_counts().to_string())
print(f'Representatives (unique source frames): {pi.is_rep.sum()}  |  '
      f'augmentation copies: {(~pi.is_rep & (pi.drop_reason == "")).sum()}')


# %% [4] Stratified, video-disjoint group split ----------------------------------------
reps = pi[pi.is_rep]
comp = reps.groupby('component').agg(images=('img_idx', 'size'),
                                     positives=('n_boxes', lambda s: int((s > 0).sum())),
                                     boxes=('n_boxes', 'sum'),
                                     small_boxes=('n_small', 'sum'),
                                     videos=('seq_prefix', 'nunique'))
feat_names = list(FEATURE_WEIGHTS)
F = comp[feat_names].to_numpy(float)
tot = F.sum(0)
w = np.array([FEATURE_WEIGHTS[f] for f in feat_names])
r = np.array([TARGET[s] for s in TARGET])[:, None]
split_names = list(TARGET)

def run_trial(order):
    cur = np.zeros((len(split_names), len(feat_names)))
    assign = np.empty(len(order), dtype=int)
    for ci in order:
        f = F[ci]
        dev_now = np.abs(cur / tot - r)
        dev_new = np.abs((cur + f) / tot - r)
        delta = ((dev_new - dev_now) * w).sum(1)
        s = int(np.argmin(delta + 1e-9 * rng.random(len(delta))))
        cur[s] += f; assign[ci] = s
    return assign, float((np.abs(cur / tot - r) * w).sum())

t0 = time.time()
best_assign, best_cost = None, np.inf
for t in range(N_TRIALS):
    keyv = F[:, 0] * rng.uniform(0.3, 1.7, len(F))      # perturbed largest-first order
    assign, cost = run_trial(np.argsort(-keyv))
    if np.bincount(assign, minlength=3).min() < MIN_COMPONENTS:
        continue
    if cost < best_cost:
        best_assign, best_cost = assign, cost
assert best_assign is not None, 'No feasible split - lower MIN_COMPONENTS.'
print(f'Best of {N_TRIALS} trials: cost={best_cost:.4f}  ({time.time() - t0:.0f}s)')

comp['new_split'] = [split_names[i] for i in best_assign]
pi['new_split'] = pi.component.map(comp['new_split'])

# final membership: representatives always; copies only in train if allowed
keep = (pi.drop_reason == '') & (pi.is_rep | ((pi.new_split == 'train') & KEEP_AUG_COPIES_IN_TRAIN))
pi.loc[(pi.drop_reason == '') & ~keep, 'drop_reason'] = 'augmentation_copy'
pi.loc[~keep, 'new_split'] = ''
final = pi[keep].copy()

def split_stats(d, col):
    rows = []
    for s in split_names:
        g = d[d[col] == s]
        rows.append({'split': s, 'images': len(g), 'share_%': round(100 * len(g) / len(d), 1),
                     'positive_imgs': int((g.n_boxes > 0).sum()),
                     'positive_%': round(100 * (g.n_boxes > 0).mean(), 1),
                     'boxes': int(g.n_boxes.sum()),
                     'boxes_share_%': round(100 * g.n_boxes.sum() / d.n_boxes.sum(), 1),
                     'small_box_%': round(100 * g.n_small.sum() / max(1, g.n_boxes.sum()), 1),
                     'components': g.component.nunique(), 'videos': g.seq_prefix.nunique(),
                     'low_light_imgs': int((g.mean_luma < 60).sum())})
    return pd.DataFrame(rows).set_index('split')

orig_stats = split_stats(pi[pi.split.isin(split_names)], 'split')
new_stats = split_stats(final, 'new_split')
print('\nORIGINAL split\n', orig_stats.to_string(), '\n\nV2 split\n', new_stats.to_string())


# %% [5] Leakage checks on the v2 split -------------------------------------------------
checks = {}
for key in ('component', 'seq_prefix', 'source_name', 'md5'):
    n_multi = int((final.groupby(key).new_split.nunique() > 1).sum())
    checks[f'{key}_shared_across_splits'] = n_multi
    print(f'{key:12s} shared across splits: {n_multi}')
assert checks['component_shared_across_splits'] == 0
assert checks['source_name_shared_across_splits'] == 0
assert checks['md5_shared_across_splits'] == 0
if checks['seq_prefix_shared_across_splits']:
    print('WARNING: some video prefixes span splits - check seq_prefix values manually.')

# re-measure train -> val/test perceptual-hash proximity
def phash_bits(path):
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    d = cv2.dct(cv2.resize(g, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32))[:8, :8].flatten()
    return d > np.median(d[1:])

with ThreadPoolExecutor(8) as ex:
    bits = np.stack(list(ex.map(phash_bits, [EXTRACT_DIR / p for p in final.rel_path]))).astype(np.float32)
split_arr = final.new_split.to_numpy()
Btr = bits[split_arr == 'train']
nn_new = np.full(len(final), np.nan)
for i0 in range(0, len(final), 512):
    Bi = bits[i0:i0 + 512]
    D = Bi @ (1 - Btr).T + (1 - Bi) @ Btr.T
    nn_new[i0:i0 + 512] = D.min(1)
final['nn_phash_to_train_v2'] = nn_new
for s in ('valid', 'test'):
    m = split_arr == s
    checks[f'{s}_median_nn_phash_to_train_v2'] = float(np.median(nn_new[m]))
    checks[f'{s}_pct_within_{DUP_PHASH_THR}_of_train_v2'] = round(100 * float((nn_new[m] <= DUP_PHASH_THR).mean()), 1)
    o = pi[pi.split == s].nn_phash_to_train.dropna()
    checks[f'{s}_median_nn_phash_to_train_original'] = float(o.median())
    checks[f'{s}_pct_within_{DUP_PHASH_THR}_of_train_original'] = round(100 * float((o <= DUP_PHASH_THR).mean()), 1)
print(json.dumps(checks, indent=2))

fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, s in zip(axes, ('valid', 'test')):
    ax.hist(pi.loc[pi.split == s, 'nn_phash_to_train'].dropna(), bins=np.arange(0, 34), alpha=.5,
            density=True, label='original split')
    ax.hist(nn_new[split_arr == s], bins=np.arange(0, 34), alpha=.5, density=True, label='v2 split')
    ax.axvline(DUP_PHASH_THR, ls='--', c='r', lw=.8)
    ax.set(title=f'{s} -> nearest train image', xlabel='pHash Hamming distance'); ax.legend()
axes[0].set_ylabel('density'); fig.tight_layout()
fig.savefig(DRIVE_V2 / 'leakage_before_after.png', dpi=150); plt.show()


# %% [6] Materialise the YOLO dataset (with label cleaning) ----------------------------
def label_path_for(rel):
    parts = list(Path(rel).parts)
    i = max(k for k, x in enumerate(parts[:-1]) if x.lower() == 'images')
    parts[i] = 'labels'
    return (EXTRACT_DIR / Path(*parts)).with_suffix('.txt')

def clean_label(path, W, H):
    out, notes, seen = [], [], set()
    text = path.read_text(errors='ignore') if path.exists() else ''
    for ln, line in enumerate(text.splitlines(), 1):
        t = line.split()
        if not t:
            continue
        try:
            v = [float(x) for x in t]
        except ValueError:
            notes.append(f'line{ln}:non_numeric_dropped'); continue
        cls, c = int(v[0]), v[1:]
        if len(c) == 4:
            x0, y0, x1, y1 = c[0] - c[2] / 2, c[1] - c[3] / 2, c[0] + c[2] / 2, c[1] + c[3] / 2
        elif len(c) >= 6 and len(c) % 2 == 0:
            x0, x1, y0, y1 = min(c[0::2]), max(c[0::2]), min(c[1::2]), max(c[1::2])
            notes.append(f'line{ln}:polygon_to_bbox')
        else:
            notes.append(f'line{ln}:bad_token_count_dropped'); continue
        cx0, cy0, cx1, cy1 = (min(max(x0, 0), 1), min(max(y0, 0), 1), min(max(x1, 0), 1), min(max(y1, 0), 1))
        if (cx0, cy0, cx1, cy1) != (x0, y0, x1, y1):
            notes.append(f'line{ln}:clipped')
        bw, bh = cx1 - cx0, cy1 - cy0
        if bw * W < MIN_BOX_PX or bh * H < MIN_BOX_PX:
            notes.append(f'line{ln}:degenerate_dropped'); continue
        rec = f'{cls} {(cx0 + cx1) / 2:.6f} {(cy0 + cy1) / 2:.6f} {bw:.6f} {bh:.6f}'
        if rec in seen:
            notes.append(f'line{ln}:duplicate_dropped'); continue
        seen.add(rec); out.append(rec)
    return out, notes

if LOCAL_V2.exists():
    shutil.rmtree(LOCAL_V2)
for s in split_names:
    (LOCAL_V2 / s / 'images').mkdir(parents=True, exist_ok=True)
    (LOCAL_V2 / s / 'labels').mkdir(parents=True, exist_ok=True)

names = final.rel_path.map(lambda p: Path(p).name)
final['dest_name'] = np.where(names.duplicated(keep=False), final.split + '__' + names, names)

def materialise(row):
    src = EXTRACT_DIR / row.rel_path
    dst_img = LOCAL_V2 / row.new_split / 'images' / row.dest_name
    shutil.copy2(src, dst_img)
    lines, notes = clean_label(label_path_for(row.rel_path), row.width, row.height)
    (LOCAL_V2 / row.new_split / 'labels' / Path(row.dest_name).with_suffix('.txt')).write_text(
        '\n'.join(lines) + ('\n' if lines else ''))
    return len(lines), notes

t0 = time.time()
with ThreadPoolExecutor(8) as ex:
    res = list(ex.map(materialise, final.itertuples(index=False)))
final['n_boxes_clean'] = [n for n, _ in res]
fix_rows = [{'rel_path': rp, 'note': nt} for rp, (_, notes) in zip(final.rel_path, res) for nt in notes]
label_fixes = pd.DataFrame(fix_rows, columns=['rel_path', 'note'])
print(f'Copied {len(final)} images in {time.time() - t0:.0f}s | label fixes: {len(label_fixes)}')

class_names = ['pothole']
(LOCAL_V2 / 'data.yaml').write_text(
    f"# BharatPotHole v2: de-duplicated, video-disjoint, stratified (seed {SEED})\n"
    f"path: {LOCAL_V2}\ntrain: train/images\nval: valid/images\ntest: test/images\n"
    f"nc: {len(class_names)}\nnames: {json.dumps(class_names)}\n")


# %% [7] Step-3 protocol files -----------------------------------------------------------
proto = LOCAL_V2 / 'protocols'; proto.mkdir(exist_ok=True)
tr_rel = pi.loc[pi.split == 'train', 'rel_path'].iloc[0]
parts = Path(tr_rel).parts
orig_root = EXTRACT_DIR / Path(*parts[:[p.lower() for p in parts].index('train')])

(proto / 'A_original_split.yaml').write_text(
    f"# Protocol A: original frame-level split (as published)\npath: {orig_root}\n"
    f"train: train/images\nval: valid/images\ntest: test/images\nnc: 1\nnames: ['pothole']\n")

clean_test = pi[(pi.split == 'test') & pi.is_rep & (pi.drop_reason.isin(['', 'augmentation_copy']))
                & ~pi.has_dup_in_train & ~pi.source_in_train]
(proto / 'B_original_test_clean.txt').write_text(
    '\n'.join(str(EXTRACT_DIR / p) for p in clean_test.rel_path))
(proto / 'B_original_test_clean.yaml').write_text(
    f"# Protocol B: train on ORIGINAL train, test on original test minus leaked images\n"
    f"# ({len(clean_test)} of {(pi.split == 'test').sum()} test images kept; same videos still appear in train)\n"
    f"path: {orig_root}\ntrain: train/images\nval: valid/images\n"
    f"test: {proto / 'B_original_test_clean.txt'}\nnc: 1\nnames: ['pothole']\n")
shutil.copy2(LOCAL_V2 / 'data.yaml', proto / 'C_v2_video_disjoint.yaml')
print(f'Protocol B keeps {len(clean_test)} / {(pi.split == "test").sum()} original test images')


# %% [8] Save manifests + report to Drive, zip the dataset ------------------------------
manifest = pi[['rel_path', 'split', 'new_split', 'drop_reason', 'is_rep', 'component',
               'seq_prefix', 'source_name', 'md5', 'n_boxes', 'n_small']].rename(
    columns={'split': 'original_split'})
manifest = manifest.merge(final[['rel_path', 'dest_name', 'n_boxes_clean', 'nn_phash_to_train_v2']],
                          on='rel_path', how='left')
manifest.to_csv(DRIVE_V2 / 'manifest_v2.csv', index=False)
orig_stats.to_csv(DRIVE_V2 / 'split_stats_original.csv')
new_stats.to_csv(DRIVE_V2 / 'split_stats_v2.csv')
label_fixes.to_csv(DRIVE_V2 / 'label_fixes.csv', index=False)
comp.to_csv(DRIVE_V2 / 'components_v2.csv')
json.dump({'checks': checks, 'best_cost': best_cost, 'seed': SEED, 'n_trials': N_TRIALS,
           'feature_weights': FEATURE_WEIGHTS, 'keep_aug_copies_in_train': KEEP_AUG_COPIES_IN_TRAIN,
           'n_final_images': len(final), 'protocol_B_test_images': len(clean_test)},
          open(DRIVE_V2 / 'build_summary.json', 'w'), indent=2)
for f in proto.iterdir():
    shutil.copy2(f, DRIVE_V2 / f.name)
shutil.copy2(LOCAL_V2 / 'data.yaml', DRIVE_V2 / 'data.yaml')

drops = pi.drop_reason.replace('', 'kept').value_counts()
readme = f"""# BharatPotHole v2 build report
Seed {SEED}, {N_TRIALS} trials, stratified on {FEATURE_WEIGHTS}

## What was removed
{drops.to_string()}

## Original split
{orig_stats.to_string()}

## v2 split (video-disjoint, one image per source frame{' in val/test' if KEEP_AUG_COPIES_IN_TRAIN else ''})
{new_stats.to_string()}

## Leakage checks
{json.dumps(checks, indent=2)}

## Protocols for Step 3
A_original_split.yaml        - published frame-level split
B_original_test_clean.yaml   - original train, test minus near-dup/shared-source images ({len(clean_test)} imgs)
C_v2_video_disjoint.yaml     - this dataset

Protocols A and B point to {EXTRACT_DIR}; re-run cell [2] in a new session.
Restore v2 in a new session:  unzip {DRIVE_V2 / 'BharatPotHole_v2.zip'} -d /content
"""
(DRIVE_V2 / 'README_v2.md').write_text(readme)

t0 = time.time()
shutil.make_archive(str(DRIVE_V2 / 'BharatPotHole_v2'), 'zip', root_dir=LOCAL_V2.parent,
                    base_dir=LOCAL_V2.name)
print(f'Zipped to Drive in {time.time() - t0:.0f}s')
print(readme)


# %% [9] (optional) sanity check with Ultralytics ---------------------------------------
# !pip -q install ultralytics
# from ultralytics.data.utils import check_det_dataset
# print(check_det_dataset(str(LOCAL_V2 / 'data.yaml')))