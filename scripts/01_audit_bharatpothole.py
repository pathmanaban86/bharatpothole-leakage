"""
BharatPotHole dataset assessment  (Google Colab)
================================================
Audits the dataset used in iWatchRoadv2 (arXiv:2510.16375):
  1. Folder structure, class names, split sizes vs the stated 70:20:10
  2. Label integrity (missing / empty / malformed / out-of-range / duplicate boxes)
  3. Positive vs negative (background) frames, boxes per image, box-size classes
  4. Image properties (resolution, brightness as a low-light proxy, blur)
  5. Train/val/test LEAKAGE:
       - exact duplicates (MD5)
       - Roboflow augmentation copies of the same source frame in different splits
       - near-duplicate frames (perceptual hash + SSIM confirmation)
       - shared video/sequence prefixes across splits (when filenames encode them)
  6. A leakage-free GROUPED re-split proposal (70:20:10) ready for YOLO training

Results are written to the SAME Drive folder as the zip:
    /content/drive/MyDrive/POTHOLE/dataset_assessment/
(The zip itself is never modified.)

Run the cells top to bottom (each "# %%" marks a Colab cell).
"""

# %% [1] Mount Drive, imports, configuration ---------------------------------
try:                                   # Colab: mount Drive once; elsewhere set POTHOLE paths below
    import os as _os
    from google.colab import drive
    if not _os.path.ismount('/content/drive'):
        drive.mount('/content/drive')
except Exception:
    pass

import os, re, json, zipfile, hashlib, random, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

try:
    import yaml
except ImportError:
    yaml = None
try:
    from skimage.metrics import structural_similarity as ssim_fn
except ImportError:
    ssim_fn = None

# ---------------- configuration ----------------
ZIP_PATH    = Path('/content/drive/MyDrive/POTHOLE/archive (1).zip')
OUT_DIR     = ZIP_PATH.parent / 'dataset_assessment'   # saved beside the zip on Drive
EXTRACT_DIR = Path('/content/pothole_extracted')        # local disk (much faster than Drive I/O)

PHASH_CANDIDATE_THR = 8      # Hamming distance (0-64) for a candidate near-duplicate pair
PHASH_STRICT_THR    = 4      # accept as duplicate without SSIM if distance <= this
SSIM_CONFIRM_THR    = 0.80   # SSIM on 128x128 grey thumbnails to confirm a candidate pair
MAX_SSIM_PAIRS      = 60000  # cap on SSIM computations (cross-split pairs are checked first)
NIGHT_LUMA_THR      = 60     # mean grey level below which a frame is flagged low-light (heuristic)
THUMB               = 128
N_WORKERS           = 8
SEED                = 42
TARGET_RATIOS       = {'train': 0.70, 'valid': 0.20, 'test': 0.10}   # ratio stated in the paper

IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
SPLIT_ALIASES = {'train': 'train', 'training': 'train',
                 'valid': 'valid', 'val': 'valid', 'validation': 'valid',
                 'test': 'test', 'testing': 'test'}
LAYOUT_DIRS = {'images', 'labels', 'image', 'label', 'imgs', 'annotations'}

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = OUT_DIR / 'figures'; FIG_DIR.mkdir(exist_ok=True)
random.seed(SEED); np.random.seed(SEED)
print('Results folder:', OUT_DIR)


# %% [2] Extract the zip (including any nested zips) -------------------------
def extract_zip():
    assert ZIP_PATH.exists(), f'Zip not found: {ZIP_PATH}'
    marker = EXTRACT_DIR / '.extracted_ok'
    if marker.exists():
        print('Already extracted ->', EXTRACT_DIR); return
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with zipfile.ZipFile(ZIP_PATH) as zf:
        print(f'Extracting {len(zf.infolist())} entries ...')
        zf.extractall(EXTRACT_DIR)
    for inner in list(EXTRACT_DIR.rglob('*.zip')):          # nested archives
        target = inner.with_suffix('')
        if not target.exists():
            with zipfile.ZipFile(inner) as zf:
                zf.extractall(target)
    marker.touch()
    print(f'Done in {time.time() - t0:.0f}s')

extract_zip()


# %% [3] Discover structure ---------------------------------------------------
def is_junk(p: Path):
    return any(part.startswith('__MACOSX') for part in p.parts) or p.name.startswith('._')

def locate(p: Path):
    """Return (split, group). group = dataset/version folder, ignoring images/labels dirs."""
    parts = [x for x in p.relative_to(EXTRACT_DIR).parts[:-1] if not x.startswith('__MACOSX')]
    split, idx = 'unsplit', None
    for i, part in enumerate(parts):
        if part.lower() in SPLIT_ALIASES:
            split, idx = SPLIT_ALIASES[part.lower()], i
    prefix = parts[:idx] if idx is not None else parts
    group = '/'.join(x for x in prefix if x.lower() not in LAYOUT_DIRS) or '(root)'
    return split, group

def tree_summary(root: Path, max_depth=5):
    lines = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith(('__MACOSX', '.')))
        rel = Path(dirpath).relative_to(root)
        if len(rel.parts) > max_depth:
            continue
        n_img = sum(Path(f).suffix.lower() in IMG_EXT for f in filenames)
        n_txt = sum(f.lower().endswith('.txt') for f in filenames)
        lines.append(f"{'    ' * len(rel.parts)}{rel.name or root.name}/  "
                     f"[images={n_img}, txt={n_txt}, other={len(filenames) - n_img - n_txt}]")
    return '\n'.join(lines)

tree_txt = tree_summary(EXTRACT_DIR)
(OUT_DIR / 'folder_tree.txt').write_text(tree_txt)
print(tree_txt)

all_files  = [p for p in EXTRACT_DIR.rglob('*') if p.is_file() and not is_junk(p)]
img_paths  = sorted(p for p in all_files if p.suffix.lower() in IMG_EXT)
txt_paths  = [p for p in all_files if p.suffix.lower() == '.txt'
              and not p.name.lower().startswith('readme') and p.name.lower() != 'classes.txt']
yaml_paths = [p for p in all_files if p.suffix.lower() in ('.yaml', '.yml')]
print(f'\nImages: {len(img_paths)} | label-like .txt: {len(txt_paths)} | yaml: {len(yaml_paths)}')

label_index = {}
for t in txt_paths:
    sp, g = locate(t)
    label_index.setdefault((g, sp, t.stem), t)

class_names = {}
for y in yaml_paths:
    if yaml is None:
        break
    try:
        d = yaml.safe_load(y.read_text(errors='ignore'))
    except Exception:
        continue
    if isinstance(d, dict) and 'names' in d:
        names = d['names']
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names)]
        class_names[locate(y)[1]] = [str(n) for n in names]
for c in (p for p in all_files if p.name.lower() == 'classes.txt'):
    class_names.setdefault(locate(c)[1], c.read_text(errors='ignore').split())
print('Class names by group:', class_names or 'none found')


# %% [4] Per-image analysis (hashes, size, brightness, blur) -----------------
def phash_bits(gray):
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    d = cv2.dct(small)[:8, :8].flatten()
    return d > np.median(d[1:])

def analyse_image(path: Path):
    rec = {'path': str(path)}
    try:
        data = path.read_bytes()
        rec['md5'] = hashlib.md5(data).hexdigest()
        rec['file_kb'] = len(data) / 1024
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError('decode failed')
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rec.update(width=w, height=h, corrupt=False,
                   mean_luma=float(gray.mean()), std_luma=float(gray.std()),
                   blur_laplacian_var=float(cv2.Laplacian(gray, cv2.CV_64F).var()))
        thumb = cv2.resize(gray, (THUMB, THUMB), interpolation=cv2.INTER_AREA)
        return rec, phash_bits(gray), thumb
    except Exception as e:
        rec.update(corrupt=True, error=str(e))
        return rec, None, None

t0 = time.time()
records, bits, thumbs = [], [], []
with ThreadPoolExecutor(N_WORKERS) as ex:
    for k, (rec, b, th) in enumerate(ex.map(analyse_image, img_paths), 1):
        records.append(rec); bits.append(b); thumbs.append(th)
        if k % 1000 == 0:
            print(f'  {k}/{len(img_paths)} images  ({time.time() - t0:.0f}s)')
df = pd.DataFrame(records)
print(f'Image pass done in {time.time() - t0:.0f}s')


# %% [5] Labels, filename provenance ------------------------------------------
RF_RE  = re.compile(r'\.rf\.[0-9a-f]+$', re.I)             # Roboflow export hash
SEQ_RE = re.compile(r'^(.*?)[_\-\s\.]*(\d+)$')             # "<video>_<frame#>"

def source_name(stem):
    s = RF_RE.sub('', stem)
    return re.sub(r'_(jpe?g|png|bmp|webp)$', '', s, flags=re.I)

def sequence_key(src):
    m = SEQ_RE.match(src)
    return ((m.group(1) or '(numeric)'), int(m.group(2))) if m else (src, -1)

def parse_label(path):
    if path is None:
        return [], [], 'missing'
    text = path.read_text(errors='ignore').strip()
    if not text:
        return [], [], 'empty'
    boxes, issues = [], []
    for ln, line in enumerate(text.splitlines(), 1):
        t = line.split()
        if not t:
            continue
        try:
            vals = [float(x) for x in t]
        except ValueError:
            issues.append(f'line{ln}:non_numeric'); continue
        cls, coords = int(vals[0]), vals[1:]
        if len(coords) == 4:
            xc, yc, w, h = coords; fmt = 'bbox'
        elif len(coords) >= 6 and len(coords) % 2 == 0:
            xs, ys = coords[0::2], coords[1::2]
            xc, yc = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
            w, h = max(xs) - min(xs), max(ys) - min(ys); fmt = 'polygon'
        else:
            issues.append(f'line{ln}:bad_token_count({len(t)})'); continue
        if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < w <= 1 and 0 < h <= 1):
            issues.append(f'line{ln}:out_of_range')
        if w * h < 1e-6:
            issues.append(f'line{ln}:degenerate_box')
        boxes.append(dict(cls=cls, xc=xc, yc=yc, w=w, h=h, fmt=fmt))
    keys = [tuple(round(b[k], 4) for k in ('cls', 'xc', 'yc', 'w', 'h')) for b in boxes]
    if len(keys) != len(set(keys)):
        issues.append(f'duplicate_boxes({len(keys) - len(set(keys))})')
    status = 'labelled' if boxes else ('invalid_only' if issues else 'empty')
    return boxes, issues, status

splits, groups, stems, srcs, seqp, seqi = [], [], [], [], [], []
n_boxes, label_status, n_issues, label_files = [], [], [], []
box_rows, issue_rows = [], []
for i, p in enumerate(img_paths):
    sp, g = locate(p)
    lab = label_index.get((g, sp, p.stem))
    boxes, issues, status = parse_label(lab)
    src = source_name(p.stem); pre, idx = sequence_key(src)
    splits.append(sp); groups.append(g); stems.append(p.stem); srcs.append(src)
    seqp.append(pre); seqi.append(idx); n_boxes.append(len(boxes))
    label_status.append(status); n_issues.append(len(issues))
    label_files.append(str(lab) if lab else '')
    for b in boxes:
        box_rows.append({'img_idx': i, **b})
    for iss in issues:
        issue_rows.append({'image': str(p.relative_to(EXTRACT_DIR)), 'label': str(lab), 'issue': iss})

df = df.assign(rel_path=[str(p.relative_to(EXTRACT_DIR)) for p in img_paths], split=splits,
               group=groups, stem=stems, source_name=srcs, seq_prefix=seqp, seq_index=seqi,
               n_boxes=n_boxes, label_status=label_status, n_label_issues=n_issues,
               label_file=label_files)
df['low_light'] = df['mean_luma'] < NIGHT_LUMA_THR

box_df = pd.DataFrame(box_rows)
if not box_df.empty:
    box_df = box_df.merge(df[['width', 'height', 'group', 'split']], left_on='img_idx',
                          right_index=True, how='left')
    box_df['w_px'] = box_df.w * box_df.width
    box_df['h_px'] = box_df.h * box_df.height
    box_df['sqrt_area_px'] = np.sqrt(box_df.w_px * box_df.h_px)
    box_df['aspect_w_over_h'] = box_df.w_px / box_df.h_px.replace(0, np.nan)
    box_df['size_cat'] = pd.cut(box_df.sqrt_area_px, [0, 32, 96, np.inf], right=False,
                                labels=['small(<32px)', 'medium(32-96px)', 'large(>=96px)'])
print(f'Boxes: {len(box_df)} | label issues: {len(issue_rows)}')


# %% [6] Near-duplicate search (perceptual hash, within and across splits) ----
valid_idx = np.array([i for i, b in enumerate(bits) if b is not None])
B = np.stack([bits[i] for i in valid_idx]).astype(np.float32); Bc = 1 - B
n = len(valid_idx)
grp_arr = df.loc[valid_idx, 'group'].to_numpy()
spl_arr = df.loc[valid_idx, 'split'].to_numpy()
key_list = sorted(set(zip(grp_arr, spl_arr)))
key_masks = [(grp_arr == g) & (spl_arr == s) for g, s in key_list]
nn = np.full((n, len(key_list)), 99, dtype=np.int16)
pa, pb, pdist = [], [], []

t0 = time.time()
CH = 512
for i0 in range(0, n, CH):
    i1 = min(n, i0 + CH)
    D = np.rint(B[i0:i1] @ Bc.T + Bc[i0:i1] @ B.T).astype(np.int16)   # Hamming distances
    D[np.arange(i1 - i0), np.arange(i0, i1)] = 99                     # exclude self
    for k, m in enumerate(key_masks):
        nn[i0:i1, k] = D[:, m].min(axis=1)
    r, c = np.nonzero(D <= PHASH_CANDIDATE_THR)
    r = r + i0; keep = c > r
    pa.append(r[keep]); pb.append(c[keep]); pdist.append(D[r[keep] - i0, c[keep]])
print(f'pHash all-pairs done in {time.time() - t0:.0f}s')

present_splits = sorted(set(spl_arr))
for s in present_splits:                     # nearest neighbour to each split, same group only
    col = np.full(n, np.nan)
    for k, (g, sp) in enumerate(key_list):
        if sp == s:
            rows = grp_arr == g
            col[rows] = nn[rows, k]
    col[col > 64] = np.nan
    df.loc[valid_idx, f'nn_phash_to_{s}'] = col

pa = valid_idx[np.concatenate(pa)] if pa else np.array([], int)
pb = valid_idx[np.concatenate(pb)] if pb else np.array([], int)
pairs = pd.DataFrame({'a': pa, 'b': pb, 'phash_dist': np.concatenate(pdist) if pdist else []})
pairs['group_a'] = df.group.values[pairs.a]; pairs['group_b'] = df.group.values[pairs.b]
pairs['split_a'] = df.split.values[pairs.a]; pairs['split_b'] = df.split.values[pairs.b]
pairs['relation'] = np.where(pairs.group_a != pairs.group_b, 'cross_version_overlap',
                    np.where(pairs.split_a != pairs.split_b, 'CROSS_SPLIT_LEAK',
                             'same_split_redundant'))
pairs['priority'] = pairs.relation.map({'CROSS_SPLIT_LEAK': 0, 'same_split_redundant': 1,
                                        'cross_version_overlap': 2})
pairs = pairs.sort_values(['priority', 'phash_dist']).reset_index(drop=True)
print(pairs.relation.value_counts().rename('candidate pairs'))

# SSIM confirmation on thumbnails
pairs['ssim'] = np.nan
if ssim_fn is not None and len(pairs):
    m = min(len(pairs), MAX_SSIM_PAIRS)
    t0 = time.time()
    pairs.loc[:m - 1, 'ssim'] = [ssim_fn(thumbs[a], thumbs[b], data_range=255)
                                 for a, b in zip(pairs.a[:m], pairs.b[:m])]
    print(f'SSIM on {m} pairs in {time.time() - t0:.0f}s')
    if m < len(pairs):
        print(f'  NOTE: {len(pairs) - m} lower-priority pairs not SSIM-checked')
pairs['md5_identical'] = df.md5.values[pairs.a] == df.md5.values[pairs.b]
pairs['confirmed'] = (pairs.md5_identical | (pairs.phash_dist <= PHASH_STRICT_THR)
                      | (pairs.ssim >= SSIM_CONFIRM_THR))

# per-image leakage flags (same group only)
df['has_dup_in_train'] = False
df['n_confirmed_dups'] = 0
conf_same = pairs[pairs.confirmed & (pairs.group_a == pairs.group_b)]
for a, b, sa, sb in conf_same[['a', 'b', 'split_a', 'split_b']].itertuples(index=False):
    df.at[a, 'n_confirmed_dups'] += 1; df.at[b, 'n_confirmed_dups'] += 1
    if sb == 'train' and sa != 'train': df.at[a, 'has_dup_in_train'] = True
    if sa == 'train' and sb != 'train': df.at[b, 'has_dup_in_train'] = True

train_src = set(zip(df.loc[df.split == 'train', 'group'], df.loc[df.split == 'train', 'source_name']))
df['source_in_train'] = [(g, s) in train_src for g, s in zip(df.group, df.source_name)]
train_seq = set(zip(df.loc[df.split == 'train', 'group'], df.loc[df.split == 'train', 'seq_prefix']))
df['seq_prefix_in_train'] = [(g, s) in train_seq for g, s in zip(df.group, df.seq_prefix)]

seq_valid = {}
for g, gdf in df.groupby('group'):
    vc = gdf.seq_prefix.value_counts()
    seq_valid[g] = bool(len(vc) >= 10 and vc.iloc[0] < 0.3 * len(gdf))
print('Filename prefixes usable as video IDs:', seq_valid)


# %% [7] Leakage-free grouped re-split ----------------------------------------
class UF:
    def __init__(self, n): self.p = list(range(n))
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb: self.p[rb] = ra

uf = UF(len(df))
for a, b in conf_same[['a', 'b']].itertuples(index=False):
    uf.union(a, b)                                                   # near-duplicates
for _, idx in df.groupby(['group', 'source_name']).groups.items():   # Roboflow copies
    idx = list(idx)
    for j in idx[1:]: uf.union(idx[0], j)
for (g, _), idx in df.groupby(['group', 'seq_prefix']).groups.items():  # same video
    if seq_valid.get(g):
        idx = list(idx)
        for j in idx[1:]: uf.union(idx[0], j)
df['component'] = [uf.find(i) for i in range(len(df))]

rng = random.Random(SEED)
df['new_split'] = 'excluded_corrupt'
for g, gdf in df[~df.corrupt.astype(bool)].groupby('group'):
    comps = list(gdf.groupby('component').size().items())
    rng.shuffle(comps); comps.sort(key=lambda t: -t[1])
    N = len(gdf); cur = {s: 0 for s in TARGET_RATIOS}; assign = {}
    for c, sz in comps:
        s = max(TARGET_RATIOS, key=lambda s: (TARGET_RATIOS[s] * N - cur[s]) / (TARGET_RATIOS[s] * N))
        assign[c] = s; cur[s] += sz
    df.loc[gdf.index, 'new_split'] = gdf.component.map(assign)

def safe(s): return re.sub(r'[^A-Za-z0-9_.-]+', '_', s)

for g, gdf in df.groupby('group'):
    gd = OUT_DIR / 'grouped_split' / safe(g); gd.mkdir(parents=True, exist_ok=True)
    for s in TARGET_RATIOS:
        (gd / f'{s}.txt').write_text('\n'.join(gdf.loc[gdf.new_split == s, 'path']))
    names = class_names.get(g) or next(iter(class_names.values()), ['pothole'])
    (gd / 'data_grouped.yaml').write_text(
        f"# Leakage-free grouped split. Paths point to {EXTRACT_DIR} - re-run cell [2] first.\n"
        f"train: {gd / 'train.txt'}\nval: {gd / 'valid.txt'}\ntest: {gd / 'test.txt'}\n"
        f"nc: {len(names)}\nnames: {json.dumps(names)}\n")


# %% [8] Summary tables ---------------------------------------------------------
def summarize(gdf):
    pos = gdf.n_boxes > 0
    return pd.Series({
        'images': len(gdf),
        'corrupt': int(gdf.corrupt.astype(bool).sum()),
        'positive_imgs': int(pos.sum()),
        'negative_imgs(empty_label)': int((gdf.label_status == 'empty').sum()),
        'missing_label_file': int((gdf.label_status == 'missing').sum()),
        'imgs_with_label_issues': int((gdf.n_label_issues > 0).sum()),
        'boxes': int(gdf.n_boxes.sum()),
        'boxes_per_positive_img': round(gdf.loc[pos, 'n_boxes'].mean(), 2) if pos.any() else 0,
        'max_boxes_in_img': int(gdf.n_boxes.max()),
        'not_640x640': int(((gdf.width != 640) | (gdf.height != 640)).sum()),
        'low_light_%': round(100 * gdf.low_light.mean(), 1),
        'unique_source_frames': gdf.source_name.nunique(),
        'imgs_per_source_frame': round(len(gdf) / max(1, gdf.source_name.nunique()), 2),
    })

summary = df.groupby(['group', 'split']).apply(summarize)
summary['split_share_%'] = (100 * summary.images /
                            summary.groupby(level=0).images.transform('sum')).round(1)
if not box_df.empty:
    size_share = (pd.crosstab([box_df.group, box_df.split], box_df.size_cat,
                              normalize='index') * 100).round(1)
    summary = summary.join(size_share)
    class_dist = pd.crosstab([box_df.group, box_df.split], box_df.cls)
else:
    class_dist = pd.DataFrame()

leak_rows = []
for g, gdf in df.groupby('group'):
    row = {'group': g}
    if 'nn_phash_to_train' in gdf:
        row['train_median_nn_phash_to_other_train'] = gdf.loc[gdf.split == 'train', 'nn_phash_to_train'].median()
    for s in ('valid', 'test'):
        sub = gdf[gdf.split == s]
        if len(sub) == 0 or 'train' not in set(gdf.split):
            continue
        row[f'{s}_imgs'] = len(sub)
        row[f'{s}_with_confirmed_near_dup_in_train_%'] = round(100 * sub.has_dup_in_train.mean(), 1)
        row[f'{s}_sharing_source_frame_with_train_%'] = round(100 * sub.source_in_train.mean(), 1)
        if seq_valid.get(g):
            row[f'{s}_sharing_video_prefix_with_train_%'] = round(100 * sub.seq_prefix_in_train.mean(), 1)
        row[f'{s}_median_nn_phash_to_train'] = sub.nn_phash_to_train.median()
    row['exact_md5_dup_pairs'] = int(pairs[(pairs.group_a == g) & (pairs.group_b == g)].md5_identical.sum())
    row['largest_group_component_%'] = round(100 * gdf.component.value_counts().iloc[0] / len(gdf), 1)
    leak_rows.append(row)
leak_df = pd.DataFrame(leak_rows)

resplit = pd.crosstab([df.group, df.new_split], df.n_boxes > 0).rename(
    columns={True: 'positive', False: 'negative'})
resplit['total'] = resplit.sum(axis=1)
resplit['share_%'] = (100 * resplit.total / resplit.groupby(level=0).total.transform('sum')).round(1)


# %% [9] Figures -------------------------------------------------------------------
def savefig(fig, name):
    fig.tight_layout(); fig.savefig(FIG_DIR / name, dpi=150); plt.close(fig)

for g, gdf in df.groupby('group'):
    tag = safe(g); sp_list = [s for s in ['train', 'valid', 'test', 'unsplit'] if s in set(gdf.split)]
    gb = box_df[box_df.group == g] if not box_df.empty else box_df

    fig, ax = plt.subplots(figsize=(7, 4))
    mx = int(gdf.n_boxes.max()) + 2
    for s in sp_list:
        ax.hist(gdf.loc[gdf.split == s, 'n_boxes'], bins=range(0, mx), alpha=.5, density=True, label=s)
    ax.set(xlabel='boxes per image (0 = negative frame)', ylabel='fraction', title=f'Boxes per image - {g}')
    ax.legend(); savefig(fig, f'boxes_per_image_{tag}.png')

    if not gb.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        for s in sp_list:
            ax.hist(gb.loc[gb.split == s, 'sqrt_area_px'].dropna(), bins=60, alpha=.5, density=True, label=s)
        for v in (32, 96): ax.axvline(v, ls='--', c='k', lw=.8)
        ax.set(xlabel='sqrt(box area) [px]  (COCO small<32, medium<96)', ylabel='density',
               title=f'Box size - {g}'); ax.legend(); savefig(fig, f'box_size_{tag}.png')

        fig, ax = plt.subplots(figsize=(5, 5))
        h = ax.hist2d(gb.xc, gb.yc, bins=40, range=[[0, 1], [0, 1]], cmap='viridis')
        ax.invert_yaxis(); fig.colorbar(h[3], ax=ax)
        ax.set(xlabel='x centre (norm.)', ylabel='y centre (norm.)', title=f'Box centres - {g}')
        savefig(fig, f'box_centres_{tag}.png')

    fig, ax = plt.subplots(figsize=(7, 4))
    for s in sp_list:
        ax.hist(gdf.loc[gdf.split == s, 'mean_luma'].dropna(), bins=50, alpha=.5, density=True, label=s)
    ax.axvline(NIGHT_LUMA_THR, ls='--', c='k', lw=.8)
    ax.set(xlabel='mean grey level', ylabel='density', title=f'Brightness (low-light proxy) - {g}')
    ax.legend(); savefig(fig, f'brightness_{tag}.png')

    if 'nn_phash_to_train' in gdf and 'train' in sp_list:
        fig, ax = plt.subplots(figsize=(7, 4))
        for s in sp_list:
            v = gdf.loc[gdf.split == s, 'nn_phash_to_train'].dropna()
            lab = 'train -> other train (baseline)' if s == 'train' else f'{s} -> nearest train'
            ax.hist(v, bins=range(0, 34), alpha=.5, density=True, label=lab)
        ax.axvline(PHASH_CANDIDATE_THR, ls='--', c='r', lw=.8)
        ax.set(xlabel='pHash Hamming distance to nearest TRAIN image', ylabel='density',
               title=f'Train-test proximity - {g}\n(val/test overlapping the baseline = frame-level split)')
        ax.legend(); savefig(fig, f'nn_distance_to_train_{tag}.png')

    leaks = pairs[pairs.confirmed & (pairs.relation == 'CROSS_SPLIT_LEAK') & (pairs.group_a == g)]
    if len(leaks):
        sub = leaks.sample(min(8, len(leaks)), random_state=SEED)
        fig, axes = plt.subplots(len(sub), 2, figsize=(9, 2.6 * len(sub)))
        axes = np.atleast_2d(axes)
        for r, row in enumerate(sub.itertuples(index=False)):
            for c, (idx, sp) in enumerate([(row.a, row.split_a), (row.b, row.split_b)]):
                im = cv2.imread(df.at[idx, 'path']); ax = axes[r, c]; ax.axis('off')
                if im is not None: ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
                t = f"{sp}: {Path(df.at[idx, 'path']).name[:38]}"
                if c == 0: t += f"\npHash={row.phash_dist}  SSIM={row.ssim:.2f}"
                ax.set_title(t, fontsize=7)
        savefig(fig, f'leak_examples_{tag}.png')


# %% [10] Save everything + human-readable report ---------------------------------------
df.drop(columns=['path']).to_csv(OUT_DIR / 'per_image.csv', index=False)
if not box_df.empty:
    box_df.to_csv(OUT_DIR / 'per_box.csv', index=False)
pd.DataFrame(issue_rows).to_csv(OUT_DIR / 'label_issues.csv', index=False)
summary.to_csv(OUT_DIR / 'split_summary.csv')
class_dist.to_csv(OUT_DIR / 'class_distribution.csv')
leak_df.to_csv(OUT_DIR / 'leakage_summary.csv', index=False)
resplit.to_csv(OUT_DIR / 'grouped_resplit_summary.csv')
df[['rel_path', 'group', 'split', 'new_split', 'component', 'n_boxes']].to_csv(
    OUT_DIR / 'grouped_resplit_assignment.csv', index=False)
pairs_out = pairs.assign(img_a=df.rel_path.values[pairs.a], img_b=df.rel_path.values[pairs.b])
pairs_out = pd.concat([pairs_out[pairs_out.confirmed], pairs_out[~pairs_out.confirmed].head(200000)])
pairs_out.drop(columns=['a', 'b', 'priority']).to_csv(OUT_DIR / 'near_duplicate_pairs.csv', index=False)

def md(t):
    try: return t.to_markdown()
    except Exception: return '```\n' + t.to_string() + '\n```'

flags = []
for r in leak_rows:
    for s in ('valid', 'test'):
        v = r.get(f'{s}_with_confirmed_near_dup_in_train_%')
        if v is not None and v >= 5:
            flags.append(f"- **[{r['group']}] {v}% of {s} images have a confirmed near-duplicate in train** "
                         f"-> reported metrics are likely inflated; use the grouped split.")
        v = r.get(f'{s}_sharing_source_frame_with_train_%')
        if v is not None and v > 0:
            flags.append(f"- [{r['group']}] {v}% of {s} images share a Roboflow source frame with train "
                         f"(augmented copies across splits).")
    if r.get('largest_group_component_%', 0) > 20:
        flags.append(f"- [{r['group']}] one duplicate-component holds {r['largest_group_component_%']}% of images: "
                     f"lower PHASH_CANDIDATE_THR or raise SSIM_CONFIRM_THR and re-run cells 6-10.")
for (g, s), row in summary.iterrows():
    if row['missing_label_file'] or row['imgs_with_label_issues'] or row['corrupt']:
        flags.append(f"- [{g}/{s}] missing labels={row['missing_label_file']}, "
                     f"label issues={row['imgs_with_label_issues']}, corrupt={row['corrupt']}")
    if row['not_640x640']:
        flags.append(f"- [{g}/{s}] {row['not_640x640']} images are not 640x640 (paper states all are).")

report = f"""# BharatPotHole dataset assessment
Source: `{ZIP_PATH}`
Generated: {time.strftime('%Y-%m-%d %H:%M')}
Settings: pHash candidate <= {PHASH_CANDIDATE_THR}, strict <= {PHASH_STRICT_THR}, SSIM >= {SSIM_CONFIRM_THR}, low-light luma < {NIGHT_LUMA_THR}

## Paper claims to check
- ">7,000 annotated frames": found **{len(df)} images** in total ({df.n_boxes.gt(0).sum()} positive, {(df.n_boxes == 0).sum()} without boxes).
- "70:20:10 split": see `split_share_%` below.
- "640x640 resolution": {int(((df.width != 640) | (df.height != 640)).sum())} images differ.

## Automated flags
{chr(10).join(flags) if flags else '- none raised'}

## Split summary
{md(summary)}

## Leakage summary
{md(leak_df.set_index('group').T) if not leak_df.empty else 'n/a'}

How to read this: `train_median_nn_phash_to_other_train` is the baseline similarity between training frames.
If val/test medians sit at or near that baseline, the split was made at frame level from the same videos,
so test performance measures memorisation of scenes rather than generalisation to new roads.

## Proposed grouped re-split (near-duplicates, Roboflow copies{', video prefixes' if any(seq_valid.values()) else ''} kept together)
{md(resplit)}

## Class distribution (boxes)
{md(class_dist) if not class_dist.empty else 'n/a'}

## Folder tree
```
{tree_txt}
```

## Files
per_image.csv, per_box.csv, label_issues.csv, split_summary.csv, class_distribution.csv,
leakage_summary.csv, near_duplicate_pairs.csv, grouped_resplit_*.csv, grouped_split/<group>/data_grouped.yaml,
figures/*.png
"""
(OUT_DIR / 'assessment_report.md').write_text(report)
json.dump({'n_images': len(df), 'n_boxes': int(df.n_boxes.sum()), 'leakage': leak_rows,
           'seq_prefix_usable': seq_valid, 'class_names': class_names},
          open(OUT_DIR / 'summary.json', 'w'), indent=2, default=str)

print(report[:4000])
print('\nSaved to', OUT_DIR)
for p in sorted(OUT_DIR.rglob('*')):
    if p.is_file(): print('  ', p.relative_to(OUT_DIR))

# %% [8] Summary tables (pandas-version-safe) ------------------------------------
SIZE_CATS = ['small(<32px)', 'medium(32-96px)', 'large(>=96px)']
if not box_df.empty:
    box_df['size_cat'] = box_df['size_cat'].astype(str)
df['corrupt'] = df['corrupt'].fillna(False).astype(bool)

def summarize(gdf):
    nb = gdf.n_boxes.to_numpy()
    pos = nb > 0
    n_src = max(1, gdf.source_name.nunique())
    return {
        'images': len(gdf),
        'corrupt': int(gdf.corrupt.sum()),
        'positive_imgs': int(pos.sum()),
        'negative_imgs(empty_label)': int((gdf.label_status == 'empty').sum()),
        'missing_label_file': int((gdf.label_status == 'missing').sum()),
        'imgs_with_label_issues': int((gdf.n_label_issues > 0).sum()),
        'boxes': int(nb.sum()),
        'boxes_per_positive_img': round(float(nb[pos].mean()), 2) if pos.any() else 0.0,
        'max_boxes_in_img': int(nb.max()) if len(nb) else 0,
        'not_640x640': int(((gdf.width != 640) | (gdf.height != 640)).sum()),
        'low_light_%': round(100 * float(gdf.low_light.mean()), 1),
        'blur_median_laplacian': round(float(gdf.blur_laplacian_var.median()), 1),
        'unique_source_frames': gdf.source_name.nunique(),
        'imgs_per_source_frame': round(len(gdf) / n_src, 2),
        'unique_video_prefixes': gdf.seq_prefix.nunique(),
    }

rows = []
for (g, s), gdf in df.groupby(['group', 'split']):
    r = {'group': g, 'split': s, **summarize(gdf)}
    if not box_df.empty:
        gb = box_df[(box_df.group == g) & (box_df.split == s)]
        for cat in SIZE_CATS:
            r[f'{cat}_%'] = round(100 * float((gb.size_cat == cat).mean()), 1) if len(gb) else 0.0
    rows.append(r)
summary = pd.DataFrame(rows)
summary['split_share_%'] = (100 * summary.images /
                            summary.groupby('group').images.transform('sum')).round(1)
summary = summary.set_index(['group', 'split'])

class_dist = (box_df.groupby(['group', 'split', 'cls']).size().unstack('cls', fill_value=0)
              if not box_df.empty else pd.DataFrame())

leak_rows = []
for g, gdf in df.groupby('group'):
    row = {'group': g}
    has_train = 'train' in set(gdf.split)
    if has_train and 'nn_phash_to_train' in gdf.columns:
        row['train_median_nn_phash_to_other_train'] = float(
            gdf.loc[gdf.split == 'train', 'nn_phash_to_train'].median())
    for s in ('valid', 'test'):
        sub = gdf[gdf.split == s]
        if len(sub) == 0 or not has_train:
            continue
        row[f'{s}_imgs'] = len(sub)
        row[f'{s}_with_confirmed_near_dup_in_train_%'] = round(100 * float(sub.has_dup_in_train.mean()), 1)
        row[f'{s}_sharing_source_frame_with_train_%'] = round(100 * float(sub.source_in_train.mean()), 1)
        if seq_valid.get(g):
            row[f'{s}_sharing_video_prefix_with_train_%'] = round(100 * float(sub.seq_prefix_in_train.mean()), 1)
        if 'nn_phash_to_train' in sub.columns:
            row[f'{s}_median_nn_phash_to_train'] = float(sub.nn_phash_to_train.median())
    same = pairs[(pairs.group_a == g) & (pairs.group_b == g)]
    row['exact_md5_dup_pairs'] = int(same.md5_identical.sum())
    row['confirmed_cross_split_pairs'] = int((same.confirmed & (same.relation == 'CROSS_SPLIT_LEAK')).sum())
    row['confirmed_same_split_pairs'] = int((same.confirmed & (same.relation == 'same_split_redundant')).sum())
    row['n_components'] = int(gdf.component.nunique())
    row['largest_component_%'] = round(100 * gdf.component.value_counts().iloc[0] / len(gdf), 1)
    leak_rows.append(row)
leak_df = pd.DataFrame(leak_rows)

tmp = df.assign(kind=np.where(df.n_boxes > 0, 'positive', 'negative'))
resplit = tmp.groupby(['group', 'new_split', 'kind']).size().unstack('kind', fill_value=0)
for c in ('positive', 'negative'):
    if c not in resplit.columns:
        resplit[c] = 0
resplit['total'] = resplit['positive'] + resplit['negative']
resplit['share_%'] = (100 * resplit.total /
                      resplit.groupby(level='group').total.transform('sum')).round(1)
print(summary.T); print(leak_df.T); print(resplit)


# %% [9] Save tables + report (runs before figures so nothing is lost) ------------
df.drop(columns=['path']).to_csv(OUT_DIR / 'per_image.csv', index=False)
if not box_df.empty:
    box_df.to_csv(OUT_DIR / 'per_box.csv', index=False)
pd.DataFrame(issue_rows, columns=['image', 'label', 'issue']).to_csv(OUT_DIR / 'label_issues.csv', index=False)
summary.to_csv(OUT_DIR / 'split_summary.csv')
class_dist.to_csv(OUT_DIR / 'class_distribution.csv')
leak_df.to_csv(OUT_DIR / 'leakage_summary.csv', index=False)
resplit.to_csv(OUT_DIR / 'grouped_resplit_summary.csv')
df[['rel_path', 'group', 'split', 'new_split', 'component', 'seq_prefix', 'n_boxes']].to_csv(
    OUT_DIR / 'grouped_resplit_assignment.csv', index=False)
pairs_out = pairs.assign(img_a=df.rel_path.to_numpy()[pairs.a.to_numpy()],
                         img_b=df.rel_path.to_numpy()[pairs.b.to_numpy()])
pairs_out = pd.concat([pairs_out[pairs_out.confirmed], pairs_out[~pairs_out.confirmed].head(200000)])
pairs_out.drop(columns=['a', 'b', 'priority']).to_csv(OUT_DIR / 'near_duplicate_pairs.csv', index=False)

def md(t):
    try:
        return t.to_markdown()
    except Exception:
        return '```\n' + t.to_string() + '\n```'

flags = []
for r in leak_rows:
    for s in ('valid', 'test'):
        v = r.get(f'{s}_with_confirmed_near_dup_in_train_%')
        if v is not None and v >= 5:
            flags.append(f"- **[{r['group']}] {v}% of {s} images have a confirmed near-duplicate in train** "
                         f"-> reported metrics are likely inflated; use the grouped split.")
        v = r.get(f'{s}_sharing_video_prefix_with_train_%')
        if v is not None and v >= 5:
            flags.append(f"- **[{r['group']}] {v}% of {s} images come from video prefixes also in train** "
                         f"(frame-level split).")
        v = r.get(f'{s}_sharing_source_frame_with_train_%')
        if v is not None and v > 0:
            flags.append(f"- [{r['group']}] {v}% of {s} images share a Roboflow source frame with train.")
    if r.get('largest_component_%', 0) > 20:
        flags.append(f"- [{r['group']}] one duplicate/video component holds {r['largest_component_%']}% of images; "
                     f"the grouped split may not reach 70:20:10 exactly.")
for (g, s), row in summary.iterrows():
    if row['missing_label_file'] or row['imgs_with_label_issues'] or row['corrupt']:
        flags.append(f"- [{g}/{s}] missing labels={row['missing_label_file']}, "
                     f"label issues={row['imgs_with_label_issues']}, corrupt={row['corrupt']}")
    if row['not_640x640']:
        flags.append(f"- [{g}/{s}] {row['not_640x640']} images are not 640x640 (paper states all are).")

n_neg = int((df.n_boxes == 0).sum())
report = f"""# BharatPotHole dataset assessment
Source: `{ZIP_PATH}`
Generated: {time.strftime('%Y-%m-%d %H:%M')}
Settings: pHash candidate <= {PHASH_CANDIDATE_THR}, strict <= {PHASH_STRICT_THR}, SSIM >= {SSIM_CONFIRM_THR}, low-light luma < {NIGHT_LUMA_THR}

## Paper claims to check
- ">7,000 annotated frames": **{len(df)} images** ({len(df) - n_neg} with boxes, {n_neg} without), {int(df.n_boxes.sum())} boxes.
- "70:20:10 split": see `split_share_%` below.
- "640x640 resolution": {int(((df.width != 640) | (df.height != 640)).sum())} images differ.

## Automated flags
{chr(10).join(flags) if flags else '- none raised'}

## Split summary
{md(summary.T)}

## Leakage summary
{md(leak_df.set_index('group').T) if not leak_df.empty else 'n/a'}

How to read this: `train_median_nn_phash_to_other_train` is the baseline similarity between training frames.
If val/test medians sit at or near that baseline, the split was made at frame level from the same videos,
so test performance reflects memorised scenes rather than generalisation to unseen roads.

## Proposed grouped re-split
{md(resplit)}

## Class distribution (boxes)
{md(class_dist) if not class_dist.empty else 'n/a'}

## Folder tree
```
{tree_txt}
```
"""
(OUT_DIR / 'assessment_report.md').write_text(report)
json.dump({'n_images': len(df), 'n_boxes': int(df.n_boxes.sum()), 'leakage': leak_rows,
           'seq_prefix_usable': seq_valid, 'class_names': class_names},
          open(OUT_DIR / 'summary.json', 'w'), indent=2, default=str)
print(report[:5000])


# %% [10] Figures (each isolated so one failure cannot stop the rest) --------------
def savefig(fig, name):
    fig.tight_layout(); fig.savefig(FIG_DIR / name, dpi=150); plt.close(fig)

def safe_plot(name, fn):
    try:
        fn()
    except Exception as e:
        plt.close('all'); print(f'  figure {name} skipped: {type(e).__name__}: {e}')

for g, gdf in df.groupby('group'):
    tag = safe(g)
    sp_list = [s for s in ['train', 'valid', 'test', 'unsplit'] if s in set(gdf.split)]
    gb = box_df[box_df.group == g] if not box_df.empty else box_df

    def p_boxes():
        fig, ax = plt.subplots(figsize=(7, 4)); mx = int(gdf.n_boxes.max()) + 2
        for s in sp_list:
            ax.hist(gdf.loc[gdf.split == s, 'n_boxes'].to_numpy(), bins=np.arange(0, mx), alpha=.5,
                    density=True, label=s)
        ax.set(xlabel='boxes per image (0 = negative frame)', ylabel='fraction', title=f'Boxes per image - {g}')
        ax.legend(); savefig(fig, f'boxes_per_image_{tag}.png')

    def p_size():
        fig, ax = plt.subplots(figsize=(7, 4))
        for s in sp_list:
            v = gb.loc[gb.split == s, 'sqrt_area_px'].dropna().to_numpy()
            if len(v): ax.hist(v, bins=60, alpha=.5, density=True, label=s)
        for v in (32, 96): ax.axvline(v, ls='--', c='k', lw=.8)
        ax.set(xlabel='sqrt(box area) [px]  (COCO small<32, medium<96)', ylabel='density', title=f'Box size - {g}')
        ax.legend(); savefig(fig, f'box_size_{tag}.png')

    def p_centres():
        fig, ax = plt.subplots(figsize=(5, 5))
        h = ax.hist2d(gb.xc.to_numpy(), gb.yc.to_numpy(), bins=40, range=[[0, 1], [0, 1]], cmap='viridis')
        ax.invert_yaxis(); fig.colorbar(h[3], ax=ax)
        ax.set(xlabel='x centre (norm.)', ylabel='y centre (norm.)', title=f'Box centres - {g}')
        savefig(fig, f'box_centres_{tag}.png')

    def p_bright():
        fig, ax = plt.subplots(figsize=(7, 4))
        for s in sp_list:
            ax.hist(gdf.loc[gdf.split == s, 'mean_luma'].dropna().to_numpy(), bins=50, alpha=.5,
                    density=True, label=s)
        ax.axvline(NIGHT_LUMA_THR, ls='--', c='k', lw=.8)
        ax.set(xlabel='mean grey level', ylabel='density', title=f'Brightness (low-light proxy) - {g}')
        ax.legend(); savefig(fig, f'brightness_{tag}.png')

    def p_nn():
        fig, ax = plt.subplots(figsize=(7, 4))
        for s in sp_list:
            v = gdf.loc[gdf.split == s, 'nn_phash_to_train'].dropna().to_numpy()
            lab = 'train -> other train (baseline)' if s == 'train' else f'{s} -> nearest train'
            ax.hist(v, bins=np.arange(0, 34), alpha=.5, density=True, label=lab)
        ax.axvline(PHASH_CANDIDATE_THR, ls='--', c='r', lw=.8)
        ax.set(xlabel='pHash Hamming distance to nearest TRAIN image', ylabel='density',
               title=f'Train-test proximity - {g}\n(val/test overlapping baseline = frame-level split)')
        ax.legend(); savefig(fig, f'nn_distance_to_train_{tag}.png')

    def p_leaks():
        leaks = pairs[pairs.confirmed & (pairs.relation == 'CROSS_SPLIT_LEAK') & (pairs.group_a == g)]
        if leaks.empty: return
        sub = leaks.sample(min(8, len(leaks)), random_state=SEED)
        fig, axes = plt.subplots(len(sub), 2, figsize=(9, 2.6 * len(sub)))
        axes = np.atleast_2d(axes)
        for r, row in enumerate(sub.itertuples(index=False)):
            for c, (idx, sp) in enumerate([(row.a, row.split_a), (row.b, row.split_b)]):
                im = cv2.imread(df.at[idx, 'path']); ax = axes[r, c]; ax.axis('off')
                if im is not None: ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
                t = f"{sp}: {Path(df.at[idx, 'path']).name[:38]}"
                if c == 0: t += f"\npHash={row.phash_dist}  SSIM={row.ssim:.2f}"
                ax.set_title(t, fontsize=7)
        savefig(fig, f'leak_examples_{tag}.png')

    safe_plot('boxes_per_image', p_boxes)
    if not gb.empty:
        safe_plot('box_size', p_size); safe_plot('box_centres', p_centres)
    safe_plot('brightness', p_bright)
    if 'nn_phash_to_train' in gdf.columns and 'train' in sp_list:
        safe_plot('nn_distance', p_nn)
    safe_plot('leak_examples', p_leaks)

print('\nSaved to', OUT_DIR)
for p in sorted(OUT_DIR.rglob('*')):
    if p.is_file(): print('  ', p.relative_to(OUT_DIR))