"""
RDD2022 India (D40 / pothole) dataset assessment  (Google Colab)
===============================================================
Same audit as for BharatPotHole, adapted to RDD2022, plus what is needed to use it as a
CROSS-DATASET test set:

  1. Structure, annotation FORMAT detection (Pascal-VOC XML, YOLO txt, COCO json - any mix)
  2. Class inventory (D00/D10/D20/D40/...), D40 selection, images that are
     pothole / other-damage-only (hard negatives) / background / UNANNOTATED (official test)
  3. Label integrity, image properties, box sizes (native AND at the 640-px inference scale)
  4. Duplicates & leakage inside RDD (MD5, Roboflow copies, pHash+SSIM near-duplicates)
  5. Overlap check against BharatPotHole (no shared/near-identical images)
  6. Domain-shift profile RDD-India vs BharatPotHole v2 (brightness, blur, box size, box position)
  7. Outputs a clean YOLO-format D40 dataset (class 0 = pothole) with:
       eval_all.yaml          all annotated images (potholes + crack-only + background)
       eval_pothole_only.yaml only images containing at least one D40
       grouped_split.yaml     leakage-free 70:20:10 split (for optional multi-dataset training)

Results: /content/drive/MyDrive/POTHOLE/rdd_india_assessment/   (zip itself is never modified)
Run the cells top to bottom.
"""

# %% [1] Setup & configuration ----------------------------------------------------------
try:                                   # Colab: mount Drive once; elsewhere set POTHOLE paths below
    import os as _os
    from google.colab import drive
    if not _os.path.ismount('/content/drive'):
        drive.mount('/content/drive')
except Exception:
    pass

import os, re, json, zipfile, hashlib, random, time, shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict
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

POTHOLE     = Path('/content/drive/MyDrive/POTHOLE')
ZIP_PATH    = POTHOLE / 'RDD2022 India Pothole D40.zip'
OUT_DIR     = POTHOLE / 'rdd_india_assessment'
EXTRACT_DIR = Path('/content/rdd_india_extracted')
LOCAL_YOLO  = Path('/content/RDD_India_D40_yolo')             # clean eval dataset (local)

# BharatPotHole references (for overlap + domain comparison; skipped if missing)
BPH_V2_LOCAL = Path('/content/BharatPotHole_v2')
BPH_V2_ZIP   = POTHOLE / 'BharatPotHole_v2' / 'BharatPotHole_v2.zip'
BPH_ASSESS   = POTHOLE / 'dataset_assessment'
BPH_MANIFEST = POTHOLE / 'BharatPotHole_v2' / 'manifest_v2.csv'

TARGET_NAMES = {'d40', 'pothole', 'potholes', 'pot hole', 'pot-hole'}
CLASS_NAMES_OVERRIDE = None   # e.g. ['D00','D10','D20','D40'] if YOLO labels come without a yaml
INFER_SIZE   = 640            # model input size used in all experiments

PHASH_CANDIDATE_THR, PHASH_STRICT_THR, SSIM_CONFIRM_THR = 8, 4, 0.80
MAX_SSIM_PAIRS = 60000
NIGHT_LUMA_THR, THUMB, N_WORKERS, SEED = 60, 128, 8, 42
TARGET_RATIOS = {'train': 0.70, 'valid': 0.20, 'test': 0.10}
N_TRIALS = 1000

IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
SPLIT_ALIASES = {'train': 'train', 'training': 'train', 'valid': 'valid', 'val': 'valid',
                 'validation': 'valid', 'test': 'test', 'testing': 'test'}
LAYOUT_DIRS = {'images', 'labels', 'image', 'label', 'imgs', 'annotations', 'annotation',
               'xmls', 'xml', 'jpegimages'}

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = OUT_DIR / 'figures'; FIG_DIR.mkdir(exist_ok=True)
random.seed(SEED); np.random.seed(SEED)
print('Results folder:', OUT_DIR)


# %% [2] Extract (including nested zips) -------------------------------------------------
def extract(zip_path, dest):
    marker = dest / '.extracted_ok'
    if marker.exists():
        print('Already extracted ->', dest); return
    assert zip_path.exists(), f'Not found: {zip_path}'
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        print(f'Extracting {len(zf.infolist())} entries from {zip_path.name} ...')
        zf.extractall(dest)
    for inner in list(dest.rglob('*.zip')):
        target = inner.with_suffix('')
        if not target.exists():
            with zipfile.ZipFile(inner) as zf:
                zf.extractall(target)
    marker.touch()

extract(ZIP_PATH, EXTRACT_DIR)


# %% [3] Discover structure and annotation formats ---------------------------------------
def is_junk(p):
    return any(x.startswith('__MACOSX') for x in p.parts) or p.name.startswith('._')

def locate(p):
    parts = [x for x in p.relative_to(EXTRACT_DIR).parts[:-1] if not x.startswith('__MACOSX')]
    split, idx = 'unsplit', None
    for i, part in enumerate(parts):
        if part.lower() in SPLIT_ALIASES:
            split, idx = SPLIT_ALIASES[part.lower()], i
    prefix = parts[:idx] if idx is not None else parts
    group = '/'.join(x for x in prefix if x.lower() not in LAYOUT_DIRS) or '(root)'
    return split, group

def tree_summary(root, max_depth=5):
    lines = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith(('__MACOSX', '.')))
        rel = Path(dirpath).relative_to(root)
        if len(rel.parts) > max_depth:
            continue
        exts = pd.Series([Path(f).suffix.lower() for f in filenames]).value_counts()
        desc = ', '.join(f'{e or "noext"}={n}' for e, n in exts.items()) or 'empty'
        lines.append(f"{'    ' * len(rel.parts)}{rel.name or root.name}/  [{desc}]")
    return '\n'.join(lines)

tree_txt = tree_summary(EXTRACT_DIR)
(OUT_DIR / 'folder_tree.txt').write_text(tree_txt)
print(tree_txt)

all_files = [p for p in EXTRACT_DIR.rglob('*') if p.is_file() and not is_junk(p)]
img_paths = sorted(p for p in all_files if p.suffix.lower() in IMG_EXT)
xml_paths = [p for p in all_files if p.suffix.lower() == '.xml']
txt_paths = [p for p in all_files if p.suffix.lower() == '.txt'
             and not p.name.lower().startswith('readme') and p.name.lower() != 'classes.txt']
json_paths = [p for p in all_files if p.suffix.lower() == '.json']
yaml_paths = [p for p in all_files if p.suffix.lower() in ('.yaml', '.yml')]
print(f'\nimages={len(img_paths)} xml={len(xml_paths)} txt={len(txt_paths)} '
      f'json={len(json_paths)} yaml={len(yaml_paths)}')

# class names for YOLO labels, per group
yolo_names = {}
for y in yaml_paths:
    if yaml is None:
        break
    try:
        d = yaml.safe_load(y.read_text(errors='ignore'))
    except Exception:
        continue
    if isinstance(d, dict) and 'names' in d:
        n = d['names']
        n = [n[k] for k in sorted(n)] if isinstance(n, dict) else n
        yolo_names[locate(y)[1]] = [str(x) for x in n]
for c in (p for p in all_files if p.name.lower() == 'classes.txt'):
    yolo_names.setdefault(locate(c)[1], c.read_text(errors='ignore').split())
print('YOLO class names found:', yolo_names or 'none')

# ---- annotation index: key (group, split, stem) -> dict(format, W, H, boxes[list]) ----
ann_index, ann_conflicts, parse_issues = {}, 0, []

def add_ann(key, rec):
    global ann_conflicts
    if key in ann_index:
        ann_conflicts += 1; return
    ann_index[key] = rec

for p in xml_paths:                                             # Pascal VOC
    try:
        root = ET.parse(p).getroot()
    except ET.ParseError as e:
        parse_issues.append({'file': str(p), 'issue': f'xml_parse_error:{e}'}); continue
    if root.tag != 'annotation':
        continue
    W = float(root.findtext('size/width') or 0); H = float(root.findtext('size/height') or 0)
    boxes = []
    for o in root.findall('object'):
        name = (o.findtext('name') or '').strip()
        bb = o.find('bndbox')
        try:
            x0, y0, x1, y1 = [float(bb.findtext(k)) for k in ('xmin', 'ymin', 'xmax', 'ymax')]
        except Exception:
            parse_issues.append({'file': str(p), 'issue': f'bad_bndbox:{name}'}); continue
        boxes.append({'name': name, 'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1, 'unit': 'px'})
    sp, g = locate(p)
    stem = Path(root.findtext('filename') or p.stem).stem
    add_ann((g, sp, stem), {'format': 'voc', 'W': W, 'H': H, 'boxes': boxes, 'file': str(p)})

for p in json_paths:                                            # COCO json (e.g. Roboflow export)
    try:
        d = json.loads(p.read_text(errors='ignore'))
    except Exception:
        continue
    if not (isinstance(d, dict) and 'images' in d and 'annotations' in d):
        continue
    cats = {c['id']: c['name'] for c in d.get('categories', [])}
    by_img = defaultdict(list)
    for a in d['annotations']:
        x, y, w, h = a['bbox']
        by_img[a['image_id']].append({'name': cats.get(a['category_id'], str(a['category_id'])),
                                      'x0': x, 'y0': y, 'x1': x + w, 'y1': y + h, 'unit': 'px'})
    sp, g = locate(p)
    for im in d['images']:
        add_ann((g, sp, Path(im['file_name']).stem),
                {'format': 'coco', 'W': im.get('width', 0), 'H': im.get('height', 0),
                 'boxes': by_img.get(im['id'], []), 'file': str(p)})

def yolo_class_name(g, cid):
    names = CLASS_NAMES_OVERRIDE or yolo_names.get(g) or next(iter(yolo_names.values()), None)
    if names and 0 <= cid < len(names):
        return names[cid]
    return f'class_{cid}'

for p in txt_paths:                                             # YOLO txt
    sp, g = locate(p)
    boxes = []
    for ln, line in enumerate(p.read_text(errors='ignore').splitlines(), 1):
        t = line.split()
        if not t:
            continue
        try:
            v = [float(x) for x in t]
        except ValueError:
            parse_issues.append({'file': str(p), 'issue': f'line{ln}:non_numeric'}); continue
        cid, c = int(v[0]), v[1:]
        if len(c) == 4:
            x0, y0, x1, y1 = c[0] - c[2] / 2, c[1] - c[3] / 2, c[0] + c[2] / 2, c[1] + c[3] / 2
        elif len(c) >= 6 and len(c) % 2 == 0:
            x0, x1, y0, y1 = min(c[0::2]), max(c[0::2]), min(c[1::2]), max(c[1::2])
        else:
            parse_issues.append({'file': str(p), 'issue': f'line{ln}:bad_token_count'}); continue
        boxes.append({'name': yolo_class_name(g, cid), 'cid': cid,
                      'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1, 'unit': 'norm'})
    add_ann((g, sp, p.stem), {'format': 'yolo', 'W': 0, 'H': 0, 'boxes': boxes, 'file': str(p)})

fmt_counts = pd.Series([r['format'] for r in ann_index.values()]).value_counts()
print('Annotation files by format:\n', fmt_counts.to_string() if len(fmt_counts) else 'none')
print('Duplicate annotations for the same image (first kept):', ann_conflicts)
if not yolo_names and not CLASS_NAMES_OVERRIDE and 'yolo' in fmt_counts:
    ids = sorted({b['cid'] for r in ann_index.values() if r['format'] == 'yolo' for b in r['boxes']})
    print(f'WARNING: YOLO labels without class names; ids present = {ids}.')
    if ids == [0]:
        CLASS_NAMES_OVERRIDE = ['pothole']
        for r in ann_index.values():
            for b in r['boxes']:
                if r['format'] == 'yolo':
                    b['name'] = 'pothole'
        print('  Only class 0 present and the zip is a D40 export -> treating class 0 as pothole.')
    else:
        print('  Set CLASS_NAMES_OVERRIDE in cell [1] and re-run from cell [3].')


# %% [4] Per-image analysis -------------------------------------------------------------
def phash_bits(gray):
    d = cv2.dct(cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32))[:8, :8].flatten()
    return d > np.median(d[1:])

def analyse_image(path):
    rec = {'path': str(path)}
    try:
        data = path.read_bytes()
        rec['md5'] = hashlib.md5(data).hexdigest()
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError('decode failed')
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rec.update(width=w, height=h, corrupt=False, mean_luma=float(gray.mean()),
                   blur_laplacian_var=float(cv2.Laplacian(gray, cv2.CV_64F).var()))
        return rec, phash_bits(gray), cv2.resize(gray, (THUMB, THUMB), interpolation=cv2.INTER_AREA)
    except Exception as e:
        rec.update(corrupt=True, error=str(e))
        return rec, None, None

t0 = time.time()
records, bits, thumbs = [], [], []
with ThreadPoolExecutor(N_WORKERS) as ex:
    for k, (rec, b, th) in enumerate(ex.map(analyse_image, img_paths), 1):
        records.append(rec); bits.append(b); thumbs.append(th)
        if k % 1000 == 0:
            print(f'  {k}/{len(img_paths)} ({time.time() - t0:.0f}s)')
df = pd.DataFrame(records)
df['corrupt'] = df['corrupt'].fillna(False).astype(bool)
print(f'Image pass: {time.time() - t0:.0f}s')


# %% [5] Join annotations, class inventory, box geometry ---------------------------------
RF_RE, SEQ_RE = re.compile(r'\.rf\.[0-9a-f]+$', re.I), re.compile(r'^(.*?)[_\-\s\.]*(\d+)$')

def source_name(stem):
    return re.sub(r'_(jpe?g|png|bmp|webp)$', '', RF_RE.sub('', stem), flags=re.I)

rows_box, cols = [], defaultdict(list)
for i, p in enumerate(img_paths):
    sp, g = locate(p)
    ann = ann_index.get((g, sp, p.stem))
    W, H = df.at[i, 'width'], df.at[i, 'height']
    src = source_name(p.stem)
    m = SEQ_RE.match(src)
    cols['split'].append(sp); cols['group'].append(g); cols['rel_path'].append(str(p.relative_to(EXTRACT_DIR)))
    cols['source_name'].append(src); cols['seq_prefix'].append((m.group(1) or '(numeric)') if m else src)
    cols['ann_format'].append(ann['format'] if ann else '')
    size_mismatch = bool(ann and ann['W'] and not df.at[i, 'corrupt']
                         and (abs(ann['W'] - W) > 1 or abs(ann['H'] - H) > 1))
    cols['ann_size_mismatch'].append(size_mismatch)
    n_t = n_o = n_bad = 0; others = set()
    for b in (ann['boxes'] if ann else []):
        if b['unit'] == 'px':
            if not W or not H:
                n_bad += 1; continue
            x0, y0, x1, y1 = b['x0'] / W, b['y0'] / H, b['x1'] / W, b['y1'] / H
        else:
            x0, y0, x1, y1 = b['x0'], b['y0'], b['x1'], b['y1']
        oob = min(x0, y0) < -1e-3 or max(x1, y1) > 1 + 1e-3
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, 1), min(y1, 1)
        is_t = b['name'].strip().lower() in TARGET_NAMES
        degenerate = (x1 - x0) * (W or 1) < 2 or (y1 - y0) * (H or 1) < 2
        if is_t:
            n_t += 1
        else:
            n_o += 1; others.add(b['name'])
        scale = INFER_SIZE / max(W, H) if W and H else np.nan
        rows_box.append({'img_idx': i, 'split': sp, 'group': g, 'class_name': b['name'], 'is_target': is_t,
                         'xc': (x0 + x1) / 2, 'yc': (y0 + y1) / 2, 'w': x1 - x0, 'h': y1 - y0,
                         'out_of_bounds': oob, 'degenerate': degenerate,
                         'sqrt_area_native_px': np.sqrt((x1 - x0) * W * (y1 - y0) * H) if W else np.nan,
                         'sqrt_area_640_px': np.sqrt((x1 - x0) * W * (y1 - y0) * H) * scale if W else np.nan})
    cols['n_target'].append(n_t); cols['n_other'].append(n_o); cols['n_bad_boxes'].append(n_bad)
    cols['other_classes'].append(','.join(sorted(others)))
    if ann is None:
        cat = 'unannotated'
    elif n_t:
        cat = 'pothole'
    elif n_o:
        cat = 'other_damage_only'
    else:
        cat = 'background'
    cols['category'].append(cat)
df = df.assign(**cols)
box_df = pd.DataFrame(rows_box)
if not box_df.empty:
    box_df['size_cat_640'] = pd.cut(box_df.sqrt_area_640_px, [0, 32, 96, np.inf], right=False,
                                    labels=['small', 'medium', 'large']).astype(str)
print(df.category.value_counts().to_string())
print('\nClass inventory (boxes):\n', box_df.class_name.value_counts().to_string() if len(box_df) else 'none')


# %% [6] Near-duplicates / leakage inside RDD -------------------------------------------
valid_idx = np.array([i for i, b in enumerate(bits) if b is not None])
B = np.stack([bits[i] for i in valid_idx]).astype(np.float32); Bc = 1 - B
n = len(valid_idx)
grp_arr = df.loc[valid_idx, 'group'].to_numpy(); spl_arr = df.loc[valid_idx, 'split'].to_numpy()
nn_same = np.full(n, 99, dtype=np.int16); nn_train = np.full(n, 99, dtype=np.int16)
train_mask = spl_arr == 'train'
pa, pb, pdist = [], [], []
for i0 in range(0, n, 512):
    i1 = min(n, i0 + 512)
    D = np.rint(B[i0:i1] @ Bc.T + Bc[i0:i1] @ B.T).astype(np.int16)
    D[np.arange(i1 - i0), np.arange(i0, i1)] = 99
    nn_same[i0:i1] = D.min(1)
    if train_mask.any():
        nn_train[i0:i1] = D[:, train_mask].min(1)
    r, c = np.nonzero(D <= PHASH_CANDIDATE_THR); r = r + i0; keep = c > r
    pa.append(r[keep]); pb.append(c[keep]); pdist.append(D[r[keep] - i0, c[keep]])
df.loc[valid_idx, 'nn_phash_any'] = np.where(nn_same > 64, np.nan, nn_same)
df.loc[valid_idx, 'nn_phash_to_train'] = np.where(nn_train > 64, np.nan, nn_train)

pairs = pd.DataFrame({'a': valid_idx[np.concatenate(pa)], 'b': valid_idx[np.concatenate(pb)],
                      'phash_dist': np.concatenate(pdist)}) if pa else pd.DataFrame(columns=['a', 'b', 'phash_dist'])
pairs['split_a'] = df.split.values[pairs.a.astype(int)]; pairs['split_b'] = df.split.values[pairs.b.astype(int)]
pairs['same_group'] = df.group.values[pairs.a.astype(int)] == df.group.values[pairs.b.astype(int)]
pairs['relation'] = np.where(~pairs.same_group, 'cross_group',
                    np.where(pairs.split_a != pairs.split_b, 'CROSS_SPLIT_LEAK', 'same_split_redundant'))
pairs = pairs.sort_values(['relation', 'phash_dist']).reset_index(drop=True)
pairs['ssim'] = np.nan
if ssim_fn is not None and len(pairs):
    m = min(len(pairs), MAX_SSIM_PAIRS)
    pairs.loc[:m - 1, 'ssim'] = [ssim_fn(thumbs[a], thumbs[b], data_range=255)
                                 for a, b in zip(pairs.a[:m], pairs.b[:m])]
pairs['md5_identical'] = df.md5.values[pairs.a.astype(int)] == df.md5.values[pairs.b.astype(int)]
pairs['confirmed'] = pairs.md5_identical | (pairs.phash_dist <= PHASH_STRICT_THR) | (pairs.ssim >= SSIM_CONFIRM_THR)
conf = pairs[pairs.confirmed & pairs.same_group]
df['n_confirmed_dups'] = 0; df['has_dup_in_train'] = False
for a, b, sa, sb in conf[['a', 'b', 'split_a', 'split_b']].itertuples(index=False):
    df.at[a, 'n_confirmed_dups'] += 1; df.at[b, 'n_confirmed_dups'] += 1
    if sb == 'train' and sa != 'train': df.at[a, 'has_dup_in_train'] = True
    if sa == 'train' and sb != 'train': df.at[b, 'has_dup_in_train'] = True
print(pairs.groupby('relation').confirmed.agg(['size', 'sum']).rename(columns={'size': 'candidates', 'sum': 'confirmed'}))

seq_counts = df.seq_prefix.value_counts()
seq_valid = bool(len(seq_counts) >= 10 and seq_counts.iloc[0] < 0.3 * len(df))
print('Filename prefixes usable as sequence IDs:', seq_valid, '| top prefixes:', seq_counts.head(5).to_dict())


# %% [7] Overlap with BharatPotHole + domain reference -----------------------------------
bph_imgs = []
if not (BPH_V2_LOCAL / 'data.yaml').exists() and BPH_V2_ZIP.exists():
    with zipfile.ZipFile(BPH_V2_ZIP) as zf:
        zf.extractall('/content')
if BPH_V2_LOCAL.exists():
    bph_imgs = sorted(p for p in BPH_V2_LOCAL.rglob('*') if p.suffix.lower() in IMG_EXT)
cross = {'bph_images_compared': len(bph_imgs)}
if bph_imgs:
    def bits_only(p):
        g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        return phash_bits(g) if g is not None else None
    with ThreadPoolExecutor(N_WORKERS) as ex:
        bb = [x for x in ex.map(bits_only, bph_imgs) if x is not None]
    Bb = np.stack(bb).astype(np.float32)
    nn_x = np.full(n, 99, dtype=np.int16)
    for i0 in range(0, n, 512):
        Bi = B[i0:i0 + 512]
        nn_x[i0:i0 + 512] = np.rint(Bi @ (1 - Bb).T + (1 - Bi) @ Bb.T).min(1)
    df.loc[valid_idx, 'nn_phash_to_bharatpothole'] = nn_x
    cross.update({'rdd_images_within_4_of_bph': int((nn_x <= PHASH_STRICT_THR).sum()),
                  'rdd_images_within_8_of_bph': int((nn_x <= PHASH_CANDIDATE_THR).sum()),
                  'median_nn_to_bph': float(np.median(nn_x))})
print('Cross-dataset overlap:', cross)

# BharatPotHole v2 statistics for the domain comparison
bph_img_stats = bph_box_stats = None
if (BPH_ASSESS / 'per_image.csv').exists():
    bpi = pd.read_csv(BPH_ASSESS / 'per_image.csv'); bpi['img_idx'] = np.arange(len(bpi))
    if BPH_MANIFEST.exists():
        kept = set(pd.read_csv(BPH_MANIFEST).query("new_split.notna() and new_split != ''", engine='python').rel_path)
        bpi = bpi[bpi.rel_path.isin(kept)]
    bph_img_stats = bpi
    if (BPH_ASSESS / 'per_box.csv').exists():
        bpb = pd.read_csv(BPH_ASSESS / 'per_box.csv')
        bph_box_stats = bpb[bpb.img_idx.isin(bpi.img_idx)]


# %% [8] Clean YOLO D40 dataset + grouped split -----------------------------------------
annot = df[(df.category != 'unannotated') & ~df.corrupt].copy()
annot = annot[~annot.duplicated('md5', keep='first')]                   # exact duplicates
annot['luma_dev'] = (annot.mean_luma - annot.groupby('source_name').mean_luma.transform('median')).abs()
annot['has_rf'] = annot.rel_path.map(lambda s: bool(RF_RE.search(Path(s).stem)))
rf_rows = annot[annot.has_rf].sort_values('luma_dev').drop_duplicates('source_name')  # one per Roboflow source frame
annot = pd.concat([annot[~annot.has_rf], rf_rows]).sort_index()   # non-Roboflow names are never merged by name

class UF:
    def __init__(self, idx): self.p = {i: i for i in idx}
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def union(self, a, b):
        if a in self.p and b in self.p:
            ra, rb = self.find(a), self.find(b)
            if ra != rb: self.p[rb] = ra

uf = UF(annot.index)
for a, b in conf[['a', 'b']].itertuples(index=False):
    uf.union(a, b)
if seq_valid:
    for _, idx in annot.groupby('seq_prefix').groups.items():
        idx = list(idx)
        for j in idx[1:]: uf.union(idx[0], j)
annot['component'] = [uf.find(i) for i in annot.index]

comp = annot.groupby('component').agg(images=('md5', 'size'), positives=('n_target', lambda s: int((s > 0).sum())),
                                      boxes=('n_target', 'sum'))
F = comp.to_numpy(float); tot = np.maximum(F.sum(0), 1)
r = np.array(list(TARGET_RATIOS.values()))[:, None]; w = np.array([1.0, 1.0, 0.5])
rng = np.random.default_rng(SEED); best, best_cost = None, np.inf
for _ in range(N_TRIALS):
    order = np.argsort(-F[:, 0] * rng.uniform(0.3, 1.7, len(F)))
    cur = np.zeros((3, F.shape[1])); asg = np.empty(len(F), int)
    for ci in order:
        d = ((np.abs((cur + F[ci]) / tot - r) - np.abs(cur / tot - r)) * w).sum(1)
        s = int(np.argmin(d + 1e-9 * rng.random(3))); cur[s] += F[ci]; asg[ci] = s
    cost = float((np.abs(cur / tot - r) * w).sum())
    if cost < best_cost:
        best, best_cost = asg, cost
comp['grouped_split'] = [list(TARGET_RATIOS)[i] for i in best]
annot['grouped_split'] = annot.component.map(comp.grouped_split)

if LOCAL_YOLO.exists():
    shutil.rmtree(LOCAL_YOLO)
(LOCAL_YOLO / 'images').mkdir(parents=True); (LOCAL_YOLO / 'labels').mkdir(parents=True)
names = annot.rel_path.map(lambda s: Path(s).name)
annot['dest_name'] = np.where(names.duplicated(keep=False), annot.split + '__' + names, names)
tb = box_df[box_df.is_target & ~box_df.degenerate] if len(box_df) else box_df
tb_by_img = {k: g for k, g in tb.groupby('img_idx')} if len(tb) else {}

def write_one(i):
    row = annot.loc[i]
    shutil.copy2(EXTRACT_DIR / row.rel_path, LOCAL_YOLO / 'images' / row.dest_name)
    g = tb_by_img.get(i)
    lines = [] if g is None else [f'0 {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}'
                                  for x, y, bw, bh in g[['xc', 'yc', 'w', 'h']].itertuples(index=False)]
    (LOCAL_YOLO / 'labels' / Path(row.dest_name).with_suffix('.txt')).write_text(
        '\n'.join(lines) + ('\n' if lines else ''))
with ThreadPoolExecutor(N_WORKERS) as ex:
    list(ex.map(write_one, annot.index))

def write_list(name, sel):
    (LOCAL_YOLO / name).write_text('\n'.join(str(LOCAL_YOLO / 'images' / d) for d in annot.loc[sel, 'dest_name']))
write_list('eval_all.txt', annot.index)
write_list('eval_pothole_only.txt', annot.index[annot.n_target > 0])
for s in TARGET_RATIOS:
    write_list(f'split_{s}.txt', annot.index[annot.grouped_split == s])
for yname, lst in [('eval_all', 'eval_all.txt'), ('eval_pothole_only', 'eval_pothole_only.txt')]:
    (LOCAL_YOLO / f'{yname}.yaml').write_text(
        f"# RDD2022 India D40 cross-dataset TEST set ({lst}); train/val keys only satisfy Ultralytics\n"
        f"path: {LOCAL_YOLO}\ntrain: {LOCAL_YOLO / lst}\nval: {LOCAL_YOLO / lst}\ntest: {LOCAL_YOLO / lst}\n"
        f"nc: 1\nnames: ['pothole']\n")
(LOCAL_YOLO / 'grouped_split.yaml').write_text(
    f"# RDD2022 India D40 leakage-free grouped split (seed {SEED})\npath: {LOCAL_YOLO}\n"
    f"train: {LOCAL_YOLO / 'split_train.txt'}\nval: {LOCAL_YOLO / 'split_valid.txt'}\n"
    f"test: {LOCAL_YOLO / 'split_test.txt'}\nnc: 1\nnames: ['pothole']\n")
print(f'Clean YOLO set: {len(annot)} images, {int(annot.n_target.sum())} D40 boxes -> {LOCAL_YOLO}')


# %% [9] Tables, figures, report, save to Drive ------------------------------------------
def summarize(g, bx):
    nb = g.n_target.to_numpy()
    return {'images': len(g), 'corrupt': int(g.corrupt.sum()),
            'pothole_imgs': int((g.category == 'pothole').sum()),
            'other_damage_only_imgs': int((g.category == 'other_damage_only').sum()),
            'background_imgs': int((g.category == 'background').sum()),
            'unannotated_imgs': int((g.category == 'unannotated').sum()),
            'd40_boxes': int(nb.sum()), 'other_boxes': int(g.n_other.sum()),
            'd40_per_pothole_img': round(float(nb[nb > 0].mean()), 2) if (nb > 0).any() else 0,
            'ann_size_mismatch': int(g.ann_size_mismatch.sum()),
            'low_light_%': round(100 * float((g.mean_luma < NIGHT_LUMA_THR).mean()), 1),
            'blur_median': round(float(g.blur_laplacian_var.median()), 1),
            'small_d40_at640_%': round(100 * float((bx.size_cat_640 == 'small').mean()), 1) if len(bx) else np.nan,
            'median_d40_sqrt_area_at640': round(float(bx.sqrt_area_640_px.median()), 1) if len(bx) else np.nan,
            'unique_source_frames': g.source_name.nunique()}

rows = []
for (grp, s), g in df.groupby(['group', 'split']):
    bx = box_df[(box_df.group == grp) & (box_df.split == s) & box_df.is_target] if len(box_df) else box_df
    rows.append({'group': grp, 'split': s, **summarize(g, bx)})
summary = pd.DataFrame(rows).set_index(['group', 'split'])
class_tab = (box_df.groupby(['split', 'class_name']).size().unstack(fill_value=0) if len(box_df) else pd.DataFrame())
img_sizes = (df.width.astype('Int64').astype(str) + 'x' + df.height.astype('Int64').astype(str)).value_counts().head(10)

# domain comparison
dom = {}
rdd_t = box_df[box_df.is_target] if len(box_df) else box_df
dom['RDD2022-India (annotated)'] = {
    'images': len(annot), 'positive_%': round(100 * float((annot.n_target > 0).mean()), 1),
    'boxes_per_pos_img': round(float(annot.loc[annot.n_target > 0, 'n_target'].mean()), 2),
    'mean_luma': round(float(annot.mean_luma.mean()), 1), 'blur_median': round(float(annot.blur_laplacian_var.median()), 1),
    'small_box_%_at640': round(100 * float((rdd_t.size_cat_640 == 'small').mean()), 1) if len(rdd_t) else np.nan,
    'median_box_sqrt_area_at640': round(float(rdd_t.sqrt_area_640_px.median()), 1) if len(rdd_t) else np.nan,
    'median_box_yc': round(float(rdd_t.yc.median()), 3) if len(rdd_t) else np.nan}
if bph_img_stats is not None:
    bbx = bph_box_stats if bph_box_stats is not None else pd.DataFrame()
    dom['BharatPotHole v2'] = {
        'images': len(bph_img_stats), 'positive_%': round(100 * float((bph_img_stats.n_boxes > 0).mean()), 1),
        'boxes_per_pos_img': round(float(bph_img_stats.loc[bph_img_stats.n_boxes > 0, 'n_boxes'].mean()), 2),
        'mean_luma': round(float(bph_img_stats.mean_luma.mean()), 1),
        'blur_median': round(float(bph_img_stats.blur_laplacian_var.median()), 1),
        'small_box_%_at640': round(100 * float(bbx.size_cat.astype(str).str.startswith('small').mean()), 1) if len(bbx) else np.nan,
        'median_box_sqrt_area_at640': round(float(bbx.sqrt_area_px.median()), 1) if len(bbx) else np.nan,
        'median_box_yc': round(float(bbx.yc.median()), 3) if len(bbx) else np.nan}
domain = pd.DataFrame(dom)

def savefig(fig, name):
    fig.tight_layout(); fig.savefig(FIG_DIR / name, dpi=150); plt.close(fig)
try:
    if len(rdd_t):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(rdd_t.sqrt_area_640_px.dropna(), bins=60, alpha=.5, density=True, label='RDD-India D40')
        if bph_box_stats is not None:
            ax.hist(bph_box_stats.sqrt_area_px.dropna(), bins=60, alpha=.5, density=True, label='BharatPotHole v2')
        for v in (32, 96): ax.axvline(v, ls='--', c='k', lw=.8)
        ax.set(xlabel=f'sqrt(box area) at {INFER_SIZE}-px input', ylabel='density', title='Pothole box size'); ax.legend()
        savefig(fig, 'box_size_rdd_vs_bph.png')
        fig, axes = plt.subplots(1, 2 if bph_box_stats is not None else 1, figsize=(10, 4.5))
        axes = np.atleast_1d(axes)
        for ax, (lab, d) in zip(axes, [('RDD-India D40', rdd_t), ('BharatPotHole v2', bph_box_stats)]):
            if d is None: continue
            h = ax.hist2d(d.xc, d.yc, bins=40, range=[[0, 1], [0, 1]], cmap='viridis'); ax.invert_yaxis()
            fig.colorbar(h[3], ax=ax); ax.set(title=f'Box centres - {lab}', xlabel='x', ylabel='y')
        savefig(fig, 'box_centres_rdd_vs_bph.png')
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(annot.mean_luma, bins=50, alpha=.5, density=True, label='RDD-India')
    if bph_img_stats is not None:
        ax.hist(bph_img_stats.mean_luma, bins=50, alpha=.5, density=True, label='BharatPotHole v2')
    ax.axvline(NIGHT_LUMA_THR, ls='--', c='k', lw=.8); ax.set(xlabel='mean grey level', title='Brightness'); ax.legend()
    savefig(fig, 'brightness_rdd_vs_bph.png')
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df.nn_phash_any.dropna(), bins=np.arange(0, 34), alpha=.6, density=True, label='RDD -> nearest RDD')
    if 'nn_phash_to_bharatpothole' in df:
        ax.hist(df.nn_phash_to_bharatpothole.dropna(), bins=np.arange(0, 34), alpha=.6, density=True,
                label='RDD -> nearest BharatPotHole')
    ax.axvline(PHASH_CANDIDATE_THR, ls='--', c='r', lw=.8); ax.set(xlabel='pHash distance', title='Redundancy & overlap'); ax.legend()
    savefig(fig, 'nn_distance_rdd.png')
except Exception as e:
    plt.close('all'); print('figure step skipped:', e)

df.drop(columns=['path']).to_csv(OUT_DIR / 'per_image.csv', index=False)
box_df.to_csv(OUT_DIR / 'per_box.csv', index=False)
pd.DataFrame(parse_issues, columns=['file', 'issue']).to_csv(OUT_DIR / 'parse_issues.csv', index=False)
summary.to_csv(OUT_DIR / 'split_summary.csv'); class_tab.to_csv(OUT_DIR / 'class_distribution.csv')
domain.to_csv(OUT_DIR / 'domain_comparison.csv')
pairs.assign(img_a=df.rel_path.values[pairs.a.astype(int)], img_b=df.rel_path.values[pairs.b.astype(int)]) \
     .drop(columns=['a', 'b']).to_csv(OUT_DIR / 'near_duplicate_pairs.csv', index=False)
annot[['rel_path', 'split', 'category', 'n_target', 'n_other', 'component', 'grouped_split', 'dest_name']] \
     .to_csv(OUT_DIR / 'clean_set_manifest.csv', index=False)

flags = []
if (df.category == 'unannotated').any():
    flags.append(f"- {int((df.category == 'unannotated').sum())} images have NO annotation "
                 f"(likely the official RDD test set) - excluded from all evaluation sets.")
if not len(rdd_t):
    flags.append('- **No D40/pothole boxes identified - check class names (CLASS_NAMES_OVERRIDE).**')
if (df.category == 'other_damage_only').any():
    flags.append(f"- {int((df.category == 'other_damage_only').sum())} images contain only cracks (D00/D10/D20): "
                 f"kept as hard negatives in eval_all; excluded from eval_pothole_only.")
if len(conf):
    flags.append(f"- {len(conf)} confirmed near-duplicate pairs inside RDD "
                 f"({int((conf.relation == 'CROSS_SPLIT_LEAK').sum())} across its own splits).")
if cross.get('rdd_images_within_4_of_bph'):
    flags.append(f"- **{cross['rdd_images_within_4_of_bph']} RDD images are near-identical to BharatPotHole images - inspect.**")
if int(df.ann_size_mismatch.sum()):
    flags.append(f"- {int(df.ann_size_mismatch.sum())} annotations declare a size different from the image file.")
if len(box_df) and int(box_df.out_of_bounds.sum()):
    flags.append(f"- {int(box_df.out_of_bounds.sum())} boxes extend outside the image (clipped).")
if len(parse_issues):
    flags.append(f"- {len(parse_issues)} annotation parse issues (parse_issues.csv).")

md = lambda t: t.to_markdown() if hasattr(t, 'to_markdown') else t.to_string()
try:
    md(pd.DataFrame({'a': [1]}))
except Exception:
    md = lambda t: '```\n' + t.to_string() + '\n```'
report = f"""# RDD2022 India D40 assessment
Source: `{ZIP_PATH}`  |  Generated {time.strftime('%Y-%m-%d %H:%M')}

## Flags
{chr(10).join(flags) if flags else '- none'}

## Annotation formats
{md(fmt_counts.to_frame('files')) if len(fmt_counts) else 'none'}

## Image categories
{md(df.category.value_counts().to_frame('images'))}

## Class inventory (boxes per split)
{md(class_tab) if len(class_tab) else 'n/a'}

## Split summary
{md(summary.T)}

## Image sizes (top 10)
{md(img_sizes.to_frame('images'))}

## Overlap with BharatPotHole
{json.dumps(cross, indent=2)}

## Domain comparison (what a BharatPotHole-trained model will face)
{md(domain)}

## Clean YOLO D40 set ({LOCAL_YOLO.name})
- {len(annot)} annotated images after de-duplication, {int(annot.n_target.sum())} D40 boxes (class 0 = pothole)
- eval_all.yaml: all annotated images (potholes + crack-only hard negatives + background)
- eval_pothole_only.yaml: {int((annot.n_target > 0).sum())} images with >=1 pothole
- grouped_split.yaml: leakage-free split for optional multi-dataset training
{md(annot.groupby('grouped_split').agg(images=('md5', 'size'), pothole_imgs=('n_target', lambda s: int((s > 0).sum())), d40_boxes=('n_target', 'sum')))}

## Folder tree
```
{tree_txt}
```
"""
(OUT_DIR / 'assessment_report.md').write_text(report)
for f in LOCAL_YOLO.glob('*.yaml'):
    shutil.copy2(f, OUT_DIR / f.name)
shutil.make_archive(str(OUT_DIR / 'RDD_India_D40_yolo'), 'zip', root_dir=LOCAL_YOLO.parent, base_dir=LOCAL_YOLO.name)
print(report[:6000])
print('\nSaved to', OUT_DIR)