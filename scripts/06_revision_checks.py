#!/usr/bin/env python3
"""
Revision checks for the BharatPotHole leakage manuscript (Google Colab, T4).
No training. Re-uses the cached predictions of 05_protocol_a_and_evaluation.py; the only new inference is
on augmentation copies of C-test frames (a few hundred images, about 1-2 min on a T4).

  R1  Composition of every training and test set: images, positive images, boxes, videos
  R2  Leakage criteria in A-test: counts and overlaps (MD5/source frame, pHash, SSIM-confirmed)
  R3  Sensitivity of the A-test vs near-duplicate-free gap to the pHash threshold (2, 4, 6, 8)
  R4  Bootstrap check: 5,000 video resamples, number of discarded (empty) resamples
  R6  Subset results for B-clean (stated rule) and B-strict, and AP50 by d_H bin (positive images)
  R5  Substitution controls: (a) copies of UNSEEN C-test frames instead of training frames,
      (b) 20 random draws at rho = 100%, (c) box count and size of each substitution pool

HOW TO RUN
  Cell 1:  from google.colab import drive; drive.mount('/content/drive')
  Cell 2:  !python "/content/drive/MyDrive/POTHOLE/scripts/06_revision_checks.py"
           With no arguments all checks run (R1-R6). A subset: add e.g. R2 R5
OUTPUT:  .../experiments/leakage_paper/revision/REVISION_CHECKS.md  (+ CSV files)
Paste REVISION_CHECKS.md back into the chat.
"""
import os, sys, json, time, pickle, zipfile, subprocess
from pathlib import Path
import numpy as np

ULTRA_VERSION = '8.4.162'
POTHOLE   = Path(os.environ.get('POTHOLE_DRIVE', '/content/drive/MyDrive/POTHOLE'))
OUT       = POTHOLE / 'experiments' / 'leakage_paper'
REV       = OUT / 'revision'
FAMILY    = POTHOLE / 'experiments' / 'family_study_v2'
ASSESS    = POTHOLE / 'dataset_assessment'
V2_DRIVE  = POTHOLE / 'BharatPotHole_v2'
ORIG_ZIP  = POTHOLE / 'archive (1).zip'
ORIG_DIR  = Path('/content/pothole_extracted')
V2_DIR    = Path('/content/BharatPotHole_v2')

SEED = 0
IMGSZ, CONF, NMS_IOU, MAX_DET = 640, 0.001, 0.7, 300
MIN_BOX = 2 / 640
BOOT_MAIN, BOOT_CHECK = 1000, 5000
RUN = {a.upper() for a in sys.argv[1:]} or {'R1', 'R2', 'R3', 'R4', 'R5', 'R6'}
DOSE_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0]
DOSE_REPS = 20
TAUS = [2, 4, 6, 8]
IOUS = np.linspace(0.5, 0.95, 10)
_GT = {}
MD = []


def log(m):
    print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)


def md(line=''):
    MD.append(line)
    print(line, flush=True)


def ensure_ultralytics():
    try:
        from importlib.metadata import version
        v = version('ultralytics')
    except Exception:
        v = None
    if v != ULTRA_VERSION:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', f'ultralytics=={ULTRA_VERSION}'], check=True)


def setup_data():
    if not (ORIG_DIR / '.extracted_ok').exists():
        log('Extracting original release ...')
        with zipfile.ZipFile(ORIG_ZIP) as z:
            z.extractall(ORIG_DIR)
        (ORIG_DIR / '.extracted_ok').touch()
    if not (V2_DIR / 'data.yaml').exists():
        log('Extracting v2 split ...')
        with zipfile.ZipFile(V2_DRIVE / 'BharatPotHole_v2.zip') as z:
            z.extractall(V2_DIR.parent)


# ------------------------------------------------------------------ evaluator (identical to main script)
def label_path(img):
    s = str(img); i = s.rfind('/images/')
    return Path(s[:i] + '/labels/' + s[i + 8:]).with_suffix('.txt')


def read_gt(img):
    if img in _GT:
        return _GT[img]
    boxes, p = [], label_path(img)
    if p.exists():
        for line in p.read_text(errors='ignore').splitlines():
            t = line.split()
            if len(t) < 5:
                continue
            try:
                v = [float(x) for x in t]
            except ValueError:
                continue
            c = v[1:]
            if len(c) == 4:
                x0, y0, x1, y1 = c[0] - c[2] / 2, c[1] - c[3] / 2, c[0] + c[2] / 2, c[1] + c[3] / 2
            elif len(c) >= 6 and len(c) % 2 == 0:
                x0, x1, y0, y1 = min(c[0::2]), max(c[0::2]), min(c[1::2]), max(c[1::2])
            else:
                continue
            x0, y0, x1, y1 = [min(max(a, 0.0), 1.0) for a in (x0, y0, x1, y1)]
            if x1 - x0 >= MIN_BOX and y1 - y0 >= MIN_BOX:
                boxes.append((round(x0, 6), round(y0, 6), round(x1, 6), round(y1, 6)))
    b = np.array(sorted(set(boxes)), dtype=np.float32).reshape(-1, 4)
    _GT[img] = b
    return b


def box_iou(a, b):
    ix0 = np.maximum(a[:, None, 0], b[None, :, 0]); iy0 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix1 = np.minimum(a[:, None, 2], b[None, :, 2]); iy1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix1 - ix0, 0, None) * np.clip(iy1 - iy0, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-12)


def match(pb, pc, gb):
    n = len(pc)
    tp = np.zeros((n, len(IOUS)), bool)
    if n == 0 or len(gb) == 0:
        return tp
    order = np.argsort(-pc, kind='stable')
    iou = box_iou(pb[order], gb)
    rows = np.where(iou.max(1) >= IOUS[0])[0]
    for j, t in enumerate(IOUS):
        used = np.zeros(len(gb), bool)
        for i in rows:
            ok = np.where((iou[i] >= t) & ~used)[0]
            if len(ok):
                k = ok[np.argmax(iou[i, ok])]
                used[k] = True
                tp[order[i], j] = True
    return tp


def ap_curve(r, p):
    mr = np.concatenate([[0.0], r, [1.0]]); mp = np.concatenate([[1.0], p, [0.0]])
    mp = np.flip(np.maximum.accumulate(np.flip(mp)))
    i = np.where(mr[1:] != mr[:-1])[0]
    return float(np.sum((mr[i + 1] - mr[i]) * mp[i + 1]))


def metrics(keys, rec):
    ngt = int(sum(rec[k][2] for k in keys))
    out = {'n_images': len(keys), 'n_pos': int(sum(rec[k][2] > 0 for k in keys)), 'n_gt': ngt}
    if ngt == 0:
        return {**out, 'AP50': np.nan, 'AP50_95': np.nan, 'R': np.nan}
    conf = np.concatenate([rec[k][0] for k in keys])
    tp = np.concatenate([rec[k][1] for k in keys]).astype(float)
    o = np.argsort(-conf, kind='stable'); tp = tp[o]
    ctp, cfp = np.cumsum(tp, 0), np.cumsum(1 - tp, 0)
    rc = ctp / ngt; pr = ctp / np.maximum(ctp + cfp, 1e-12)
    aps = [ap_curve(rc[:, j], pr[:, j]) for j in range(len(IOUS))]
    f1 = 2 * pr[:, 0] * rc[:, 0] / np.maximum(pr[:, 0] + rc[:, 0], 1e-12)
    i = int(np.argmax(f1)) if len(f1) else 0
    return {**out, 'AP50': aps[0], 'AP50_95': float(np.mean(aps)), 'R': float(rc[i, 0]) if len(f1) else 0.0}


def bootstrap(groups, rec, n_boot, seed=SEED):
    rng = np.random.default_rng(seed)
    gl = list(groups)
    names = sorted({n for d in groups.values() for n in d})
    res = {n: np.empty(n_boot) for n in names}
    for b in range(n_boot):
        pick = rng.integers(0, len(gl), len(gl))
        for n in names:
            keys = [k for i in pick for k in groups[gl[i]].get(n, ())]
            res[n][b] = metrics(keys, rec)['AP50'] if keys else np.nan
    return res


def ci(a):
    a = a[~np.isnan(a)]
    return (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))) if len(a) else (np.nan, np.nan)


def predict_cached(weights, keys, tag):
    f = OUT / 'cache' / f'{tag}.pkl'
    cache = pickle.load(open(f, 'rb')) if f.exists() else {}
    todo = [k for k in dict.fromkeys(keys) if k not in cache]
    if todo:
        from ultralytics import YOLO
        log(f'{tag}: predicting {len(todo)} new images ({len(cache)} cached)')
        model = YOLO(str(weights))
        import torch
        dev = 0 if torch.cuda.is_available() else 'cpu'
        for i in range(0, len(todo), 64):
            chunk = todo[i:i + 64]
            for attempt in (0, 1):
                try:
                    out = {}
                    for k, r in zip(chunk, model.predict(source=chunk, imgsz=IMGSZ, conf=CONF, iou=NMS_IOU,
                                                         max_det=MAX_DET, batch=4, device=dev, verbose=False,
                                                         stream=True)):
                        assert Path(r.path).name == Path(k).name
                        out[k] = (r.boxes.xyxyn.cpu().numpy().astype(np.float32),
                                  r.boxes.conf.cpu().numpy().astype(np.float32))
                    cache.update(out)
                    break
                except torch.OutOfMemoryError:
                    if attempt:
                        raise
                    log('GPU out of memory (another process holds the GPU) - continuing on CPU')
                    torch.cuda.empty_cache(); dev = 'cpu'
                    model = YOLO(str(weights))
        tmp = f.with_suffix('.tmp'); pickle.dump(cache, open(tmp, 'wb')); os.replace(tmp, f)
    return cache


def records(cache, keys):
    rec = {}
    for k in dict.fromkeys(keys):
        pb, pc = cache[k]
        gb = read_gt(k)
        rec[k] = (pc, match(pb, pc, gb), len(gb))
    return rec


def box_stats(keys):
    n_img = len(keys)
    b = [read_gt(k) for k in keys]
    nb = np.array([len(x) for x in b])
    allb = np.concatenate([x for x in b if len(x)]) if nb.sum() else np.zeros((0, 4))
    side = np.sqrt((allb[:, 2] - allb[:, 0]) * (allb[:, 3] - allb[:, 1])) * 640
    return {'images': n_img, 'boxes': int(nb.sum()), 'boxes_per_image': round(float(nb.mean()), 2) if n_img else np.nan,
            'small_boxes_%': round(100 * float((side < 32).mean()), 1) if len(side) else np.nan,
            'median_box_px': round(float(np.median(side)), 1) if len(side) else np.nan}


def table(rows, cols):
    md('| ' + ' | '.join(cols) + ' |')
    md('|' + '---|' * len(cols))
    for r in rows:
        md('| ' + ' | '.join(str(r.get(c, '')) for c in cols) + ' |')
    md()


# ------------------------------------------------------------------ main
def main():
    import pandas as pd
    ensure_ultralytics(); setup_data()
    REV.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    pi = pd.read_csv(ASSESS / 'per_image.csv')
    man = pd.read_csv(V2_DRIVE / 'manifest_v2.csv')
    for c in ('has_dup_in_train', 'source_in_train'):
        pi[c] = pi[c].fillna(False).astype(bool)
    md(f'# Revision checks ({time.strftime("%Y-%m-%d %H:%M")})')
    md(f'per_image.csv columns: {list(pi.columns)}')
    md(f'manifest_v2.csv columns: {list(man.columns)}')
    md()

    # ---------------- sets (as in the main script)
    t = pi[pi.split == 'test'].copy()
    t['key'] = [str(ORIG_DIR / p) for p in t.rel_path]
    A_test = list(t.key)
    video = dict(zip(t.key, t.seq_prefix))
    nn = dict(zip(t.key, t.nn_phash_to_train))
    dup = dict(zip(t.key, t.has_dup_in_train)); src = dict(zip(t.key, t.source_in_train))
    leaked = {k for k in A_test if dup[k] or src[k]}
    B_clean = [l.strip() for l in (V2_DRIVE / 'B_original_test_clean.txt').read_text().splitlines() if l.strip()]
    B_clean = [k for k in B_clean if k in video]
    cm = man[man.new_split == 'test']
    C_test = [str(V2_DIR / 'test' / 'images' / d) for d in cm.dest_name]
    video.update(zip(C_test, cm.seq_prefix))
    reps = man[man.is_rep.astype(bool) & man.new_split.notna()]
    src2split = dict(zip(reps.source_name, reps.new_split))
    copies = man[(man.drop_reason == 'augmentation_copy') & (man.n_boxes > 0)].copy()
    copies['src_split'] = copies.source_name.map(src2split)
    tr_cp = copies[copies.src_split == 'train'].sample(frac=1, random_state=SEED).drop_duplicates('source_name')
    te_cp = copies[copies.src_split == 'test'].sample(frac=1, random_state=SEED).drop_duplicates('source_name')
    POOL_TRAIN = [str(ORIG_DIR / p) for p in tr_cp.rel_path]
    POOL_TEST = [str(ORIG_DIR / p) for p in te_cp.rel_path]
    log(f'A_test {len(A_test)} | B_clean {len(B_clean)} | leaked {len(leaked)} | C_test {len(C_test)} | '
        f'train-copy pool {len(POOL_TRAIN)} | test-copy pool {len(POOL_TEST)}')

    # ---------------- R1 composition
    if 'R1' in RUN:
        md('## R1 Composition of training and test sets')
        rows = []
        a_tr = pi[pi.split == 'train']
        a_keys = [str(ORIG_DIR / p) for p in a_tr.rel_path]
        s = box_stats(a_keys)
        pos_a = [k for k in a_keys if len(read_gt(k))]
        row = {'set': 'Protocol A train', **s, 'positive_images': len(pos_a), 'videos': a_tr.seq_prefix.nunique()}
        if 'rel_path' in man.columns and 'source_name' in man.columns:
            m2 = man.drop_duplicates('rel_path').set_index('rel_path')['source_name']
            srcs = {m2.get(p) for p, k in zip(a_tr.rel_path, a_keys) if len(read_gt(k)) and p in m2.index}
            row['unique_positive_source_frames'] = len({x for x in srcs if isinstance(x, str)})
        rows.append(row)
        c_tr = man[(man.new_split == 'train') & man.is_rep.astype(bool)]
        c_keys = [str(V2_DIR / 'train' / 'images' / d) for d in c_tr.dest_name]
        s = box_stats(c_keys)
        rows.append({'set': 'Protocol C train', **s, 'positive_images': sum(len(read_gt(k)) > 0 for k in c_keys),
                     'videos': c_tr.seq_prefix.nunique(), 'unique_positive_source_frames': sum(len(read_gt(k)) > 0 for k in c_keys)})
        for name, keys in (('A-test', A_test), ('Leaked', sorted(leaked)), ('B-clean', B_clean), ('C-test', C_test)):
            s = box_stats(keys)
            pos = [k for k in keys if len(read_gt(k))]
            pv = pd.Series([video[k] for k in pos]).value_counts() if pos else pd.Series(dtype=int)
            rows.append({'set': name, **s, 'positive_images': len(pos), 'videos': len({video[k] for k in keys}),
                         'videos_with_positives': len(pv),
                         'pos_per_video_median': float(pv.median()) if len(pv) else 0,
                         'pos_per_video_max': int(pv.max()) if len(pv) else 0})
        cols = ['set', 'images', 'positive_images', 'boxes', 'boxes_per_image', 'small_boxes_%', 'median_box_px',
                'videos', 'videos_with_positives', 'pos_per_video_median', 'pos_per_video_max',
                'unique_positive_source_frames']
        table(rows, cols)
        pd.DataFrame(rows).to_csv(REV / 'R1_composition.csv', index=False)
        trv = set(a_tr.seq_prefix)
        md(f'A-test images from videos NOT in Protocol A training: {sum(video[k] not in trv for k in A_test)}')
        md()

    # ---------------- R2 leakage criteria
    if 'R2' in RUN:
        md('## R2 Leakage criteria in A-test (n = %d)' % len(A_test))
        d = np.array([nn[k] for k in A_test], dtype=float)
        D = np.array([dup[k] for k in A_test]); S = np.array([src[k] for k in A_test])
        crit = [
            ('confirmed near-duplicate (pHash rule: d<=4, or d 5-8 with SSIM>=0.80)', D.sum()),
            ('shared source frame', S.sum()),
            ('d_H <= 4', np.nansum(d <= 4)),
            ('d_H 5-8', np.nansum((d >= 5) & (d <= 8))),
            ('d_H 5-8 confirmed by SSIM', np.nansum(D & (d >= 5) & (d <= 8))),
            ('confirmed AND shared source', (D & S).sum()),
            ('confirmed only', (D & ~S).sum()),
            ('shared source only', (~D & S).sum()),
            ('leaked (either)', (D | S).sum()),
            ('d_H <= 4 but not leaked', np.nansum((d <= 4) & ~(D | S))),
            ('leaked but d_H > 4', np.nansum((d > 4) & (D | S))),
        ]
        table([{'criterion': c, 'images': int(v), '%': round(100 * v / len(A_test), 1)} for c, v in crit],
              ['criterion', 'images', '%'])

        bset = set(B_clean)
        neither = [k for k in A_test if k not in leaked and k not in bset]
        md(f'A-test images in neither Leaked nor B-clean: {len(neither)}')
        if neither:
            dn = np.array([nn[k] for k in neither], dtype=float)
            md(f'  their d_H: min {np.nanmin(dn):.0f}, median {np.nanmedian(dn):.0f}, max {np.nanmax(dn):.0f}; '
               f'd_H<=4: {int(np.nansum(dn <= 4))}, 5-8: {int(np.nansum((dn >= 5) & (dn <= 8)))}, >8: {int(np.nansum(dn > 8))}; '
               f'positive: {sum(len(read_gt(k)) > 0 for k in neither)}')
            md(f'  example files: {[Path(k).name for k in neither[:5]]}')
            tt = t.set_index('key')
            trn = pi[pi.split == 'train']; val = pi[pi.split == 'valid']
            comp_tr = set(trn['component'].dropna()); src_val = set(val['source_name'].dropna())
            src_tr = set(trn['source_name'].dropna())
            sub = tt.loc[neither]
            diag = {
                'component shared with a training image': int(sub['component'].isin(comp_tr).sum()),
                'source frame in validation': int(sub['source_name'].isin(src_val).sum()),
                'source frame in training (by name)': int(sub['source_name'].isin(src_tr).sum()),
                'nn_phash_to_valid <= 4': int((sub['nn_phash_to_valid'] <= 4).sum()),
                'n_confirmed_dups > 0': int((sub['n_confirmed_dups'].fillna(0) > 0).sum()),
                'seq_prefix_in_train': int(sub['seq_prefix_in_train'].fillna(False).astype(bool).sum()),
                'label_status != ok': int((sub['label_status'].astype(str) != 'ok').sum()),
                'n_label_issues > 0': int((sub['n_label_issues'].fillna(0) > 0).sum()),
                'low_light': int(sub['low_light'].fillna(False).astype(bool).sum()),
            }
            md('  diagnostics: ' + '; '.join(f'{k} = {v}' for k, v in diag.items()))
            md(f"  label_status values: {sub['label_status'].astype(str).value_counts().to_dict()}")
            sb = tt.loc[[k for k in B_clean if k in tt.index]]
            md('  same diagnostics for B-clean (n=%d): component shared with train = %d; label_status != ok = %d; '
               'n_confirmed_dups > 0 = %d' % (len(sb), int(sb['component'].isin(comp_tr).sum()),
                                              int((sb['label_status'].astype(str) != 'ok').sum()),
                                              int((sb['n_confirmed_dups'].fillna(0) > 0).sum())))
        md(f'B-clean images that are also flagged Leaked: {len(bset & leaked)}')
        md()

    # ---------------- Protocol A model records
    if 'R3' in RUN:
        wA = OUT / 'runs' / 'yolo26n_A_s0' / 'weights' / 'best.pt'
        cacheA = predict_cached(wA, A_test, 'yolo26n_A')
        recA = records(cacheA, A_test)
        pos_set = {k for k in A_test if recA[k][2] > 0}

    # ---------------- R3 threshold sensitivity (positive images only)
    if 'R3' in RUN:
        md('## R3 Sensitivity to the pHash threshold (positive A-test images, YOLO26n Protocol A)')
        md('Near-duplicate-free subset at threshold tau: d_H > tau and no shared source frame.')
        rows = []
        for tau in TAUS:
            clean = [k for k in A_test if k in pos_set and not src[k] and not (nn[k] <= tau)]
            allp = [k for k in A_test if k in pos_set]
            g = {}
            for k in allp:
                g.setdefault(video[k], {}).setdefault('A', []).append(k)
                if k in clean:
                    g[video[k]].setdefault('B', []).append(k)
            bs = bootstrap(g, recA, BOOT_MAIN)
            mA, mB = metrics(allp, recA)['AP50'], metrics(clean, recA)['AP50']
            lo, hi = ci(bs['A'] - bs['B'])
            rows.append({'tau': tau, 'n_clean_pos': len(clean), 'clean_videos': len({video[k] for k in clean}),
                         'AP50_A': round(mA, 3), 'AP50_clean': round(mB, 3), 'diff': round(mA - mB, 3),
                         'CI_lo': round(lo, 3), 'CI_hi': round(hi, 3), 'rel_%': round(100 * (mA - mB) / mB, 1)})
        # reference: the manuscript definition (confirmed near-duplicate or shared source)
        clean = [k for k in B_clean if k in pos_set]
        allp = [k for k in A_test if k in pos_set]
        g = {}
        for k in allp:
            g.setdefault(video[k], {}).setdefault('A', []).append(k)
            if k in set(clean):
                g[video[k]].setdefault('B', []).append(k)
        bs1 = bootstrap(g, recA, BOOT_MAIN)
        mA, mB = metrics(allp, recA)['AP50'], metrics(clean, recA)['AP50']
        lo, hi = ci(bs1['A'] - bs1['B'])
        rows.append({'tau': 'manuscript rule', 'n_clean_pos': len(clean), 'clean_videos': len({video[k] for k in clean}),
                     'AP50_A': round(mA, 3), 'AP50_clean': round(mB, 3), 'diff': round(mA - mB, 3),
                     'CI_lo': round(lo, 3), 'CI_hi': round(hi, 3), 'rel_%': round(100 * (mA - mB) / mB, 1)})
        table(rows, ['tau', 'n_clean_pos', 'clean_videos', 'AP50_A', 'AP50_clean', 'diff', 'CI_lo', 'CI_hi', 'rel_%'])
        pd.DataFrame(rows).to_csv(REV / 'R3_threshold_sensitivity.csv', index=False)

    # ---------------- R4 bootstrap check
    if 'R4' in RUN and 'R3' in RUN:
        md('## R4 Bootstrap check (positive images, manuscript rule)')
        md(f'Videos with positive A-test images: {len(g)}')
        md(f'1,000 resamples: discarded (no boxes in B-clean) = {int(np.isnan(bs1["B"]).sum())}; '
           f'CI diff = [{ci(bs1["A"] - bs1["B"])[0]:.3f}, {ci(bs1["A"] - bs1["B"])[1]:.3f}]')
        bs5 = bootstrap(g, recA, BOOT_CHECK, seed=1)
        md(f'5,000 resamples (seed 1): discarded = {int(np.isnan(bs5["B"]).sum())}; '
           f'CI A = [{ci(bs5["A"])[0]:.3f}, {ci(bs5["A"])[1]:.3f}]; CI B = [{ci(bs5["B"])[0]:.3f}, {ci(bs5["B"])[1]:.3f}]; '
           f'CI diff = [{ci(bs5["A"] - bs5["B"])[0]:.3f}, {ci(bs5["A"] - bs5["B"])[1]:.3f}]; '
           f'share of resamples with diff <= 0 = {np.nanmean((bs5["A"] - bs5["B"]) <= 0):.4f}')
        md()

    # ---------------- R6 final subset table: B-clean by the stated rule vs the stricter list
    if 'R6' in RUN:
        md('## R6 Subset results (YOLO26n, Protocol A)')
        md('B-rule = A-test minus Leaked (no confirmed near-duplicate, no shared source frame in training); '
           'B-strict = the 337-image list used so far.')
        wA = OUT / 'runs' / 'yolo26n_A_s0' / 'weights' / 'best.pt'
        recA = records(predict_cached(wA, A_test, 'yolo26n_A'), A_test)
        bstrict = set(B_clean)
        brule = [k for k in A_test if k not in leaked]
        extra = [k for k in brule if k not in bstrict]
        tt = t.set_index('key')
        md(f'B-rule {len(brule)} images; B-strict {len(bstrict)}; in B-rule only {len(extra)}')
        for nm, ks in (('in B-rule only', extra), ('B-strict', sorted(bstrict))):
            sub = tt.loc[ks]
            md(f"  {nm}: nn_phash_to_valid<=4 = {int((sub['nn_phash_to_valid'] <= 4).sum())}; "
               f"nn_phash_to_test<=4 = {int((sub['nn_phash_to_test'] <= 4).sum())}; "
               f"n_confirmed_dups>0 = {int((sub['n_confirmed_dups'].fillna(0) > 0).sum())}; "
               f"min nn_phash_to_train = {np.nanmin(sub['nn_phash_to_train']):.0f}")
        for scope in ('all images', 'positive images'):
            keep = (lambda k: True) if scope == 'all images' else (lambda k: recA[k][2] > 0)
            sets = {'A-test': [k for k in A_test if keep(k)],
                    'Leaked': [k for k in A_test if k in leaked and keep(k)],
                    'B-rule': [k for k in brule if keep(k)],
                    'B-strict': [k for k in A_test if k in bstrict and keep(k)]}
            g = {}
            for n_, ks in sets.items():
                for k in ks:
                    g.setdefault(video[k], {}).setdefault(n_, []).append(k)
            rng_b = np.random.default_rng(SEED)
            gl = list(g)
            boot = {n_: {'AP50': np.empty(BOOT_MAIN), 'AP50_95': np.empty(BOOT_MAIN)} for n_ in sets}
            for b in range(BOOT_MAIN):
                pick = rng_b.integers(0, len(gl), len(gl))
                for n_ in sets:
                    ks = [k for i in pick for k in g[gl[i]].get(n_, ())]
                    m = metrics(ks, recA) if ks else {'AP50': np.nan, 'AP50_95': np.nan}
                    boot[n_]['AP50'][b], boot[n_]['AP50_95'][b] = m['AP50'], m['AP50_95']
            rows = []
            for n_, ks in sets.items():
                m = metrics(ks, recA)
                l50, h50 = ci(boot[n_]['AP50']); l95, h95 = ci(boot[n_]['AP50_95'])
                row = {'subset': n_, 'images': m['n_images'], 'pos': m['n_pos'], 'gt': m['n_gt'],
                       'videos': len({video[k] for k in ks}),
                       'AP50': round(m['AP50'], 3), 'AP50_CI': f'[{l50:.3f}, {h50:.3f}]',
                       'AP50_95': round(m['AP50_95'], 3), 'AP50_95_CI': f'[{l95:.3f}, {h95:.3f}]',
                       'R': round(m['R'], 3)}
                for ref in ('B-rule', 'B-strict'):
                    if n_ in ('A-test', 'Leaked'):
                        d = boot[n_]['AP50'] - boot[ref]['AP50']
                        d95 = boot[n_]['AP50_95'] - boot[ref]['AP50_95']
                        mr = metrics(sets[ref], recA)
                        row[f'dAP50 vs {ref}'] = (f"{m['AP50'] - mr['AP50']:+.3f} [{ci(d)[0]:.3f}, {ci(d)[1]:.3f}] "
                                                  f"({100 * (m['AP50'] - mr['AP50']) / mr['AP50']:.0f}%)")
                        row[f'dAP50_95 vs {ref}'] = (f"{m['AP50_95'] - mr['AP50_95']:+.3f} "
                                                     f"[{ci(d95)[0]:.3f}, {ci(d95)[1]:.3f}]")
                rows.append(row)
            md(f'### {scope}')
            table(rows, ['subset', 'images', 'pos', 'gt', 'videos', 'AP50', 'AP50_CI', 'AP50_95', 'AP50_95_CI', 'R',
                         'dAP50 vs B-rule', 'dAP50_95 vs B-rule', 'dAP50 vs B-strict', 'dAP50_95 vs B-strict'])
            pd.DataFrame(rows).to_csv(REV / f'R6_subsets_{scope.split()[0]}.csv', index=False)

        # AP50 by d_H to the nearest training image, positive images (Fig. 3b, Table 5)
        md('### AP50 by d_H bin (positive images)')
        rows = []
        for lo_, hi_, lab in ((0, 4, '0-4'), (5, 8, '5-8'), (9, 12, '9-12'), (13, 64, '>12')):
            ks = [k for k in A_test if recA[k][2] > 0 and pd.notna(nn[k]) and lo_ <= nn[k] <= hi_]
            g = {}
            for k in ks:
                g.setdefault(video[k], {}).setdefault('bin', []).append(k)
            bb = bootstrap(g, recA, BOOT_MAIN)['bin']
            m = metrics(ks, recA)
            lo, hi = ci(bb)
            rows.append({'bin': lab, 'pos': m['n_pos'], 'gt': m['n_gt'], 'videos': len(g),
                         'AP50': round(m['AP50'], 3), 'AP50_lo': round(lo, 3), 'AP50_hi': round(hi, 3),
                         'AP50_95': round(m['AP50_95'], 3), 'R': round(m['R'], 3)})
        table(rows, ['bin', 'pos', 'gt', 'videos', 'AP50', 'AP50_lo', 'AP50_hi', 'AP50_95', 'R'])
        pd.DataFrame(rows).to_csv(REV / 'R6_bins_positive.csv', index=False)

    # ---------------- R5 substitution controls
    if 'R5' in RUN:
        md('## R5 Substitution controls (Protocol C models, C-test)')
        md('Pools: copies of training frames (seen by the model) vs copies of C-test frames (unseen).')
        table([{'pool': 'C-test positives (original)', **box_stats([k for k in C_test if len(read_gt(k))])},
               {'pool': 'copies of training frames', **box_stats(POOL_TRAIN)},
               {'pool': 'copies of C-test frames', **box_stats(POOL_TEST)}],
              ['pool', 'images', 'boxes', 'boxes_per_image', 'small_boxes_%', 'median_box_px'])
        rows = []
        for m in ('yolo26n', 'yolo26s'):
            w = FAMILY / 'runs' / f'{m}_s0' / 'weights' / 'best.pt'
            cache = predict_cached(w, C_test + POOL_TRAIN + POOL_TEST, f'{m}_C')
            rec = records(cache, C_test + POOL_TRAIN + POOL_TEST)
            pos = [k for k in C_test if rec[k][2] > 0]
            neg = [k for k in C_test if rec[k][2] == 0]
            for pool_name, pool in (('train copies (leaked)', POOL_TRAIN), ('C-test copies (unseen)', POOL_TEST)):
                if not pool:
                    md(f'{pool_name}: empty pool, skipped'); continue
                for p in DOSE_LEVELS:
                    vals = []
                    for _ in range(DOSE_REPS if p > 0 else 1):
                        n = int(round(p * len(pos)))
                        drop = set(rng.choice(len(pos), n, replace=False)) if n else set()
                        add = list(rng.choice(len(pool), n, replace=n > len(pool))) if n else []
                        keys = neg + [k for i, k in enumerate(pos) if i not in drop] + [pool[j] for j in add]
                        vals.append(metrics(keys, rec)['AP50'])
                    v = np.array(vals)
                    rows.append({'model': m, 'pool': pool_name, 'rho_%': int(100 * p), 'reps': len(v),
                                 'AP50_mean': round(float(v.mean()), 3), 'AP50_sd': round(float(v.std()), 3)})
            if POOL_TEST:
                rows.append({'model': m, 'pool': 'all C-test copies (no substitution)', 'rho_%': '-', 'reps': 1,
                             'AP50_mean': round(metrics(POOL_TEST, rec)['AP50'], 3), 'AP50_sd': 0})
        table(rows, ['model', 'pool', 'rho_%', 'reps', 'AP50_mean', 'AP50_sd'])
        pd.DataFrame(rows).to_csv(REV / 'R5_substitution_controls.csv', index=False)

    name = 'REVISION_CHECKS.md' if len(RUN) == 6 else f'REVISION_CHECKS_{"_".join(sorted(RUN))}.md'
    (REV / name).write_text('\n'.join(MD))
    log(f'Done. Paste {REV / name} into the chat.')


if __name__ == '__main__':
    main()
