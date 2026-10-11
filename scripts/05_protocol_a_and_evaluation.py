#!/usr/bin/env python3
"""
BharatPotHole leakage paper - lean experiment runner (Google Colab, T4)
=======================================================================
"How much does data leakage inflate road-defect detection accuracy?
 Evidence from a re-split of BharatPotHole"

What this script does (only ONE new training run per model, everything else is inference):
  STAGE train : trains YOLO26n on the ORIGINAL published split (Protocol A), resumable.
  STAGE eval  : reuses existing outputs on Drive and evaluates, with bootstrap CIs by video:
      T1  Protocol A model on the original test  vs  the leakage-free subset (Protocol B)
          -> same model, same training data: the gap is the leakage inflation
      T2  Protocol C models (already trained in family_study_v2) on the v2 video-disjoint test
      T3  All models on RDD2022-India D40 (external yardstick)
      T4  Leakage dose-response: v2 test with 0-100% of positive frames replaced by
          augmentation copies of v2 TRAIN frames (no retraining)
      T5  Protocol A model: AP by pHash distance to the nearest training image
  STAGE summary : collects the audit numbers already on Drive + all new results into one folder.

Carried forward from earlier work (nothing is recomputed):
  dataset_assessment/per_image.csv            leakage flags, video ids, pHash distances
  BharatPotHole_v2/manifest_v2.csv            dedup + v2 split, augmentation copies
  BharatPotHole_v2/B_original_test_clean.txt  Protocol B test list (337 images)
  BharatPotHole_v2/BharatPotHole_v2.zip       v2 dataset
  rdd_india_assessment/RDD_India_D40_yolo.zip external test set
  experiments/family_study_v2/runs/*_s0       Protocol C weights (YOLO26n, YOLO26s)
  experiments/family_study_v2/pretrained/     identical starting weights (same MD5)
  Ultralytics 8.4.162                         same version as the family study

HOW TO RUN (Colab, GPU runtime)
  Cell 1:  from google.colab import drive; drive.mount('/content/drive')
  Cell 2:  !python "/content/drive/MyDrive/POTHOLE/scripts/05_protocol_a_and_evaluation.py"
  After a disconnect, run Cell 1 + Cell 2 again: training resumes, finished parts are skipped,
  cached predictions are reused.
  Optional: --stage train | eval | summary      (default: all)
            MODELS_A=yolo26n,yolo26s  (env var) to also train YOLO26s on Protocol A

OUTPUT: /content/drive/MyDrive/POTHOLE/experiments/leakage_paper/
  runs/<model>_A_s0/          Protocol A training run (weights, results.csv)
  cache/*.pkl                 per-image predictions (re-used on re-run)
  results/*.csv               all tables      figures/*.png   all figures
  carried_forward/            copies of earlier audit tables      SUMMARY.md
"""
import os
import sys
import json
import time
import pickle
import shutil
import zipfile
import argparse
import subprocess
from pathlib import Path

# ----------------------------------------------------------------------------- config
ULTRA_VERSION = '8.4.162'
POTHOLE   = Path(os.environ.get('POTHOLE_DRIVE', '/content/drive/MyDrive/POTHOLE'))
OUT       = POTHOLE / 'experiments' / 'leakage_paper'
FAMILY    = POTHOLE / 'experiments' / 'family_study_v2'
ASSESS    = POTHOLE / 'dataset_assessment'
V2_DRIVE  = POTHOLE / 'BharatPotHole_v2'
RDD_DRIVE = POTHOLE / 'rdd_india_assessment'

ORIG_ZIP  = POTHOLE / 'archive (1).zip'
ORIG_DIR  = Path('/content/pothole_extracted')
V2_DIR    = Path('/content/BharatPotHole_v2')
RDD_DIR   = Path('/content/RDD_India_D40_yolo')

MODELS_A  = [m.strip() for m in os.environ.get('MODELS_A', 'yolo26n').split(',') if m.strip()]
MODELS_C  = ['yolo26n', 'yolo26s']          # already trained (family_study_v2, seed 0)
SEED      = 0
TRAIN_ARGS = dict(epochs=100, patience=30, imgsz=640, batch=16, optimizer='auto',
                  pretrained=True, deterministic=True, amp=True, workers=2, plots=True)

IMGSZ, CONF, NMS_IOU, MAX_DET = 640, 0.001, 0.7, 300
MIN_BOX   = 2 / 640                          # drop degenerate GT boxes (< 2 px)
BOOT      = 1000                             # bootstrap resamples (by video)
DOSE_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0]
DOSE_REPS = 20
SIM_BINS  = [(0, 4, '0-4 (near-identical)'), (5, 8, '5-8'), (9, 12, '9-12'), (13, 64, '>12 (dissimilar)')]
CHUNK     = 256


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def ensure_ultralytics():
    try:
        from importlib.metadata import version
        v = version('ultralytics')
    except Exception:
        v = None
    if v != ULTRA_VERSION:
        log(f'Installing ultralytics=={ULTRA_VERSION} (found {v}) to match the family study')
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', f'ultralytics=={ULTRA_VERSION}'],
                       check=True)


def extract(zip_path, target_check, dest):
    if target_check.exists():
        return
    assert zip_path.exists(), f'Missing on Drive: {zip_path}'
    log(f'Extracting {zip_path.name} ...')
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest)


def setup_data():
    if not (ORIG_DIR / '.extracted_ok').exists():
        extract(ORIG_ZIP, ORIG_DIR / '.extracted_ok', ORIG_DIR)
        (ORIG_DIR / '.extracted_ok').touch()
    extract(V2_DRIVE / 'BharatPotHole_v2.zip', V2_DIR / 'data.yaml', V2_DIR.parent)
    extract(RDD_DRIVE / 'RDD_India_D40_yolo.zip', RDD_DIR / 'images', RDD_DIR.parent)


def original_root():
    import pandas as pd
    pi = pd.read_csv(ASSESS / 'per_image.csv')
    parts = Path(pi.loc[pi.split == 'train', 'rel_path'].iloc[0]).parts
    return ORIG_DIR / Path(*parts[:[p.lower() for p in parts].index('train')])


# ============================================================================ STAGE: train
def train_protocol_a():
    import torch
    from ultralytics import YOLO
    assert torch.cuda.is_available(), 'No GPU: Runtime > Change runtime type > T4 GPU'
    root = original_root()
    data = Path('/content/data_A.yaml')
    data.write_text(f"path: {root}\ntrain: train/images\nval: valid/images\ntest: test/images\n"
                    f"nc: 1\nnames: ['pothole']\n")
    for m in MODELS_A:
        name = f'{m}_A_s{SEED}'
        run = OUT / 'runs' / name
        done, last = run / 'DONE', run / 'weights' / 'last.pt'
        if done.exists():
            log(f'{name}: already trained - skipping'); continue
        if last.exists():
            log(f'{name}: resuming from {last}')
            try:
                YOLO(str(last)).train(resume=True)
            except Exception as e:                      # raised when training had already finished
                log(f'{name}: nothing to resume ({e})')
        else:
            pre = FAMILY / 'pretrained' / f'{m}.pt'      # identical starting weights as Protocol C
            log(f'{name}: training on the ORIGINAL split from {pre if pre.exists() else m + ".pt"}')
            YOLO(str(pre) if pre.exists() else f'{m}.pt').train(
                data=str(data), seed=SEED, device=0, project=str(OUT / 'runs'), name=name,
                exist_ok=True, **TRAIN_ARGS)
        if (run / 'weights' / 'best.pt').exists():
            done.write_text(time.strftime('%Y-%m-%d %H:%M'))
            log(f'{name}: DONE')


# ============================================================================ evaluation core
import numpy as np                                     # noqa: E402

IOUS = np.linspace(0.5, 0.95, 10)
_GT = {}


def label_path(img):
    s = str(img)
    i = s.rfind('/images/')
    return Path(s[:i] + '/labels/' + s[i + 8:]).with_suffix('.txt')


def read_gt(img):
    """YOLO labels -> unique, clipped xyxy (normalised). Bbox and polygon lines accepted."""
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
    """Greedy COCO-style matching at 10 IoU thresholds. Returns tp[n_det, 10] in input order."""
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


def metrics(keys, rec, full=True):
    ngt = int(sum(rec[k][2] for k in keys))
    out = {'n_images': len(keys), 'n_pos_images': int(sum(rec[k][2] > 0 for k in keys)), 'n_gt': ngt}
    if ngt == 0:
        return {**out, 'AP50': np.nan, 'AP50_95': np.nan, 'P': np.nan, 'R': np.nan, 'F1': np.nan}
    conf = np.concatenate([rec[k][0] for k in keys])
    tp = np.concatenate([rec[k][1] for k in keys]).astype(float)
    order = np.argsort(-conf, kind='stable')
    tp = tp[order]
    ctp, cfp = np.cumsum(tp, 0), np.cumsum(1 - tp, 0)
    rc = ctp / ngt
    pr = ctp / np.maximum(ctp + cfp, 1e-12)
    aps = [ap_curve(rc[:, j], pr[:, j]) for j in range(len(IOUS))] if full else \
          [ap_curve(rc[:, 0], pr[:, 0])] + [np.nan] * 9
    out.update(AP50=aps[0], AP50_95=float(np.mean(aps)) if full else np.nan)
    if len(conf):
        f1 = 2 * pr[:, 0] * rc[:, 0] / np.maximum(pr[:, 0] + rc[:, 0], 1e-12)
        i = int(np.argmax(f1))
        out.update(P=float(pr[i, 0]), R=float(rc[i, 0]), F1=float(f1[i]))
    else:
        out.update(P=0.0, R=0.0, F1=0.0)
    return out


def bootstrap(groups, rec, seed=SEED):
    """groups: {group_id: {set_name: [keys]}} -> {set_name: {'AP50': arr, 'AP50_95': arr}}.
    Groups (videos) are resampled with replacement; all sets share the same draw (paired)."""
    rng = np.random.default_rng(seed)
    gl = list(groups)
    names = sorted({n for d in groups.values() for n in d})
    res = {n: {'AP50': np.empty(BOOT), 'AP50_95': np.empty(BOOT)} for n in names}
    for b in range(BOOT):
        pick = rng.integers(0, len(gl), len(gl))
        for n in names:
            keys = [k for i in pick for k in groups[gl[i]].get(n, ())]
            m = metrics(keys, rec) if keys else {'AP50': np.nan, 'AP50_95': np.nan}
            res[n]['AP50'][b], res[n]['AP50_95'][b] = m['AP50'], m['AP50_95']
    return res


def ci(a):
    a = a[~np.isnan(a)]
    return (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))) if len(a) else (np.nan, np.nan)


def predict_cached(weights, keys, tag):
    """Per-image predictions, cached on Drive so re-runs never repeat inference."""
    from ultralytics import YOLO
    f = OUT / 'cache' / f'{tag}.pkl'
    f.parent.mkdir(parents=True, exist_ok=True)
    cache = pickle.load(open(f, 'rb')) if f.exists() else {}
    todo = [k for k in dict.fromkeys(keys) if k not in cache]
    if todo:
        log(f'{tag}: predicting {len(todo)} images ({len(cache)} cached)')
        model = YOLO(str(weights))
        for i in range(0, len(todo), CHUNK):
            chunk = todo[i:i + CHUNK]
            results = model.predict(source=chunk, imgsz=IMGSZ, conf=CONF, iou=NMS_IOU, max_det=MAX_DET,
                                    batch=16, device=0, verbose=False, stream=True)
            for k, r in zip(chunk, results):
                assert Path(r.path).name == Path(k).name, 'prediction order mismatch'
                cache[k] = (r.boxes.xyxyn.cpu().numpy().astype(np.float32),
                            r.boxes.conf.cpu().numpy().astype(np.float32))
        tmp = f.with_suffix('.tmp')
        pickle.dump(cache, open(tmp, 'wb'))
        os.replace(tmp, f)
    return cache


def records(cache, keys):
    rec = {}
    for k in dict.fromkeys(keys):
        pb, pc = cache[k]
        gb = read_gt(k)
        rec[k] = (pc, match(pb, pc, gb), len(gb))
    return rec


# ============================================================================ STAGE: eval
def evaluate():
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(SEED)
    (OUT / 'results').mkdir(parents=True, exist_ok=True)
    (OUT / 'figures').mkdir(parents=True, exist_ok=True)

    # ---- evaluation sets, all built from files already on Drive -----------------------
    pi = pd.read_csv(ASSESS / 'per_image.csv')
    man = pd.read_csv(V2_DRIVE / 'manifest_v2.csv')
    for c in ('has_dup_in_train', 'source_in_train'):
        pi[c] = pi[c].fillna(False).astype(bool)

    t = pi[pi.split == 'test'].copy()
    t['key'] = [str(ORIG_DIR / p) for p in t.rel_path]
    A_test = list(t.key)
    video = dict(zip(t.key, t.seq_prefix))
    nn_dist = dict(zip(t.key, t.nn_phash_to_train))
    leaked = set(t.key[t.has_dup_in_train | t.source_in_train])
    B_clean = [l.strip() for l in (V2_DRIVE / 'B_original_test_clean.txt').read_text().splitlines() if l.strip()]
    B_clean = [k for k in B_clean if k in video]
    A_leaked = [k for k in A_test if k in leaked]

    cm = man[man.new_split == 'test']
    C_test = [str(V2_DIR / 'test' / 'images' / d) for d in cm.dest_name]
    video.update(zip(C_test, cm.seq_prefix))

    reps = man[man.is_rep.astype(bool) & man.new_split.notna()]
    src2split = dict(zip(reps.source_name, reps.new_split))
    copies = man[(man.drop_reason == 'augmentation_copy') & (man.n_boxes > 0)].copy()
    copies = copies[copies.source_name.map(src2split) == 'train']
    copies = copies.sample(frac=1, random_state=SEED).drop_duplicates('source_name')
    POOL = [str(ORIG_DIR / p) for p in copies.rel_path]        # copies of frames the C models trained on

    RDD = sorted(str(p) for p in (RDD_DIR / 'images').glob('*') if p.suffix.lower() in ('.jpg', '.jpeg', '.png'))
    log(f'Sets: A_test {len(A_test)} | B_clean {len(B_clean)} | A_leaked {len(A_leaked)} | '
        f'C_test {len(C_test)} | leaked pool {len(POOL)} | RDD {len(RDD)}')

    A_models = {m: OUT / 'runs' / f'{m}_A_s{SEED}' / 'weights' / 'best.pt' for m in MODELS_A}
    A_models = {m: w for m, w in A_models.items() if w.exists()}
    C_models = {m: FAMILY / 'runs' / f'{m}_s{SEED}' / 'weights' / 'best.pt' for m in MODELS_C}
    C_models = {m: w for m, w in C_models.items() if w.exists()}
    if not A_models:
        log('WARNING: no Protocol A weights yet - run the train stage first. Continuing with C models only.')

    rows, infl_rows, dose_rows, sim_rows = [], [], [], []

    def add_row(model, protocol, test_set, keys, rec, boots=None):
        m = metrics(keys, rec)
        lo, hi = ci(boots['AP50']) if boots else (np.nan, np.nan)
        lo95, hi95 = ci(boots['AP50_95']) if boots else (np.nan, np.nan)
        rows.append({'model': model, 'trained_on': protocol, 'test_set': test_set, **m,
                     'AP50_lo': lo, 'AP50_hi': hi, 'AP50_95_lo': lo95, 'AP50_95_hi': hi95})

    # ---- T1 / T3 / T5 : Protocol A models ------------------------------------------------
    for m, w in A_models.items():
        tag = f'{m}_A'
        cache = predict_cached(w, A_test + RDD, tag)
        rec = records(cache, A_test + RDD)
        groups, b_set = {}, set(B_clean)
        for k in A_test:
            g = groups.setdefault(video[k], {})
            g.setdefault('A_test', []).append(k)
            if k in b_set:
                g.setdefault('B_clean', []).append(k)
            if k in leaked:
                g.setdefault('A_leaked', []).append(k)
        bs = bootstrap(groups, rec)
        add_row(m, 'A (original split)', 'original test (A)', A_test, rec, bs['A_test'])
        add_row(m, 'A (original split)', 'leakage-free subset (B)', B_clean, rec, bs['B_clean'])
        add_row(m, 'A (original split)', 'leaked subset', A_leaked, rec, bs['A_leaked'])
        for metric in ('AP50', 'AP50_95'):
            d = bs['A_test'][metric] - bs['B_clean'][metric]
            pa, pb = metrics(A_test, rec)[metric], metrics(B_clean, rec)[metric]
            lo, hi = ci(d)
            infl_rows.append({'model': m, 'metric': metric, 'original_test': pa, 'leakage_free_B': pb,
                              'abs_inflation': pa - pb, 'abs_lo': lo, 'abs_hi': hi,
                              'rel_inflation_%': 100 * (pa - pb) / pb if pb else np.nan})
        brdd = bootstrap({k: {'RDD': [k]} for k in RDD}, rec)
        add_row(m, 'A (original split)', 'RDD2022-India (external)', RDD, rec, brdd['RDD'])

        # T5: AP vs pHash distance to nearest training image
        for lo_, hi_, lab in SIM_BINS:
            keys = [k for k in A_test if pd.notna(nn_dist[k]) and lo_ <= nn_dist[k] <= hi_]
            if not keys:
                continue
            g = {}
            for k in keys:
                g.setdefault(video[k], {}).setdefault('bin', []).append(k)
            b = bootstrap(g, rec)['bin']
            mm = metrics(keys, rec)
            sim_rows.append({'model': m, 'phash_bin': lab, **mm,
                             'AP50_lo': ci(b['AP50'])[0], 'AP50_hi': ci(b['AP50'])[1]})

    # ---- T2 / T3 / T4 : Protocol C models ------------------------------------------------
    for m, w in C_models.items():
        tag = f'{m}_C'
        cache = predict_cached(w, C_test + POOL + RDD, tag)
        rec = records(cache, C_test + POOL + RDD)
        g = {}
        for k in C_test:
            g.setdefault(video[k], {}).setdefault('C_test', []).append(k)
        add_row(m, 'C (video-disjoint v2)', 'v2 test (C)', C_test, rec, bootstrap(g, rec)['C_test'])
        add_row(m, 'C (video-disjoint v2)', 'copies of train frames (100% leaked)', POOL, rec)
        brdd = bootstrap({k: {'RDD': [k]} for k in RDD}, rec)
        add_row(m, 'C (video-disjoint v2)', 'RDD2022-India (external)', RDD, rec, brdd['RDD'])

        # T4: dose-response - replace a fraction of positive test frames with leaked copies
        pos = [k for k in C_test if rec[k][2] > 0]
        neg = [k for k in C_test if rec[k][2] == 0]
        for p in DOSE_LEVELS:
            vals = []
            for r in range(DOSE_REPS if 0 < p < 1 else 1):
                n = int(round(p * len(pos)))
                drop = set(rng.choice(len(pos), n, replace=False)) if n else set()
                add = list(rng.choice(len(POOL), n, replace=n > len(POOL))) if n else []
                keys = neg + [k for i, k in enumerate(pos) if i not in drop] + [POOL[j] for j in add]
                mm = metrics(keys, rec)
                vals.append((mm['AP50'], mm['AP50_95'], mm['R']))
            v = np.array(vals)
            dose_rows.append({'model': m, 'leaked_fraction_of_positives': p, 'reps': len(v),
                              'AP50_mean': v[:, 0].mean(), 'AP50_sd': v[:, 0].std(),
                              'AP50_95_mean': v[:, 1].mean(), 'AP50_95_sd': v[:, 1].std(),
                              'R_mean': v[:, 2].mean()})

    # ---- save tables ------------------------------------------------------------------
    R = OUT / 'results'
    main = pd.DataFrame(rows); main.to_csv(R / 'T1_T3_main_results.csv', index=False)
    infl = pd.DataFrame(infl_rows); infl.to_csv(R / 'T1_inflation.csv', index=False)
    dose = pd.DataFrame(dose_rows); dose.to_csv(R / 'T4_dose_response.csv', index=False)
    sim = pd.DataFrame(sim_rows); sim.to_csv(R / 'T5_similarity_bins.csv', index=False)
    log(f'Tables saved to {R}')

    # ---- figures ----------------------------------------------------------------------
    F = OUT / 'figures'
    if len(main):
        fig, ax = plt.subplots(figsize=(9, 4.5))
        lab = main.model + ' | ' + main.trained_on.str.split(' ').str[0] + ' | ' + main.test_set
        err = np.vstack([main.AP50 - main.AP50_lo, main.AP50_hi - main.AP50]).clip(0)
        ax.barh(lab, main.AP50, xerr=np.nan_to_num(err), color='#4C72B0', capsize=3)
        ax.invert_yaxis(); ax.set_xlabel('AP50 (95% CI, bootstrap by video)')
        ax.set_title('Same detector, different test protocols')
        fig.tight_layout(); fig.savefig(F / 'fig_protocols.png', dpi=200); plt.close(fig)
    if len(dose):
        fig, ax = plt.subplots(figsize=(6, 4))
        for m, d in dose.groupby('model'):
            x = 100 * d.leaked_fraction_of_positives
            ax.plot(x, d.AP50_mean, marker='o', label=m)
            ax.fill_between(x, d.AP50_mean - d.AP50_sd, d.AP50_mean + d.AP50_sd, alpha=.2)
        ax.set(xlabel='Positive test frames replaced by copies of training frames (%)',
               ylabel='AP50', title='Leakage dose-response (Protocol C models)')
        ax.legend(); fig.tight_layout(); fig.savefig(F / 'fig_dose_response.png', dpi=200); plt.close(fig)
    if len(sim):
        fig, ax = plt.subplots(figsize=(6, 4))
        for i, (m, d) in enumerate(sim.groupby('model')):
            x = np.arange(len(d)) + 0.35 * i
            err = np.vstack([d.AP50 - d.AP50_lo, d.AP50_hi - d.AP50]).clip(0)
            ax.bar(x, d.AP50, width=0.35, yerr=np.nan_to_num(err), capsize=3, label=m)
            ax.set_xticks(np.arange(len(d)) + 0.175 * i, d.phash_bin, rotation=15)
        ax.set(xlabel='pHash distance of test image to nearest training image', ylabel='AP50',
               title='Protocol A: accuracy vs similarity to training data')
        ax.legend(); fig.tight_layout(); fig.savefig(F / 'fig_similarity.png', dpi=200); plt.close(fig)
    log(f'Figures saved to {F}')


# ============================================================================ STAGE: summary
def summary():
    import pandas as pd
    cf = OUT / 'carried_forward'; cf.mkdir(parents=True, exist_ok=True)
    for src in [ASSESS / 'leakage_summary.csv', ASSESS / 'split_summary.csv', ASSESS / 'assessment_report.md',
                V2_DRIVE / 'build_summary.json', V2_DRIVE / 'split_stats_original.csv',
                V2_DRIVE / 'split_stats_v2.csv', V2_DRIVE / 'leakage_before_after.png',
                RDD_DRIVE / 'domain_comparison.csv', RDD_DRIVE / 'assessment_report.md',
                FAMILY / 'runs_summary.csv', POTHOLE / 'diagnostics' / 'yolo26s_s0']:
        if src.is_dir():
            shutil.copytree(src, cf / src.name, dirs_exist_ok=True)
        elif src.exists():
            shutil.copy2(src, cf / (src.parent.name + '__' + src.name))

    man = pd.read_csv(V2_DRIVE / 'manifest_v2.csv')
    kept = man[man.new_split.notna()]
    audit = {
        'images_released': len(man),
        'exact_md5_duplicates_removed': int((man.drop_reason == 'exact_md5_duplicate').sum()),
        'augmentation_copies_removed': int((man.drop_reason == 'augmentation_copy').sum()),
        'unique_frames': len(kept),
        'duplicate_share_%': round(100 * (1 - len(kept) / len(man)), 1),
        'positive_images_released': int((man.n_boxes > 0).sum()),
        'positive_images_unique': int((kept.n_boxes_clean > 0).sum()),
        'negative_images_released': int((man.n_boxes == 0).sum()),
        'negative_images_unique': int((kept.n_boxes_clean == 0).sum()),
        'boxes_released': int(man.n_boxes.sum()),
        'boxes_unique_clean': int(kept.n_boxes_clean.sum()),
        'v2_split_sizes': kept.new_split.value_counts().to_dict(),
    }
    json.dump(audit, open(OUT / 'results' / 'T0_audit_numbers.json', 'w'), indent=2)

    def md(path):
        if not path.exists():
            return '_not yet available_'
        d = pd.read_csv(path)
        try:
            return d.round(4).to_markdown(index=False)
        except Exception:
            return '```\n' + d.round(4).to_string(index=False) + '\n```'

    R = OUT / 'results'
    txt = f"""# Leakage paper - results summary ({time.strftime('%Y-%m-%d %H:%M')})
Ultralytics {ULTRA_VERSION} | seed {SEED} | bootstrap {BOOT} resamples by video | AP computed with an own
all-point evaluator (conf {CONF}), identical for every model and test set.

## T0 Audit (carried forward)
```
{json.dumps(audit, indent=2)}
```

## T1 Inflation: same Protocol A model, original test vs leakage-free subset
{md(R / 'T1_inflation.csv')}

## T1-T3 All protocols and test sets
{md(R / 'T1_T3_main_results.csv')}

## T4 Leakage dose-response (Protocol C models)
{md(R / 'T4_dose_response.csv')}

## T5 Accuracy vs similarity to training data (Protocol A)
{md(R / 'T5_similarity_bins.csv')}

Earlier audit tables, v2 build report, RDD domain comparison, family-study validation summary and the
dataset diagnostics are copied in carried_forward/.
"""
    (OUT / 'SUMMARY.md').write_text(txt)
    log(f'Summary written: {OUT / "SUMMARY.md"}')


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='all', choices=['all', 'train', 'eval', 'summary'])
    a, _ = ap.parse_known_args()
    if not POTHOLE.exists():
        sys.exit('Drive not mounted: run  from google.colab import drive; drive.mount("/content/drive")')
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'results').mkdir(exist_ok=True)
    ensure_ultralytics()
    setup_data()
    if a.stage in ('all', 'train'):
        train_protocol_a()
    if a.stage in ('all', 'eval'):
        evaluate()
    if a.stage in ('all', 'eval', 'summary'):
        summary()
    log('Finished.')


if __name__ == '__main__':
    main()
