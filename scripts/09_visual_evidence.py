#!/usr/bin/env python3
"""
Visual evidence (Online Resource 2) and a small sample dataset - no training, CPU is enough.

Figures (PDF + 300-dpi PNG) in experiments/leakage_paper/publication/supplementary/:
  FigS1_leaked_pairs        published test images next to their nearest training image (d_H, SSIM)
  FigS2_export_copies       one source frame whose export copies fall into different splits
  Fig3_distance_examples    MAIN-TEXT Fig. 3: one test image per d_H bin and its nearest training image
  FigS3_predictions_A       Protocol A detector on Leaked vs B-clean test images (GT and predictions)
  FigS4_contamination       Protocol C detector on C-test images, unseen copies and training copies
  FigS5_annotation          near-identical unique frames with different annotations
  FigS6_domain_shift        BharatPotHole vs RDD2022-India examples
  Online_Resource_2.docx    the supplementary figures, numbered S1-S6 in the order above (S3 excluded)

Sample dataset in release/sample_dataset/ (+ .zip): the images shown above plus random B-clean and
C-test images, their YOLO labels, and sample_manifest.csv (role, split, video, MD5, d_H, SSIM).
CHECK THE BHARATPOTHOLE AND RDD2022 LICENCES BEFORE PUBLISHING ANY IMAGES.

Inputs: outputs of scripts 01-06 on Drive (per_image.csv, manifest_v2.csv, cached predictions).
Run:  !python "/content/drive/MyDrive/POTHOLE/scripts/09_visual_evidence.py"
"""
import importlib.util
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('rc', HERE / '06_revision_checks.py')
rc = importlib.util.module_from_spec(_spec)
sys.argv = sys.argv[:1]                                  # keep 06's argument parsing quiet
_spec.loader.exec_module(rc)
sys.path.insert(0, str(HERE))
import pubstyle as P                                     # noqa: E402

SUP = rc.OUT / 'publication' / 'supplementary'
SAMPLE = rc.POTHOLE / 'release' / 'sample_dataset'
RDD_ZIP = rc.POTHOLE / 'rdd_india_assessment' / 'RDD_India_D40_yolo.zip'
RDD_DIR = Path('/content/RDD_India_D40_yolo')
CONF_SHOW = 0.25                                         # predictions drawn at this confidence
C_GT, C_PR = '#009E73', '#D55E00'                        # ground truth / prediction outlines
RNG = np.random.default_rng(rc.SEED)
CAPTIONS = {}
SAMPLE_ROWS = []


# ------------------------------------------------------------------ image helpers
def load_rgb(path, size=None):
    import cv2
    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA) if size else im


def grey(path, size):
    import cv2
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA)


def phash64(path):
    """Same pHash as script 01: 32x32 grey (INTER_AREA), 8x8 DCT block, bit = coef > median (no DC)."""
    import cv2
    d = cv2.dct(grey(path, 32).astype(np.float32))[:8, :8].flatten()
    bits = d > np.median(d[1:])
    return np.packbits(bits).view('>u8')[0]


_POP = np.array([bin(i).count('1') for i in range(256)], np.uint8)


def hamming(a, B):
    """a: uint64 scalar or (n,) ; B: (m,) -> (n, m) Hamming distances."""
    x = np.bitwise_xor(np.atleast_1d(a)[:, None].astype(np.uint64), B[None, :].astype(np.uint64))
    return _POP[x.view(np.uint8)].reshape(x.shape + (8,)).sum(-1)


def ssim(p, q):
    from skimage.metrics import structural_similarity
    return float(structural_similarity(grey(p, 128), grey(q, 128), data_range=255))


def draw(ax, path, preds=None, title=None, note=None, title_color=P.C_INK):
    from matplotlib.patches import Rectangle
    im = load_rgb(path)
    h, w = im.shape[:2]
    ax.imshow(im)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    for x0, y0, x1, y1 in rc.read_gt(str(path)):
        ax.add_patch(Rectangle((x0 * w, y0 * h), (x1 - x0) * w, (y1 - y0) * h, fill=False, ec=C_GT, lw=0.9))
    if preds is not None:
        pb, pc = preds
        for (x0, y0, x1, y1), c in zip(pb, pc):
            if c >= CONF_SHOW:
                ax.add_patch(Rectangle((x0 * w, y0 * h), (x1 - x0) * w, (y1 - y0) * h, fill=False, ec=C_PR,
                                       lw=0.9, ls=(0, (2, 1))))
    if title:
        ax.set_title(title, fontsize=6.5, pad=2, color=title_color)
    if note:
        ax.text(0.5, -0.03, note, transform=ax.transAxes, ha='center', va='top', fontsize=6.3)


def grid(nrows, ncols, height_per_row=None):
    plt = P.setup_style()
    w = P.W2
    h = height_per_row or (w / ncols) * 1.12
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, h * nrows), squeeze=False)
    return plt, fig, axes


def save(fig, name, caption, supp=True):
    """supp=False: main-text figure (not added to Online Resource 2)."""
    SUP.mkdir(parents=True, exist_ok=True)
    fig.savefig(SUP / f'{name}.pdf')
    fig.savefig(SUP / f'{name}.png', dpi=300)
    if supp:
        CAPTIONS[name] = caption
    print('  saved', name)


def add_sample(path, role, **meta):
    SAMPLE_ROWS.append({'role': role, 'file': Path(path).name, 'src': str(path), **meta})


# ------------------------------------------------------------------ main
def main():
    import matplotlib
    matplotlib.use('Agg')
    rc.setup_data()
    if RDD_ZIP.exists() and not (RDD_DIR / 'images').exists():
        with zipfile.ZipFile(RDD_ZIP) as z:
            z.extractall(RDD_DIR.parent)

    pi = pd.read_csv(rc.ASSESS / 'per_image.csv')
    man = pd.read_csv(rc.V2_DRIVE / 'manifest_v2.csv')
    for c in ('has_dup_in_train', 'source_in_train'):
        pi[c] = pi[c].fillna(False).astype(bool)
    pi['key'] = [str(rc.ORIG_DIR / p) for p in pi.rel_path]
    t = pi[pi.split == 'test'].copy()
    tr = pi[pi.split == 'train'].copy()
    t['leaked'] = t.has_dup_in_train | t.source_in_train
    t['pos'] = [len(rc.read_gt(k)) > 0 for k in t.key]

    # nearest training image for every test image (identity is not stored in per_image.csv)
    print('Hashing training and test images ...')
    Htr = np.array([phash64(k) for k in tr.key], dtype=np.uint64)
    Hte = np.array([phash64(k) for k in t.key], dtype=np.uint64)
    D = hamming(Hte, Htr)
    t['nn_idx'] = D.argmin(1)
    t['nn_d'] = D.min(1)
    t['nn_key'] = tr.key.to_numpy()[t.nn_idx]
    agree = float((t.nn_d == t.nn_phash_to_train).mean())
    print(f'  recomputed d_H agrees with per_image.csv for {100 * agree:.1f}% of test images')

    # ---------------- S1 leaked pairs
    lk = t[t.leaked & t.pos]
    pick = pd.concat([lk[lk.nn_d == 0].sample(min(4, (lk.nn_d == 0).sum()), random_state=0),
                      lk[lk.nn_d.between(1, 4)].sample(min(2, lk.nn_d.between(1, 4).sum()), random_state=0)])
    plt, fig, ax = grid(3, 4)
    for i, (_, r) in enumerate(pick.head(6).iterrows()):
        a, b = ax[i // 2][2 * (i % 2)], ax[i // 2][2 * (i % 2) + 1]
        s = ssim(r.key, r.nn_key)
        draw(a, r.key, title='Test (published split)')
        draw(b, r.nn_key, title='Nearest training image',
             note=f'$\\mathit{{d}}_{{\\mathrm{{H}}}}$ = {int(r.nn_d)}, SSIM = {s:.2f}')
        add_sample(r.key, 'S1_leaked_test', split='test', video=r.seq_prefix, d_H=int(r.nn_d), ssim=round(s, 3))
        add_sample(r.nn_key, 'S1_nearest_train', split='train', video=r.seq_prefix, d_H=int(r.nn_d), ssim=round(s, 3))
    for k in range(len(pick.head(6)), 6):
        ax[k // 2][2 * (k % 2)].axis('off'); ax[k // 2][2 * (k % 2) + 1].axis('off')
    save(fig, 'FigS1_leaked_pairs',
         'Examples of published test images (left of each pair) and their nearest training image (right). '
         'Green boxes: ground truth. dH: Hamming distance between perceptual hashes; SSIM on 128 × 128 grey '
         'thumbnails')
    plt.close(fig)

    # ---------------- S2 export copies of one source frame across splits
    g = pi.groupby('source_name')
    span = g.split.nunique()
    cand = span[span >= 2].index
    sizes = g.size().loc[cand].sort_values(ascending=False)
    src = sizes.index[0] if len(sizes) else g.size().idxmax()
    rows = pi[pi.source_name == src].sort_values('split').head(6)
    plt, fig, ax = grid(1, len(rows), height_per_row=P.W2 / max(len(rows), 4) * 1.15)
    col = {'train': P.C_INK, 'valid': P.C_NEUTRAL, 'test': P.C_LEAK}
    for a, (_, r) in zip(ax[0], rows.iterrows()):
        draw(a, r.key, title=r.split.capitalize(), title_color=col.get(r.split, P.C_INK))
        add_sample(r.key, 'S2_export_copy', split=r.split, video=r.seq_prefix, source_name=src, md5=r.md5)
    save(fig, 'FigS2_export_copies',
         f'All released files that share one source frame ({len(rows)} export copies), with the subset of the '
         'published split each was assigned to. Green boxes: ground truth')
    plt.close(fig)

    # ---------------- main-text Fig. 3: distance ladder
    bins = [(0, 4), (5, 8), (9, 12), (13, 64)]
    plt, fig, ax = grid(2, 4)
    for j, (lo, hi) in enumerate(bins):
        c = t[t.pos & t.nn_d.between(lo, hi)]
        if c.empty:
            ax[0][j].axis('off'); ax[1][j].axis('off'); continue
        r = c.sample(1, random_state=1).iloc[0]
        lab = f'{lo}–{hi}' if hi < 64 else f'> {lo - 1}'
        draw(ax[0][j], r.key, title=f'Test, $\\mathit{{d}}_{{\\mathrm{{H}}}}$ {lab}')
        draw(ax[1][j], r.nn_key, title='Nearest training image',
             note=f'$\\mathit{{d}}_{{\\mathrm{{H}}}}$ = {int(r.nn_d)}, SSIM = {ssim(r.key, r.nn_key):.2f}')
        add_sample(r.key, 'S3_test', split='test', video=r.seq_prefix, d_H=int(r.nn_d))
    save(fig, 'Fig3_distance_examples',
         'One positive test image per dH bin (top) and its nearest training image (bottom). Images at dH ≤ 4 '
         'are near-copies; at larger distances the nearest training image shows a different scene. '
         'Green boxes: ground truth', supp=False)
    plt.close(fig)

    # ---------------- S4 Protocol A predictions on Leaked vs B-clean
    cacheA = rc.OUT / 'cache' / 'yolo26n_A.pkl'
    if cacheA.exists():
        import pickle
        cA = pickle.load(open(cacheA, 'rb'))
        Lk = t[t.leaked & t.pos].sample(3, random_state=2)
        Bc = t[~t.leaked & t.pos].sample(3, random_state=2)
        plt, fig, ax = grid(2, 3)
        for j, (_, r) in enumerate(Lk.iterrows()):
            draw(ax[0][j], r.key, preds=cA.get(r.key), title='Leaked' if j == 1 else None, title_color=P.C_LEAK)
        for j, (_, r) in enumerate(Bc.iterrows()):
            draw(ax[1][j], r.key, preds=cA.get(r.key), title='B-clean' if j == 1 else None, title_color=P.C_CLEAN)
            add_sample(r.key, 'S4_b_clean', split='test', video=r.seq_prefix, d_H=int(r.nn_d))
        save(fig, 'FigS3_predictions_A',
             f'YOLO26n trained on the published split (Protocol A) on test images with a near-duplicate in '
             f'training (Leaked, top) and without (B-clean, bottom). Green: ground truth; orange dashed: '
             f'predictions with confidence ≥ {CONF_SHOW}')
        plt.close(fig)

    # ---------------- S5 contamination test
    cacheC = rc.OUT / 'cache' / 'yolo26n_C.pkl'
    if cacheC.exists():
        import pickle
        cC = pickle.load(open(cacheC, 'rb'))
        cm = man[man.new_split == 'test']
        Ctest = [str(rc.V2_DIR / 'test' / 'images' / d) for d in cm.dest_name]
        Cpos = [k for k in Ctest if len(rc.read_gt(k))]
        reps = man[man.is_rep.astype(bool) & man.new_split.notna()]
        s2s = dict(zip(reps.source_name, reps.new_split))
        cp = man[(man.drop_reason == 'augmentation_copy') & (man.n_boxes > 0)].copy()
        cp['src_split'] = cp.source_name.map(s2s)
        pool_tr = [str(rc.ORIG_DIR / p) for p in cp[cp.src_split == 'train'].drop_duplicates('source_name').rel_path]
        pool_te = [str(rc.ORIG_DIR / p) for p in cp[cp.src_split == 'test'].drop_duplicates('source_name').rel_path]
        rowsets = [('C-test (unseen video)', [Cpos[i] for i in RNG.choice(len(Cpos), min(3, len(Cpos)), replace=False)], P.C_CLEAN),
                   ('Copy of a C-test frame (unseen)', [pool_te[i] for i in RNG.choice(len(pool_te), min(3, len(pool_te)), replace=False)], P.C_ALT),
                   ('Copy of a training frame (leaked)', [pool_tr[i] for i in RNG.choice(len(pool_tr), min(3, len(pool_tr)), replace=False)], P.C_LEAK)]
        plt, fig, ax = grid(3, 3)
        for i, (lab, keys, c) in enumerate(rowsets):
            for j, k in enumerate(keys):
                draw(ax[i][j], k, preds=cC.get(k), title=lab if j == 1 else None, title_color=c)
                add_sample(k, f'S5_{lab.split(" (")[0].replace(" ", "_")}', split='v2-test' if i == 0 else 'copy')
        save(fig, 'FigS4_contamination',
             'Controlled contamination test: YOLO26n trained on the video-disjoint split (Protocol C) on '
             'C-test images (top), export copies of unseen C-test frames (middle) and export copies of its '
             f'own training frames (bottom). Green: ground truth; orange dashed: predictions with confidence ≥ {CONF_SHOW}')
        plt.close(fig)

    # ---------------- S6 annotation inconsistency among unique frames
    reps2 = man[man.is_rep.astype(bool) & man.new_split.notna()].copy()
    reps2['key'] = [str(rc.V2_DIR / s / 'images' / d) for s, d in zip(reps2.new_split, reps2.dest_name)]
    reps2['nb'] = [len(rc.read_gt(k)) for k in reps2.key]
    lab2 = reps2[reps2.nb > 0].reset_index(drop=True)
    H2 = np.array([phash64(k) for k in lab2.key], dtype=np.uint64)
    pairs = []
    for i0 in range(0, len(H2), 400):
        Dm = hamming(H2[i0:i0 + 400], H2)
        for a_, b_ in zip(*np.where(Dm <= 4)):
            a_ += i0
            if a_ < b_ and abs(lab2.nb[a_] - lab2.nb[b_]) >= 2:
                pairs.append((a_, b_, int(Dm[a_ - i0, b_])))
    print(f'  near-identical labelled pairs with >= 2 boxes difference: {len(pairs)}')
    if pairs:
        sel = [pairs[i] for i in RNG.choice(len(pairs), min(3, len(pairs)), replace=False)]
        plt, fig, ax = grid(len(sel), 2)
        for i, (a_, b_, d_) in enumerate(sel):
            ra, rb = lab2.iloc[a_], lab2.iloc[b_]
            draw(ax[i][0], ra.key, note=f'{ra.nb} boxes')
            draw(ax[i][1], rb.key, note=f'{rb.nb} boxes ($\\mathit{{d}}_{{\\mathrm{{H}}}}$ = {d_})')
            add_sample(ra.key, 'S6_annotation_pair', split=ra.new_split, video=ra.seq_prefix, d_H=d_)
            add_sample(rb.key, 'S6_annotation_pair', split=rb.new_split, video=rb.seq_prefix, d_H=d_)
        save(fig, 'FigS5_annotation',
             'Pairs of near-identical unique frames (dH ≤ 4) whose annotations differ by two or more boxes. '
             'Green boxes: ground truth')
        plt.close(fig)

    # ---------------- S7 domain shift
    rdd = sorted((RDD_DIR / 'images').glob('*')) if (RDD_DIR / 'images').exists() else []
    if rdd:
        bp = [k for k in reps2[reps2.nb > 0].key]
        bsel = [bp[i] for i in RNG.choice(len(bp), 4, replace=False)]
        rsel = [str(rdd[i]) for i in RNG.choice(len(rdd), 4, replace=False)]
        plt, fig, ax = grid(2, 4)
        for j, k in enumerate(bsel):
            draw(ax[0][j], k, title='BharatPotHole' if j == 0 else None)
        for j, k in enumerate(rsel):
            draw(ax[1][j], k, title='RDD2022-India' if j == 0 else None)
            add_sample(k, 'S7_rdd_india', split='external')
        save(fig, 'FigS6_domain_shift',
             'Random positive images from BharatPotHole (top) and the RDD2022-India pothole subset (bottom). '
             'Green boxes: ground truth (D40 for RDD2022)')
        plt.close(fig)

    write_docx()
    write_sample(t, man)


def write_docx():
    import docx
    from docx.shared import Pt, Cm
    d = docx.Document()
    s = d.sections[0]; s.page_width, s.page_height = Cm(21), Cm(29.7)
    for m in ('left_margin', 'right_margin', 'top_margin', 'bottom_margin'):
        setattr(s, m, Cm(2))
    p = d.add_paragraph(); r = p.add_run('Online Resource 2 – Visual evidence'); r.bold = True; r.font.size = Pt(13)
    d.add_paragraph('How much does data leakage inflate road-defect detection performance? Evidence from a '
                    're-split of BharatPotHole')
    from PIL import Image
    for i, (name, cap) in enumerate(CAPTIONS.items(), 1):
        jpg = SUP / f'{name}.jpg'
        Image.open(SUP / f'{name}.png').convert('RGB').save(jpg, quality=90, optimize=True)
        d.add_picture(str(jpg), width=Cm(12 if 'annotation' in name else 17))
        d.paragraphs[-1].paragraph_format.keep_with_next = True
        p = d.add_paragraph(); r = p.add_run(f'Fig. S{i} '); r.bold = True; r.font.size = Pt(9)
        r2 = p.add_run(cap); r2.font.size = Pt(9)
    d.save(SUP / 'Online_Resource_2.docx')
    print('  saved Online_Resource_2.docx')


def write_sample(t, man):
    """Images shown in the figures + 10 random B-clean and 10 random C-test images, with labels."""
    for _, r in t[~t.leaked].sample(10, random_state=3).iterrows():
        add_sample(r.key, 'random_b_clean', split='test', video=r.seq_prefix, d_H=int(r.nn_d))
    cm = man[man.new_split == 'test'].sample(10, random_state=3)
    for d_, v in zip(cm.dest_name, cm.seq_prefix):
        add_sample(str(rc.V2_DIR / 'test' / 'images' / d_), 'random_c_test', split='v2-test', video=v)
    if SAMPLE.exists():
        shutil.rmtree(SAMPLE)
    (SAMPLE / 'images').mkdir(parents=True); (SAMPLE / 'labels').mkdir()
    df = pd.DataFrame(SAMPLE_ROWS).drop_duplicates('src')
    for src in df.src:
        p = Path(src)
        shutil.copy2(p, SAMPLE / 'images' / p.name)
        lp = rc.label_path(src)
        if lp.exists():
            shutil.copy2(lp, SAMPLE / 'labels' / lp.name)
        else:
            (SAMPLE / 'labels' / (p.stem + '.txt')).write_text('')
    import hashlib
    df['md5'] = [hashlib.md5(Path(s).read_bytes()).hexdigest() for s in df.src]
    df.drop(columns='src').to_csv(SAMPLE / 'sample_manifest.csv', index=False)
    (SAMPLE / 'README.txt').write_text(
        'Sample of BharatPotHole (and RDD2022-India) images used in Online Resource 2, with YOLO labels.\n'
        'Images remain under the licences of their original datasets (see docs/DATA.md).\n'
        'sample_manifest.csv: role, file, split, video, d_H to the nearest training image, SSIM, MD5.\n')
    z = shutil.make_archive(str(SAMPLE), 'zip', SAMPLE)
    print(f'  sample dataset: {len(df)} images -> {z}')


if __name__ == '__main__':
    main()
