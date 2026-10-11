#!/usr/bin/env python3
"""
Collect the release artifacts (no images) from Drive into one folder and one zip, ready to attach
to a GitHub release or to upload to Zenodo.

  splits/manifest_v2.csv          every released file: MD5, video, source frame, new split or drop reason
  splits/excluded_images.csv      files removed before splitting, with the reason
  splits/a_test_subsets.csv       published test set: video, d_H to training, Leaked / B-clean / B-strict flags
  audit/                          per-image audit table and summaries (script 01)
  rdd_india/                      RDD2022-India audit tables (script 03), images excluded
  results/                        tables written by scripts 05 and 06 (CSV, JSON, Markdown)
  predictions/ (optional)         cached per-image predictions (*.pkl), add --with-predictions
  SHA256SUMS.txt

Images are not redistributed. Every image is identified by its released file name and MD5 hash,
so the split can be rebuilt from the public BharatPotHole release.

Run:  !python "/content/drive/MyDrive/POTHOLE/scripts/08_export_release.py" [--with-predictions]
Output: MyDrive/POTHOLE/release/bharatpothole-leakage-artifacts/ and ...-artifacts.zip
"""
import argparse
import hashlib
import os
import shutil
from pathlib import Path

import pandas as pd

POTHOLE = Path(os.environ.get('POTHOLE_DRIVE', '/content/drive/MyDrive/POTHOLE'))
OUT = POTHOLE / 'release' / 'bharatpothole-leakage-artifacts'
SMALL = ('.csv', '.json', '.md', '.txt', '.yaml')


def copy_small(src: Path, dst: Path):
    if not src.exists():
        print('  (missing, skipped)', src)
        return
    dst.mkdir(parents=True, exist_ok=True)
    for f in sorted(src.iterdir()):
        if f.is_file() and f.suffix.lower() in SMALL:
            shutil.copy2(f, dst / f.name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--with-predictions', action='store_true')
    a = ap.parse_args()
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / 'splits').mkdir(parents=True)

    man = pd.read_csv(POTHOLE / 'BharatPotHole_v2' / 'manifest_v2.csv')
    man.to_csv(OUT / 'splits' / 'manifest_v2.csv', index=False)
    cols = [c for c in ('rel_path', 'md5', 'original_split', 'seq_prefix', 'source_name', 'drop_reason') if c in man]
    if 'drop_reason' in man:
        man[man.drop_reason.notna()][cols].to_csv(OUT / 'splits' / 'excluded_images.csv', index=False)

    pi = pd.read_csv(POTHOLE / 'dataset_assessment' / 'per_image.csv')
    t = pi[pi.split == 'test'].copy()
    for c in ('has_dup_in_train', 'source_in_train'):
        t[c] = t[c].fillna(False).astype(bool)
    t['leaked'] = t.has_dup_in_train | t.source_in_train
    t['b_clean'] = ~t.leaked                                   # stated rule (main analysis)
    strict = POTHOLE / 'BharatPotHole_v2' / 'B_original_test_clean.txt'
    names = {Path(l.strip()).name for l in strict.read_text().splitlines() if l.strip()} if strict.exists() else set()
    t['b_strict'] = t.rel_path.map(lambda p: Path(p).name in names)
    keep = [c for c in ('rel_path', 'md5', 'seq_prefix', 'source_name', 'n_boxes', 'nn_phash_to_train',
                        'nn_phash_to_valid', 'has_dup_in_train', 'source_in_train', 'leaked', 'b_clean',
                        'b_strict') if c in t]
    t[keep].to_csv(OUT / 'splits' / 'a_test_subsets.csv', index=False)
    print(f'A-test {len(t)} | Leaked {int(t.leaked.sum())} | B-clean {int(t.b_clean.sum())} | '
          f'B-strict {int(t.b_strict.sum())}')

    copy_small(POTHOLE / 'dataset_assessment', OUT / 'audit')
    copy_small(POTHOLE / 'rdd_india_assessment', OUT / 'rdd_india')
    copy_small(POTHOLE / 'experiments' / 'leakage_paper' / 'results', OUT / 'results')
    copy_small(POTHOLE / 'experiments' / 'leakage_paper' / 'revision', OUT / 'results' / 'revision')
    if a.with_predictions:
        src = POTHOLE / 'experiments' / 'leakage_paper' / 'cache'
        (OUT / 'predictions').mkdir()
        for f in sorted(src.glob('*.pkl')):
            shutil.copy2(f, OUT / 'predictions' / f.name)

    lines = []
    for f in sorted(p for p in OUT.rglob('*') if p.is_file()):
        lines.append(f'{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.relative_to(OUT)}')
    (OUT / 'SHA256SUMS.txt').write_text('\n'.join(lines) + '\n')
    z = shutil.make_archive(str(OUT), 'zip', OUT)
    print(f'{len(lines)} files -> {OUT}\nzip -> {z}')


if __name__ == '__main__':
    main()
