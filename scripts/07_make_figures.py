#!/usr/bin/env python3
"""
Publication figures (Figs. 1-5) from the result files written by scripts 01-06.
White background, no grid, minimal text, Okabe-Ito colours, 84 mm (single) / 174 mm (double) width.
Each figure is saved as PDF (vector), TIFF (600 dpi, LZW) and PNG (300 dpi).

Inputs (Drive, POTHOLE_DRIVE overrides the root):
  experiments/leakage_paper/results/T0_audit_numbers.json        (script 05)
  experiments/leakage_paper/results/T1_T3_main_results.csv        (script 05)
  experiments/leakage_paper/revision/R5_substitution_controls.csv (script 06)
  experiments/leakage_paper/revision/R6_subsets_positive.csv      (script 06)
  experiments/leakage_paper/revision/R6_bins_positive.csv         (script 06)
  dataset_assessment/per_image.csv                                 (script 01)
  BharatPotHole_v2/manifest_v2.csv                                 (script 02)
Output: experiments/leakage_paper/publication/figures/

Run:  !python "/content/drive/MyDrive/POTHOLE/scripts/07_make_figures.py"
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pubstyle as P                                     # noqa: E402  (style, colours, notation)

REV = P.POTHOLE / 'experiments' / 'leakage_paper' / 'revision'


def need(path):
    if not Path(path).exists():
        sys.exit(f'Missing input: {path}\nRun the script that produces it first (see README).')
    return path


def fig1_workflow(plt, audit, split):
    """Audit, deduplication and splitting workflow with the counts of this dataset."""
    from matplotlib.patches import FancyBboxPatch
    rel, md5, cp = audit['images_released'], audit['exact_md5_duplicates_removed'], audit['augmentation_copies_removed']
    pos_r, neg_r = audit['positive_images_released'], audit['negative_images_released']
    uniq = audit['unique_frames']
    pos_u_pre = pos_r - md5 - cp                         # all removed files were positive images
    neg_u_pre = uniq - pos_u_pre
    pos_c, neg_c = audit['positive_images_unique'], audit['negative_images_unique']
    fig = plt.figure(figsize=(P.W2, 48 * P.MM)); fig.set_layout_engine('none')
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 174); ax.set_ylim(0, 48); ax.axis('off')
    ec = '#404040'

    def box(x, y, w, h, title, sub=None):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=1.2', fc='white', ec=ec, lw=0.6))
        ax.text(x + w / 2, y + (h * 0.70 if sub else h / 2), title, ha='center', va='center', fontweight='bold')
        if sub:
            ax.text(x + w / 2, y + h * 0.27, sub, ha='center', va='center', fontsize=6.5, linespacing=1.25)

    def arrow(x0, x1, y, lab):
        ax.annotate('', xy=(x1, y), xytext=(x0, y),
                    arrowprops=dict(arrowstyle='-|>', lw=0.6, color=ec, mutation_scale=7))
        ax.text((x0 + x1) / 2, y - 3.2, lab, ha='center', va='top', fontsize=6.2, linespacing=1.25)

    f = lambda v: f'{v:,}'
    y, h = 14, 20
    box(2, y, 30, h, f'Released {f(rel)}', f'{f(pos_r)} positive\n{f(neg_r)} negative\n{f(audit["boxes_released"])} boxes')
    arrow(32, 48, y + h / 2, f'−{f(md5)} exact\nduplicates\n−{f(cp)} export\ncopies')
    box(48, y, 30, h, f'Unique {f(uniq)}', f'{f(pos_u_pre)} positive\n{f(neg_u_pre)} negative')
    arrow(78, 92, y + h / 2, 'label\ncleaning')
    box(92, y, 30, h, f'Cleaned {f(uniq)}', f'{f(pos_c)} positive\n{f(neg_c)} negative\n{f(audit["boxes_unique_clean"])} boxes')
    arrow(122, 136, y + h / 2, 'grouped\nsplit')
    for i, (n, v) in enumerate([('Training', split['train']), ('Validation', split['valid']), ('Test', split['test'])]):
        yy = 33 - i * 13
        ax.add_patch(FancyBboxPatch((136, yy), 36, 10, boxstyle='round,pad=0,rounding_size=1.2', fc='white', ec=ec, lw=0.6))
        ax.text(154, yy + 5, f'{n}  {f(v)}', ha='center', va='center', fontweight='bold')
    ax.text(154, 45, 'Video-disjoint', ha='center', va='center', fontsize=6.5)
    P.save(fig, 'Fig1_workflow')
    plt.close(fig)


def parse_ci(s):
    lo, hi = re.findall(r'-?\d+\.\d+', str(s))[:2]
    return float(lo), float(hi)


def fig3_same_model(plt, sub, bins):
    """a: Leaked / A-test / B-clean (positive images); b: AP50 by d_H bin."""
    sub = sub.set_index('subset')
    fig, (a, b) = plt.subplots(1, 2, figsize=(P.W2, 2.2), gridspec_kw={'width_ratios': [1, 1.25]})
    bars = [('Leaked', P.C_LEAK), ('A-test', P.C_NEUTRAL), ('B-rule', P.C_CLEAN)]
    for i, (k, c) in enumerate(bars):
        v = float(sub.loc[k, 'AP50']); lo, hi = parse_ci(sub.loc[k, 'AP50_CI'])
        a.bar(i, v, width=0.6, color=c)
        a.errorbar(i, v, yerr=[[v - lo], [hi - v]], color=P.C_INK, lw=0.8, capsize=2.5, capthick=0.8)
    a.set_xticks(range(3), ['Leaked', 'A-test', 'B-clean']); a.tick_params(axis='x', length=0)
    a.set_ylabel(P.AP50); a.set_ylim(0, 1.0)
    diff = float(sub.loc['A-test', 'AP50']) - float(sub.loc['B-rule', 'AP50'])
    top = parse_ci(sub.loc['A-test', 'AP50_CI'])[1] + 0.06
    a.plot([1, 1, 2, 2], [top - 0.02, top, top, top - 0.02], color=P.C_INK, lw=0.7)
    a.text(1.5, top + 0.015, f'{P.DAP50} = {diff:.2f}', ha='center', va='bottom', fontsize=7)
    P.panel(a, 'a')
    y, lo, hi = bins.AP50.to_numpy(), bins.AP50_lo.to_numpy(), bins.AP50_hi.to_numpy()
    x = np.arange(len(bins))
    b.errorbar(x, y, yerr=[y - lo, hi - y], color=P.C_INK, lw=1.0, marker='o', ms=4, mfc=P.C_INK,
               capsize=2.5, capthick=0.8)
    b.set_xticks(x, [s.replace('-', '–') for s in bins['bin']])
    b.set_xlabel(P.DH + ' to nearest training image'); b.set_ylabel(P.AP50)
    b.set_ylim(0, 1.0); b.set_xlim(-0.4, len(bins) - 0.6)
    P.panel(b, 'b')
    P.save(fig, 'Fig3_same_model')
    plt.close(fig)


def fig4_contamination(plt, r5, ref_ap50):
    """AP50 versus rho for copies of training frames (leaked) and of unseen C-test frames (control)."""
    r5 = r5[r5['rho_%'].astype(str) != '-'].copy()
    r5['rho'] = r5['rho_%'].astype(float)
    fig, ax = plt.subplots(figsize=(P.W1, 2.3))
    sty = {'yolo26n': (P.C_CLEAN, 'o', 'YOLO26n'), 'yolo26s': (P.C_ALT, 's', 'YOLO26s')}
    for m, (c, mk, nm) in sty.items():
        for pool, ls, unseen in (('train copies (leaked)', '-', False), ('C-test copies (unseen)', (0, (3, 1.5)), True)):
            g = r5[(r5.model == m) & (r5.pool == pool)].sort_values('rho')
            if g.empty:
                continue
            ax.fill_between(g.rho, g.AP50_mean - g.AP50_sd, g.AP50_mean + g.AP50_sd, color=c,
                            alpha=0.10 if unseen else 0.15, lw=0)
            ax.plot(g.rho, g.AP50_mean, color=c, ls=ls, marker=mk, ms=3.2, mfc='white' if unseen else c, mew=0.8,
                    label=f'{nm}, ' + ('unseen copies' if unseen else 'training copies'))
    ax.axhline(ref_ap50, color=P.C_LEAK, lw=0.8, ls=(0, (3, 2)))
    ax.text(2, ref_ap50 + 0.012, 'YOLO26n, Protocol A, A-test', color=P.C_LEAK, fontsize=6.5, va='bottom')
    ax.set_xlabel(P.RHO + ' (%)'); ax.set_ylabel(P.AP50)
    ax.set_xlim(0, 100); ax.set_ylim(0, 0.75)
    ax.legend(loc='upper left', bbox_to_anchor=(0.0, 0.80), fontsize=6.3)
    P.save(fig, 'Fig4_contamination')
    plt.close(fig)


def main():
    P.FIG.mkdir(parents=True, exist_ok=True)
    audit = json.load(open(need(P.RES / 'T0_audit_numbers.json')))
    main_ = pd.read_csv(need(P.RES / 'T1_T3_main_results.csv'))
    main_['set'] = main_.test_set.map(P.SET_NAME)
    main_['protocol'] = main_.trained_on.str[0]
    d = {'main': main_,
         'pi': pd.read_csv(need(P.POTHOLE / 'dataset_assessment' / 'per_image.csv')),
         'man': pd.read_csv(need(P.POTHOLE / 'BharatPotHole_v2' / 'manifest_v2.csv'))}
    plt = P.setup_style()
    print('Figures ->', P.FIG)
    fig1_workflow(plt, audit, audit['v2_split_sizes'])
    P.fig2_distance(plt, d)
    fig3_same_model(plt, pd.read_csv(need(REV / 'R6_subsets_positive.csv')),
                    pd.read_csv(need(REV / 'R6_bins_positive.csv')))
    ref = P.row(main_, 'yolo26n', 'A', 'A-test')
    fig4_contamination(plt, pd.read_csv(need(REV / 'R5_substitution_controls.csv')),
                       float(ref.AP50) if ref is not None else 0.615)
    P.fig5_external(plt, d)


if __name__ == '__main__':
    main()
