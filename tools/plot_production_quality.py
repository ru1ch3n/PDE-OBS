"""Render only recorded production residual summaries, without synthetic samples.

Requires matplotlib and a TeX installation providing pdflatex + times. No model,
solver, training data, or prediction arrays are loaded. Range bars are extrema,
not confidence intervals. The PDF uses the manuscript's Times LaTeX typography.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('pgf')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ORDER = ['darcy', 'poisson', 'helmholtz', 'heat', 'reaction_diffusion', 'burgers', 'navier_stokes']
LABELS = ['Darcy', 'Poisson', 'Helmholtz', 'Heat', 'Reaction--diffusion', 'Burgers', 'Navier--Stokes']
COLORS = ['#4E6096', '#667FA5', '#849BBB', '#529B8E', '#6DA596', '#D08B64', '#B36979']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--summary', type=Path, required=True)
    p.add_argument('--pdf', type=Path, required=True)
    p.add_argument('--receipt', type=Path, required=True)
    a = p.parse_args()
    if a.pdf.exists() or a.receipt.exists():
        raise FileExistsError('Use fresh output paths; never overwrite an audited figure.')
    q = json.loads(a.summary.read_text())
    assert q['total_evaluated_slice_records'] == 14000
    for pde in ORDER:
        r = q['evaluated_slice'][pde]
        assert r['records'] == 2000 and r['condition'] == 'smooth_grf'
        assert r['normalization_thresholds'] == [0.05]
        assert 0 < r['min_normalized_residual'] <= r['mean_normalized_residual'] <= r['max_normalized_residual'] < .05
    plt.rcParams.update({'pgf.texsystem': 'pdflatex', 'pgf.rcfonts': False,
                         'pgf.preamble': r'\usepackage{times}\usepackage{amsmath}',
                         'font.family': 'serif', 'font.size': 9, 'axes.labelsize': 9,
                         'xtick.labelsize': 8.2, 'ytick.labelsize': 8,
                         'axes.linewidth': .55, 'xtick.major.width': .5, 'ytick.major.width': .5})
    fig, ax = plt.subplots(figsize=(6.5, 1.8))
    fig.subplots_adjust(left=.1, right=.99, bottom=.24, top=.8)
    for i, (pde, color) in enumerate(zip(ORDER, COLORS)):
        r = q['evaluated_slice'][pde]
        ax.vlines(i, r['min_normalized_residual'], r['max_normalized_residual'], color=color, lw=1.4, zorder=2)
        ax.hlines([r['min_normalized_residual'], r['max_normalized_residual']], i-.08, i+.08, color=color, lw=.8)
        ax.plot(i, r['mean_normalized_residual'], 'o', ms=4.1, color=color, markeredgecolor='white', markeredgewidth=.35, zorder=3)
    ax.axhline(.05, color='#7C7C7C', lw=.8, ls=(0,(4,3)), zorder=1)
    ax.axvline(2.5, color='#D5D8DD', lw=.65)
    ax.set_yscale('log')
    ax.set_ylim(4e-9, .12)
    ax.set_yticks([1e-8, 1e-6, 1e-4, 1e-2])
    ax.set_yticklabels([r'$10^{-8}$', r'$10^{-6}$', r'$10^{-4}$', r'$10^{-2}$'])
    ax.minorticks_off()
    ax.set_xlim(-.5,6.5)
    ax.set_xticks(range(7), LABELS)
    ax.tick_params(axis='x', length=0, pad=5)
    ax.set_ylabel('Normalized residual', labelpad=5)
    ax.grid(axis='y', color='#E5E7EB', linestyle=':', linewidth=.55)
    ax.set_axisbelow(True)
    ax.spines[['top','right']].set_visible(False)
    ax.spines[['left','bottom']].set_color('#A4A9B0')
    legend = [Line2D([],[],marker='o',lw=0,color='#4E6096',ms=4,label='Record mean'),
              Line2D([],[],marker='|',lw=.9,color='#75849D',ms=6,label='Minimum--maximum'),
              Line2D([],[],ls=(0,(4,3)),lw=.8,color='#7C7C7C',label='Generation gate')]
    fig.legend(handles=legend, loc='upper center', bbox_to_anchor=(.55,1.02), ncol=3,
               frameon=False, handlelength=1.5, columnspacing=2, fontsize=8)
    a.pdf.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(a.pdf, metadata={'Title':'Recorded production consistency', 'Author':'', 'CreationDate':None, 'ModDate':None})
    plt.close(fig)
    receipt = {'schema':'pdeobs-production-residual-figure/v1',
               'summary_sha256':hashlib.sha256(a.summary.read_bytes()).hexdigest(),
               'source_summary_sha256':q['source_summary_sha256'],
               'figure_sha256':hashlib.sha256(a.pdf.read_bytes()).hexdigest(),
               'records':14000, 'records_per_family':2000, 'family_order':ORDER,
               'marks':'Mean and observed min/max across records; no fitted distribution or confidence interval.',
               'scope':'Original generation-time, family-specific discrete consistency; not independent continuum solution error.',
               'fonts':'pdflatex with times, matching manuscript preamble',
               'style_reference':'Beyond Matryoshka (ICML 2025), Figure 3: compact axes, restrained color, thin rules; no source data or graphics copied.',
               'style_reference_url':'https://arxiv.org/pdf/2503.01776',
               'new_solves':0, 'new_inference':0}
    with a.receipt.open('x',encoding='utf-8') as f:
        json.dump(receipt,f,indent=2,sort_keys=True)
    print(json.dumps(receipt,indent=2))


if __name__ == '__main__':
    main()
