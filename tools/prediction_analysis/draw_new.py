"""Compact 1 x 4 Figure 2; unchanged scores and every contrast retained.

PDF text uses the manuscript's pdflatex + times package. The PNG is rendered
from that PDF. The editable SVG uses Times-family text and Computer Modern math.
"""
from pathlib import Path
import json
import shutil
import subprocess

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent
DATA = ROOT
OUT = ROOT.parents[1] / "figures"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "Nimbus Roman"],
    "font.size": 7.2, "axes.labelsize": 7.2, "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0, "axes.linewidth": .55,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "mathtext.fontset": "cm", "pgf.texsystem": "pdflatex", "pgf.rcfonts": False,
    "pgf.preamble": r"\usepackage{times}\usepackage{amsmath,amssymb}",
    "text.color": "#214A55", "axes.labelcolor": "#214A55",
    "xtick.color": "#34404C", "ytick.color": "#34404C",
    "savefig.facecolor": "white", "figure.facecolor": "white",
})


def trim(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_color("#83929E")
    ax.tick_params(length=2.0, width=.5, pad=1.6)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#DBE1E5", lw=.4, ls=(0, (2, 3)))


def boxes(ax, groups, positions, colors, seed):
    rng = np.random.default_rng(seed)
    for vals, x, col in zip(groups, positions, colors):
        # Every contrast is shown, including boxplot outliers. Jitter is x-only.
        ax.scatter(x + rng.uniform(-.20, .20, len(vals)), vals,
                   s=2.1, c=col, alpha=.33, linewidths=0, zorder=2)
        ax.boxplot([vals], positions=[x], widths=.42, showfliers=False,
                   patch_artist=True, manage_ticks=False, whis=1.5, zorder=3,
                   boxprops={"facecolor": col, "edgecolor": col,
                             "alpha": .35, "linewidth": .65},
                   medianprops={"color": "#102A43", "linewidth": 1.0},
                   whiskerprops={"color": col, "linewidth": .65},
                   capprops={"color": col, "linewidth": .65})
    ax.axhline(0, c="#35434E", lw=.75, ls=(0, (4, 2.5)), zorder=1)
    trim(ax)


def matrix_axes(ax, n):
    ax.tick_params(which="both", length=0, pad=1.8)
    ax.set_xticks(np.arange(-.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", lw=.5)
    for spine in ax.spines.values():
        spine.set_visible(False)


def colorbar(fig, im, left, width, ticks, label):
    cax = fig.add_axes([left, .078, width, .025])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=ticks)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=1.8, width=.5, pad=1, labelsize=7)
    cb.ax.set_title(label, fontsize=7.2, pad=3)


def build(pdftoppm):
    result=json.loads((DATA / "statistics.json").read_text())
    if not result["strict_rescoring"] or result["blocks"]!=3969:raise ValueError("complete verified result cut required")
    a=pd.DataFrame(result["pairs"]);tr=pd.DataFrame(result["transfers"]);de=pd.DataFrame(result["density"])
    array={k:np.asarray(v) for k,v in result["arrays"].items()}
    # Exactly the unmodified ICLR text width, with four panels on one row.
    fig = plt.figure(figsize=(5.5, 2.45))
    ink, blue, coral, teal = "#214A55", "#437DA6", "#C97258", "#378A88"
    cmap = LinearSegmentedColormap.from_list(
        "pdeobs_transfer", ["#FCF6EE", "#F0CDAA", "#D99474", "#B75C50", "#713747"])
    bottom, height = .335, .49
    title_y = .985
    for left, title in [(.008, "(a) PDE / method"), (.278, "(b) Transfer"),
                        (.538, "(c) Layout"), (.790, "(d) Density")]:
        fig.text(left, title_y, title, fontsize=8.1, weight="bold", va="top")

    a_vals = a.C_over_D.to_numpy().reshape(7, 7)
    ax = fig.add_axes([.065, bottom, .176, height])
    vmax = max(2., float(np.ceil(np.log10(a_vals).max() * 2) / 2))
    im = ax.imshow(np.log10(a_vals), cmap=cmap, norm=Normalize(min(0.,float(np.log10(a_vals).min())), vmax),
                   aspect="auto", interpolation="none")
    ax.set_xticks(range(7), ["U-FNO", "FNO", "CNO", "DON", "GNOT", "Trans.", "PINO"])
    plt.setp(ax.get_xticklabels(), rotation=65, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(7), ["Darcy", "Poiss.", "Helm.", "Heat", "R-D", "Burg.", "N-S"])
    matrix_axes(ax, 7)
    ax.axhline(2.5, c=ink, lw=.8)
    marks = a.fixed_original500.to_numpy().reshape(7, 7)
    for i, j in np.argwhere(marks):
        col = "white" if np.log10(a_vals[i, j]) > vmax * .65 else ink
        ax.plot(j, i, "o", ms=1.8, mec="none", mfc=col)
    colorbar(fig, im, .077, .156, [0, 1, 2, 3], r"$\log_{10}(C/D)$")

    axb = fig.add_axes([.318, bottom, .178, height])
    b_vals = array["median_R"]
    bmax = max(1.5, float(np.ceil(np.log10(b_vals).max() * 2) / 2))
    imb = axb.imshow(np.log10(b_vals), cmap=cmap, norm=Normalize(min(0.,float(np.log10(b_vals).min())), bmax),
                     aspect="auto", interpolation="none")
    views = ["R50", "H", "V", "CL", "BL", "LI", "BD", "R65", "R80"]
    axb.set_xticks(range(9), views)
    plt.setp(axb.get_xticklabels(), rotation=65, ha="right", rotation_mode="anchor")
    axb.set_yticks(range(9), views)
    matrix_axes(axb, 9)
    for pos in [3.5, 6.5]:
        axb.axhline(pos, c=ink, lw=.65)
        axb.axvline(pos, c=ink, lw=.65)
    fig.text(.407, .867, r"Train $\downarrow$ / test $\rightarrow$",
             ha="center", fontsize=7.0)
    colorbar(fig, imb, .329, .156, [0, 1, 2], r"Median $\log_{10} R_{v\to w}$")

    axc = fig.add_axes([.558, bottom, .182, height])
    groups = [tr.loc[tr.count_group.eq(k) & tr.task.eq(t), "log10_ratio"].to_numpy()
              for k in [8192, 8284] for t in ["recovery", "rollout"]]
    positions = [0, 1, 2.5, 3.5]
    boxes(axc, groups, positions, [blue, coral, blue, coral], 20260924)
    axc.set_ylim(min(-1.65,float(np.min(np.concatenate(groups)))-.2),max(4.2,float(np.max(np.concatenate(groups)))+.2))
    axc.set_xlim(-.52, 4.02)
    axc.set_yticks([-1, 0, 1, 2, 3, 4])
    axc.set_xticks(positions, ["Rec.", "Roll.", "Rec.", "Roll."])
    plt.setp(axc.get_xticklabels(), rotation=60, ha="right", rotation_mode="anchor")
    axc.axvline(1.75, color="#DDE4E8", lw=.6, zorder=0)
    fig.text(.649, .867, r"$\log_{10}(E_{v,w}/E_{w,w})$", ha="center", fontsize=7.2)
    for center, label in [(.5, "Group I"), (3, "Group II")]:
        axc.text(center, -.285, label, transform=axc.get_xaxis_transform(),
                 ha="center", fontsize=7.0, clip_on=False)

    axd = fig.add_axes([.811, bottom, .182, height])
    density_groups = [de.loc[de.train_view.eq(v) & de.test_view.eq(w), "log10_ratio"].to_numpy()
                      for v in ["R50", "R65", "R80"] for w in ["R65", "R80"]]
    pos = [0, 1, 2.5, 3.5, 5, 6]
    boxes(axd, density_groups, pos, ["#83B5B0", teal] * 3, 20260925)
    axd.set_ylim(min(-3.25,float(np.min(np.concatenate(density_groups)))-.2),max(3.1,float(np.max(np.concatenate(density_groups)))+.2))
    axd.set_xlim(-.6, 6.6)
    axd.set_yticks([-3, -2, -1, 0, 1, 2, 3])
    axd.set_xticks([.5, 3, 5.5], ["R50", "R65", "R80"])
    axd.set_xlabel("Training pattern", labelpad=2)
    for x in [1.75, 4.25]:
        axd.axvline(x, c="#DDE4E8", lw=.6, zorder=0)
    fig.text(.902, .867, r"$\log_{10}(E_{v,w}/E_{v,\mathrm{R50}})$",
             ha="center", fontsize=7.2)
    axd.legend(handles=[Patch(facecolor="#83B5B0", label="R65"),
                        Patch(facecolor=teal, label="R80")],
               title="Test pattern", title_fontsize=7.0, fontsize=7.0,
               loc="upper center", bbox_to_anchor=(.5, -.29), ncol=2,
               frameon=False, borderpad=0, handlelength=.8, handletextpad=.3,
               columnspacing=.7, labelspacing=.15)

    assert len(a) == 49 and len(tr) == 3528 and len(de) == 294
    assert sum(map(len, groups)) == 882
    for axis, values in [(axc, groups), (axd, density_groups)]:
        lo, hi = axis.get_ylim()
        assert all(np.isfinite(vals).all() and (vals > lo).all() and (vals < hi).all()
                   for vals in values), "All data must remain inside the displayed axes."

    OUT.mkdir(parents=True, exist_ok=True)
    pdf = OUT / "fig_cross_observation_results.pdf"
    # PGF uses the same Times text / Computer Modern mathematics as the paper.
    fig.savefig(pdf, backend="pgf", metadata={
        "Title": "PDE-OBS: Evaluation across observation patterns", "Author": "",
        "Subject": "Complete strict prediction scores; compact four-panel layout",
        "Creator": "Matplotlib and LaTeX"})
    subprocess.run([pdftoppm, "-singlefile", "-png", "-r", "400", str(pdf),
                    str(pdf.with_suffix(""))], check=True, capture_output=True)
    fig.savefig(OUT / "fig_cross_observation_results.svg", metadata={
        "Title": "PDE-OBS evaluation across observation patterns", "Creator": "Matplotlib"})
    svg_path = OUT / "fig_cross_observation_results.svg"
    svg_path.write_text("\n".join(line.rstrip() for line in
                                svg_path.read_text(encoding="utf-8").splitlines()) + "\n",
                        encoding="utf-8")
    metadata = {
        "fixed_original500_marks":int(a.fixed_original500.sum()),"new_score_only":True,"layout": "1x4", "width_inches": 5.5, "height_inches": 2.45,
        "minimum_font_pt": 7.0, "pdf_text_font": "LaTeX times (Nimbus Roman)",
        "pdf_math_font": "Computer Modern", "png_rendered_from_pdf": True,
        "cell_value_annotations": False, "distribution_count_annotations": False,
        "points_in_panel_c": sum(map(len, groups)),
        "points_in_panel_d": sum(map(len, density_groups)),
        "a_color_log10_max": vmax, "b_color_log10_max": bmax,
        "b_ratio_min": float(b_vals.min()), "b_ratio_max": float(b_vals.max()),
        "outliers_clipped": False,
        "axes_c_limits": list(axc.get_ylim()), "axes_d_limits": list(axd.get_ylim()),
    }
    (OUT / "figure_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, default=DATA)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--pdftoppm", default=shutil.which("pdftoppm"))
    args = parser.parse_args()
    if not args.pdftoppm or not shutil.which("pdflatex"):
        parser.error("pdflatex and pdftoppm are required for manuscript-matched PDF/PNG export")
    DATA, OUT = args.analysis_dir, args.output_dir
    build(args.pdftoppm)
