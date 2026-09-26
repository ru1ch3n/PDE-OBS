"""Regenerate the figures used in the documentation, from this repository's own code.

The figures are not decoration: they show the exact masks the benchmark freezes and the exact
quantity the strict scorer measures. Everything here is produced by calling the library, so a
reader can check a figure by regenerating it.

    python -m pip install ".[analysis]"        # matplotlib
    python docs/figures/make_figures.py        # writes the PNGs next to this script

Deterministic: the mask seed is fixed below, and the field is produced by the same generator the
CLI uses, with a fixed seed. Nothing here reads a dataset, downloads anything or trains a model.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import colors  # noqa: E402

from pdeobs.api import specs  # noqa: E402
from pdeobs.masks import apply_mask, generate_mask  # noqa: E402

RESOLUTION = 128
MASK_SEED = 20260804
OBSERVED = "#1b3a6b"
HIDDEN = "#e8eaee"
FIELD_CMAP = "RdBu_r"
INK = "#0d1729"
PANEL = "#14213a"
ACCENT = "#36c5d8"
MUTED = "#9fb0c9"
EDGE = "#24344f"


def paper_views() -> dict[str, dict]:
    return dict(specs.PAPER_VIEWS)


def build_view_mask(view: dict) -> np.ndarray:
    spec = dict(view["mask"])
    protocol = spec.pop("protocol")
    return generate_mask(protocol, RESOLUTION, seed=MASK_SEED, **spec)


def figure_observation_views(out: Path) -> Path:
    """The nine frozen views, with their realized observed-cell counts."""
    views = paper_views()
    fig, axes = plt.subplots(3, 3, figsize=(8.2, 9.2))
    cmap = colors.ListedColormap([HIDDEN, OBSERVED])
    for ax, (name, view) in zip(axes.ravel(), views.items()):
        mask = build_view_mask(view)
        count = int(mask.sum())
        ax.imshow(mask, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(f"{view['label']}  ·  {name}", fontsize=9, pad=6)
        ax.set_xlabel(f"{count:,} of {RESOLUTION * RESOLUTION:,} cells observed"
                      f"  ({count / RESOLUTION ** 2:.0%})", fontsize=7.5)
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ax.spines.values():
            side.set_edgecolor("#c3c7ce")
    fig.suptitle("The nine frozen observation views, at 128 x 128", fontsize=12, y=0.975)
    fig.text(0.5, 0.012, "dark = observed, light = hidden.  Counts are realized, not nominal: "
                         "a ratio is rounded once to an exact cell count.", ha="center", fontsize=8, color="#444")
    fig.tight_layout(rect=(0, 0.03, 1, 0.955))
    fig.subplots_adjust(hspace=0.42)
    path = out / "observation_views.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def solve_one_record() -> np.ndarray:
    """Generate one real Poisson record with the same entry point the CLI uses, and return its state.

    This is an actual solve, not a drawn picture: the figure would be misleading otherwise.
    """
    import glob
    import os
    import subprocess
    import tempfile

    import h5py

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src")] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    with tempfile.TemporaryDirectory() as tmp:
        done = subprocess.run(
            [sys.executable, "-m", "pdeobs", "generate-case", "--pde", "poisson", "--boundary", "periodic",
             "--setting", "smooth_grf", "--param-regime", "low", "--num-samples", "1",
             "--resolution", str(RESOLUTION), "--seed", str(MASK_SEED), "--root", tmp],
            capture_output=True, text=True, cwd=str(ROOT), env=env,
        )
        if done.returncode:
            raise RuntimeError("generate-case failed; run it from the source root or install the package: " + done.stderr[-800:])
        shard = sorted(glob.glob(f"{tmp}/**/*.h5", recursive=True))[0]
        with h5py.File(shard, "r") as handle:
            trajectory = np.asarray(handle["trajectory"][0])   # (T, H, W, C)
    return np.asarray(trajectory[0, :, :, 0], dtype=float)


def figure_task(out: Path) -> Path:
    """What one task instance looks like: complete field, observation, and the hidden part scored."""
    field = solve_one_record()
    scale = float(np.abs(field).max())
    if scale:
        field = field / scale

    view = paper_views()["block_observed_50pct"]
    mask = build_view_mask(view)
    observed = apply_mask(field, mask, fill_value=np.nan)
    hidden = np.where(mask, np.nan, field)

    fig, axes = plt.subplots(1, 3, figsize=(10.6, 4.5))
    limit = float(np.abs(field).max())
    shared = dict(cmap=FIELD_CMAP, vmin=-limit, vmax=limit, interpolation="nearest")
    titles = ("Complete record\n(what the solver produced)",
              f"Observation ({view['label']})\n(what the method receives)",
              "Hidden cells\n(reported as a separate diagnostic)")
    for ax, data, title in zip(axes, (field, observed, hidden), titles):
        ax.imshow(np.ma.masked_invalid(data), **shared)
        ax.set_title(title, fontsize=9.5, pad=8)
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ax.spines.values():
            side.set_edgecolor("#c3c7ce")
    fig.text(0.5, 0.02, "The headline metric is the per-identity relative L2 over the COMPLETE field; "
                        "the hidden-cell error is reported separately, never as the headline.",
             ha="center", fontsize=8.5, color="#444")
    fig.tight_layout(rect=(0, 0.06, 1, 0.99))
    fig.subplots_adjust(top=0.80)
    path = out / "task_example.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def figure_grid(out: Path) -> Path:
    """The 441-row grid: 7 families x 7 public learned methods x 9 views, and its evaluation fan-out."""
    from pdeobs.one_setting import PDE_ORDER, PUBLIC_LEARNED_METHODS

    families = list(PDE_ORDER)
    methods = list(PUBLIC_LEARNED_METHODS)
    views = list(paper_views())
    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    cell = np.zeros((len(families), len(methods) * 1))
    ax.imshow(cell, cmap=colors.ListedColormap(["#eef1f6"]), aspect="auto")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, fontsize=8.5, rotation=30, ha="right")
    ax.set_yticks(range(len(families)))
    ax.set_yticklabels([f.replace("_", " ") for f in families], fontsize=8.5)
    for i in range(len(families)):
        for j in range(len(methods)):
            ax.text(j, i, str(len(views)), ha="center", va="center", fontsize=8, color="#1b3a6b")
    for side in ax.spines.values():
        side.set_edgecolor("#c3c7ce")
    ax.set_title(f"{len(families)} families x {len(methods)} public learned methods x {len(views)} frozen views "
                 f"= {len(families) * len(methods) * len(views)} training rows", fontsize=11, pad=10)
    ax.set_xlabel(f"each cell trains {len(views)} checkpoints, one per training view; each checkpoint is then "
                  f"evaluated on all {len(views)} test views "
                  f"({len(families) * len(methods) * len(views)} x {len(views)} = "
                  f"{len(families) * len(methods) * len(views) * len(views)} possible blocks)", fontsize=8.5, labelpad=10)
    fig.tight_layout()
    path = out / "benchmark_grid.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def figure_hero(out: Path) -> Path:
    """A wide title card. The three panels on the right are a real solve, not an illustration."""
    field = solve_one_record()
    scale = float(np.abs(field).max())
    if scale:
        field = field / scale
    mask = build_view_mask(paper_views()["block_observed_50pct"])
    observed = np.where(mask, field, np.nan)
    hidden = np.where(mask, np.nan, field)

    fig = plt.figure(figsize=(16.0, 4.2))
    fig.patch.set_facecolor(INK)
    fig.add_artist(plt.Rectangle((0, 0.965), 1, 0.035, transform=fig.transFigure,
                                 facecolor=ACCENT, edgecolor="none", zorder=5))

    fig.text(0.045, 0.74, "P D E  ·  O B S", fontsize=13, color=ACCENT, weight="bold")
    fig.text(0.045, 0.50, "Partial-observation", fontsize=34, color="white", weight="bold", va="center")
    fig.text(0.045, 0.31, "PDE benchmark", fontsize=34, color="white", weight="bold", va="center")
    fig.text(0.045, 0.15, "deterministic records  \u00b7  frozen observation views  \u00b7  strict scoring",
             fontsize=12.5, color=MUTED, va="center")

    titles = ("observed", "masked", "recovered target")
    for i, (data, title) in enumerate(zip((observed, np.where(mask, np.nan, 0.0 * field), hidden), titles)):
        ax = fig.add_axes([0.545 + i * 0.145, 0.24, 0.13, 0.58])
        ax.set_facecolor(PANEL)
        if i == 1:
            ax.imshow(mask.astype(float), cmap=colors.ListedColormap([ACCENT, EDGE]),
                      vmin=0, vmax=1, interpolation="nearest")
        else:
            ax.imshow(np.ma.masked_invalid(data), cmap=FIELD_CMAP, vmin=-1, vmax=1, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title.upper(), fontsize=9, color=MUTED, pad=7)
        for side in ax.spines.values():
            side.set_edgecolor(EDGE)
    fig.text(0.7425, 0.11, "one record, one of the nine frozen views, and the cells a method must recover",
             fontsize=9, color=MUTED, ha="center")

    path = out / "hero.png"
    fig.savefig(path, dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def main() -> None:
    for build in (figure_hero, figure_observation_views, figure_task, figure_grid):
        path = build(HERE)
        print(f"wrote {path.relative_to(ROOT)}  ({path.stat().st_size // 1024} KiB)")


if __name__ == "__main__":
    main()
