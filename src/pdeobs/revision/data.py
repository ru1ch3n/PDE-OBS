"""V2 data access for the October 2026 revision.

It provides:
- the experiment configuration of a frozen recipe pointed at a V2 view;
- split selection with exact per-regime counts;
- per-parent observation layouts (`pdeobs.revision.v2masks`);
- the nearest-observation input fill.

This mirrors the harness that produced every October run, so a released checkpoint can be re-evaluated on exactly the
records, masks and inputs it was scored on.

A *recipe* is the dict of one entry of `configs/paper/october/recipes.yaml` completed with the run's
`pde`, `dataset_id`, `protocol` and `seed` (see `reproduce/october/replay_inference.py`).
"""
from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import yaml

from ..dataset import BenchmarkDataset
from . import v2masks as VM
from .fill import nearest_index
from .v2spec import BOUNDARY, ELL_OF, parse_dataset_id

STATIC = {"poisson", "darcy", "helmholtz"}
CAMPAIGN_SEED = 20260804                       # mask bank of the historical grid (unused by V2 layouts, kept in the config)
V2_SPLIT_COUNTS = {"train": 3000, "validation": 300, "test": 600, "confirm": 200}
V2_REGIMES = ("low", "medium", "high")
V2_EXTERNAL = {"rpb": {"counts": {"train": 1024, "validation": 128, "test": 256, "ood": 256}, "regimes": ("rpb",),
                       "setting": "rpb_sin_frequency"}}


def campaign_and_registry(repo_root: str | Path = ".") -> tuple[dict, dict]:
    from ..one_setting import load_campaign
    root = Path(repo_root)
    campaign = load_campaign(root / "configs/campaign/all_pde_one_setting_10method_9x9.yaml")
    registry = yaml.safe_load((root / "configs/method/all_pde_source_faithful_registry.yaml").read_text(encoding="utf-8"))
    return campaign, registry


def experiment_config(recipe: Mapping[str, Any], data_root: str | Path, repo_root: str | Path = ".") -> dict:
    """The frozen recipe of (pde, method) with the recipe's overrides (seed, epochs, learning rate, batch size)."""
    from ..one_setting import build_experiment_config
    campaign, registry = campaign_and_registry(repo_root)
    cfg = build_experiment_config(campaign, registry, dataset_root=data_root, pde=recipe["pde"],
                                  method=recipe.get("recipe") or recipe["method"], view="random_50pct", epochs=500)
    cfg["name"] = str(recipe.get("id", "october-run"))
    cfg["training"]["seed"] = int(recipe["seed"])
    cfg["data"]["mask_seed"] = CAMPAIGN_SEED
    if int(recipe.get("epochs", 500)) != 500:
        cfg["training"]["epochs"] = int(recipe["epochs"])
    if recipe.get("lr") is not None:
        cfg["training"]["learning_rate"] = float(recipe["lr"])
    if recipe.get("batch_size"):
        bs = int(recipe["batch_size"])
        cfg["training"]["batch_size"] = bs
        cfg["training"]["scheduler_steps_per_epoch"] = math.ceil(int(cfg["training"].get("records_per_epoch", 1800)) / bs)
    if recipe.get("param_channel"):
        base = cfg["method"]["base"] if "base" in cfg["method"] else cfg["method"]
        base["kwargs"]["geometry_channels"] = int(base["kwargs"].get("geometry_channels", 1)) + 1
    return cfg


def v2_layout(dataset_id: str) -> tuple[Mapping[str, int], tuple[str, ...]]:
    ellcode = str(dataset_id).split("_")[2]
    if ellcode in V2_EXTERNAL:
        return V2_EXTERNAL[ellcode]["counts"], V2_EXTERNAL[ellcode]["regimes"]
    return V2_SPLIT_COUNTS, V2_REGIMES


def v2_view(recipe: Mapping[str, Any]) -> str:
    """Directory of the V2 view: <dataset_id>[__grid<N>][__t4] (temporal families use the compact t4 view)."""
    grid = int(recipe.get("grid") or 128)
    view = recipe["dataset_id"] + ("" if grid == 128 else f"__grid{grid}")
    return view + ("" if recipe["pde"] in STATIC else "__t4")


def v2_config(recipe: Mapping[str, Any], data_root: str | Path, repo_root: str | Path = ".") -> dict:
    """Experiment config of a recipe pointed at its V2 view under data_root (the V2 store)."""
    pde, ellcode = parse_dataset_id(recipe["dataset_id"])
    if pde != recipe["pde"]:
        raise ValueError(f"dataset {recipe['dataset_id']} does not belong to {recipe['pde']}")
    cfg = experiment_config(recipe, data_root, repo_root)
    setting = V2_EXTERNAL[ellcode]["setting"] if ellcode in V2_EXTERNAL else ELL_OF[ellcode][1]
    cfg["data"]["root"] = str(Path(data_root) / v2_view(recipe))
    cfg["data"]["train_glob"] = "**/*.h5"
    cfg["data"]["filters"] = {"pde": pde, "boundary": BOUNDARY[pde], "setting": setting}
    counts, regimes = v2_layout(recipe["dataset_id"])
    n = int(recipe.get("records_per_epoch") or min(int(recipe.get("n_train", 9000)), counts["train"] * len(regimes)))
    cfg["training"]["scheduler_steps_per_epoch"] = math.ceil(n / int(cfg["training"]["batch_size"]))
    return cfg


class V2Dataset(BenchmarkDataset):
    """BenchmarkDataset with per-parent observation layouts and the optional nearest-observation input fill."""

    layout_fn: Callable[[Mapping[str, Any], tuple, int], tuple[np.ndarray, dict]] | None = None
    fill: str | None = None                       # "euclid" (static, Dirichlet) or "torus" (periodic)
    epoch = 0

    def _mask(self, target, metadata, *, dataset_index):
        if self.layout_fn is None:
            return super()._mask(target, metadata, dataset_index=dataset_index)
        spatial = tuple(target.shape[:2] if target.ndim <= 3 else target.shape[-3:-1])
        mask, info = self.layout_fn(metadata, spatial, self.epoch)
        metadata.update(info)
        metadata["observation_count"] = int(np.count_nonzero(mask))
        metadata["observation_ratio"] = float(np.mean(mask))
        return mask

    def __getitem__(self, index):
        item = super().__getitem__(index)
        if self.fill is not None:
            obs = np.asarray(item["observations"], dtype=np.float32)
            mask = np.asarray(item["mask"]) > 0.5
            frame = obs.reshape(-1, *obs.shape[-3:])[0]
            m2 = mask.reshape(-1, *mask.shape[-3:])[0][..., 0]
            idx = nearest_index(m2, self.fill == "torus")
            h, w = m2.shape
            filled = frame.reshape(h * w, -1)[idx.ravel()].reshape(frame.shape)
            item["observations"] = filled.reshape(obs.shape).astype(np.float32)
        return item


def v2_split(cfg: Mapping[str, Any], recipe: Mapping[str, Any], part: str) -> tuple[V2Dataset, dict]:
    """(dataset of one split ordered by sample id, receipt). Every planned parent must be present exactly once."""
    from ..one_setting import subset_dataset
    from ..runner import _dataset
    data = _dataset(copy.deepcopy(dict(cfg)), None)
    keep = [i for i, md in enumerate(data.metadata) if md.get("split") == part]
    counts, regimes = v2_layout(recipe["dataset_id"])
    by: dict[str, int] = {}
    for i in keep:
        by[str(data.metadata[i]["regime"])] = by.get(str(data.metadata[i]["regime"]), 0) + 1
    if set(by) != set(regimes) or any(v != counts[part] for v in by.values()):
        raise RuntimeError(f"{recipe['dataset_id']}/{part} incomplete: {by}")
    keep = sorted(keep, key=lambda i: data.metadata[i]["sample_id"])
    ids = [str(data.metadata[i]["sample_id"]) for i in keep]
    out = subset_dataset(data, keep)
    out.__class__ = V2Dataset
    receipt = {"dataset_id": recipe["dataset_id"], "view": v2_view(recipe), "part": part, "records": len(keep),
               "sample_ids_sha256": hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()}
    return out, receipt


def apply_layout(ds: V2Dataset, protocol: str, *, pde: str, rep: int = 0) -> None:
    """Fixed per-parent layout of replicate `rep` (U / C / L / G <K>, nested U<K>, N3 shifted / jittered grids)."""
    if VM.N3_PROTO.match(protocol):
        fixed = "_r" in protocol or protocol.startswith("GT")
        if not fixed:
            raise ValueError("GJ<r>_K<K> without a realization is a training augmentation, not a test layout")
        cache: dict = {}

        def fn(md, spatial, epoch, protocol=protocol):
            key = int(spatial[0])
            if key not in cache:
                cache[key] = VM.n3_mask(protocol, pde, "", 0, n=key)
            m, info = cache[key]
            return m.copy(), {"mask_id": f"v2:{protocol}", "mask_contract": VM.MASK_CONTRACT}
        ds.layout_fn = fn
        return
    if protocol in VM.MIXTURES:
        raise ValueError("mixtures are training protocols; evaluate a mixture-trained model on single layouts")
    VM.describe(protocol, pde)
    cache2: dict = {}

    def fn2(md, spatial, epoch, protocol=protocol):
        key = (str(md["parent_id"]), int(spatial[0]))
        if key not in cache2:
            cache2[key] = VM.protocol_mask(protocol, pde, key[0], rep=rep, n=key[1])
        return cache2[key], {"mask_id": f"v2:{protocol}:rep{rep}", "mask_parent_id": key[0], "mask_contract": VM.MASK_CONTRACT}
    ds.layout_fn = fn2


def input_setup(recipe: Mapping[str, Any], ds: V2Dataset) -> None:
    """The run's input representation: nearest-observation fill for Voronoi-CNN and for `fill_input: nearest`."""
    if recipe.get("param_channel"):
        raise NotImplementedError("the V4-E viscosity channel is not part of the public replay")
    if (recipe.get("input_representation") == "voronoi_mask_geometry" or recipe.get("fill_input") == "nearest"
            or (recipe.get("adapter") == "rev8k" and recipe.get("method") == "voronoi_cnn")):
        ds.fill = "euclid" if recipe["pde"] in STATIC else "torus"


__all__ = ["V2Dataset", "apply_layout", "campaign_and_registry", "experiment_config", "input_setup", "v2_config",
           "v2_layout", "v2_split", "v2_view"]
