"""V2 data specification (rev8k-v2, plan sections 5.1-5.2): slices, parents, record ids, splits and seeds.

* 4 PDEs (Darcy, Poisson, Burgers, Navier-Stokes) x 3 GRF correlation lengths (0.22, 0.09, 0.025) = 12 slices.
* Per PDE 11,700 parents = 3 physical regimes x 3,900. Within a regime the ordinals 0-2999 are train, 3000-3299
  validation and 3300-3899 test (fixed before generation; parent level, so the three ell records of a parent always
  share a split).
* The three records of a parent share one white-noise draw: the generator seed is
  ``v2seed("parent", pde_key, regime, ordinal)`` WITHOUT the construction, and every paper GRF setting draws its noise
  first from the same RNG stream, so ell only changes the spectral filter (plan 5.2: same parent noise, one
  normalisation per record).
* The paper boundary, regime parameters, time coordinates and the 128^2 grid of each family are unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass

from .contracts import v2seed

PDES = ("darcy", "poisson", "burgers", "navier_stokes")
PDE_KEY = {"darcy": "darcy", "poisson": "poisson", "burgers": "burgers", "navier_stokes": "ns"}
FAMILY_OF = {v: k for k, v in PDE_KEY.items()}
BOUNDARY = {"darcy": "dirichlet", "poisson": "dirichlet", "burgers": "periodic", "navier_stokes": "periodic"}
STATIC = frozenset({"darcy", "poisson"})
ELLS = (("ell022", 0.22, "smooth_grf"), ("ell009", 0.09, "medium_grf"), ("ell0025", 0.025, "rough_grf"))
ELL_OF = {code: (ell, setting) for code, ell, setting in ELLS}
REGIMES = ("low", "medium", "high")
SPLITS = (("train", 0, 3000), ("validation", 3000, 3300), ("test", 3300, 3900))
PARENTS_PER_REGIME = 3900
# V4-H (D-0061): independent confirmation parents 3900-4099 per regime, outside every train / validation / test split,
# written to their own store (never mixed into the 11,700-parent pool); same seeds rule v2seed("parent", ...).
CONFIRM = ("confirm", 3900, 4100)
GRID = 128
STORED_FRAMES = 15
DENSE_T = {"burgers": 267, "navier_stokes": 71}        # paper corpus (numerics_full_t15.yaml) dense solver frames
FINAL_T = {"burgers": 0.35, "navier_stokes": 0.4}      # generator defaults used by the paper corpus
# resolution addon (V2-R1): extra native grids of the rough Poisson / Navier-Stokes slices
ADDON_GRIDS = {"poisson": (64, 256), "navier_stokes": (64, 256)}
ADDON_ELL = "ell0025"


def canonical_pde(name: str) -> str:
    name = str(name).strip().lower()
    if name in PDE_KEY:
        return name
    if name in FAMILY_OF:
        return FAMILY_OF[name]
    raise ValueError(f"unknown V2 PDE {name!r}; expected one of {PDES}")


def dataset_id(pde: str, ellcode: str) -> str:
    """Matches manifests/dataset_plan.csv of the V2 plan, e.g. v2_ns_ell0025_ref128."""
    if ellcode not in ELL_OF:
        raise ValueError(f"unknown ell code {ellcode!r}")
    return f"v2_{PDE_KEY[canonical_pde(pde)]}_{ellcode}_ref128"


def parse_dataset_id(ds: str) -> tuple[str, str]:
    parts = str(ds).split("_")
    if len(parts) != 4 or parts[0] != "v2" or parts[3] != "ref128":
        raise ValueError(f"not a V2 dataset id: {ds!r}")
    return canonical_pde(parts[1]), parts[2]


def parent_id(pde: str, regime: str, ordinal: int) -> str:
    return f"rev8k-v2/{PDE_KEY[canonical_pde(pde)]}/{regime}/{int(ordinal):05d}"


def record_id(pde: str, regime: str, ordinal: int, ellcode: str) -> str:
    return f"{parent_id(pde, regime, ordinal)}/{ellcode}"


def split_of(ordinal: int) -> str:
    for name, lo, hi in SPLITS + (CONFIRM,):
        if lo <= int(ordinal) < hi:
            return name
    raise ValueError(f"ordinal {ordinal} outside 0..{PARENTS_PER_REGIME - 1}")


def split_range(split: str) -> tuple[int, int]:
    for name, lo, hi in SPLITS + (CONFIRM,):
        if name == split:
            return lo, hi
    raise ValueError(f"unknown split {split!r}")


def parent_seed(pde: str, regime: str, ordinal: int) -> int:
    """Generator seed of a parent (shared by its three ell records)."""
    return v2seed("parent", PDE_KEY[canonical_pde(pde)], str(regime), int(ordinal)) % (2 ** 63)


def ordered_parents(name: str, pde: str, regime: str, ordinals) -> list[int]:
    """Ordinals ordered by v2seed(name, pde_key, regime, parent_id) (ties by ordinal)."""
    key = PDE_KEY[canonical_pde(pde)]
    return sorted((int(o) for o in ordinals), key=lambda o: (v2seed(name, key, str(regime), parent_id(pde, regime, o)), o))


def nested_train(pde: str, regime: str, n_per_regime: int) -> list[int]:
    """Plan 5.2: nested training subsets N = 1800 / 5400 / 9000 (600 / 1800 / 3000 per regime) by a per-regime hash
    order of the training parents; validation and test never take part in the selection."""
    lo, hi = split_range("train")
    if not 0 < int(n_per_regime) <= hi - lo:
        raise ValueError("n_per_regime must be in 1..3000")
    return ordered_parents("nested_train", pde, regime, range(lo, hi))[: int(n_per_regime)]


def pilot_parents(pde: str, regime: str, k: int = 10) -> list[int]:
    """Plan 5.3 pilot: k parents per regime from train + validation (never test), by a fixed hash order."""
    return ordered_parents("pilot_case", pde, regime, range(0, split_range("validation")[1]))[: int(k)]


@dataclass(frozen=True)
class Slice:
    pde: str
    ellcode: str

    @property
    def dataset_id(self) -> str:
        return dataset_id(self.pde, self.ellcode)

    @property
    def ell(self) -> float:
        return ELL_OF[self.ellcode][0]

    @property
    def setting(self) -> str:
        return ELL_OF[self.ellcode][1]

    @property
    def boundary(self) -> str:
        return BOUNDARY[self.pde]


def slices(pdes=PDES) -> list[Slice]:
    return [Slice(canonical_pde(p), code) for p in pdes for code, _, _ in ELLS]


def plan_counts() -> dict:
    """Arithmetic of the V2 data plan (for receipts and the plan command)."""
    per_slice = {name: (hi - lo) * len(REGIMES) for name, lo, hi in SPLITS}
    return {"slices": len(PDES) * len(ELLS), "records_per_slice": per_slice, "records_per_slice_total": sum(per_slice.values()),
            "parents_per_pde": PARENTS_PER_REGIME * len(REGIMES), "parents_total": PARENTS_PER_REGIME * len(REGIMES) * len(PDES),
            "records_total": PARENTS_PER_REGIME * len(REGIMES) * len(PDES) * len(ELLS),
            "addon_views": sum(len(g) for g in ADDON_GRIDS.values()) * PARENTS_PER_REGIME * len(REGIMES)}


__all__ = ["ADDON_ELL", "ADDON_GRIDS", "BOUNDARY", "CONFIRM", "DENSE_T", "ELLS", "ELL_OF", "FINAL_T", "GRID", "PARENTS_PER_REGIME",
           "PDES", "PDE_KEY", "REGIMES", "SPLITS", "STATIC", "STORED_FRAMES", "Slice", "canonical_pde", "dataset_id",
           "nested_train", "ordered_parents", "parent_id", "parent_seed", "parse_dataset_id", "pilot_parents", "plan_counts",
           "record_id", "slices", "split_of", "split_range"]
