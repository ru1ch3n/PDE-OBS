"""Shared, dependency-light helpers for the revision tools (CPU only, no model loading).

Everything here is deterministic: file digests, canonical JSON hashes, the frozen grid constants,
loaders for the declared result sources and small CSV/JSON writers.  No archive fallback exists
anywhere in this package: a missing declared input is an error.
"""
from __future__ import annotations

import csv
import datetime as _dt
import gzip
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

REPO = Path(__file__).resolve().parents[2]
PDES = ("darcy", "poisson", "helmholtz", "heat", "reaction_diffusion", "burgers", "navier_stokes")
STATIONARY_PDES = ("darcy", "poisson", "helmholtz")
METHODS = ("ufno_2d", "fno", "cno", "deeponet", "gnot", "transolver", "pino")
VIEWS = ("random_50pct", "random_65pct", "random_80pct", "block_observed_50pct", "line_sensors_50pct",
         "horizontal_lines_50pct", "vertical_lines_50pct", "boundary_band_50pct", "clustered_50pct")
SCORING_VERSION = "pdeobs-strict-v1"
TEST_IDENTITIES_PER_BLOCK = 200
TRAIN_IDENTITIES = 1800


class SourceError(ValueError):
    """A declared input is missing, inconsistent or forbidden."""


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    """Same canonical JSON digest as ``pdeobs.strict_score._canonical_hash`` (compact separators, no NaN)."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def ids_hash(ids: Iterable[str]) -> str:
    """Digest of the newline-joined sorted identities, as written in every split manifest."""
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def git_head(repo: Path = REPO) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=False).stdout.strip()
    return {"commit": run("rev-parse", "HEAD") or None, "dirty_paths": [p for p in run("status", "--short").splitlines() if p]}


def load_yaml(path: Path) -> Any:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SourceError(message)


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO / candidate


def load_sources(path: str | Path) -> dict[str, Any]:
    sources = load_yaml(resolve(path))
    require(sources.get("schema") == "pdeobs-revision-result-sources/v1", "unsupported result_sources schema")
    return sources


def verified_main_grid(sources: Mapping[str, Any]) -> dict[str, Any]:
    """Load the main-grid index and check every declared digest against the files on disk."""
    grid = sources["main_grid"]
    manifest_path = resolve(grid["manifest"])
    require(sha256_file(manifest_path) == grid["manifest_sha256"], "main-grid manifest digest differs from the declaration")
    manifest = load_json(manifest_path)
    require(manifest.get("schema") == sources["release_input_version"], "release input version differs from the manifest schema")
    root = resolve(grid["root"])
    digests: dict[str, str] = {}
    for name, entry in manifest["files"].items():
        actual = sha256_file(root / name)
        require(actual == entry["sha256"], f"digest mismatch for {name}")
        digests[name] = actual
    index = load_json(resolve(grid["index"]))
    records = index["records"]
    expected = {f"{p}/{m}/{v}" for p in PDES for m in METHODS for v in VIEWS}
    require(set(records) == expected and len(records) == 441, "the main-grid index is not the complete 441-identity grid")
    for identity, record in records.items():
        require(record["scoring_version"] == SCORING_VERSION, f"{identity}: scoring version is not {SCORING_VERSION}")
        require(record["evaluation_status"] == grid["required_evaluation_status"], f"{identity}: evaluation status not accepted")
        require(set(record["blocks"]) == set(VIEWS), f"{identity}: not the full nine-view block set")
        for view, block in record["blocks"].items():
            require(block["joint"]["n"] == TEST_IDENTITIES_PER_BLOCK, f"{identity}/{view}: not 200 identities")
    return {"index": index, "records": records, "manifest": manifest, "digests": digests}


def load_per_identity(sources: Mapping[str, Any], pde: str) -> dict[tuple[str, str], dict[str, float]]:
    """Per-record joint errors of the main grid: {(model, test_view): {identity: rel_l2_joint}}."""
    path = resolve(sources["main_grid"]["per_identity"].format(pde=pde))
    blocks: dict[tuple[str, str], dict[str, float]] = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            block = blocks.setdefault((row["model"], row["test_view"]), {})
            require(row["identity"] not in block, f"duplicate identity {row['identity']} in {row['model']}/{row['test_view']}")
            block[row["identity"]] = float(row["rel_l2_joint"])
    for key, block in blocks.items():
        require(len(block) == TEST_IDENTITIES_PER_BLOCK, f"{key}: per-identity block is not complete")
    return blocks


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="raise", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in fieldnames})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def geometric_mean(values: Sequence[float]) -> float:
    import math

    require(len(values) > 0 and all(v > 0 for v in values), "geometric mean needs positive values")
    return math.exp(sum(math.log(v) for v in values) / len(values))
