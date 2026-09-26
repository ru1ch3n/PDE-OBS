"""Evidence bundles: link a paper table row to the artifacts that produced it, and check the links.

A bundle is a small JSON manifest (``pdeobs-evidence-bundle/v1``) that names, for one experiment,
the resolved configuration, the identity manifest, the split receipt, the final checkpoint, and the
per-view strict evaluation blocks (contract, prediction artifact, score report).  The tool has three
verbs:

``plan``    read a paper-row output directory and write the manifest; existence checks only.
``check``   verify the manifest structure, hash every present artifact, cross-check the metadata
            (PDE, method, views, task, seeds, cohort, epochs) across manifest, receipt, resolved
            configuration, completion record and contracts, recompute the configuration identities
            with the writer's own rules, verify checkpoint and manifest bindings, re-score the
            prediction artifacts with the frozen strict scorer, and check the split receipt against
            the declared cohort.  Read-only.
``export``  copy the small JSON artifacts and a file index (with SHA-256) into a NEW directory.
            Large binaries are referenced by hash unless ``--copy-large`` is given.  The export can
            be re-checked directly (``check --bundle <export>/evidence_bundle.json``).

Every check reports its own status.  A passed check never implies another: matching hashes prove
file identity, not scientific correctness; a re-scored metric proves the stored number follows
from the stored arrays, not that the arrays came from the claimed training.  Numerical-reference
evidence and external reproduction are recorded as separate, explicitly unverified categories.

Nothing here trains, generates data, downloads, or launches a job.  The tool never reads
``--root`` files outside the bundle's directory tree and never overwrites an existing output.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping

BUNDLE_VERSION = "pdeobs-evidence-bundle/v1"
CHECK_VERSION = "pdeobs-evidence-check/v1"
ERROR_VERSION = "pdeobs-evidence-error/v1"

# schema strings owned by the frozen modules; imported by value so a mismatch is visible here
ROW_VERSION = "pdeobs-paper-row-v1"
COMPLETION_VERSION = "pdeobs-paper-row-training-completion-v1"
MANIFEST_VERSION = "pdeobs-identity-manifest-v1"
CONTRACT_VERSION = "pdeobs-strict-contract-v1"
SCORING_VERSION = "pdeobs-strict-v1"
ARTIFACT_VERSION = "pdeobs-strict-inference-v1"
SPLIT_VERSIONS = {"original500": "pdeobs.one-setting-split/v1", "demo": "pdeobs-paper-row-demo-split-v1"}
SPLIT_ALGORITHMS = {"original500": "sha256_rank_within_regime_largest_remainder_exact_ten_percent_test",
                    "demo": "sha256_rank_one_test_per_regime_demo_only"}
PURPOSES = ("paper-evidence", "software-demo")
CHECKS = ("manifest_structure", "artifact_completeness", "metadata_consistency", "configuration_identity",
          "identity_shape", "finiteness", "metric_recomputation", "split_provenance", "checkpoint_binding",
          "purpose_guard", "numerical_reference_evidence", "external_reproduction")
CORE_CHECKS = CHECKS[:10]
REQUIRED_TOP = ("schema_version", "experiment_id", "purpose", "task", "pde", "method", "training_view",
                "training_protocol", "test_protocol", "seeds", "dataset", "artifacts", "scorer_version")
REQUIRED_ARTIFACTS = ("receipt", "resolved_config", "training_config", "training_completion", "checkpoint")
REQUIRED_BLOCK = ("test_view", "contract", "predictions", "score")
# trainer-file keys that legitimately differ from the resolved training section (paths, placement)
TRAINER_KEYS_NOT_COMPARED = {"checkpoint_dir", "resume_from", "device", "num_workers", "pin_memory", "persistent_workers"}
LARGE_SUFFIXES = {".pt", ".h5", ".hdf5", ".npz"}
EXIT_COMPLETE, EXIT_FAILED, EXIT_INCOMPLETE = 0, 2, 3


class EvidenceError(ValueError):
    def __init__(self, category: str, reason: str, next_step: str):
        super().__init__(reason)
        self.category, self.reason, self.next_step = category, reason, next_step


# ----------------------------------------------------------------------------- helpers

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def recipe_hash(experiment: Mapping[str, Any]) -> str:
    """The writer's training-recipe identity: the resolved experiment minus seed, data root and placement keys."""
    recipe = copy.deepcopy(dict(experiment))
    recipe.pop("seed", None)
    if isinstance(recipe.get("data"), dict):
        recipe["data"].pop("root", None)
    if isinstance(recipe.get("training"), dict):
        for key in ("device", "num_workers", "pin_memory", "persistent_workers"):
            recipe["training"].pop(key, None)
    return canonical_hash(recipe)


def _read_json(path: Path, what: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise EvidenceError("unreadable_artifact", f"{what} at {path} is not readable JSON: {exc}",
                            "regenerate or re-copy the file; do not hand-edit JSON artifacts") from exc


def _contained(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise EvidenceError("bad_path", f"artifact path {relative!r} must be a non-empty path relative to the bundle root",
                            "write paths relative to --root (or the manifest directory)")
    candidate = (root / relative).resolve()
    if root.resolve() not in candidate.parents and candidate != root.resolve():
        raise EvidenceError("bad_path", f"artifact path {relative!r} escapes the bundle root {root}",
                            "keep every artifact inside the bundle root; symlinks and '..' are not followed")
    return candidate


def _status(failed: bool, blocked: bool) -> str:
    if failed:
        return "failed"
    if blocked:
        return "blocked"
    return "passed"


def _close(a: Any, b: Any, rel: float = 1.0e-9) -> bool:
    try:
        return math.isclose(float(a), float(b), rel_tol=rel, abs_tol=0.0)
    except (TypeError, ValueError):
        return False


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


# ----------------------------------------------------------------------------- plan

def plan_from_row_dir(root: str | Path, *, purpose: str | None = None, paper_reference: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Build a manifest from a ``paper-row`` output directory.  Existence checks only; no hashing."""
    root = Path(root).resolve()
    receipt_path = root / "receipt.json"
    if not receipt_path.is_file():
        raise EvidenceError("not_a_row_directory", f"{root} has no receipt.json",
                            "point --root at a directory written by `pdeobs paper-row`, or write the manifest by hand")
    receipt = _read_json(receipt_path, "paper-row receipt")
    if receipt.get("schema_version") != ROW_VERSION:
        raise EvidenceError("unsupported_receipt", f"receipt schema {receipt.get('schema_version')!r} is not {ROW_VERSION}",
                            "only paper-row receipts are auto-planned; other layouts need a hand-written manifest")
    cohort = receipt.get("cohort")
    derived_purpose = "paper-evidence" if cohort == "original500" else "software-demo"
    if purpose is None:
        purpose = derived_purpose
    blocks = []
    for entry in receipt.get("evaluation_blocks") or []:
        view = entry.get("test_view")
        base = f"evaluation/{view}"
        blocks.append({"test_view": view, "contract": f"{base}/contract.json",
                       "predictions": f"{base}/predictions.h5", "score": f"{base}/score.json"})
    manifest: dict[str, Any] = {
        "schema_version": BUNDLE_VERSION,
        "experiment_id": "/".join(str(receipt.get(k, "?")) for k in ("run_id", "attempt_id")),
        "purpose": purpose,
        "task": receipt.get("task"), "pde": receipt.get("pde"), "method": receipt.get("method"),
        "model_adaptation": f"{receipt.get('method')} as registered in the source-faithful method registry (adaptation, not upstream reproduction)",
        "training_view": (receipt.get("train_view") or (receipt.get("row") or {}).get("train_view")),
        "training_protocol": {"cohort": cohort, "epochs": receipt.get("epochs"), "actual_epochs": receipt.get("actual_epochs"),
                              "checkpoint_selection": receipt.get("checkpoint_selection"), "stop_reason": receipt.get("stop_reason"),
                              "registry_sha256": receipt.get("registry_sha256"), "campaign_sha256": receipt.get("campaign_sha256")},
        "test_protocol": {"split": "paper-test" if cohort == "original500" else "demo",
                          "expected_identity_count": 200 if cohort == "original500" else None,
                          "test_views": [b["test_view"] for b in blocks]},
        "seeds": {"split_seed": receipt.get("split_seed"), "training_seed": receipt.get("training_seed")},
        "dataset": {"version": receipt.get("dataset_manifest_sha256"), "identity_manifest": "identity_manifest.json",
                    "split_manifest": "split.json"},
        "artifacts": {"receipt": "receipt.json", "resolved_config": "row_resolved.json",
                      "training_config": "checkpoints/training_config.json",
                      "training_completion": "training_completion.json", "checkpoint": "checkpoints/last.pt",
                      "blocks": blocks},
        "scorer_version": SCORING_VERSION,
        "metric": {"name": "rel_l2_joint_mean", "definition": "per-identity L2(pred-target)/max(L2(target),1e-12), "
                   "arithmetic mean over every expected identity; rollout norms cover all target frames jointly",
                   "unit": "relative L2 ratio (dimensionless, not a percentage)", "precision": "float64"},
        "paper_reference": dict(paper_reference or {}),
        "notes": ["Generated by `python -m pdeobs.evidence plan`; edit paper_reference to point at the table/row/cell.",
                  "purpose is derived from the receipt cohort; a demo cohort can never be paper evidence."],
    }
    present, missing = [], []
    for rel in _all_paths(manifest):
        (present if (root / rel).is_file() else missing).append(rel)
    return {"manifest": manifest, "root": str(root), "present": present, "missing": missing,
            "derived_purpose": derived_purpose}


def _all_paths(manifest: Mapping[str, Any]) -> list[str]:
    paths: list[str] = []
    ds = manifest.get("dataset") or {}
    if isinstance(ds, dict):
        for key in ("identity_manifest", "split_manifest"):
            if ds.get(key):
                paths.append(ds[key])
    art = manifest.get("artifacts") or {}
    if isinstance(art, dict):
        for key in REQUIRED_ARTIFACTS:
            if art.get(key):
                paths.append(art[key])
        for block in art.get("blocks") or []:
            if isinstance(block, dict):
                for key in ("contract", "predictions", "score"):
                    if block.get(key):
                        paths.append(block[key])
    ref = manifest.get("numerical_reference") or {}
    if isinstance(ref, dict) and ref.get("report"):
        paths.append(ref["report"])
    return paths


# ----------------------------------------------------------------------------- check

def load_manifest(path: str | Path) -> dict[str, Any]:
    """Load a bundle manifest, or the envelope that ``export`` writes (``evidence_bundle.json``)."""
    payload = _read_json(Path(path), "evidence bundle manifest")
    if isinstance(payload, dict) and isinstance(payload.get("manifest"), dict) and "export" in payload and "artifacts" not in payload:
        payload = payload["manifest"]
    if not isinstance(payload, dict) or payload.get("schema_version") != BUNDLE_VERSION:
        raise EvidenceError("unsupported_manifest", f"bundle schema {payload.get('schema_version') if isinstance(payload, dict) else None!r} is not {BUNDLE_VERSION}",
                            "generate the manifest with `python -m pdeobs.evidence plan --root <row dir>`")
    if payload.get("purpose") not in PURPOSES:
        raise EvidenceError("bad_purpose", f"purpose must be one of {PURPOSES}", "set purpose explicitly; demo cohorts are software-demo")
    if not isinstance(payload.get("artifacts"), dict):
        raise EvidenceError("bad_manifest", "manifest lacks an 'artifacts' object", "run plan again or add the artifacts section")
    return payload


def _structure_problems(manifest: Mapping[str, Any]) -> list[str]:
    problems = []
    for key in REQUIRED_TOP:
        if manifest.get(key) in (None, "", {}, []):
            problems.append(f"missing manifest field {key!r}")
    ds = manifest.get("dataset") or {}
    for key in ("identity_manifest", "split_manifest"):
        if not isinstance(ds, dict) or not ds.get(key):
            problems.append(f"missing dataset.{key}")
    art = manifest.get("artifacts") or {}
    for key in REQUIRED_ARTIFACTS:
        if not isinstance(art, dict) or not art.get(key):
            problems.append(f"missing artifacts.{key}")
    blocks = art.get("blocks") if isinstance(art, dict) else None
    if not isinstance(blocks, list) or not blocks:
        problems.append("no evaluation blocks are declared; nothing links this row to a score")
    else:
        views = []
        for i, block in enumerate(blocks):
            if not isinstance(block, dict):
                problems.append(f"block {i} is not an object"); continue
            for key in REQUIRED_BLOCK:
                if not block.get(key):
                    problems.append(f"block {i} lacks {key!r}")
            views.append(block.get("test_view"))
        if len(set(views)) != len(views):
            problems.append(f"duplicate test views declared: {views}")
        declared = (manifest.get("test_protocol") or {}).get("test_views") if isinstance(manifest.get("test_protocol"), dict) else None
        if isinstance(declared, list) and set(declared) != set(views):
            problems.append(f"test_protocol.test_views {sorted(map(str, declared))} differ from the declared blocks {sorted(map(str, views))}")
    for key in ("training_protocol", "test_protocol", "seeds"):
        if not isinstance(manifest.get(key), dict):
            problems.append(f"manifest field {key!r} must be an object")
    return problems


def check_bundle(manifest: Mapping[str, Any], root: str | Path, *, rescore: bool = True) -> dict[str, Any]:
    """Run every check category against the artifacts under ``root``.  Read-only."""
    root = Path(root).resolve()
    checks: dict[str, dict[str, Any]] = {name: {"status": "not_attempted", "details": []} for name in CHECKS}
    files: dict[str, dict[str, Any]] = {}

    def fail(name: str, message: str) -> None:
        checks[name]["details"].append(message); checks[name]["status"] = "failed"

    def block(name: str, message: str) -> None:
        checks[name]["details"].append(message)
        if checks[name]["status"] != "failed":
            checks[name]["status"] = "blocked"

    def note(name: str, message: str) -> None:
        checks[name]["details"].append(message)
        if checks[name]["status"] == "not_attempted":
            checks[name]["status"] = "passed"

    # 0. manifest structure -------------------------------------------------------------------
    problems = _structure_problems(manifest)
    if problems:
        for p in problems:
            fail("manifest_structure", p)
    else:
        note("manifest_structure", "every required field, artifact and block declaration is present")

    # 1. artifact completeness + hashes -------------------------------------------------------
    missing, present = [], []
    for rel in _all_paths(manifest):
        path = _contained(root, rel)
        if path.is_file():
            files[rel] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            present.append(rel)
        else:
            files[rel] = {"sha256": None, "bytes": None, "missing": True}
            missing.append(rel)
    note("artifact_completeness", f"{len(present)} declared artifact(s) present, {len(missing)} missing")
    for rel in missing:
        block("artifact_completeness", f"missing: {rel}")
    art = manifest.get("artifacts") if isinstance(manifest.get("artifacts"), dict) else {}
    blocks = [b for b in (art.get("blocks") or []) if isinstance(b, dict)]

    def loaded(rel: str | None, what: str, version: str | None = None, key: str = "schema_version") -> Any:
        if not rel or rel in missing or rel not in files:
            return None
        payload = _read_json(_contained(root, rel), what)
        if version is not None and isinstance(payload, dict) and payload.get(key) != version:
            fail("artifact_completeness", f"{what} {rel}: schema {payload.get(key)!r} is not {version}")
        return payload

    receipt = loaded(art.get("receipt"), "receipt", ROW_VERSION)
    completion = loaded(art.get("training_completion"), "training completion", COMPLETION_VERSION)
    resolved = loaded(art.get("resolved_config"), "resolved configuration")
    trainer = loaded(art.get("training_config"), "trainer configuration")
    identity_manifest = loaded((manifest.get("dataset") or {}).get("identity_manifest"), "identity manifest", MANIFEST_VERSION)
    split = loaded((manifest.get("dataset") or {}).get("split_manifest"), "split receipt")
    tp = manifest.get("training_protocol") if isinstance(manifest.get("training_protocol"), dict) else {}
    seeds = manifest.get("seeds") if isinstance(manifest.get("seeds"), dict) else {}
    cohort = tp.get("cohort") or (receipt or {}).get("cohort")
    experiment = (resolved or {}).get("experiment") if isinstance(resolved, dict) else None
    row_cfg = (resolved or {}).get("row") if isinstance(resolved, dict) else None
    contracts: dict[str, Any] = {}
    scores: dict[str, Any] = {}
    for b in blocks:
        view = str(b.get("test_view"))
        contracts[view] = loaded(b.get("contract"), f"contract[{view}]", CONTRACT_VERSION)
        scores[view] = loaded(b.get("score"), f"score[{view}]", SCORING_VERSION, key="scoring_version")

    # 2. metadata consistency across manifest / receipt / resolved / completion / contracts ----
    def agree(name: str, label: str, values: dict[str, Any]) -> None:
        known = {k: v for k, v in values.items() if v is not None}
        if len(known) < 2:
            block("metadata_consistency", f"{label}: recorded in fewer than two places ({sorted(known)})")
        elif len({json.dumps(v, sort_keys=True) for v in known.values()}) != 1:
            fail("metadata_consistency", f"{label} disagree: {known}")

    agree("pde", "pde", {"manifest": manifest.get("pde"), "receipt": (receipt or {}).get("pde"), "resolved.row": (row_cfg or {}).get("pde"),
                         **{f"contract[{v}]": (c or {}).get("pde") for v, c in contracts.items()}})
    agree("method", "method", {"manifest": manifest.get("method"), "receipt": (receipt or {}).get("method"), "resolved.row": (row_cfg or {}).get("method"),
                               "resolved.experiment.method.name": ((experiment or {}).get("method") or {}).get("name") if isinstance(experiment, dict) else None,
                               **{f"contract[{v}]": (c or {}).get("method") for v, c in contracts.items()}})
    agree("training_view", "training view", {"manifest": manifest.get("training_view"), "receipt": (receipt or {}).get("train_view"), "resolved.row": (row_cfg or {}).get("train_view"),
                                             **{f"contract[{v}]": (c or {}).get("train_view") for v, c in contracts.items()}})
    agree("task", "task", {"manifest": manifest.get("task"), "receipt": (receipt or {}).get("task"), "resolved.experiment": (experiment or {}).get("task") if isinstance(experiment, dict) else None,
                           **{f"contract[{v}]": (c or {}).get("task") for v, c in contracts.items()}})
    agree("cohort", "training cohort", {"manifest": tp.get("cohort"), "receipt": (receipt or {}).get("cohort"), "completion": (completion or {}).get("cohort"),
                                        "resolved.row": (row_cfg or {}).get("cohort"), **{f"contract[{v}]": (c or {}).get("training_cohort") for v, c in contracts.items()}})
    agree("actual_epochs", "actual epochs", {"manifest": tp.get("actual_epochs"), "receipt": (receipt or {}).get("actual_epochs"), "completion": (completion or {}).get("actual_epochs"),
                                             **{f"contract[{v}]": (c or {}).get("actual_epochs") for v, c in contracts.items()}})
    agree("training_seed", "training seed", {"manifest": seeds.get("training_seed"), "receipt": (receipt or {}).get("training_seed"), "resolved.row": (row_cfg or {}).get("training_seed"),
                                             "resolved.experiment.seed": (experiment or {}).get("seed") if isinstance(experiment, dict) else None,
                                             **{f"contract[{v}]": (c or {}).get("training_seed") for v, c in contracts.items()}})
    agree("split_seed", "split seed", {"manifest": seeds.get("split_seed"), "receipt": (receipt or {}).get("split_seed"), "split": (split or {}).get("seed") if isinstance(split, dict) else None})
    agree("run_id", "run id", {"receipt": (receipt or {}).get("run_id"), "resolved.row": (row_cfg or {}).get("run_id"), **{f"contract[{v}]": (c or {}).get("run_id") for v, c in contracts.items()}})
    agree("attempt_id", "attempt id", {"receipt": (receipt or {}).get("attempt_id"), "resolved.row": (row_cfg or {}).get("attempt_id"), **{f"contract[{v}]": (c or {}).get("attempt_id") for v, c in contracts.items()}})
    if receipt is not None and manifest.get("experiment_id") and manifest.get("experiment_id") != f"{receipt.get('run_id')}/{receipt.get('attempt_id')}":
        fail("metadata_consistency", f"experiment_id {manifest.get('experiment_id')!r} is not receipt run_id/attempt_id")
    for view, c in contracts.items():
        if c is None:
            block("metadata_consistency", f"{view}: contract missing; view labels not cross-checked"); continue
        if c.get("test_view") != view or c.get("observation_id") != view:
            fail("metadata_consistency", f"block labelled {view!r} points at a contract for test_view={c.get('test_view')!r} observation_id={c.get('observation_id')!r}")
    if receipt is not None:
        inventory = {e.get("test_view"): e for e in (receipt.get("evaluation_blocks") or []) if isinstance(e, dict)}
        if set(inventory) != set(contracts):
            fail("metadata_consistency", f"receipt evaluation_blocks {sorted(map(str, inventory))} differ from the declared blocks {sorted(contracts)}")
        for view, entry in inventory.items():
            rel = next((b.get("score") for b in blocks if str(b.get("test_view")) == str(view)), None)
            actual = (files.get(rel) or {}).get("sha256") if rel else None
            if entry.get("score_sha256") and actual and entry["score_sha256"] != actual:
                fail("metadata_consistency", f"{view}: score.json hash differs from the receipt's score_sha256")
            elif entry.get("score_sha256") and actual:
                pass
            if entry.get("status") and scores.get(view) and entry.get("status") != scores[view].get("status"):
                fail("metadata_consistency", f"{view}: receipt block status {entry.get('status')!r} differs from score.json {scores[view].get('status')!r}")
    if receipt is not None and completion is not None:
        for key in ("training_records", "validation_records"):
            if receipt.get(key) is not None and completion.get(key) is not None and receipt.get(key) != completion.get(key):
                fail("metadata_consistency", f"{key} differs between receipt and completion")
    if checks["metadata_consistency"]["status"] == "not_attempted":
        note("metadata_consistency", "pde, method, training view, task, cohort, epochs, seeds, run/attempt ids and the block inventory agree across manifest, receipt, resolved configuration, completion record and contracts; every score.json hashes to the receipt's score_sha256")

    # 3. configuration identity: recompute the writer's hashes from the resolved configuration --
    if not isinstance(experiment, dict):
        block("configuration_identity", "resolved configuration (row_resolved.json with an 'experiment' object) missing; training_config_sha256 cannot be recomputed")
    else:
        try:
            config_hash, rec_hash = canonical_hash(experiment), recipe_hash(experiment)
        except (TypeError, ValueError) as exc:
            fail("configuration_identity", f"resolved configuration is not canonically hashable: {exc}")
            config_hash = rec_hash = None
        if config_hash:
            claims = {"receipt": (receipt or {}).get("training_config_sha256"), "completion": (completion or {}).get("training_config_sha256"),
                      **{f"contract[{v}]": (c or {}).get("training_config_sha256") for v, c in contracts.items()}}
            recipes = {"receipt": (receipt or {}).get("training_recipe_sha256"), "completion": (completion or {}).get("training_recipe_sha256"),
                       **{f"contract[{v}]": (c or {}).get("training_recipe_sha256") for v, c in contracts.items()}}
            for label, table, value in (("training_config_sha256", claims, config_hash), ("training_recipe_sha256", recipes, rec_hash)):
                known = {k: v for k, v in table.items() if v}
                wrong = sorted(k for k, v in known.items() if v != value)
                if not known:
                    block("configuration_identity", f"no record carries {label}")
                elif wrong:
                    fail("configuration_identity", f"{label} recomputed from row_resolved.json ({value[:12]}...) differs from {wrong}")
            if isinstance(trainer, dict) and isinstance(experiment.get("training"), dict):
                section = experiment["training"]
                diffs = [k for k in sorted(set(trainer) & set(section) - TRAINER_KEYS_NOT_COMPARED) if trainer.get(k) != section.get(k)]
                if diffs:
                    fail("configuration_identity", f"checkpoints/training_config.json differs from the resolved training section in {diffs}")
                if trainer.get("seed") is not None and experiment.get("seed") is not None and trainer.get("seed") != experiment.get("seed"):
                    fail("configuration_identity", "checkpoints/training_config.json seed differs from the resolved experiment seed")
                if trainer.get("task") is not None and experiment.get("task") is not None and trainer.get("task") != experiment.get("task"):
                    fail("configuration_identity", "checkpoints/training_config.json task differs from the resolved experiment task")
            elif trainer is None:
                block("configuration_identity", "checkpoints/training_config.json missing; trainer settings not cross-checked")
            if checks["configuration_identity"]["status"] == "not_attempted":
                note("configuration_identity", f"training_config_sha256 {config_hash[:12]}... and training_recipe_sha256 {rec_hash[:12]}... recomputed from row_resolved.json with the writer's rules match every record; the trainer file agrees with the resolved training section on every shared key")

    # 4. identities / shapes, 5. finiteness, 6. metric recomputation -------------------------
    expected_count = (manifest.get("test_protocol") or {}).get("expected_identity_count") if isinstance(manifest.get("test_protocol"), dict) else None
    contract_ids: dict[str, list[str]] = {}
    if not blocks:
        block("identity_shape", "no blocks declared"); block("finiteness", "no blocks declared"); block("metric_recomputation", "no blocks declared")
    for b in blocks:
        view = str(b.get("test_view"))
        contract, score = contracts.get(view), scores.get(view)
        if contract is None or score is None:
            block("identity_shape", f"{view}: contract or score report missing; identity checks blocked")
            block("finiteness", f"{view}: contract or score report missing"); block("metric_recomputation", f"{view}: contract or score report missing")
            continue
        ids = list(contract.get("expected_ids") or [])
        contract_ids[view] = ids
        if not ids:
            fail("identity_shape", f"{view}: contract lists no expected identities")
        if len(ids) != len(set(ids)):
            fail("identity_shape", f"{view}: duplicate expected_ids in contract")
        if expected_count is not None and len(ids) != expected_count:
            fail("identity_shape", f"{view}: contract has {len(ids)} identities, bundle expects {expected_count}")
        if contract.get("split") == "paper-test" and len(ids) != 200:
            fail("identity_shape", f"{view}: paper-test contract without exactly 200 identities")
        if score.get("status") != "valid":
            fail("identity_shape", f"{view}: stored score status is {score.get('status')!r}: {score.get('errors')}")
        for key in ("expected_identity_count", "scored_identity_count", "actual_prediction_identity_count", "actual_target_identity_count"):
            if score.get(key) != len(ids):
                fail("identity_shape", f"{view}: score.{key}={score.get(key)} but the contract lists {len(ids)} identities")
        per = score.get("per_identity") or []
        if len(per) != len(ids) or {p.get("identity") for p in per} != set(ids):
            fail("identity_shape", f"{view}: per-identity results do not cover exactly the contract identities")
        pred_rel = b.get("predictions")
        if not pred_rel or pred_rel in missing:
            block("identity_shape", f"{view}: prediction artifact missing; stored shape and identity count not verified")
            block("finiteness", f"{view}: prediction artifact missing; finiteness not verified")
            block("metric_recomputation", f"{view}: prediction artifact missing; cannot re-score")
            continue
        pred_path = _contained(root, pred_rel)
        try:
            import h5py
            with h5py.File(pred_path, "r") as handle:
                shape = list(handle["prediction"].shape)
                n_ids = int(handle["prediction_ids"].shape[0])
                version = handle.attrs.get("artifact_version")
        except (OSError, KeyError, ImportError, ValueError) as exc:
            fail("identity_shape", f"{view}: prediction artifact unreadable ({exc})")
            fail("finiteness", f"{view}: prediction artifact unreadable; finiteness could not be checked")
            fail("metric_recomputation", f"{view}: prediction artifact unreadable; re-scoring impossible")
            continue
        if version != ARTIFACT_VERSION:
            fail("identity_shape", f"{view}: prediction artifact version {version!r} is not {ARTIFACT_VERSION}")
        if shape[1:] != list(contract.get("expected_shape") or []):
            fail("identity_shape", f"{view}: stored prediction shape {shape[1:]} differs from contract {contract.get('expected_shape')}")
        if n_ids != len(ids) or shape[0] != len(ids):
            fail("identity_shape", f"{view}: artifact holds {n_ids} identities / {shape[0]} rows, contract lists {len(ids)}")
        if checks["identity_shape"]["status"] == "not_attempted" or not any(view in d for d in checks["identity_shape"]["details"]):
            note("identity_shape", f"{view}: {len(ids)} identities, stored shape {shape[1:]} as contracted, artifact version {ARTIFACT_VERSION}")
        if not rescore:
            block("finiteness", f"{view}: re-scoring skipped by request; finiteness of the stored arrays NOT verified")
            block("metric_recomputation", f"{view}: re-scoring skipped by request (--no-rescore)")
            continue
        from .strict_score import score_prediction_file
        fresh = score_prediction_file(pred_path, contract)
        if fresh.get("status") != "valid":
            fail("finiteness", f"{view}: strict re-validation of the stored arrays failed: {fresh.get('errors')}")
            fail("metric_recomputation", f"{view}: re-scoring the stored arrays is {fresh.get('status')!r}: {fresh.get('errors')}")
            continue
        note("finiteness", f"{view}: every stored prediction and target array is finite under {SCORING_VERSION}")
        stored_mean = (score.get("summary") or {}).get("rel_l2_joint_mean")
        fresh_mean = (fresh.get("summary") or {}).get("rel_l2_joint_mean")
        ok = True
        if not _finite_number(stored_mean) or not _close(stored_mean, fresh_mean):
            fail("metric_recomputation", f"{view}: stored rel_l2_joint_mean={stored_mean} but re-scoring gives {fresh_mean}"); ok = False
        stored_per = {p.get("identity"): p.get("rel_l2_joint") for p in per}
        fresh_per = {p.get("identity"): p.get("rel_l2_joint") for p in fresh.get("per_identity") or []}
        mismatched = [i for i in stored_per if not _finite_number(stored_per[i]) or not _close(stored_per[i], fresh_per.get(i))]
        if mismatched:
            fail("metric_recomputation", f"{view}: {len(mismatched)} per-identity value(s) differ from re-scoring, e.g. {mismatched[:3]}"); ok = False
        if fresh.get("artifact_contract_binding") != "pdeobs-strict-inference-v1_sha256_bound":
            fail("metric_recomputation", f"{view}: artifact is not SHA-256 bound to its contract ({fresh.get('artifact_contract_binding')})"); ok = False
        if ok:
            note("metric_recomputation", f"{view}: re-scored {len(ids)} identities with {SCORING_VERSION}; mean and per-identity values reproduce the stored report")

    # 7. split provenance ---------------------------------------------------------------------
    if split is None:
        block("split_provenance", "split receipt missing")
    elif cohort not in SPLIT_VERSIONS:
        fail("split_provenance", f"cohort {cohort!r} has no executable split rule in this release (fixed200/min120/recovered are metadata only)")
    else:
        if split.get("schema_version") != SPLIT_VERSIONS[cohort]:
            fail("split_provenance", f"split schema {split.get('schema_version')!r} is not {SPLIT_VERSIONS[cohort]} for cohort {cohort}")
        if split.get("algorithm") != SPLIT_ALGORITHMS[cohort]:
            fail("split_provenance", f"split algorithm {split.get('algorithm')!r} is not the {cohort} algorithm")
        seed = seeds.get("split_seed")
        if seed is not None and split.get("seed") != seed:
            fail("split_provenance", f"split seed {split.get('seed')} differs from the bundle's split_seed {seed}")
        ident_str = split.get("macrodomain_identity")
        pde = manifest.get("pde")
        if not isinstance(ident_str, str) or ident_str.count("|") != 2 or (pde and not ident_str.startswith(f"{pde}|")):
            fail("split_provenance", f"macrodomain_identity {ident_str!r} is not 'pde|boundary|setting' for pde {pde!r}")
        train_ids, test_ids = split.get("train_ids") or [], split.get("test_ids") or []
        if set(train_ids) & set(test_ids):
            fail("split_provenance", "train and test identities overlap")
        if receipt is not None and receipt.get("training_records") is not None and receipt.get("training_records") != len(train_ids):
            fail("split_provenance", f"receipt training_records {receipt.get('training_records')} differ from the split's {len(train_ids)} train identities")
        if receipt is not None and receipt.get("test_records") is not None and receipt.get("test_records") != len(test_ids):
            fail("split_provenance", f"receipt test_records {receipt.get('test_records')} differ from the split's {len(test_ids)} test identities")
        if cohort == "original500":
            counts = split.get("regime_counts") or {}
            tests = sorted(int(v.get("test", -1)) for v in counts.values()) if isinstance(counts, dict) else []
            trains = sorted(int(v.get("train", -1)) for v in counts.values()) if isinstance(counts, dict) else []
            if split.get("records") != 2000 or len(train_ids) != 1800 or len(test_ids) != 200:
                fail("split_provenance", f"original500 needs 2000 records = 1800 train + 200 test, got {split.get('records')}/{len(train_ids)}/{len(test_ids)}")
            if trains != [600, 600, 600] or tests != [66, 67, 67]:
                fail("split_provenance", f"original500 regime allocation must be train 600/600/600 and test 67/67/66, got train {trains} test {tests}")
        if identity_manifest is not None:
            universe = set(identity_manifest.get("expected_ids") or [])
            if not set(train_ids) <= universe or not set(test_ids) <= universe:
                fail("split_provenance", "split identities are not all listed in the identity manifest")
        else:
            block("split_provenance", "identity manifest missing; split identities not checked against it")
        for view, ids in contract_ids.items():
            if list(ids) != list(test_ids):
                fail("split_provenance", f"{view}: contract expected_ids differ from split test_ids (set or order)")
        if checks["split_provenance"]["status"] == "not_attempted":
            note("split_provenance", f"{cohort}: {len(train_ids)} train / {len(test_ids)} test identities, disjoint, seed {split.get('seed')}, "
                 f"macrodomain {ident_str}; every block contract lists exactly the split's test identities in order")

    # 8. checkpoint + manifest bindings --------------------------------------------------------
    ckpt_rel = art.get("checkpoint")
    ckpt_hash = files.get(ckpt_rel, {}).get("sha256") if ckpt_rel else None
    if ckpt_hash is None:
        block("checkpoint_binding", "checkpoint file missing; binding to contracts cannot be verified")
    claimed = {"receipt": (receipt or {}).get("checkpoint_id"), "training_completion": (completion or {}).get("checkpoint_id"),
               **{f"contract[{v}]": (c or {}).get("checkpoint_id") for v, c in contracts.items()}}
    for source, value in claimed.items():
        if value is None:
            block("checkpoint_binding", f"{source}: no checkpoint_id recorded")
        elif ckpt_hash is not None and value != ckpt_hash:
            fail("checkpoint_binding", f"{source}: checkpoint_id {value[:12]}... differs from the file's SHA-256 {ckpt_hash[:12]}...")
    if identity_manifest is not None:
        manifest_hash = canonical_hash(identity_manifest)
        for source, payload in (("receipt", receipt), ("training_completion", completion)):
            value = (payload or {}).get("dataset_manifest_sha256")
            if value and value != manifest_hash:
                fail("checkpoint_binding", f"{source}: dataset_manifest_sha256 does not match the identity manifest content")
        for view, c in contracts.items():
            value = (c or {}).get("dataset_manifest_sha256")
            if value and value != manifest_hash:
                fail("checkpoint_binding", f"contract[{view}]: dataset_manifest_sha256 does not match the identity manifest content")
        declared = (manifest.get("dataset") or {}).get("version")
        if declared and declared != manifest_hash:
            fail("checkpoint_binding", "manifest dataset.version does not match the identity manifest content")
    else:
        block("checkpoint_binding", "identity manifest missing; dataset binding cannot be verified")
    if completion is not None and split is not None:
        value = completion.get("split_sha256")
        if value and value != canonical_hash(split):
            fail("checkpoint_binding", "training_completion.split_sha256 does not match the split receipt content")
    if completion is not None and completion.get("checkpoint_selection") != "final_last_only":
        fail("checkpoint_binding", f"checkpoint_selection is {completion.get('checkpoint_selection')!r}, not final_last_only")
    if checks["checkpoint_binding"]["status"] == "not_attempted":
        note("checkpoint_binding", f"checkpoint {ckpt_hash[:12]}... is the checkpoint_id in the receipt, the completion record and every contract; "
             "dataset_manifest_sha256 equals the canonical hash of the identity manifest; split_sha256 equals the canonical hash of the split receipt; final_last_only selection recorded")

    # 9. purpose guard --------------------------------------------------------------------------
    derived = "paper-evidence" if cohort == "original500" else "software-demo"
    signals = []
    if cohort != "original500":
        signals.append(f"cohort={cohort!r}")
    for view, c in contracts.items():
        if c is None:
            continue
        if c.get("split") != "paper-test":
            signals.append(f"contract[{view}].split={c.get('split')!r}")
        if "not a paper" in str(c.get("scope", "")).lower():
            signals.append(f"contract[{view}].scope marks a non-paper evaluation")
    if (root / "ai_smoke_receipt.json").is_file():
        signals.append("ai_smoke_receipt.json present in the root")
    if signals and manifest.get("purpose") == "paper-evidence":
        fail("purpose_guard", "manifest claims paper-evidence but the artifacts carry demo/smoke markers: " + "; ".join(signals))
    else:
        note("purpose_guard", f"purpose {manifest.get('purpose')!r} is consistent with the artifacts (derived {derived!r})")
        checks["purpose_guard"]["details"].append("this guard catches accidental mislabeling; it is not a tamper-proof provenance authentication")

    # 10./11. categories this tool cannot establish ------------------------------------------
    ref = manifest.get("numerical_reference") or {}
    nr = checks["numerical_reference_evidence"]
    if isinstance(ref, dict) and ref.get("report"):
        rel = ref["report"]
        if rel in missing:
            nr["details"].append(f"declared report {rel} is missing"); nr["status"] = "blocked"
        else:
            nr["details"].append(f"report {rel} present (sha256 {files[rel]['sha256'][:12]}...); its scientific content is NOT verified by this tool")
            nr["status"] = "recorded_unverified"
    else:
        nr["details"].append("no numerical-reference report declared; independent accuracy of the evaluated targets is not established here")
    checks["external_reproduction"]["details"].append(
        "not attempted: this tool checks internal consistency of the supplied artifacts only; an external reproduction "
        "needs an independent run from the resolved configuration, identity manifest and split receipt")

    # compatibility alias for readers of the first overlay-2 report: passed only when both halves passed
    halves = (checks["identity_shape"]["status"], checks["finiteness"]["status"])
    checks["identity_shape_finiteness"] = {"status": "passed" if halves == ("passed", "passed") else ("failed" if "failed" in halves else "blocked"),
                                          "details": ["alias of identity_shape + finiteness; see those categories"]}
    statuses = {name: checks[name]["status"] for name in CORE_CHECKS}
    if any(s == "failed" for s in statuses.values()):
        overall, code = "failed", EXIT_FAILED
    elif any(s in ("blocked", "not_attempted") for s in statuses.values()):
        overall, code = "incomplete", EXIT_INCOMPLETE
    else:
        overall, code = "complete", EXIT_COMPLETE
    return {
        "schema_version": CHECK_VERSION, "bundle_schema": BUNDLE_VERSION, "root": str(root),
        "experiment_id": manifest.get("experiment_id"), "purpose": manifest.get("purpose"), "cohort": cohort,
        "verification_status": overall, "exit_code": code, "checks": checks, "files": files,
        "claims": {
            "local_artifact_consistency": overall,
            "scientific_reproduction": "not established by this tool",
            "note": "complete means the supplied files are mutually consistent, their metadata and configuration identities agree, and the stored "
                    "score follows from the stored arrays; it does not authenticate data origin, numerical fidelity, or that the run matches the paper table",
        },
    }


# ----------------------------------------------------------------------------- export

def export_bundle(manifest: Mapping[str, Any], root: str | Path, out: str | Path, *, report: Mapping[str, Any],
                  copy_large: bool = False, allow_demo: bool = False) -> dict[str, Any]:
    root = Path(root).resolve()
    destination = Path(out).resolve()
    if destination.exists() or destination.is_symlink():
        raise EvidenceError("output_exists", f"{destination} already exists", "choose a new output directory; nothing was overwritten")
    if root == destination or root in destination.parents:
        raise EvidenceError("bad_output", "output must not be inside the bundle root", "export to a sibling directory")
    if manifest.get("purpose") != "paper-evidence" and not allow_demo:
        raise EvidenceError("demo_into_paper_export", f"bundle purpose is {manifest.get('purpose')!r}; the paper export path refuses it",
                            "pass --allow-demo to export a software-demo bundle into a clearly labelled demo directory")
    if report.get("checks", {}).get("purpose_guard", {}).get("status") == "failed":
        raise EvidenceError("purpose_guard_failed", "the artifacts carry demo/smoke markers but the manifest claims paper evidence",
                            "fix the manifest purpose or supply the real production artifacts")
    destination.mkdir(parents=True, exist_ok=False)
    index = []
    for rel, info in report.get("files", {}).items():
        source = _contained(root, rel)
        entry = {"path": rel, "sha256": info.get("sha256"), "bytes": info.get("bytes"), "copied": False}
        if source.is_file():
            large = source.suffix.lower() in LARGE_SUFFIXES
            if not large or copy_large:
                target = destination / "artifacts" / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
                entry["copied"] = True
        else:
            entry["missing"] = True
        index.append(entry)
    label = "software-demo export, not paper evidence" if manifest.get("purpose") != "paper-evidence" else "paper evidence export"
    payload = {"schema_version": BUNDLE_VERSION, "manifest": dict(manifest), "verification": report,
               "export": {"copied_large_files": bool(copy_large), "allow_demo": bool(allow_demo), "label": label,
                          "reimport": "python -m pdeobs.evidence check --bundle <this directory>/evidence_bundle.json (root defaults to <this directory>/artifacts); "
                                      "large files not copied are reported as blocked, exactly as they are absent"}}
    (destination / "evidence_bundle.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    (destination / "manifest.json").write_text(json.dumps(dict(manifest), indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    (destination / "file_index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    (destination / "checks.json").write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return {"out": str(destination), "files": len(index), "copied": sum(1 for e in index if e["copied"]),
            "referenced_only": sum(1 for e in index if not e["copied"] and not e.get("missing")),
            "verification_status": report.get("verification_status"), "label": label}


# ----------------------------------------------------------------------------- CLI

def _emit(payload: Any) -> None:
    sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def _fail(exc: EvidenceError) -> int:
    sys.stderr.write(json.dumps({"schema_version": ERROR_VERSION, "status": "error", "category": exc.category,
                                 "reason": exc.reason, "next_step": exc.next_step}) + "\n")
    return EXIT_FAILED


def _write_new(path: Path, payload: Any) -> None:
    if path.exists():
        raise EvidenceError("output_exists", f"{path} already exists", "choose a new file name; nothing was overwritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _default_root(bundle_path: Path) -> Path:
    """An exported envelope keeps its files under <export>/artifacts; a planned manifest sits in the row directory."""
    folder = bundle_path.resolve().parent
    if bundle_path.name == "evidence_bundle.json" and (folder / "artifacts").is_dir():
        return folder / "artifacts"
    if bundle_path.name == "manifest.json" and (folder / "artifacts").is_dir() and (folder / "evidence_bundle.json").is_file():
        return folder / "artifacts"
    return folder


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m pdeobs.evidence", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan", help="write an evidence-bundle manifest from a paper-row output directory (existence checks only)")
    p.add_argument("--root", required=True, help="paper-row output directory")
    p.add_argument("--out", help="write the manifest to this NEW file; omitted: print only")
    p.add_argument("--purpose", choices=PURPOSES, help="override the purpose derived from the receipt cohort (a demo cohort is always checked as demo)")
    p.add_argument("--paper-ref", help='JSON string, e.g. \'{"table": "2", "row": "poisson/fno", "cell": "R50"}\'')
    c = sub.add_parser("check", help="verify every link in a bundle; read-only; exit 0 complete, 3 incomplete, 2 failed")
    c.add_argument("--bundle", required=True, help="manifest written by plan, by hand, or an exported evidence_bundle.json")
    c.add_argument("--root", help="bundle root; default: the manifest's directory (an export's artifacts/ directory)")
    c.add_argument("--out", help="also write the check report to this NEW file")
    c.add_argument("--no-rescore", action="store_true", help="skip re-scoring the prediction artifacts (finiteness and metric recomputation are then reported as blocked)")
    e = sub.add_parser("export", help="check, then copy the small artifacts and a hashed file index into a NEW directory")
    e.add_argument("--bundle", required=True)
    e.add_argument("--root")
    e.add_argument("--out", required=True)
    e.add_argument("--copy-large", action="store_true", help="also copy checkpoints and prediction artifacts (default: reference by hash)")
    e.add_argument("--allow-demo", action="store_true", help="permit exporting a software-demo bundle (never into a paper table)")
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            ref = json.loads(args.paper_ref) if args.paper_ref else None
            result = plan_from_row_dir(args.root, purpose=args.purpose, paper_reference=ref)
            if args.out:
                _write_new(Path(args.out), result["manifest"])
                result["written"] = str(Path(args.out).resolve())
            _emit(result)
            return EXIT_COMPLETE if not result["missing"] else EXIT_INCOMPLETE
        manifest_path = Path(args.bundle)
        manifest = load_manifest(manifest_path)
        root = Path(args.root) if args.root else _default_root(manifest_path)
        report = check_bundle(manifest, root, rescore=not getattr(args, "no_rescore", False))
        if args.command == "check":
            if args.out:
                _write_new(Path(args.out), report)
            _emit(report)
            return int(report["exit_code"])
        result = export_bundle(manifest, root, args.out, report=report, copy_large=args.copy_large, allow_demo=args.allow_demo)
        _emit(result)
        return int(report["exit_code"])
    except EvidenceError as exc:
        return _fail(exc)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return _fail(EvidenceError("unexpected_input", f"{type(exc).__name__}: {exc}",
                                   "inspect the named artifact; the tool made no changes"))


if __name__ == "__main__":
    raise SystemExit(main())
