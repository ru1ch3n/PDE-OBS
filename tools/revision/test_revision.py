"""Acceptance tests for the revision tools (CPU, synthetic fixtures; no production data needed).

    python -m unittest discover -s tools/revision -p 'test_*.py'
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))

import build_results as br  # noqa: E402
import audit_training as at  # noqa: E402
from common import SCORING_VERSION, VIEWS, SourceError, canonical_hash, ids_hash  # noqa: E402

IDS = [f"seed-1/poisson/dirichlet/smooth_grf/{regime}/{i:06d}" for regime, n in (("low", 67), ("medium", 67), ("high", 66)) for i in range(n)]
CAMPAIGN = {"views": {v: {"mask": {"protocol": "random_3pct", "ratio": 0.5}} for v in VIEWS}, "problem_settings": {"poisson": {"task": "recovery"}}}


def synthetic_block(folder: Path, view: str, values: dict[str, float], *, checkpoint: str = "c" * 64, kind: str = "frozen_paper_view", mask=None) -> None:
    ids = list(values)
    contract = {"schema_version": "pdeobs-strict-contract-v1", "task": "recovery", "split": "paper-test", "expected_ids": ids, "expected_shape": [4, 4, 1],
                "target_time_indices": [0], "observation_id": view, "view_kind": kind, "mask_config": mask or CAMPAIGN["views"][VIEWS[0]]["mask"],
                "checkpoint_id": checkpoint, "projection": False}
    per_identity = [{"identity": i, "rel_l2_joint": values[i]} for i in ids]
    score = {"status": "valid", "scoring_version": SCORING_VERSION, "scored_identity_count": len(ids), "config": contract, "config_sha256": canonical_hash(contract),
             "identity_set_sha256": canonical_hash(sorted(ids)), "artifact_contract_binding": "pdeobs-strict-inference-v1_sha256_bound",
             "summary": {"rel_l2_joint_mean": sum(values.values()) / len(values)}, "per_identity": per_identity}
    dest = folder / "blocks" / view
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "contract.json").write_text(json.dumps(contract))
    (dest / "score.json").write_text(json.dumps(score))
    (dest / "completion.json").write_text(json.dumps({"predictions_sha256": "p" * 64, "realized_observed_fraction": 0.5}))


def synthetic_records(mix_values: dict[str, float], diagonal: float = 0.1, off: float = 0.5) -> dict:
    records = {}
    for v in VIEWS:
        blocks = {w: {"joint": {"mean": diagonal if v == w else off, "std": 0.01, "n": 200}, "contract_sha256": "k" * 64} for w in VIEWS}
        records[f"poisson/fno/{v}"] = {"blocks": blocks, "training_cohort": "original500", "actual_epochs": 500, "stop_reason": None,
                                       "cohort_evidence": {"training_protocol": None}, "checkpoint_original_sha256": "o" * 64}
    return records


class BuildResultsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self.tmp.name)
        self.values = {i: 0.2 + 0.001 * k for k, i in enumerate(IDS)}
        for v in VIEWS:
            synthetic_block(self.folder, v, self.values)
        self.blocks = {v: br.load_strict_block(self.folder, v) for v in VIEWS}
        self.records = synthetic_records(self.values)
        self.grid_pi = {(f"poisson/fno/{v}", v): {i: 0.1 for i in IDS} for v in VIEWS}

    def tearDown(self):
        self.tmp.cleanup()

    def test_missing_score_is_an_error(self):
        with self.assertRaises(SourceError):
            br.load_strict_block(self.folder, "no_such_view")

    def test_duplicate_or_missing_identity_is_an_error(self):
        dest = self.folder / "blocks" / VIEWS[0]
        score = json.loads((dest / "score.json").read_text())
        score["per_identity"][1]["identity"] = score["per_identity"][0]["identity"]
        (dest / "score.json").write_text(json.dumps(score))
        with self.assertRaises(SourceError):
            br.load_strict_block(self.folder, VIEWS[0])

    def test_identity_set_mismatch_between_mix_and_specialist_fails(self):
        wrong = {(f"poisson/fno/{v}", v): {i + "x": 0.1 for i in IDS} for v in VIEWS}
        with self.assertRaises(SourceError):
            br.compare_pair(pde="poisson", method="fno", task="recovery", mix_blocks=self.blocks, records=self.records, grid_per_identity=wrong, legacy_blocks={})

    def test_forbidden_source_is_refused(self):
        with self.assertRaises(SourceError):
            br.assert_no_forbidden_sources({"forbidden_sources": ["results/archived_campaign"], "mixed_models": [{"strict_scores": "results/archived_campaign/x"}]})
        br.assert_no_forbidden_sources({"forbidden_sources": ["results/archived_campaign"], "mixed_models": [{"strict_scores": "results/revision_v2/mix_strict/x"}]})

    def test_ratios_and_restricted_population_use_exactly_the_declared_views(self):
        rows = br.compare_pair(pde="poisson", method="fno", task="recovery", mix_blocks=self.blocks, records=self.records, grid_per_identity=self.grid_pi, legacy_blocks={})
        mean = sum(self.values.values()) / len(self.values)
        for r in rows:
            self.assertAlmostEqual(r["ratio_mix_over_specialist"], mean / 0.1, places=12)
            self.assertEqual(r["transfer_n"], 8)
        restricted = br.population_summary(rows, [VIEWS[3], VIEWS[7], VIEWS[8]], "recipe_matched")
        self.assertEqual(restricted["views"], [VIEWS[3], VIEWS[7], VIEWS[8]])
        self.assertEqual(restricted["n"], 3)
        self.assertAlmostEqual(restricted["geomean_ratio"], mean / 0.1, places=12)
        self.assertAlmostEqual(restricted["mean_mix"], mean, places=12)
        self.assertAlmostEqual(restricted["mean_specialist"], 0.1, places=12)
        with self.assertRaises(SourceError):
            br.population_summary(rows[:2], list(VIEWS), "all_nine_descriptive")

    def test_frozen_view_contract_checks(self):
        block = self.blocks[VIEWS[0]]
        digest = ids_hash(IDS)
        br.check_frozen_view_contract(block, pde="poisson", view=VIEWS[0], task="recovery", campaign=CAMPAIGN, checkpoint_sha256="c" * 64, test_ids_digest=digest)
        with self.assertRaises(SourceError):
            br.check_frozen_view_contract(block, pde="poisson", view=VIEWS[0], task="recovery", campaign=CAMPAIGN, checkpoint_sha256="d" * 64, test_ids_digest=digest)
        with self.assertRaises(SourceError):
            br.check_frozen_view_contract(block, pde="poisson", view=VIEWS[0], task="rollout", campaign=CAMPAIGN, checkpoint_sha256="c" * 64, test_ids_digest=digest)


class AuditTrainingTests(unittest.TestCase):
    SPEC = ("random_3pct(ratio=0.5) + random_3pct(ratio=0.65) + random_3pct(ratio=0.8) + block_missing(missing_fraction=0.49) + "
            "line_sensors(num_lines=76, orientation=both) + line_sensors(num_lines=64, orientation=horizontal) + "
            "line_sensors(num_lines=64, orientation=vertical) + boundary_sensors(width=19) + clustered_sensors(ratio=0.5)")

    def test_assignment_is_reproducible_and_order_independent(self):
        counts = {"low": {"records": 30}, "medium": {"records": 30}, "high": {"records": 30}}
        identities = at.reconstruct_identities("poisson", "dirichlet", "smooth_grf", counts, 7)
        split_of = {r["sample_id"]: "train" for r in identities}
        first = at.assign_mixture(self.SPEC, 7, identities, (128, 128), split_of)
        second = at.assign_mixture(self.SPEC, 7, list(reversed(identities)), (128, 128), split_of)
        by_id = {r["physical_identity"]: r for r in second}
        for r in first:
            self.assertEqual((r["assigned_component"], r["component_mask_hash"], r["observed_count"]), (by_id[r["physical_identity"]]["assigned_component"], by_id[r["physical_identity"]]["component_mask_hash"], by_id[r["physical_identity"]]["observed_count"]))
        self.assertEqual(len(first), 90)
        self.assertEqual(len({r["assigned_component"] for r in first}), 9)
        # a different mask seed changes masks but the cycle structure still assigns one component per identity
        other = at.assign_mixture(self.SPEC, 8, identities, (128, 128), split_of)
        self.assertNotEqual([r["component_mask_hash"] for r in first], [r["component_mask_hash"] for r in other])

    def test_unknown_updates_are_not_derived_from_planned_budget(self):
        self.assertEqual(at.unknown(None), "unknown")
        self.assertEqual(at.steps_per_epoch(4), 450)
        self.assertEqual(at.steps_per_epoch(8), 225)
        self.assertEqual(at.steps_per_epoch(2), 900)


if __name__ == "__main__":
    unittest.main()
