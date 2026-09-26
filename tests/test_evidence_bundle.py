"""Evidence-bundle tool: plan / check / export against a real paper-row demo output.

The fixture is produced by the repository's own bounded CPU demo (nine 16x16 records, one epoch,
two test views).  It is software evidence for the TOOL; the numbers it scores are not paper results.
Tamper cases edit copies of the fixture, never the fixture itself.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

from pdeobs import evidence  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def demo_row(tmp_path_factory) -> Path:
    from pdeobs.paper_row import create_paper_row_demo_data, run_paper_row

    base = tmp_path_factory.mktemp("evidence-demo")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    create_paper_row_demo_data(base / "data", "poisson")
    receipt = run_paper_row(ROOT / "configs/demo/paper_row_static.yaml", data_root=base / "data",
                            manifest_path=base / "data/identities.json", output=base / "row", device="cpu")
    assert receipt["status"] == "complete", receipt
    return base / "row"


def _copy(src: Path, dst: Path) -> Path:
    shutil.copytree(src, dst)
    return dst


def _plan_and_check(root: Path, **kw):
    manifest = evidence.plan_from_row_dir(root)["manifest"]
    return manifest, evidence.check_bundle(manifest, root, **kw)


def test_plan_lists_every_artifact_and_derives_demo_purpose(demo_row):
    plan = evidence.plan_from_row_dir(demo_row)
    assert plan["missing"] == []
    assert plan["derived_purpose"] == "software-demo" and plan["manifest"]["purpose"] == "software-demo"
    views = [b["test_view"] for b in plan["manifest"]["artifacts"]["blocks"]]
    assert views == ["random_50pct", "random_80pct"]
    assert plan["manifest"]["scorer_version"] == "pdeobs-strict-v1"
    assert plan["manifest"]["test_protocol"]["split"] == "demo"


def test_check_is_read_only_and_complete_on_untouched_row(demo_row):
    before = sorted(p.relative_to(demo_row).as_posix() for p in demo_row.rglob("*") if p.is_file())
    manifest, report = _plan_and_check(demo_row)
    after = sorted(p.relative_to(demo_row).as_posix() for p in demo_row.rglob("*") if p.is_file())
    assert before == after, "check must not write into the bundle root"
    assert report["verification_status"] == "complete" and report["exit_code"] == 0
    for name in ("manifest_structure", "artifact_completeness", "metadata_consistency", "configuration_identity", "identity_shape",
                 "finiteness", "metric_recomputation", "split_provenance", "checkpoint_binding", "purpose_guard"):
        assert report["checks"][name]["status"] == "passed", (name, report["checks"][name])
    assert report["checks"]["external_reproduction"]["status"] == "not_attempted"
    assert report["checks"]["numerical_reference_evidence"]["status"] == "not_attempted"
    assert report["claims"]["scientific_reproduction"] == "not established by this tool"
    assert all(info["sha256"] for info in report["files"].values())


def test_tampered_score_summary_fails_metric_recomputation_only(demo_row, tmp_path):
    root = _copy(demo_row, tmp_path / "row")
    score_path = root / "evaluation/random_50pct/score.json"
    score = json.loads(score_path.read_text())
    score["summary"]["rel_l2_joint_mean"] = 0.001  # a "better" number that the arrays do not support
    score_path.write_text(json.dumps(score))
    _, report = _plan_and_check(root)
    assert report["checks"]["metric_recomputation"]["status"] == "failed"
    assert any("re-scoring gives" in d for d in report["checks"]["metric_recomputation"]["details"])
    assert report["verification_status"] == "failed" and report["exit_code"] == 2
    assert report["checks"]["split_provenance"]["status"] == "passed"  # one failed check does not fail the others


def test_missing_prediction_artifact_is_blocked_not_failed(demo_row, tmp_path):
    root = _copy(demo_row, tmp_path / "row")
    (root / "evaluation/random_80pct/predictions.h5").unlink()
    _, report = _plan_and_check(root)
    assert report["checks"]["artifact_completeness"]["status"] == "blocked"
    assert report["checks"]["metric_recomputation"]["status"] == "blocked"
    assert report["verification_status"] == "incomplete" and report["exit_code"] == 3
    assert report["files"]["evaluation/random_80pct/predictions.h5"]["sha256"] is None


def test_swapped_checkpoint_breaks_binding(demo_row, tmp_path):
    root = _copy(demo_row, tmp_path / "row")
    ckpt = root / "checkpoints/last.pt"
    ckpt.write_bytes(ckpt.read_bytes() + b"\0")  # different bytes, same name
    _, report = _plan_and_check(root)
    assert report["checks"]["checkpoint_binding"]["status"] == "failed"
    assert any("differs from the file's SHA-256" in d for d in report["checks"]["checkpoint_binding"]["details"])


def test_demo_row_cannot_be_labelled_paper_evidence(demo_row):
    manifest = evidence.plan_from_row_dir(demo_row)["manifest"]
    manifest["purpose"] = "paper-evidence"
    report = evidence.check_bundle(manifest, demo_row)
    assert report["checks"]["purpose_guard"]["status"] == "failed"
    assert report["verification_status"] == "failed"


def test_synthetic_original500_split_rules(demo_row, tmp_path):
    """Synthetic fixture: a demo row relabelled as original500 must fail the paper split rules."""
    root = _copy(demo_row, tmp_path / "row")
    manifest = evidence.plan_from_row_dir(root)["manifest"]
    manifest["training_protocol"]["cohort"] = "original500"
    manifest["test_protocol"] = {"split": "paper-test", "expected_identity_count": 200, "test_views": ["random_50pct", "random_80pct"]}
    report = evidence.check_bundle(manifest, root, rescore=False)
    details = " ".join(report["checks"]["split_provenance"]["details"])
    assert report["checks"]["split_provenance"]["status"] == "failed"
    assert "is not pdeobs.one-setting-split/v1" in details and "1800 train + 200 test" in details
    assert report["checks"]["identity_shape"]["status"] == "failed"
    assert any("bundle expects 200" in d for d in report["checks"]["identity_shape"]["details"])
    assert report["checks"]["finiteness"]["status"] == "blocked"  # rescore=False never certifies finiteness


def test_export_refuses_existing_dir_and_demo_without_flag(demo_row, tmp_path):
    manifest, report = _plan_and_check(demo_row)
    with pytest.raises(evidence.EvidenceError, match="refuses"):
        evidence.export_bundle(manifest, demo_row, tmp_path / "paper-export", report=report)
    taken = tmp_path / "taken"
    taken.mkdir()
    with pytest.raises(evidence.EvidenceError, match="already exists"):
        evidence.export_bundle(manifest, demo_row, taken, report=report, allow_demo=True)
    assert sorted(p.name for p in taken.iterdir()) == []


def test_export_references_large_files_by_hash_unless_asked(demo_row, tmp_path):
    manifest, report = _plan_and_check(demo_row)
    result = evidence.export_bundle(manifest, demo_row, tmp_path / "demo-export", report=report, allow_demo=True)
    out = Path(result["out"])
    payload = json.loads((out / "evidence_bundle.json").read_text())
    assert payload["export"]["label"].startswith("software-demo")
    index = {e["path"]: e for e in json.loads((out / "file_index.json").read_text())}
    assert index["checkpoints/last.pt"]["copied"] is False and index["checkpoints/last.pt"]["sha256"]
    assert index["evaluation/random_50pct/predictions.h5"]["copied"] is False
    assert index["receipt.json"]["copied"] is True and (out / "artifacts/receipt.json").is_file()
    assert not (out / "artifacts/checkpoints/last.pt").exists()
    result2 = evidence.export_bundle(manifest, demo_row, tmp_path / "demo-export-large", report=report, allow_demo=True, copy_large=True)
    assert (Path(result2["out"]) / "artifacts/checkpoints/last.pt").is_file()


def test_cli_errors_are_json_on_stderr_with_exit_2(tmp_path):
    run = subprocess.run([sys.executable, "-m", "pdeobs.evidence", "check", "--bundle", str(tmp_path / "nope.json")],
                         capture_output=True, text=True, timeout=60, cwd=str(ROOT), env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
    assert run.returncode == 2 and run.stdout == ""
    err = json.loads(run.stderr.strip().splitlines()[-1])
    assert err["schema_version"] == "pdeobs-evidence-error/v1" and err["category"] and err["next_step"]
    run = subprocess.run([sys.executable, "-m", "pdeobs.evidence", "plan", "--root", str(tmp_path)],
                         capture_output=True, text=True, timeout=60, cwd=str(ROOT), env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
    assert run.returncode == 2
    assert json.loads(run.stderr.strip().splitlines()[-1])["category"] == "not_a_row_directory"


def test_cli_check_exit_codes_and_stdout_json(demo_row, tmp_path):
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    plan = subprocess.run([sys.executable, "-m", "pdeobs.evidence", "plan", "--root", str(demo_row), "--out", str(tmp_path / "bundle.json")],
                          capture_output=True, text=True, timeout=120, cwd=str(ROOT), env=env)
    assert plan.returncode == 0, plan.stderr
    check = subprocess.run([sys.executable, "-m", "pdeobs.evidence", "check", "--bundle", str(tmp_path / "bundle.json"), "--root", str(demo_row)],
                           capture_output=True, text=True, timeout=300, cwd=str(ROOT), env=env)
    assert check.returncode == 0, check.stderr
    report = json.loads(check.stdout)
    assert report["schema_version"] == "pdeobs-evidence-check/v1" and report["verification_status"] == "complete"
    again = subprocess.run([sys.executable, "-m", "pdeobs.evidence", "plan", "--root", str(demo_row), "--out", str(tmp_path / "bundle.json")],
                           capture_output=True, text=True, timeout=120, cwd=str(ROOT), env=env)
    assert again.returncode == 2 and json.loads(again.stderr.strip().splitlines()[-1])["category"] == "output_exists"
