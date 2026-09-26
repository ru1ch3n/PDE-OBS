"""Second-audit regressions: the evidence checker and the archived-results checker must report the disagreements
they are asked to detect.  Adapted from the independent reviewer's standalone file; every case mutates a COPY of a
tiny CPU demo row or purely synthetic archive metadata.  No source, production artifact or shipped value changes."""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
from pathlib import Path

import pytest

pytest.importorskip("torch")

from pdeobs import evidence  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def row(tmp_path_factory) -> Path:
    import torch
    from pdeobs.paper_row import create_paper_row_demo_data, run_paper_row

    torch.set_num_threads(1)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    base = tmp_path_factory.mktemp("second-audit-demo")
    create_paper_row_demo_data(base / "data", "poisson")
    result = run_paper_row(ROOT / "configs/demo/paper_row_static.yaml", data_root=base / "data",
                           manifest_path=base / "data/identities.json", output=base / "row", device="cpu")
    assert result["status"] == "complete"
    return base / "row"


def change_json(path: Path, edit) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    edit(value)
    path.write_text(json.dumps(value), encoding="utf-8")


def copy_row(row: Path, tmp_path: Path):
    dst = tmp_path / "row"
    shutil.copytree(row, dst)
    return dst, evidence.plan_from_row_dir(dst)["manifest"]


def test_untouched_evidence_passes(row):
    m = evidence.plan_from_row_dir(row)["manifest"]
    report = evidence.check_bundle(m, row)
    assert report["verification_status"] == "complete", report["checks"]
    for name in ("manifest_structure", "metadata_consistency", "configuration_identity", "identity_shape", "finiteness"):
        assert report["checks"][name]["status"] == "passed", (name, report["checks"][name])


@pytest.mark.parametrize("case", ["training_config", "resolved_model", "bundle_model", "training_view", "test_view", "omitted_configs",
                                  "receipt_inventory", "bundle_seed", "trainer_seed", "score_hash"])
def test_inconsistent_metadata_must_not_pass(row, tmp_path, case):
    dst, m = copy_row(row, tmp_path)
    if case == "training_config":
        change_json(dst / "checkpoints/training_config.json", lambda d: d.update(learning_rate=123.0))
    elif case == "resolved_model":
        change_json(dst / "row_resolved.json", lambda d: d["experiment"]["method"].update(name="transolver"))
    elif case == "bundle_model":
        m["method"] = "transolver"
    elif case == "training_view":
        m["training_view"] = "clustered_50pct"
    elif case == "test_view":
        m["artifacts"]["blocks"][0]["test_view"] = "clustered_50pct"
    elif case == "omitted_configs":
        for k in ("resolved_config", "training_config"):
            (dst / m["artifacts"].pop(k)).unlink()
    elif case == "receipt_inventory":
        change_json(dst / "receipt.json", lambda d: d["evaluation_blocks"].pop())
    elif case == "bundle_seed":
        m["seeds"]["training_seed"] = 1
    elif case == "trainer_seed":
        change_json(dst / "checkpoints/training_config.json", lambda d: d.update(seed=7))
    elif case == "score_hash":
        change_json(dst / "evaluation/random_50pct/score.json", lambda d: d.update(note="edited"))
    report = evidence.check_bundle(m, dst)
    assert report["verification_status"] != "complete", f"{case} was falsely accepted as complete"
    assert report["verification_status"] == "failed", (case, report["checks"])


def test_no_rescore_cannot_certify_finiteness(row, tmp_path):
    import h5py

    dst, m = copy_row(row, tmp_path)
    with h5py.File(dst / m["artifacts"]["blocks"][0]["predictions"], "r+") as h:
        h["prediction"][0, 0, 0, 0] = float("nan")
    report = evidence.check_bundle(m, dst, rescore=False)
    assert report["checks"]["finiteness"]["status"] == "blocked"
    assert report["checks"]["metric_recomputation"]["status"] == "blocked"
    assert report["verification_status"] == "incomplete"
    report = evidence.check_bundle(m, dst)
    assert report["checks"]["finiteness"]["status"] == "failed" and report["verification_status"] == "failed"


def test_unreadable_prediction_cannot_pass_recomputation(row, tmp_path):
    import h5py

    dst, m = copy_row(row, tmp_path)
    with h5py.File(dst / m["artifacts"]["blocks"][0]["predictions"], "r+") as h:
        del h["prediction"]
    report = evidence.check_bundle(m, dst)
    assert report["checks"]["metric_recomputation"]["status"] == "failed"
    assert report["checks"]["finiteness"]["status"] == "failed"
    assert report["verification_status"] == "failed"


def test_export_can_be_rechecked_directly(row, tmp_path):
    m = evidence.plan_from_row_dir(row)["manifest"]
    report = evidence.check_bundle(m, row)
    out = tmp_path / "export"
    evidence.export_bundle(m, row, out, report=report, allow_demo=True, copy_large=True)
    assert (out / "manifest.json").is_file()
    rc = evidence.main(["check", "--bundle", str(out / "evidence_bundle.json")])
    assert rc == 0
    again = evidence.check_bundle(evidence.load_manifest(out / "evidence_bundle.json"), out / "artifacts")
    assert again["verification_status"] == "complete"
    # without the large files the re-check is honest about what is absent
    out2 = tmp_path / "export-small"
    evidence.export_bundle(m, row, out2, report=report, allow_demo=True)
    partial = evidence.check_bundle(evidence.load_manifest(out2 / "evidence_bundle.json"), out2 / "artifacts")
    assert partial["verification_status"] == "incomplete"
    assert partial["checks"]["checkpoint_binding"]["status"] == "blocked"


@pytest.fixture(scope="module")
def archive_tool():
    spec = importlib.util.spec_from_file_location("archived_results_under_test", ROOT / "tools/archived_results.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def archive_fixture(ar):
    """Synthetic structural metadata; 0.5 is NOT a result from a paper or experiment."""
    ck, mh, ids, ds, ca = ("a" * 64, "b" * 64, "c" * 64, "d" * 64, "e" * 64)
    split = {"algorithm": ar.SPLIT_ALGORITHM, "schema_version": "pdeobs.one-setting-split/v1", "seed": 20260804, "records": 2000,
             "train_records": 1800, "test_records": 200, "validation_records": 0,
             "regime_counts": {"low": {"train": 600, "test": 67}, "medium": {"train": 600, "test": 67}, "high": {"train": 600, "test": 66}},
             "macrodomain_identity": "poisson|dirichlet|smooth_grf", "test_sample_ids_sha256": ids}
    sel = {"completion": {"status": "completed", "checkpoint_sha256": ck, "dataset_summary_sha256": ds, "campaign_sha256": ca,
                          "policy_evidence": {"checkpoint_sha256": ck}, "blocks": {v: {"metrics_sha256": mh} for v in ar.VIEWS}},
           "blocks": {v: {"samples": 200, "test_sample_ids_sha256": ids, "checkpoint_sha256": ck,
                          "metrics": {"relative_l2": 0.5, "nonfinite_rate": 0.0}} for v in ar.VIEWS},
           "files": {"completion.json": {"sha256": "f" * 64}, **{"blocks/%s/metrics.json" % v: {"sha256": mh} for v in ar.VIEWS}}}
    train = {"records": {"completion.json": {"status": "completed", "planned_epochs": 500, "epochs_completed": 500,
                                             "checkpoint_sha256": ck, "dataset_summary_sha256": ds},
                         "identity.json": {"dataset_summary_sha256": ds, "campaign_sha256": ca},
                         "health.json": {}, "split_manifest.json": split}, "files": {}}
    return sel, train


def test_untouched_archive_metadata_passes(archive_tool):
    s, t = archive_fixture(archive_tool)
    assert archive_tool.build_record("poisson/fno/random_50pct", s, t)["verification_status"] == "consistent"


@pytest.mark.parametrize("case", ["split_hash", "nan_metric", "metrics_file_hash", "missing_ids_hash", "test_records_touched"])
def test_archive_corruption_must_propagate(archive_tool, case):
    s, t = archive_fixture(archive_tool)
    if case == "split_hash":
        t["records"]["split_manifest.json"]["test_sample_ids_sha256"] = "0" * 64
    elif case == "nan_metric":
        s["blocks"][archive_tool.VIEWS[0]]["metrics"]["relative_l2"] = float("nan")
    elif case == "metrics_file_hash":
        s["files"]["blocks/%s/metrics.json" % archive_tool.VIEWS[0]]["sha256"] = "0" * 64
    elif case == "missing_ids_hash":
        for v in archive_tool.VIEWS:
            s["blocks"][v]["test_sample_ids_sha256"] = None
    elif case == "test_records_touched":
        t["records"]["health.json"] = {"test_records_touched": 3}
    report = archive_tool.build_record("poisson/fno/random_50pct", s, t)
    assert report["verification_status"] == "inconsistent", f"{case} was reported {report['verification_status']}"
    json.dumps(report, allow_nan=False)  # the record itself must stay strict JSON
