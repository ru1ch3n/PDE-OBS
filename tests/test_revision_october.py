"""Semantic tests of the October 2026 revision modules (observation layouts, covariance reference, data-consistency split,
paired statistics, RPB constants). CPU only, synthetic data, no downloads."""
import numpy as np
import pytest

from pdeobs.revision import v2masks as VM
from pdeobs.revision.covariance import CovarianceReconstructor, pod_basis
from pdeobs.revision.dc import dc_split
from pdeobs.revision.stats import ParentResamples, matched_offpattern, paired_difference


@pytest.mark.parametrize("n", [64, 128])
@pytest.mark.parametrize("proto", ["U256", "C256", "U1024", "C1024", "G256"])
def test_layouts_exact_count_unique_and_eligible(n, proto):
    m = VM.protocol_mask(proto, "poisson", "parent-0", rep=0, n=n)
    assert m.shape == (n, n) and m.dtype == bool
    assert int(m.sum()) == int(proto[1:])                     # exact K, a boolean mask cannot hold duplicates
    el = VM.eligible("poisson", n)
    assert not (m & ~el).any()                                 # Dirichlet boundary never observed


def test_layout_depends_on_parent_and_replicate_only():
    a = VM.protocol_mask("U256", "poisson", "p", rep=0, n=128)
    assert np.array_equal(a, VM.protocol_mask("U256", "poisson", "p", rep=0, n=128))
    assert not np.array_equal(a, VM.protocol_mask("U256", "poisson", "q", rep=0, n=128))
    assert not np.array_equal(a, VM.protocol_mask("U256", "poisson", "p", rep=1, n=128))


@pytest.mark.parametrize("proto", ["L256", "L1024"])
def test_line_layouts_refused_on_64_grid(proto):
    with pytest.raises(ValueError):                            # refused, never silently replaced by another layout
        VM.protocol_mask(proto, "poisson", "p", rep=0, n=64)


def test_mxr2_components_and_balance():
    assert VM.MIXTURES["MXR2_K256"] == ("U256", "C256")
    pos = VM.mixture_positions({"rpb": [f"p{i}" for i in range(1024)]}, "poisson", 2)
    js = np.array([j % 2 for j in pos.values()])
    assert abs(int((js == 0).sum()) - int((js == 1).sum())) <= 1
    m0, i0 = VM.mixture_mask("MXR2_K256", "poisson", "p1", pos["p1"], 0, n=64)
    m1, i1 = VM.mixture_mask("MXR2_K256", "poisson", "p1", pos["p1"], 1, n=64)
    assert int(m0.sum()) == int(m1.sum()) == 256 and i0["component"] != i1["component"]


def _fields(n=300, d=64, r=8, seed=0):
    rng = np.random.default_rng(seed)
    basis = np.linalg.qr(rng.standard_normal((d, r)))[0]
    return (rng.standard_normal((n, r)) * np.linspace(3, 0.3, r)) @ basis.T + 0.5


def test_covariance_latent_and_kspace_agree_and_exact_in_span():
    X = _fields()
    cr = CovarianceReconstructor(rank=32).fit(X[:250])
    rng = np.random.default_rng(1)
    for t in X[250:260]:
        m = np.zeros(64, bool)
        m[rng.choice(64, 20, replace=False)] = True
        a = cr.predict(t[m], m, reg=1e-8, solve="latent")
        b = cr.predict(t[m], m, reg=1e-8, solve="kspace")
        assert np.linalg.norm(a - b) <= 1e-8 * np.linalg.norm(a)
        assert np.linalg.norm(a - t) <= 1e-5 * np.linalg.norm(t)     # fields in the training span, enough sensors


def test_covariance_is_affine_equivariant_and_selects_on_validation():
    X = _fields(seed=2)
    cr = CovarianceReconstructor(rank=32).fit(X[:250])
    cr2 = CovarianceReconstructor(rank=32).fit(2.0 * X[:250] + 0.5)      # affinely transformed training fields
    t = X[260]
    m = np.zeros(64, bool)
    m[::3] = True
    u1 = cr.predict(t[m], m, reg=1e-3)
    u2 = cr2.predict(2.0 * t[m] + 0.5, m, reg=1e-3)
    assert np.allclose(u2, 2.0 * u1 + 0.5, atol=1e-9)                    # relative regularisation keeps equivariance
    sel = cr.select_reg([(x, m) for x in X[250:280]], [1e-8, 1e-4, 1e-1])
    assert sel["reg"] in (1e-8, 1e-4, 1e-1) and sel["records"] == 30


def test_pod_basis_orthonormal():
    mu, U, s = pod_basis(_fields(seed=3), 16)
    assert np.allclose(U.T @ U, np.eye(U.shape[1]), atol=1e-8)


def test_dc_split_identity_and_hidden_definition():
    rng = np.random.default_rng(4)
    t = rng.standard_normal((5, 1, 8, 8))
    p = t + 0.1 * rng.standard_normal(t.shape)
    m = rng.random((5, 8, 8, 1)) > 0.6
    d = dc_split(p, t, m)
    assert np.allclose(d["raw"] ** 2, d["obs"] ** 2 + d["hid"] ** 2, rtol=1e-12)
    dc_pred = np.where(m.reshape(5, 1, 8, 8), t, p)              # data-consistent prediction
    assert np.allclose(dc_split(dc_pred, t, m)["raw"], d["hid"], rtol=1e-12)
    with pytest.raises(ValueError):
        dc_split(p, t, np.ones((5, 4, 4, 1)))


def test_paired_statistics_shared_resamples():
    ids = [f"r{i}" for i in range(200)]
    reg = {i: ("a" if k < 100 else "b") for k, i in enumerate(ids)}
    res = ParentResamples(ids, reg, boot=500, seed=0)
    assert res.regimes == {"a": 100, "b": 100}
    rng = np.random.default_rng(5)
    base = rng.random(200)
    better = res.vector(dict(zip(ids, base)))
    worse = res.vector(dict(zip(ids, base + 0.1)))
    out = paired_difference(res, worse, better)
    assert out["n"] == 200 and out["ci95"][0] > 0 and abs(out["mean_diff"] - 0.1) < 1e-12
    e = {(p, q): res.vector(dict(zip(ids, base + (0.0 if p == q else 0.2)))) for p in ("U", "C") for q in ("U", "C")}
    mo = matched_offpattern(e, ("U", "C"), res)
    assert mo["C_minus_D_ci95"][0] > 0.19 and mo["C_over_D"] > 1


def test_rpb_constants():
    from pdeobs.revision.external import rpb
    assert rpb.COUNTS == {"train": 1024, "validation": 128, "test": 256, "ood": 256}
    assert "L256" not in rpb.FEASIBLE_VIEWS and rpb.MIXTURE == "MXR2_K256"
    for v in rpb.FEASIBLE_VIEWS:
        assert int(VM.protocol_mask(v, "poisson", "rpb/poisson/in/00000", rep=0, n=64).sum()) == int(v[1:])


def test_cufft_compatible_irfft2_matches_native_on_hermitian_spectra():
    torch = pytest.importorskip("torch")
    from pdeobs.revision.fftcompat import irfft2_cufft
    x = torch.fft.rfft2(torch.randn(3, 2, 16, 12, generator=torch.Generator().manual_seed(0)), norm="ortho")
    ref = torch.fft.irfft2(x, s=(16, 12), norm="ortho")
    assert torch.allclose(irfft2_cufft(x, s=(16, 12), norm="ortho"), ref, atol=1e-6)


def test_cufft_compatible_irfft2_keeps_the_redundant_imaginary_parts():
    torch = pytest.importorskip("torch")
    from pdeobs.revision.fftcompat import irfft2_cufft
    h, w = 8, 8
    sign = torch.tensor([(-1.0) ** j for j in range(w)])
    x = torch.zeros(h, w // 2 + 1, dtype=torch.complex64)
    x[0, 0] = 1j                                               # Im of the DC bin: dropped on CPU, kept by large-batch cuFFT
    assert torch.allclose(torch.fft.irfft2(x, s=(h, w), norm="ortho"), torch.zeros(h, w), atol=1e-7)
    assert torch.allclose(irfft2_cufft(x, s=(h, w), norm="ortho"), (-sign / (h * w) ** 0.5).expand(h, w), atol=1e-7)
    x = torch.zeros(h, w // 2 + 1, dtype=torch.complex64)
    x[0, w // 2] = 1j                                          # Im of the Nyquist bin
    assert torch.allclose(irfft2_cufft(x, s=(h, w), norm="ortho"), torch.full((h, w), -1.0 / (h * w) ** 0.5), atol=1e-7)
    with pytest.raises(NotImplementedError):
        irfft2_cufft(x, s=(h, w), norm="backward")


def test_cufft_c2r_context_patches_cpu_calls_only_inside_the_block():
    torch = pytest.importorskip("torch")
    from pdeobs.revision.fftcompat import cufft_c2r_on_cpu, irfft2_cufft
    native = torch.fft.irfft2
    x = torch.zeros(4, 3, dtype=torch.complex64)
    x[0, 0] = 1j
    with cufft_c2r_on_cpu():
        assert torch.allclose(torch.fft.irfft2(x, s=(4, 4), norm="ortho"), irfft2_cufft(x, s=(4, 4), norm="ortho"))
    assert torch.fft.irfft2 is native


class _Records:
    """Minimal stand-in for a V2 split: records with a stored trajectory, addressed through base / indices."""

    def __init__(self, trajectories):
        from types import SimpleNamespace
        self.base = [SimpleNamespace(trajectory=t) for t in trajectories]
        self.indices = list(range(len(trajectories)))

    def __len__(self):
        return len(self.indices)


def test_train_scalar_stats_pools_every_training_value():
    from pdeobs.revision.data import train_scalar_stats
    a = np.arange(8, dtype=np.float64).reshape(2, 2, 2)
    b = np.full((2, 2, 2), 3.0)
    st = train_scalar_stats(_Records([a, b]), "poisson")          # static: the solution (last stored frame) only
    assert st["values"] == 8 and st["records"] == 2
    assert st["mu"] == pytest.approx(4.25) and st["sigma"] == pytest.approx(2.1875 ** 0.5)
    st = train_scalar_stats(_Records([a, b]), "navier_stokes")    # temporal: every stored frame of the t4 view
    assert st["values"] == 16
    assert st["mu"] == pytest.approx(3.25) and st["sigma"] == pytest.approx(2.6875 ** 0.5)


def _replay_module():
    import importlib.util
    import pathlib
    path = pathlib.Path(__file__).resolve().parents[1] / "reproduce" / "october" / "replay_inference.py"
    spec = importlib.util.spec_from_file_location("replay_inference_for_tests", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_recipe_overrides_parse_json_and_strings():
    parse = _replay_module().parse_overrides
    assert parse(["v4_norm=train_scalar", "lr=0.0003", "epochs=2"]) == {"v4_norm": "train_scalar", "lr": 0.0003, "epochs": 2}
    assert parse([]) == {}
    with pytest.raises(SystemExit):
        parse(["v4_norm"])


# ---------------------------------------------------------------- reproduce/october/score_method.py and rebuild_tables.py --latex
def _october_scripts():
    import importlib.util
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1] / "reproduce" / "october"
    mods = {}
    for name in ("replay_inference", "score_method", "rebuild_tables"):
        spec = importlib.util.spec_from_file_location(name, root / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        import sys
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        mods[name] = mod
    return mods


def test_score_method_recipe_and_predictor_parsing():
    sm = _october_scripts()["score_method"]
    assert sm.parse_dataset_id("v2_ns_ell009_ref128") == ("navier_stokes", "ell009")
    assert sm.parse_dataset_id("v2_poisson_rpb_ref128") == ("poisson", "rpb")
    rec = sm.recipe_for("v2_poisson_ell009_ref128", "nearest")
    assert rec["pde"] == "poisson" and rec["fill_input"] == "nearest" and rec["method"] == "fno"
    assert "fill_input" not in sm.recipe_for("v2_darcy_ell0025_ref128", None)
    assert rec["grid"] == 128
    assert sm.recipe_for("v2_poisson_rpb_ref128", None)["grid"] == 64          # the converted RPB view is 64^2
    assert sm.recipe_for("v2_poisson_ell009_ref128", None, 256)["grid"] == 256  # N8 resolution view
    assert sm.load_predictor("covariance") == "covariance"
    assert sm.load_predictor("json:dumps") is __import__("json").dumps
    import pytest
    with pytest.raises(SystemExit):
        sm.load_predictor("not-a-builtin")


def test_score_method_persistence_shapes():
    torch = __import__("pytest").importorskip("torch")
    sm = _october_scripts()["score_method"]
    static_in = torch.arange(2 * 1 * 4 * 4, dtype=torch.float32).reshape(2, 1, 4, 4)
    assert torch.equal(sm.persistence(static_in, torch.zeros(2, 1, 4, 4)), static_in)
    hist = torch.arange(2 * 1 * 1 * 4 * 4, dtype=torch.float32).reshape(2, 1, 1, 4, 4)
    pred = sm.persistence(hist, torch.zeros(2, 3, 1, 4, 4))
    assert pred.shape == (2, 3, 1, 4, 4)
    assert torch.equal(pred[:, 2], hist[:, -1])


def test_rebuild_tables_latex_writer(tmp_path):
    rt = _october_scripts()["rebuild_tables"]
    assert rt.tex_escape("a_b & 5% <x>") == "a\\_b \\& 5\\% $<$x$>$"
    rt.write_tex(str(tmp_path), "demo", ["PDE", "view"], [["Poisson", "U256"], ["NS", "C1024"]], "1. Demo", note="a note")
    text = (tmp_path / "demo.tex").read_text(encoding="utf-8")
    assert text.startswith("% 1. Demo") and "\\begin{tabular}{ll}" in text and "Poisson & U256 \\\\" in text and text.endswith("\\end{tabular}\n")


def test_rebuild_tables_reproduces_frozen_cells(tmp_path):
    """Path B: the committed run-level results rebuild the published table values (a few frozen cells; standard library)."""
    import csv
    import pathlib
    import subprocess
    import sys
    root = pathlib.Path(__file__).resolve().parents[1]
    res = root / "results" / "revision_october"
    if not (res / "runs_long.part-A.csv").exists():
        __import__("pytest").skip("results/revision_october is not part of this checkout")
    r = subprocess.run([sys.executable, str(root / "reproduce" / "october" / "rebuild_tables.py"), "--results", str(res), "--out", str(tmp_path), "--latex"],
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-800:]
    rows = {(x["PDE"], x["slice"], x["view"]): x for x in csv.DictReader(open(tmp_path / "static_calibration.csv", encoding="utf-8"))}
    cell = rows[("Poisson", "medium", "U256")]
    assert cell["covariance"] == "8.0e-05" and cell["best neural (envelope)"] == "0.020" and cell["model"] == "P4 voronoi_cnn U256"
    assert rows[("Darcy", "rough", "U1024")]["best neural (envelope)"] == "0.015"
    dens = {(x["model"], x["trained on"]): x for x in csv.DictReader(open(tmp_path / "ns_medium_density.csv", encoding="utf-8"))}
    assert dens[("fno", "MXR6_K256_K1024")]["256"] == "0.054" and dens[("fno", "MXR6_K256_K1024")]["1,024"] == "0.027"
    rpb = {(x["population"], x["method"], x["trained on"]): x for x in csv.DictReader(open(tmp_path / "rpb.csv", encoding="utf-8"))}
    assert rpb[("ID test", "fno", "MXR2_K256")]["U256"] == "0.376"
    assert sum(1 for _ in open(tmp_path / "failures.csv", encoding="utf-8")) - 1 == 53
    assert (tmp_path / "static_calibration.tex").read_text(encoding="utf-8").count("\\\\") >= 40
    assert "Inputs verified against the manifests" in (tmp_path / "tables.md").read_text(encoding="utf-8")


# ---------------------------------------------------------------- protocol enforcement of the October scripts (2026-10-28)
def test_score_method_context_hands_over_the_task_context_only():
    sm = _october_scripts()["score_method"]
    base = {"sample_id": "rev8k-v2/poisson/high/00001/ell009", "parent_id": "rev8k-v2/poisson/high/00001", "dataset_id": "v2_poisson_ell009_ref128",
            "split": "test", "pde": "poisson", "boundary": "dirichlet", "setting": "s", "state_representation": "scalar", "resolution": [128, 128],
            "T": 1, "mask_id": "v2:U256:rep0", "mask_parent_id": "rev8k-v2/poisson/high/00001", "mask_contract": "c", "observation_count": 256,
            "observation_ratio": 256 / 16384, "parameters": {"nu": 0.01}, "seed": 12345, "generation_seed": "v2/parent", "label_route": "r",
            "native_vs_reference_target": 1e-6, "regime": "high", "regime_sample_index": 1, "solver_version": "v", "tier": "rev8k-v2"}
    ctx = sm.predictor_context([base])[0]
    assert set(ctx) == set(sm.PREDICTOR_CONTEXT_KEYS) & set(base)       # the temporal keys exist on temporal records only
    assert ctx["sample_id"] == base["sample_id"] and ctx["observation_count"] == 256 and ctx["mask_id"] == "v2:U256:rep0"
    temporal = dict(base, T=4, view="t4", stored_frame_indices=[0, 5, 10, 15], stored_time_values=[0.0, 0.1, 0.2, 0.3])
    assert sm.predictor_context([temporal])[0]["stored_time_values"] == [0.0, 0.1, 0.2, 0.3]
    for hidden in ("parameters", "seed", "generation_seed", "label_route", "native_vs_reference_target", "regime", "regime_sample_index", "tier"):
        assert hidden not in ctx
    changed = dict(base, parameters={"nu": 0.5}, seed=999, generation_seed="other", native_vs_reference_target=0.3, regime="low", label_route="q")
    assert sm.predictor_context([changed]) == [ctx]           # the hidden fields change, the context does not
    assert sm.predictor_context(None) == []


def test_score_method_rejects_non_finite_and_wrong_shape_predictions():
    torch = pytest.importorskip("torch")
    sm = _october_scripts()["score_method"]
    target = torch.ones(2, 1, 4, 4)
    sm.validate_prediction(torch.zeros(2, 1, 4, 4), target, "ok")
    for bad in (float("nan"), float("inf"), -float("inf")):
        pred = torch.zeros(2, 1, 4, 4)
        pred[1, 0, 2, 3] = bad
        with pytest.raises(sm.ScoreInvalid, match="non-finite"):
            sm.validate_prediction(pred, target, "view")
    with pytest.raises(sm.ScoreInvalid, match="shape"):
        sm.validate_prediction(torch.zeros(2, 1, 4, 5), target)
    with pytest.raises(sm.ScoreInvalid, match="not a tensor"):
        sm.validate_prediction([[0.0]], target)
    with pytest.raises(sm.ScoreInvalid, match="floating"):
        sm.validate_prediction(torch.zeros(2, 1, 4, 4, dtype=torch.int64), target)
    with pytest.raises(sm.ScoreInvalid, match="target"):
        sm.validate_prediction(torch.zeros(2, 1, 4, 4), target * float("nan"))


def test_score_method_identity_checks_and_strict_json():
    import hashlib
    sm = _october_scripts()["score_method"]
    ids = ["b", "a", "c"]
    digest = hashlib.sha256("a\nb\nc".encode()).hexdigest()
    assert sm.check_identities(ids, 3, digest) == digest
    assert sm.check_identities(ids, 3, None) == digest
    with pytest.raises(sm.ScoreInvalid, match="repeated"):
        sm.check_identities(["a", "a", "b"], 3, None)
    with pytest.raises(sm.ScoreInvalid, match="expected"):
        sm.check_identities(ids, 4, None)
    with pytest.raises(sm.ScoreInvalid, match="receipt"):
        sm.check_identities(ids, 3, "0" * 64)
    assert '"mean": 0.5' in sm.dump_report({"views": {"U256__rep0": {"mean": 0.5}}})
    with pytest.raises(ValueError):
        sm.dump_report({"views": {"U256__rep0": {"mean": float("nan")}}})
    assert sm.EXIT_INVALID == 2


def test_replay_verdict_needs_every_view_finite_and_complete():
    rp = _replay_module()
    ok = {"replay_mean": 0.02, "stored_mean": 0.02, "records": 1800, "stored_records": 1800, "rel_diff": 0.0, "per_record": None}
    assert rp.decide({"U256__rep0": ok, "C256__rep0": dict(ok, rel_diff=5e-5)}, 1e-4)["status"] == "agrees"
    missing = rp.decide({"U256__rep0": ok, "C256__rep0": dict(ok, stored_mean=None, rel_diff=None)}, 1e-4)
    assert missing["status"] == "incomplete" and not missing["agrees"] and missing["missing_views"] == ["C256__rep0"]
    nan_last = rp.decide({"U256__rep0": ok, "C256__rep0": dict(ok, replay_mean=float("nan"), rel_diff=float("nan"))}, 1e-4)
    nan_first = rp.decide({"U256__rep0": dict(ok, replay_mean=float("nan"), rel_diff=float("nan")), "C256__rep0": ok}, 1e-4)
    assert nan_last["status"] == nan_first["status"] == "disagrees"
    assert rp.decide({"U256__rep0": dict(ok, stored_mean=None, rel_diff=None)}, 1e-4)["status"] == "incomplete"
    assert rp.decide({"U256__rep0": dict(ok, rel_diff=2e-4)}, 1e-4)["status"] == "disagrees"
    assert rp.decide({"U256__rep0": dict(ok, records=1799)}, 1e-4)["status"] == "incomplete"
    pr = {"compared": 1800, "missing": 0, "repeated_rows": 0, "non_finite": 0, "max_rel_diff": 2e-4, "mean_rel_diff": 1e-5}
    assert rp.decide({"U256__rep0": dict(ok, per_record=pr)}, 1e-4, 1e-3)["status"] == "agrees"
    assert rp.decide({"U256__rep0": dict(ok, per_record=dict(pr, max_rel_diff=2e-3))}, 1e-4, 1e-3)["status"] == "disagrees"
    assert rp.decide({"U256__rep0": dict(ok, per_record=dict(pr, missing=3))}, 1e-4, 1e-3)["status"] == "incomplete"
    assert rp.decide({"U256__rep0": dict(ok, per_record=dict(pr, repeated_rows=1800))}, 1e-4, 1e-3)["status"] == "incomplete"
    assert rp.decide({"U256__rep0": dict(ok, per_record=None)}, 1e-4, 1e-3)["status"] == "incomplete"
    assert rp.decide({"U256__rep0": dict(ok, per_record=dict(pr, non_finite=1, max_rel_diff=None))}, 1e-4, 1e-3)["status"] == "disagrees"
    assert rp.decide({"U256__rep0": dict(ok, per_record=pr)}, 1e-4)["status"] == "agrees"      # no per-record file: view means only


def test_replay_per_record_alignment(tmp_path):
    import gzip
    rp = _replay_module()
    p = tmp_path / "v2eval7.csv.gz"
    with gzip.open(p, "wt", encoding="utf-8", newline="") as fh:
        fh.write("run_id,view,sample_id,error,regime,obs,hid,hid_rel\n")
        fh.write("R__s1101,U256__rep0,id/1,0.10,high,,,\n")
        fh.write("R__s1101,U256__rep0,id/2,0.20,high,,,\n")
        fh.write("R__s1101,U256__rep0,id/2,0.25,high,,,\n")
        fh.write("OTHER,U256__rep0,id/1,0.99,high,,,\n")
    table, dups = rp.load_per_record(str(p), {"R__s1101"})
    assert table == {("R__s1101", "U256__rep0", "id/1"): 0.10, ("R__s1101", "U256__rep0", "id/2"): 0.20}
    assert dups[("R__s1101", "U256__rep0")] == 1
    cmp = rp.compare_per_record(table, dups, "R__s1101", "U256__rep0", ["id/1", "id/2", "id/3"], [0.1, 0.2002, 0.3])
    assert cmp["compared"] == 2 and cmp["missing"] == 1 and cmp["repeated_rows"] == 1 and cmp["non_finite"] == 0
    assert cmp["max_rel_diff"] == pytest.approx(0.001)
    q = tmp_path / "bad.csv"
    q.write_text("run_id,view,error\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        rp.load_per_record(str(q), {"R"})


def _tiny_results(tmp_path, runs_rows, manifest=True):
    import hashlib
    import json
    header = ("run_id,cohort,source,pde,dataset_id,test_dataset_id,grid,method,protocol,seed,parent_run_id,suite,view,n,mean,median,p90,max,valid,"
              "dc_raw,dc_obs,dc_hid,dc_hid_rel,run_status,final_train_loss,gpu_seconds,v4_norm,lr,batch_size,epochs,fill_input,param_channel")
    runs = header + "\n" + "".join(runs_rows)
    cls = "task,dataset_id,view,method,mean,median,skill_mu,selection\n"
    (tmp_path / "runs_long.part-A.csv").write_text(runs, encoding="utf-8", newline="")
    (tmp_path / "classical_long.part-A.csv").write_text(cls, encoding="utf-8", newline="")
    if manifest:
        m = {"schema": "pdeobs-october-results/v1", "part": "A", "committed_files": {
            "runs_long.part-A.csv": {"sha256": hashlib.sha256(runs.encode()).hexdigest(), "rows": len(runs_rows)},
            "classical_long.part-A.csv": {"sha256": hashlib.sha256(cls.encode()).hexdigest(), "rows": 0}}}
        (tmp_path / "manifest.part-A.json").write_text(json.dumps(m), encoding="utf-8")


def _row(run, view, mean, seed="1101", parent=""):
    return (f"{run},P4,native,poisson,v2_poisson_ell009_ref128,,128,fno,U256,{seed},{parent},v2eval7,{view},1800,{mean},{mean},{mean},{mean},True,"
            f",,,,completed,,,,,,,,\n")


def test_rebuild_tables_verifies_manifests_and_refuses_conflicting_rows(tmp_path):
    rt = _october_scripts()["rebuild_tables"]
    run = "P4__poisson__ell009__fno__U256__s1101"
    _tiny_results(tmp_path, [_row(run, "U256__rep0", 0.1), _row(run, "C256__rep0", 0.2)])
    checked = rt.verify_manifests(str(tmp_path))
    assert checked["runs_long.part-A.csv"]["rows"] == 2 and checked["classical_long.part-A.csv"]["rows"] == 0
    runs, cls = rt.load(str(tmp_path))
    assert len(runs) == 2 and cls == []
    assert rt.blocks(runs, ("P4",), "v2eval7")[("P4", "poisson", "ell009", "fno", "U256")]["U256__rep0"] == {"1101": 0.1}
    # an identical repeated row is dropped; a row with different values for the same key is refused
    _tiny_results(tmp_path, [_row(run, "U256__rep0", 0.1), _row(run, "U256__rep0", 0.1)])
    assert len(rt.load(str(tmp_path))[0]) == 1
    _tiny_results(tmp_path, [_row(run, "U256__rep0", 0.1), _row(run, "U256__rep0", 0.9)])
    with pytest.raises(SystemExit, match="different values"):
        rt.load(str(tmp_path))
    # two results of the same cell (an evaluation row of the run beside the run's own row) with different values are refused
    _tiny_results(tmp_path, [_row(run, "U256__rep0", 0.1), _row("E2ev__v2eval7__" + run, "U256__rep0", 0.3, parent=run)])
    with pytest.raises(SystemExit, match="feed one table cell"):
        rt.blocks(rt.load(str(tmp_path))[0], ("P4",), "v2eval7")
    # a valid block with a non-finite mean is refused
    _tiny_results(tmp_path, [_row(run, "U256__rep0", "nan")])
    with pytest.raises(SystemExit, match="non-finite"):
        rt.load(str(tmp_path))
    # a file that differs from its manifest, a result file without a manifest entry, and a missing manifest are refused
    _tiny_results(tmp_path, [_row(run, "U256__rep0", 0.1)])
    text = (tmp_path / "runs_long.part-A.csv").read_text(encoding="utf-8")
    (tmp_path / "runs_long.part-A.csv").write_text(text.replace(",0.1,", ",0.2,", 1), encoding="utf-8", newline="")
    with pytest.raises(SystemExit, match="differ"):
        rt.verify_manifests(str(tmp_path))
    _tiny_results(tmp_path, [_row(run, "U256__rep0", 0.1)])
    (tmp_path / "runs_long.part-B.csv").write_text("run_id\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="without a manifest entry"):
        rt.verify_manifests(str(tmp_path))
    (tmp_path / "runs_long.part-B.csv").unlink()
    (tmp_path / "manifest.part-A.json").unlink()
    with pytest.raises(SystemExit, match="no manifest"):
        rt.verify_manifests(str(tmp_path))


def test_hf_inventory_listing_and_reconciliation_logic(tmp_path):
    import hashlib
    import importlib.util
    import json
    import pathlib
    path = pathlib.Path(__file__).resolve().parents[1] / "reproduce" / "october" / "hf_inventory.py"
    spec = importlib.util.spec_from_file_location("hf_inventory_for_tests", path)
    hi = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hi)
    A, B, C, M, Q, W, X = "a" * 64, "b" * 64, "c" * 64, "d" * 64, "e" * 64, "f" * 64, "0" * 64
    pages = {
        "https://huggingface.co/api/datasets/x/y/tree/deadbee?recursive=true&expand=true": (
            [{"type": "directory", "path": "data"},
             {"type": "file", "path": "data/a/s0.h5", "size": 10, "lfs": {"oid": A, "size": 10}},
             {"type": "file", "path": "data/a/s0.manifest.json", "size": 5, "oid": "1234"}],
            {"link": '<https://huggingface.co/page2>; rel="next"'}),
        "https://huggingface.co/page2": ([{"type": "file", "path": "october2026/v2/view1/high/test/p1.h5", "size": 20, "lfs": {"oid": B, "size": 20}}], {}),
        "https://huggingface.co/api/datasets/x/y/revision/main": ({"sha": "deadbeef0", "lastModified": "t"}, {}),
    }

    def get(url, timeout=0, retries=0):
        body, headers = pages[url]
        return json.dumps(body).encode(), headers

    assert hi.resolve_revision("dataset", "x/y", "main", get)["commit"] == "deadbeef0"
    files, n = hi.list_tree("dataset", "x/y", "deadbee", get)
    assert n == 2 and [f["path"] for f in files] == ["data/a/s0.h5", "data/a/s0.manifest.json", "october2026/v2/view1/high/test/p1.h5"]
    assert files[0]["sha256"] == A and files[1]["blob"] == "1234" and files[1]["sha256"] is None
    csv_text = hi.inventory_csv(files).decode()
    assert csv_text.splitlines()[0] == "path,size,lfs,sha256,blob" and csv_text.count("\n") == 4

    def f(path, size, sha=None):
        return {"path": path, "size": size, "lfs": sha is not None, "sha256": sha, "blob": None if sha else "blob"}

    tree = [f("data/a/s0.h5", 10, A), f("data/a/s0.manifest.json", 5), f("summary.json", 9, C),
            f("october2026/v2/view1/high/test/p1.h5", 20, B), f("october2026/v2/view1/high/test/p1.sha256", 3),
            f("october2026/results/part-B/manifest.part-B.json", 2), f("october2026/results/part-B/classical_long.csv.gz", 8, C),
            f("october2026/results/part-B/update/export_manifest.json", 2), f("october2026/results/part-B/update/classical_long.csv.gz", 8, C),
            f("models/cluster-A/pde/m/view/checkpoints/last.pt", 9, W), f("models/cluster-A/pde/m/view/completion.json", 7),
            f("october2026/models/V4C/R__s1/weights.pt", 11, X), f("october2026/models/V4C/R__s1/model.json", 4),
            f("october2026/retrains/present/weights.pt", 11, X), f("README.md", 1)]
    shard_digest = hashlib.sha256(f"v2/view1/high/test/p1.h5 {B}".encode()).hexdigest()
    committed = b"run_id,x\nr1,1\n"
    docs = {
        "release_manifest.json": {"schema_version": 1, "files": [
            {"path": "a/s0.h5", "sha256": A, "size": 10, "url": "https://h/resolve/main/data/a/s0.h5"},
            {"path": "a/s0.manifest.json", "sha256": M, "size": 5, "url": "https://h/resolve/main/data/a/s0.manifest.json"}]},
        "scrub-manifest.json": {"shards": {"a/s0.h5": {"bytes": 10, "scrubbed_sha256": A}}, "summary_files": {"summary.json": {"scrubbed_sha256": C}}},
        "october2026/data_manifest.part-1.json": {
            "format": "pdeobs-october2026-data-manifest/1",
            "views": {"v2/view1": {"files": 2, "bytes": 23, "shards": 1, "types": {"h5": 1, "sha256": 1}, "shard_digest": shard_digest}},
            "shards": [{"path": "v2/view1/high/test/p1.h5", "bytes": 20, "sha256": B}]},
        "october2026/results/part-B/manifest.part-B.json": {
            "schema": "pdeobs-october-results/v1",
            "committed_files": {"runs_long.part-B.csv": {"sha256": hashlib.sha256(committed).hexdigest(), "rows": 1}},
            "export_files": {"classical_long.csv.gz": {"sha256": C, "bytes": 8}, "runs_long.csv.gz": {"sha256": Q, "bytes": 8}}, "excluded": ["2 rows"]},
        "october2026/results/part-B/update/export_manifest.json": {
            "schema": "pdeobs-october-export/v1", "published_under": "october2026/results/part-B/update",
            "files": {"classical_long.csv.gz": {"sha256": C, "bytes": 8}, "runs_long.csv.gz": {"sha256": Q, "bytes": 8}},
            "published_files": ["classical_long.csv.gz"], "not_published": ["runs_long.csv.gz"]},
        "models_manifest.cluster-A.json": {
            "schema_version": "pdeobs-anonymized-release/v3", "host_label": "cluster-A",
            "rows": [{"release_path": "pde/m/view", "checkpoint_sha256": W,
                      "files": {"checkpoints/last.pt": {"bytes": 9, "sha256": W}, "completion.json": {"bytes": 7, "sha256": Q}}}]},
        "october2026/models_manifest.part-1.json": {
            "format": "pdeobs-october2026-models-manifest/1", "n_models": 1,
            "models": [{"run_id": "R__s1", "cohort": "V4C", "recipe_sha256": M, "weights_sha256": X, "weights_bytes": 11,
                        "files": ["weights.pt", "model.json", "history.json"]}],
            "not_in_this_part": ["R__s2"]},
        "october2026/retrains/manifest.json": {
            "format": "pdeobs-october2026-retrains-manifest/1", "models": [],
            "not_published": [{"name": "failed", "status": "failed"}, {"name": "present", "status": "failed"}]},
    }
    manifests = {p: hi.manifest_entries(d, p) for p, d in docs.items()}
    assert [m["schema"] for m in manifests.values()] == ["release", "scrub_dataset", "data", "results", "export", "cluster_models", "october_models", "retrains"]
    code_root = tmp_path / "checkout"
    (code_root / "results" / "revision_october").mkdir(parents=True)
    (code_root / "results" / "revision_october" / "runs_long.part-B.csv").write_bytes(committed)
    rec = hi.reconcile(tree, manifests, {"data/a/s0.manifest.json": M, "models/cluster-A/pde/m/view/completion.json": "9" * 64}, str(code_root))
    r = rec["manifests"]
    assert r["release_manifest.json"]["digest_verified"] == 2 and r["release_manifest.json"]["missing_count"] == 0
    assert r["scrub-manifest.json"]["present"] == 2 and r["scrub-manifest.json"]["digest_verified"] == 2
    v = r["october2026/data_manifest.part-1.json"]["views"]["v2/view1"]
    assert all(x["agree"] for x in v.values()) and r["october2026/data_manifest.part-1.json"]["views_disagreeing"] == []
    rb = r["october2026/results/part-B/manifest.part-B.json"]
    assert rb["present"] == 1 and rb["missing_count"] == 0 and rb["not_published_count"] == 1     # declared by the export manifest below it
    assert rb["not_published"][0]["declared_by"].endswith("export_manifest.json")
    assert rb["code_files"]["runs_long.part-B.csv"]["status"] == "verified" and rb["code_files"]["runs_long.part-B.csv"]["checkout"]["rows"] == 1
    ex = r["october2026/results/part-B/update/export_manifest.json"]
    assert ex["present"] == 1 and ex["absent_ok"] == 1 and ex["absent_present_count"] == 0
    cl = r["models_manifest.cluster-A.json"]
    assert cl["digest_verified"] == 1 and cl["digest_mismatch_count"] == 1 and cl["digest_mismatch"][0]["path"].endswith("completion.json")
    om = r["october2026/models_manifest.part-1.json"]
    assert om["present"] == 2 and om["missing"] == ["october2026/models/V4C/R__s1/history.json"]
    assert om["digest_verified"] == 1 and om["listed_without_digest"] == 1 and om["not_in_this_part"]["missing_everywhere"] == ["R__s2"]
    rt = r["october2026/retrains/manifest.json"]
    assert rt["absent_ok"] == 1 and rt["absent_present_count"] == 1 and rt["absent_present"][0]["entry"] == "present"
    assert rec["covered_files"] == 10 and rec["uncovered"]["sidecar"]["files"] == 1 and rec["uncovered"]["manifest"]["files"] == 2
    assert rec["uncovered_other_paths"] == ["october2026/retrains/present/weights.pt"] and rec["uncovered"]["card_or_doc"]["files"] == 1
    p = hi.problems(rec)
    assert set(p) == {"models_manifest.cluster-A.json", "october2026/models_manifest.part-1.json", "october2026/retrains/manifest.json"}
    assert p["october2026/models_manifest.part-1.json"] == {"missing": 1, "run_ids_missing_everywhere": 1}
    # a regular file whose digest was not fetched is unverified, never a match; a size that differs is a mismatch
    rec2 = hi.reconcile(tree, {"release_manifest.json": manifests["release_manifest.json"]}, {})
    assert rec2["manifests"]["release_manifest.json"]["digest_unverified"] == ["data/a/s0.manifest.json"]
    wrong = hi.manifest_entries({"schema_version": 1, "files": [{"path": "a/s0.h5", "sha256": A, "size": 11}]}, "release_manifest.json")
    assert hi.reconcile(tree, {"release_manifest.json": wrong})["manifests"]["release_manifest.json"]["size_mismatch_count"] == 1
    # a committed file the checkout holds in a later state: its own manifest records that state
    entry = {"name": "runs_long.part-B.csv", "sha256": "1" * 64, "rows": 7}
    assert hi.check_code_file(str(code_root), entry, None)["status"] == "differs"
    cur = {"committed_files": {"runs_long.part-B.csv": {"sha256": hashlib.sha256(committed).hexdigest(), "rows": 1}}, "update_x": {}}
    cf = hi.check_code_file(str(code_root), entry, cur)
    assert cf["status"].startswith("superseded") and cf["checkout_manifest"]["updates"] == ["update_x"]
    assert hi.manifest_entries({"rewritten_files": ["models/x/y.json"]}, "scrub-manifest.json")["schema"] == "scrub_model"
    assert hi.classify("README.md") == "card_or_doc" and hi.classify("data/x.sha256") == "sidecar" and hi.classify("october2026/v2/a.h5") == "other"
    assert hi.classify("october2026/retrains/manifest.json") == "manifest" and hi.classify("data/a/shard_00000.manifest.json") == "sidecar"
