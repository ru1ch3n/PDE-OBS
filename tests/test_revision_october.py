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
