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
