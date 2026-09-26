"""Run these tests with the declared base dependency minima as well as current versions."""
import numpy as np
import pytest

from pdeobs.pdes import generate_sample
from pdeobs.pdes import numerics


@pytest.mark.parametrize('family,solver', [('darcy', 'cg'), ('poisson', 'cg'), ('helmholtz', 'minres')])
def test_elliptic_public_entry_uses_supported_rtol_api(monkeypatch, family, solver):
    original = getattr(numerics.spla, solver)
    calls = []

    def observed(*args, **kwargs):
        assert 'rtol' in kwargs
        assert 'tol' not in kwargs
        calls.append(kwargs['rtol'])
        return original(*args, **kwargs)

    monkeypatch.setattr(numerics.spla, solver, observed)
    result = generate_sample(family, boundary='dirichlet', setting='smooth_grf',
                             regime='low', resolution=12, seed=17)
    assert calls, f'{family} did not exercise {solver}'
    assert np.isfinite(result.trajectory).all()
    assert result.trajectory.shape == (1, 12, 12, 1)
