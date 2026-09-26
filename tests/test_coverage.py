from itertools import product

import pytest

from pdeobs.coverage import FactorCoverageError, audit_factor_coverage


def _metadata():
    return [
        {"pde": pde, "boundary": boundary, "setting": setting, "regime": regime}
        for pde, boundary, setting, regime in product(
            ("heat",),
            ("dirichlet", "neumann"),
            ("smooth", "rough"),
            ("low", "high"),
        )
    ]


def test_factor_coverage_requires_complete_cartesian_product():
    required = {
        "pde": ["heat"],
        "boundary": ["dirichlet", "neumann"],
        "setting": ["smooth", "rough"],
        "regime": ["low", "high"],
    }
    receipt = audit_factor_coverage(_metadata(), required)
    assert receipt["status"] == "passed"
    assert receipt["expected_strata"] == 8
    assert receipt["minimum_samples_per_expected_stratum"] == 1

    with pytest.raises(FactorCoverageError) as captured:
        audit_factor_coverage(_metadata()[:-1], required)
    assert captured.value.receipt["status"] == "failed"
    assert len(captured.value.receipt["missing_strata"]) == 1


def test_factor_coverage_rejects_unexpected_factor_value():
    rows = _metadata()
    rows.append({**rows[0], "boundary": "periodic"})
    with pytest.raises(FactorCoverageError) as captured:
        audit_factor_coverage(
            rows,
            {
                "pde": ["heat"],
                "boundary": ["dirichlet", "neumann"],
                "setting": ["smooth", "rough"],
                "regime": ["low", "high"],
            },
        )
    assert captured.value.receipt["unexpected_strata"] == [
        {
            "pde": "heat",
            "boundary": "periodic",
            "setting": "smooth",
            "regime": "low",
        }
    ]
