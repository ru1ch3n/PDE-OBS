"""Fail-closed audits for factorial experiment coverage."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from itertools import product
from typing import Any

FACTOR_FIELDS = ("pde", "boundary", "setting", "regime")


class FactorCoverageError(ValueError):
    """Raised when selected records do not cover the declared Cartesian product."""

    def __init__(self, message: str, receipt: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.receipt = dict(receipt)


def audit_factor_coverage(
    metadata: Sequence[Mapping[str, Any]],
    required: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    """Require every declared factor value and every Cartesian-product stratum."""

    unknown = set(required) - set(FACTOR_FIELDS)
    if unknown:
        raise ValueError(f"unknown factor coverage fields: {sorted(unknown)}")
    fields = tuple(field for field in FACTOR_FIELDS if field in required)
    if not fields:
        raise ValueError("required factor coverage must declare at least one factor")
    normalized: dict[str, tuple[str, ...]] = {}
    for field in fields:
        raw = required[field]
        if isinstance(raw, (str, bytes)):
            raise ValueError(f"required factor {field} must be a non-string sequence")
        values = tuple(str(value) for value in raw)
        if not values or len(values) != len(set(values)):
            raise ValueError(f"required factor {field} must be non-empty and unique")
        normalized[field] = values

    observed = Counter(tuple(str(row.get(field, "")) for field in fields) for row in metadata)
    expected = tuple(product(*(normalized[field] for field in fields)))
    missing = tuple(combination for combination in expected if observed[combination] == 0)
    unexpected = tuple(sorted(combination for combination in observed if combination not in expected))
    expected_counts = [observed[combination] for combination in expected]
    factor_counts = {
        field: {
            value: sum(1 for row in metadata if str(row.get(field, "")) == value)
            for value in normalized[field]
        }
        for field in fields
    }
    receipt: dict[str, Any] = {
        "schema_version": "pdeobs.factor-coverage/v1",
        "status": "passed" if not missing and not unexpected else "failed",
        "sample_count": len(metadata),
        "fields": list(fields),
        "required": {field: list(normalized[field]) for field in fields},
        "expected_strata": len(expected),
        "observed_expected_strata": len(expected) - len(missing),
        "minimum_samples_per_expected_stratum": min(expected_counts, default=0),
        "maximum_samples_per_expected_stratum": max(expected_counts, default=0),
        "factor_counts": factor_counts,
        "missing_strata": [dict(zip(fields, values, strict=True)) for values in missing],
        "unexpected_strata": [
            dict(zip(fields, values, strict=True)) for values in unexpected
        ],
    }
    if missing or unexpected:
        raise FactorCoverageError(
            f"factor coverage failed: {len(missing)} missing and "
            f"{len(unexpected)} unexpected strata",
            receipt,
        )
    return receipt


__all__ = ["FACTOR_FIELDS", "FactorCoverageError", "audit_factor_coverage"]
