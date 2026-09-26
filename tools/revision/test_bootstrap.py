"""Acceptance tests for the paired test-identity bootstrap (synthetic arrays, CPU)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bootstrap as bs  # noqa: E402
from common import METHODS, VIEWS, SourceError  # noqa: E402


def groups_for(n: int) -> dict[str, list[int]]:
    thirds = np.array_split(np.arange(n), 3)
    return {"low": thirds[0].tolist(), "medium": thirds[1].tolist(), "high": thirds[2].tolist()}


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(3)
        self.n = 60
        self.E = np.abs(self.rng.normal(0.3, 0.1, size=(len(METHODS), len(VIEWS), len(VIEWS), self.n)))
        self.groups = groups_for(self.n)

    def test_weights_are_stratified_and_sum_to_n(self):
        w = bs.resample_weights(self.groups, self.n, 50, 1)
        self.assertEqual(w.shape, (50, self.n))
        for regime, members in self.groups.items():
            self.assertTrue(np.all(w[:, members].sum(axis=1) == len(members)))

    def test_identical_errors_give_zero_paired_differences(self):
        E = self.E.copy()
        E[:, :, :, :] = E[:, 0:1, 0:1, :]          # every cell equals the same per-identity errors
        M = bs.cell_means(E, bs.resample_weights(self.groups, self.n, 200, 5))
        s = bs.pair_statistics(M)
        self.assertTrue(np.allclose(s["C_minus_D"], 0.0))
        self.assertTrue(np.allclose(s["transfer_diff"], 0.0))
        self.assertTrue(np.allclose(s["C_over_D"], 1.0))

    def test_simultaneous_identity_reordering_changes_nothing(self):
        perm = self.rng.permutation(self.n)
        E2 = self.E[..., perm]
        groups2 = {k: [int(np.where(perm == i)[0][0]) for i in v] for k, v in self.groups.items()}
        w1 = bs.resample_weights(self.groups, self.n, 300, 11)
        w2 = bs.resample_weights(groups2, self.n, 300, 11)
        # with the same seed the draws index the same physical identities under the permutation
        s1 = bs.pair_statistics(bs.cell_means(self.E, w1))
        s2 = bs.pair_statistics(bs.cell_means(E2, w2))
        self.assertTrue(np.allclose(s1["C_over_D"], s2["C_over_D"]))

    def test_fixed_seed_reproduces(self):
        a = bs.resample_weights(self.groups, self.n, 100, 42)
        b = bs.resample_weights(self.groups, self.n, 100, 42)
        self.assertTrue(np.array_equal(a, b))
        self.assertFalse(np.array_equal(a, bs.resample_weights(self.groups, self.n, 100, 43)))

    def test_ratio_of_means_not_mean_of_ratios(self):
        M = bs.cell_means(self.E, bs.resample_weights(self.groups, self.n, 20, 2))
        s = bs.pair_statistics(M)
        self.assertTrue(np.allclose(s["C_over_D"], s["C"] / s["D"]))

    def test_zero_denominator_is_flagged(self):
        samples = np.array([1.0, 2.0, np.inf, 0.5])
        row = bs.percentile_row("C_over_D", "pair", 1.0, samples, replicates=4, seed=0, n=3, groups="x")
        self.assertIn("zero denominator", row["denominator_flag"])

    def test_different_identity_sets_fail(self):
        with self.assertRaises(SourceError):
            bs.resample_weights({"low": [0, 1], "medium": [2], "high": [3]}, 5, 3, 0)


if __name__ == "__main__":
    unittest.main()
