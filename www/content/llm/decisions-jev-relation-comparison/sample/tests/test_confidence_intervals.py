"""Offline tests for reference Wilson intervals on all-record accuracy only."""
import json
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import build_comparison as comparison


class WilsonIntervalTests(unittest.TestCase):
    def test_published_four_intervals(self):
        expected = {
            1220: (0.841350787227958, 0.8774353992767379),
            1224: (0.8443170958986999, 0.8800955969582831),
            1318: (0.9149595750124766, 0.9416760171232568),
            1326: (0.9210874380192129, 0.9468011668210949),
        }
        self.assertEqual(comparison.WILSON_Z_95, 1.95996398454)
        for correct, bounds in expected.items():
            with self.subTest(correct=correct):
                actual = comparison.wilson_interval(correct, 1418)
                for got, want in zip(actual, bounds):
                    self.assertAlmostEqual(got, want, places=13)

    def test_bounds_solve_score_test_equation(self):
        # Independent characterization: inversion of the binomial score test.
        for correct, total in ((20, 100), (1220, 1418), (1326, 1418)):
            p = correct / total
            for bound in comparison.wilson_interval(correct, total):
                score = (p - bound) / math.sqrt(bound * (1 - bound) / total)
                self.assertAlmostEqual(abs(score), comparison.WILSON_Z_95, places=10)

    def test_zero_and_all_correct_stay_in_unit_interval(self):
        z2 = comparison.WILSON_Z_95 ** 2
        low, high = comparison.wilson_interval(0, 10)
        self.assertAlmostEqual(low, 0)
        self.assertAlmostEqual(high, z2 / (10 + z2))
        low, high = comparison.wilson_interval(10, 10)
        self.assertAlmostEqual(low, 10 / (10 + z2))
        self.assertAlmostEqual(high, 1)

    def test_complement_symmetry(self):
        low, high = comparison.wilson_interval(8, 20)
        other_low, other_high = comparison.wilson_interval(12, 20)
        self.assertAlmostEqual(low, 1 - other_high)
        self.assertAlmostEqual(high, 1 - other_low)

    def test_refusals_remain_in_denominator(self):
        data = json.loads((ROOT / "aggregate-results.json").read_text())
        for language in data["languages"].values():
            for provider in ("decisions", "jev_historical"):
                result = language[provider]
                self.assertEqual(result["records"], 1418)
                self.assertEqual(comparison.accuracy_interval(result),
                                 comparison.wilson_interval(result["correct"], 1418))
                self.assertAlmostEqual(result["correct"] / result["records"],
                                       result["accuracy_all_records"])
            result = language["decisions"]
            self.assertNotEqual(comparison.accuracy_interval(result),
                                comparison.wilson_interval(result["correct"], result["valid_choices"]))

    def test_invalid_inputs_are_rejected(self):
        for correct, total in ((0, 0), (-1, 10), (11, 10), (1.5, 10), (1, 10.5), (True, 10)):
            with self.subTest(correct=correct, total=total), self.assertRaises(ValueError):
                comparison.wilson_interval(correct, total)
        for z in (0, -1, float("nan"), float("inf")):
            with self.subTest(z=z), self.assertRaises(ValueError):
                comparison.wilson_interval(5, 10, z)


if __name__ == "__main__":
    unittest.main()
