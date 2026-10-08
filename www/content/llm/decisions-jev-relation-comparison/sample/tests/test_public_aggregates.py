"""Consistency tests for the published, privacy-filtered aggregate report."""
from datetime import datetime
import json
from pathlib import Path
import unittest

REPORT = Path(__file__).resolve().parents[1] / "aggregate-results.json"


class PublicAggregateTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads(REPORT.read_text(encoding="utf-8"))

    def test_primary_conditional_and_coverage_denominators(self):
        for row in self.data["languages"].values():
            d = row["decisions"]
            self.assertEqual(d["records"], 1418)
            self.assertAlmostEqual(d["accuracy_all_records"], d["correct"] / 1418)
            self.assertAlmostEqual(d["accuracy_valid_choices"], d["correct"] / d["valid_choices"])
            self.assertAlmostEqual(row["coverage"], d["valid_choices"] / 1418)
            self.assertEqual(sum(d["status_counts"].values()), 1418)

    def test_all_31_refusals_remain_visible(self):
        self.assertEqual(self.data["languages"]["ja"]["decisions"]["status_counts"]["refusal"], 4)
        self.assertEqual(self.data["languages"]["en"]["decisions"]["status_counts"]["refusal"], 27)
        self.assertTrue(self.data["diagnostic_excluded_from_benchmark"])

    def test_full_paired_counts_match_both_correct_totals(self):
        for row in self.data["languages"].values():
            p = row["paired_all_records"]
            self.assertEqual(sum(p.values()), 1418)
            self.assertEqual(p["both_correct"] + p["decisions_only_correct"], row["decisions"]["correct"])
            self.assertEqual(p["both_correct"] + p["jev_only_correct"], row["jev_historical"]["correct"])

    def test_standard_and_hypothetical_premium_costs(self):
        tokens = sum(r["cost"]["metered_input_tokens"] for r in self.data["languages"].values())
        self.assertEqual(tokens, self.data["decisions_total_input_tokens"])
        self.assertAlmostEqual(tokens * .1 / 1e6, self.data["decisions_total_standard_estimate_usd"])
        self.assertAlmostEqual(tokens * .11 / 1e6, self.data["decisions_total_hypothetical_premium_estimate_usd"])
        self.assertFalse(any(r["cost"]["invoice_verified"] for r in self.data["languages"].values()))
        self.assertFalse(any(r["cost"]["premium_applicability_verified"] for r in self.data["languages"].values()))

    def test_wall_time_matches_recorded_timestamps(self):
        seconds = (datetime.fromisoformat(self.data["overall_finished_at_utc"]) -
                   datetime.fromisoformat(self.data["overall_started_at_utc"])).total_seconds()
        self.assertAlmostEqual(seconds, self.data["total_wall_clock_seconds"])

    def test_privacy_sensitive_raw_fields_are_absent(self):
        forbidden = {"api_key", "request_id", "authorization", "subject", "object", "articles", "request_sha256"}
        def inspect(value):
            if isinstance(value, dict):
                self.assertFalse(forbidden.intersection(k.lower() for k in value))
                for child in value.values(): inspect(child)
            elif isinstance(value, list):
                for child in value: inspect(child)
        inspect(self.data)


if __name__ == "__main__":
    unittest.main()
