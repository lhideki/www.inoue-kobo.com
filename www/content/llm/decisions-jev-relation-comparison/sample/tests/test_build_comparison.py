"""Offline report-generation tests using entirely synthetic saved records."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import build_comparison as comparison


class GeneratedReportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.data_dir = Path(temporary.name) / "inputs"
        self.result_dir = Path(temporary.name) / "results"
        self.data_dir.mkdir()
        self.result_dir.mkdir()
        self.relations = [{"id": rid,
            "labels": {lang: f"{lang} relation {rid}" for lang in ("ja", "en")},
            "descriptions": {lang: f"Synthetic description {rid}" for lang in ("ja", "en")}}
            for rid in ("P1", "P2")]
        self.rows = [{"id": str(i), "expected_relation": {"id": rid},
            "subject": {"id": f"Q{i // 2}", "ja": {"label": "架空の主語"}, "en": {"label": "Synthetic subject"}},
            "object": {"ja": {"label": "架空の目的語"}, "en": {"label": "Synthetic object"}},
            "articles": {"ja": {"text": "架空の評価用文章です。"}, "en": {"text": "Synthetic evaluation text."}}}
            for i, rid in enumerate(("P1", "P1", "P2"))]
        data, choices = self.data_dir / "dataset.jsonl", self.data_dir / "relation_choices.json"
        self.write_jsonl(data, self.rows)
        choices.write_text(json.dumps({"relations": self.relations}), encoding="utf-8")
        self.dataset_hash = comparison.evaluation.sha256(data)
        self.choices_hash = comparison.evaluation.sha256(choices)
        for language, predictions in (("ja", ("P1", None, "P2")), ("en", ("P2", "P1", None))):
            decisions, jev = [], []
            for i, (row, prediction, jev_prediction) in enumerate(zip(self.rows, predictions, ("P1", "P2", "P2")), 1):
                expected = {"id": row["expected_relation"]["id"]}
                decisions.append({"id": row["id"], "expected": expected, "language": language,
                    "request_sha256": hashlib.sha256(comparison.evaluation.encoded_payload(row, self.relations, language)).hexdigest(),
                    "model_returned": comparison.evaluation.MODEL,
                    "status": "choice" if prediction else "refusal",
                    "predicted": f"{language} relation {prediction}" if prediction else None,
                    "elapsed_ms": i * 10, "usage": {"input_tokens": i * (10 if language == "ja" else 100)}})
                jev.append({"id": row["id"], "expected": expected,
                    "predicted": f"{language} relation {jev_prediction}", "elapsed_ms": i * 20})
            result = self.result_dir / f"decisions-{language}.jsonl"
            self.write_jsonl(result, decisions)
            self.write_jsonl(self.data_dir / f"jev-results-{language}.jsonl", jev)
            result.with_suffix(".manifest.json").write_text(json.dumps({
                "completed": True, "planned_records": 3, "dataset_sha256": self.dataset_hash,
                "choices_sha256": self.choices_hash, "automatic_retries": 0}), encoding="utf-8")

    @staticmethod
    def write_jsonl(path, records):
        path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

    def generate_offline(self):
        with patch.object(comparison.evaluation.http.client, "HTTPSConnection", side_effect=AssertionError("Network")), \
             patch.object(comparison.evaluation.os.environ, "get", side_effect=AssertionError("Environment read")):
            return comparison.report(self.data_dir, self.result_dir)

    def test_generated_cost_schema_matches_published_aggregate(self):
        report = self.generate_offline()
        published = json.loads((ROOT / "aggregate-results.json").read_text(encoding="utf-8"))
        for language, tokens in (("ja", 60), ("en", 600)):
            with self.subTest(language=language):
                cost = report["languages"][language]["cost"]
                self.assertEqual(set(cost), set(published["languages"][language]["cost"]))
                self.assertEqual(cost["metered_input_tokens"], tokens)
                self.assertAlmostEqual(cost["standard_rate_estimate_usd"], tokens * .1 / 1e6)
                self.assertAlmostEqual(cost["hypothetical_10_percent_premium_estimate_usd"], tokens * .11 / 1e6)
                self.assertIs(cost["premium_applicability_verified"], False)
                self.assertIs(cost["invoice_verified"], False)
        total_keys = {key for key in report if key.startswith("decisions_total_")}
        self.assertEqual(total_keys, {key for key in published if key.startswith("decisions_total_")})
        self.assertEqual(report["decisions_total_input_tokens"], 660)
        self.assertAlmostEqual(report["decisions_total_standard_estimate_usd"], 660 * .1 / 1e6)
        self.assertAlmostEqual(report["decisions_total_hypothetical_premium_estimate_usd"], 660 * .11 / 1e6)

    def test_generated_metrics_keep_refusals_and_input_provenance(self):
        report = self.generate_offline()
        self.assertEqual(report["dataset_sha256"], self.dataset_hash)
        self.assertEqual(report["choices_sha256"], self.choices_hash)
        self.assertEqual(report["records_per_language"], 3)
        for language, correct in (("ja", 2), ("en", 1)):
            with self.subTest(language=language):
                row = report["languages"][language]
                self.assertEqual(row["decisions"]["correct"], correct)
                self.assertEqual(row["decisions"]["accuracy_all_records"], correct / 3)
                self.assertEqual(row["decisions"]["accuracy_valid_choices"], correct / 2)
                self.assertEqual(row["decisions"]["status_counts"], {"choice": 2, "refusal": 1})
                self.assertEqual(sum(row["paired_all_records"].values()), 3)
                self.assertEqual(row["paired_valid_choices_only"]["not_paired_valid"], 1)
                self.assertEqual(len(row["per_relation_comparison"]), 2)


if __name__ == "__main__":
    unittest.main()
