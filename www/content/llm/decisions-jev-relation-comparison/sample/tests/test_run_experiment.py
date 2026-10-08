"""Offline-only tests for the shared-budget launcher."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run_experiment as experiment


class ExperimentTests(unittest.TestCase):
    def test_combined_budget_is_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(experiment.evaluation, "sha256", side_effect=lambda p: experiment.EXPECTED_HASHES[p.name]), \
                 patch.object(experiment.evaluation, "load_inputs", return_value=([{}] * 1418, [{}] * 36)), \
                 patch.object(experiment.evaluation, "plan", return_value={"languages": {
                     "ja": {"reserved_list_price_usd": 6}, "en": {"reserved_list_price_usd": 6}}}):
                with self.assertRaises(ValueError):
                    experiment.preflight(Path(tmp), Path(tmp) / "new", 10, 1.1)

    def test_above_authorized_limit_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            experiment.preflight(Path(tmp), Path(tmp) / "new", 10.01, 1.1)

    def test_previous_run_is_not_repeated(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            experiment.preflight(Path(tmp), Path(tmp), 10, 1.1)

    def test_offline_preflight_never_prompts(self):
        args = argparse.Namespace(data=Path("unused"), output=Path("unused"), budget_usd=10,
                                  price_multiplier=1.1, execute=False)
        with patch.object(experiment, "preflight", return_value=([], [], {}, {"ja": 4, "en": 2})), \
             patch.object(experiment, "private_key", side_effect=AssertionError("Key prompted")), \
             patch.object(experiment.evaluation, "run", side_effect=AssertionError("API run")):
            self.assertEqual(experiment.execute(args), 0)

    def test_failed_japanese_run_stops_before_english(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(data=Path(tmp), output=Path(tmp) / "results", budget_usd=10,
                                      price_multiplier=1.1, execute=True)
            plan = {"estimate_method": "synthetic", "dataset_sha256": "synthetic", "choices_sha256": "synthetic"}
            def failed_run(run_args, rows, relations, api_key=None):
                run_args.output.write_text(json.dumps({"id": "0", "status": "http_error"}) + "\n")
                run_args.output.with_suffix(".manifest.json").write_text(json.dumps({"completed": False}))
            with patch.object(experiment, "preflight", return_value=([], [], plan, {"ja": 4, "en": 2})), \
                 patch.object(experiment, "private_key", return_value="synthetic-key"), \
                 patch.object(experiment.evaluation, "run", side_effect=failed_run) as run:
                self.assertEqual(experiment.execute(args), 2)
            self.assertEqual(run.call_count, 1)
            ledger = (args.output / "experiment.json").read_text()
            self.assertIn("stopped", ledger)
            self.assertNotIn("synthetic-key", ledger)
            self.assertFalse((args.output / "decisions-en.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
