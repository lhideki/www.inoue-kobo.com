"""Offline tests. Synthetic fixtures contain no Wikipedia text or API results."""
import argparse
import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "evaluate_relations.py"
SPEC = importlib.util.spec_from_file_location("evaluation", SCRIPT)
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


def fixtures():
    relations = [{"id": "P1", "labels": {"ja": "関係A", "en": "relation A"},
                  "descriptions": {"ja": "説明A", "en": "description A"}},
                 {"id": "P2", "labels": {"ja": "関係B", "en": "relation B"},
                  "descriptions": {"ja": "説明B", "en": "description B"}}]
    rows = [{"id": str(i), "expected_relation": {"id": "P1" if i < 2 else "P2"},
             "subject": {"ja": {"label": "主語"}, "en": {"label": "subject"}},
             "object": {"ja": {"label": "目的語"}, "en": {"label": "object"}},
             "articles": {"ja": {"text": "評価用の架空文章"}, "en": {"text": "Synthetic test text."}}}
            for i in range(3)]
    return rows, relations


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.rows, self.relations = fixtures()

    def mock_http_run(self, status, body, headers, key="sk-synthetic-secret"):
        from unittest.mock import MagicMock
        with tempfile.TemporaryDirectory() as tmp:
            data, choices = Path(tmp) / "rows.jsonl", Path(tmp) / "choices.json"
            data.write_text("\n".join(json.dumps(r) for r in self.rows), encoding="utf-8")
            choices.write_text(json.dumps({"relations": self.relations}), encoding="utf-8")
            args = argparse.Namespace(execute=True, reserve_budget_usd=1, price_multiplier=1.1,
                limit=None, language="ja", dataset=data, choices=choices,
                output=Path(tmp) / "out.jsonl", timeout=1, prompt_key=False)
            response = MagicMock(status=status)
            response.getheader.side_effect = lambda name: headers.get(name)
            response.read.return_value = body
            connection = MagicMock()
            connection.getresponse.return_value = response
            stdout = io.StringIO()
            with patch.object(evaluation.http.client, "HTTPSConnection", return_value=connection), \
                 patch.object(evaluation.sys, "stdout", stdout), \
                 patch.object(evaluation.platform, "platform", return_value="synthetic-platform"), \
                 patch.object(evaluation.os.environ, "get", side_effect=AssertionError("Environment read")):
                evaluation.run(args, self.rows, self.relations, api_key=key)
            result = evaluation.read_jsonl(args.output)
            manifest = json.loads(args.output.with_suffix(".manifest.json").read_text())
            evidence = args.output.read_text() + json.dumps(manifest) + stdout.getvalue()
            self.assertNotIn(key, evidence)
            self.assertNotIn("評価用の架空文章", evidence)
            self.assertNotIn("Synthetic test text.", evidence)
            self.assertNotIn("Authorization", evidence)
            self.assertEqual(connection.request.call_count, 1)
            connection.close.assert_called_once()
            self.assertEqual(len(result), 1)
            self.assertFalse(manifest["completed"])
            self.assertEqual(manifest["automatic_retries"], 0)
            return result[0], evidence

    def test_429_codes_are_distinguished_without_messages_or_secrets(self):
        for code, kind in (("insufficient_quota", "insufficient_quota"),
                           ("rate_limit_exceeded", "tokens"),
                           ("credit_balance_exhausted", "insufficient_quota"),
                           ("organization_usage_limit_exceeded", "insufficient_quota"),
                           ("organization_spend_limit_exceeded", "insufficient_quota"),
                           ("project_spend_limit_exceeded", "insufficient_quota"),
                           ("slow_down", "rate_limit_error")):
            with self.subTest(code=code):
                body = json.dumps({"error": {"type": kind, "code": code,
                    "message": "sk-synthetic-secret Authorization private-message",
                    "param": "private-parameter"}, "request": "private-request"}).encode()
                result, evidence = self.mock_http_run(429, body,
                    {"x-request-id": "req_0123456789abcdef0123456789abcdef", "Retry-After": "60",
                     "Authorization": "Bearer sk-synthetic-secret"})
                self.assertEqual(result["status"], "http_error")
                self.assertEqual(result["http_status"], 429)
                self.assertEqual(result["api_error"], {"type": kind, "code": code, "retry_after_seconds": 60})
                self.assertEqual(result["request_id"], "req_0123456789abcdef0123456789abcdef")
                for text in ("private-message", "private-parameter", "private-request"):
                    self.assertNotIn(text, evidence)

    def test_429_code_lookalikes_remain_omitted(self):
        for code in ("credit_balance_exhausted", "organization_usage_limit_exceeded",
                     "organization_spend_limit_exceeded", "project_spend_limit_exceeded", "slow_down"):
            for value in (code.upper(), code + "_private", code + " sk-synthetic-secret"):
                with self.subTest(code=value):
                    details = evaluation.safe_http_error(
                        json.dumps({"error": {"code": value}}).encode(), None, "sk-synthetic-secret")
                    self.assertEqual(details, {"code_omitted": True})

    def test_unknown_error_fields_and_headers_are_omitted(self):
        body = json.dumps({"error": {"type": "sk-synthetic-secret", "code": "unknown_code",
            "message": "private-message", "other": "private-extra"}}).encode()
        result, evidence = self.mock_http_run(429, body,
            {"x-request-id": "req_sk-synthetic-secret", "Retry-After": "sk-synthetic-secret"})
        self.assertIsNone(result["request_id"])
        self.assertEqual(result["api_error"], {"type_omitted": True, "code_omitted": True,
            "retry_after_omitted": True})
        for text in ("unknown_code", "private-message", "private-extra"):
            self.assertNotIn(text, evidence)

    def test_malformed_error_body_remains_http_error(self):
        for body in (b'not JSON sk-synthetic-secret', b'{"error":null}', b'\xff'):
            with self.subTest(body=body):
                result, _ = self.mock_http_run(429, body, {})
                self.assertEqual(result["status"], "http_error")
                self.assertEqual(result["api_error"], {"body_unrecognized": True})

    def test_retry_after_accepts_only_bounded_seconds_or_http_date(self):
        date = "Wed, 21 Oct 2026 07:28:00 GMT"
        self.assertEqual(evaluation.safe_http_error(b'{}', date, "sk-synthetic-secret")["retry_after_utc"],
                         "2026-10-21T07:28:00+00:00")
        for value in ("999999999", "-1", "1.5", "60\nAuthorization", [], "Bearer other-secret"):
            with self.subTest(value=value):
                self.assertTrue(evaluation.safe_http_error(b'{}', value, "sk-synthetic-secret")["retry_after_omitted"])

    def test_structured_values_matching_key_are_omitted(self):
        self.assertIsNone(evaluation.safe_request_id("req_0123456789abcdef", "req_0123456789abcdef"))
        self.assertNotIn("code", evaluation.safe_http_error(
            b'{"error":{"code":"insufficient_quota"}}', None, "insufficient_quota"))
        self.assertNotIn("retry_after_seconds", evaluation.safe_http_error(b'{}', "60", "60"))

    def test_untrusted_model_name_cannot_record_a_key(self):
        result, _ = self.mock_http_run(200,
            b'{"model":"sk-synthetic-secret","usage":{"input_tokens":12},"answers":[null]}', {})
        self.assertEqual(result["status"], "invalid_response")
        self.assertNotIn("model_returned", result)
        self.assertTrue(result["model_returned_omitted"])

    def test_payload_keeps_text_instruction_choice_order(self):
        body = evaluation.payload(self.rows[0], self.relations, "ja")
        self.assertEqual(body["input"], self.rows[0]["articles"]["ja"]["text"])
        self.assertEqual(body["questions"][0]["instructions"],
            "日本語Wikipedia本文に基づき、主語「主語」と目的語「目的語」の間に成立する関係を候補から選んでください。")
        self.assertEqual(body["questions"][0]["choices"], [
            {"value": "関係A", "description": "説明A"}, {"value": "関係B", "description": "説明B"}])

    def test_gold_and_metadata_never_enter_payload(self):
        changed = copy.deepcopy(self.rows[0])
        changed.update(id="SECRET_IDENTIFIER", private_note="not model input")
        changed["expected_relation"] = {"id": "P2", "label": "GOLD_LABEL"}
        self.assertEqual(evaluation.payload(changed, self.relations, "ja"),
                         evaluation.payload(self.rows[0], self.relations, "ja"))

    def test_failures_and_missing_count_against_full_denominator(self):
        result = [{"id": "0", "expected": {"id": "P1"}, "status": "choice", "predicted": "関係A", "elapsed_ms": 10},
                  {"id": "1", "expected": {"id": "P1"}, "status": "refusal", "predicted": None, "elapsed_ms": 2}]
        for record in result:
            record["language"] = "ja"
            record["model_returned"] = evaluation.MODEL
            record["request_sha256"] = evaluation.hashlib.sha256(evaluation.encoded_payload(self.rows[int(record["id"])], self.relations, "ja")).hexdigest()
        summary = evaluation.summarize(self.rows, self.relations, result, "ja")
        self.assertEqual(summary["accuracy_all_records"], 1 / 3)
        self.assertEqual(summary["accuracy_valid_choices"], 1)
        self.assertEqual(summary["missing"], 1)
        self.assertEqual(summary["latency_valid_choices"]["count"], 1)
        self.assertEqual(summary["per_relation"]["P1"]["recall"], .5)
        self.assertEqual(summary["per_relation"]["P2"]["f1"], 0)

    def test_duplicate_and_foreign_ids_are_rejected(self):
        with self.assertRaises(ValueError):
            evaluation.unique_by_id([self.rows[0], self.rows[0]])
        with self.assertRaises(ValueError):
            evaluation.summarize(self.rows, self.relations, [{"id": "alien"}], "ja")

    def test_label_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            evaluation.summarize(self.rows, self.relations, [{"id": "0", "expected": {"id": "P2"}}], "ja")

    def test_valid_choice_schema(self):
        response = {"answers": [{"type": "choice", "name": "relation", "choice": "関係A",
                    "confidence": .4, "probabilities": [{"value": "関係A", "probability": .7},
                                                            {"value": "関係B", "probability": .3}]}]}
        result = evaluation.normalize_response(response, {"関係A", "関係B"})
        self.assertEqual(result["confidence"], .4)  # Never replace it with max probability.
        self.assertEqual(result["probabilities"]["関係A"], .7)
        response["answers"][0]["probabilities"][0]["probability"] = float("nan")
        with self.assertRaises(ValueError):
            evaluation.normalize_response(response, {"関係A", "関係B"})

    def test_refusal_needs_no_probabilities(self):
        self.assertEqual(evaluation.normalize_response({"answers": [{"type": "refusal", "name": "relation"}]},
                          {"関係A", "関係B"})["status"], "refusal")

    def test_malformed_responses_rejected_cleanly(self):
        for response in (None, [], {"answers": [None]}, {"answers": []}, {"answers": [{"name": "relation", "type": "choice", "choice": []}]}):
            with self.subTest(response=response), self.assertRaises(ValueError):
                evaluation.normalize_response(response, {"関係A", "関係B"})

    def test_mismatched_provenance_is_rejected(self):
        record = {"id": "0", "expected": {"id": "P1"}, "status": "refusal", "predicted": None,
                  "language": "ja", "request_sha256": "wrong", "elapsed_ms": 1}
        with self.assertRaises(ValueError):
            evaluation.summarize(self.rows, self.relations, [record], "ja")
        record["request_sha256"] = evaluation.hashlib.sha256(evaluation.encoded_payload(self.rows[0], self.relations, "ja")).hexdigest()
        record["language"] = "en"
        with self.assertRaises(ValueError):
            evaluation.summarize(self.rows, self.relations, [record], "ja")

    def test_bool_is_not_a_token_count(self):
        self.assertFalse(evaluation.valid_tokens(True))
        self.assertFalse(evaluation.valid_tokens(-1))
        self.assertTrue(evaluation.valid_tokens(0))

    def test_unverified_model_cannot_be_scored_as_luna(self):
        record = {"id": "0", "expected": {"id": "P1"}, "status": "refusal", "predicted": None,
                  "language": "ja", "model_returned": "unverified-model", "elapsed_ms": 1,
                  "request_sha256": evaluation.hashlib.sha256(evaluation.encoded_payload(self.rows[0], self.relations, "ja")).hexdigest()}
        with self.assertRaises(ValueError):
            evaluation.summarize(self.rows, self.relations, [record], "ja")

    def test_masked_input_never_falls_back_to_echo(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(execute=True, reserve_budget_usd=1, price_multiplier=1,
                limit=1, language="ja", output=Path(tmp) / "out.jsonl", timeout=1, prompt_key=True,
                dataset=Path(tmp) / "rows.jsonl", choices=Path(tmp) / "choices.json")
            def warn_instead_of_reading(*args):
                evaluation.warnings.warn("Cannot disable echo", evaluation.getpass.GetPassWarning)
                raise AssertionError("Would have echoed input")
            with patch.object(evaluation.sys.stdin, "isatty", return_value=True), \
                 patch.object(evaluation.getpass, "getpass", side_effect=warn_instead_of_reading), \
                 patch.object(evaluation.http.client, "HTTPSConnection", side_effect=AssertionError("Network")):
                with self.assertRaises(ValueError):
                    evaluation.run(args, self.rows, self.relations)

    def test_mock_run_preserves_invalid_attempt_and_finalizes_manifest(self):
        from unittest.mock import MagicMock
        with tempfile.TemporaryDirectory() as tmp:
            data, choices = Path(tmp) / "rows.jsonl", Path(tmp) / "choices.json"
            data.write_text("\n".join(json.dumps(r) for r in self.rows), encoding="utf-8")
            choices.write_text(json.dumps({"relations": self.relations}), encoding="utf-8")
            args = argparse.Namespace(execute=True, reserve_budget_usd=1, price_multiplier=1, limit=None,
                language="ja", dataset=data, choices=choices, output=Path(tmp) / "out.jsonl", timeout=1, prompt_key=False)
            response = MagicMock(status=200)
            response.getheader.return_value = "synthetic-request-id"
            response.read.return_value = b'{"model":"gpt-6-luna","usage":{"input_tokens":12},"answers":[null]}'
            connection = MagicMock()
            connection.getresponse.return_value = response
            with patch.dict(evaluation.os.environ, {"OPENAI_API_KEY": "synthetic-key"}, clear=True), \
                 patch.object(evaluation.http.client, "HTTPSConnection", return_value=connection):
                evaluation.run(args, self.rows, self.relations)
            result = evaluation.read_jsonl(args.output)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["status"], "invalid_response")
            self.assertEqual(result[0]["usage"]["input_tokens"], 12)
            manifest = json.loads(args.output.with_suffix(".manifest.json").read_text())
            self.assertFalse(manifest["completed"])
            self.assertIn("finished_at_utc", manifest)
            self.assertNotIn("synthetic-key", args.output.read_text() + json.dumps(manifest))
            self.assertNotIn("評価用の架空文章", args.output.read_text())
            connection.close.assert_called_once()

    def test_masked_key_requires_interactive_terminal(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(execute=True, reserve_budget_usd=1, price_multiplier=1,
                limit=1, language="ja", output=Path(tmp) / "out.jsonl", timeout=1, prompt_key=True,
                dataset=Path(tmp) / "rows.jsonl", choices=Path(tmp) / "choices.json")
            with patch.object(evaluation.sys.stdin, "isatty", return_value=False), \
                 patch.object(evaluation.getpass, "getpass", side_effect=AssertionError("Prompted")), \
                 patch.object(evaluation.http.client, "HTTPSConnection", side_effect=AssertionError("Network")):
                with self.assertRaises(ValueError):
                    evaluation.run(args, self.rows, self.relations)

    def test_incomplete_and_duplicate_probabilities_rejected(self):
        response = {"answers": [{"type": "choice", "name": "relation", "choice": "関係A", "confidence": .2,
                    "probabilities": [{"value": "関係A", "probability": .5}, {"value": "関係A", "probability": .5}]}]}
        with self.assertRaises(ValueError):
            evaluation.normalize_response(response, {"関係A", "関係B"})

    def test_nearest_rank_latency_matches_jev(self):
        self.assertEqual(evaluation.percentile([3, 1, 4, 2], .5), 2)
        self.assertEqual(evaluation.percentile([3, 1, 4, 2], .95), 4)

    def test_run_without_execute_does_not_read_key_or_network(self):
        args = argparse.Namespace(execute=False)
        with patch.object(evaluation.os.environ, "get", side_effect=AssertionError("Key read")), \
             patch.object(evaluation.http.client, "HTTPSConnection", side_effect=AssertionError("Network")):
            with self.assertRaises(ValueError):
                evaluation.run(args, self.rows, self.relations)

    def mock_cli_run(self, *, status=200, include_usage=True, interrupt=False):
        from unittest.mock import MagicMock
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, choices, output = root / "rows.jsonl", root / "choices.json", root / "out.jsonl"
            data.write_text("\n".join(json.dumps(r) for r in self.rows), encoding="utf-8")
            choices.write_text(json.dumps({"relations": self.relations}), encoding="utf-8")
            body = {"model": evaluation.MODEL, "answers": [{
                "type": "choice", "name": "relation", "choice": "関係A", "confidence": .4,
                "probabilities": [{"value": "関係A", "probability": .7},
                                  {"value": "関係B", "probability": .3}]}]}
            if include_usage:
                body["usage"] = {"input_tokens": 12}
            if status != 200:
                body = {"error": {"code": "insufficient_quota"}}
            response = MagicMock(status=status)
            response.getheader.return_value = None
            response.read.return_value = json.dumps(body).encode()
            connection = MagicMock()
            connection.getresponse.return_value = response
            if interrupt:
                connection.request.side_effect = KeyboardInterrupt
            argv = [str(SCRIPT), "run", "--dataset", str(data), "--choices", str(choices),
                    "--output", str(output), "--execute", "--reserve-budget-usd", "1"]
            with patch.object(evaluation.sys, "argv", argv), \
                 patch.object(evaluation.sys, "stdout", io.StringIO()), \
                 patch.dict(evaluation.os.environ, {"OPENAI_API_KEY": "synthetic-key"}, clear=True), \
                 patch.object(evaluation.http.client, "HTTPSConnection", return_value=connection):
                exit_code = evaluation.main()
            manifest = json.loads(output.with_suffix(".manifest.json").read_text())
            records = evaluation.read_jsonl(output)
            self.assertNotIn("synthetic-key", output.read_text() + json.dumps(manifest))
            connection.close.assert_called_once()
            return exit_code, manifest, records, connection.request.call_count

    def test_cli_http_failure_returns_nonzero(self):
        code, manifest, records, attempts = self.mock_cli_run(status=429)
        self.assertEqual(code, 2)
        self.assertFalse(manifest["completed"])
        self.assertEqual(records[0]["status"], "http_error")
        self.assertEqual(attempts, 1)

    def test_cli_missing_usage_returns_nonzero(self):
        code, manifest, records, attempts = self.mock_cli_run(include_usage=False)
        self.assertEqual(code, 2)
        self.assertFalse(manifest["completed"])
        self.assertEqual(records[0]["status"], "choice")
        self.assertEqual(attempts, 1)

    def test_cli_interruption_returns_nonzero(self):
        code, manifest, records, attempts = self.mock_cli_run(interrupt=True)
        self.assertEqual(code, 2)
        self.assertFalse(manifest["completed"])
        self.assertEqual(records[0]["status"], "interrupted")
        self.assertEqual(attempts, 1)

    def test_cli_complete_run_returns_zero(self):
        code, manifest, records, attempts = self.mock_cli_run()
        self.assertEqual(code, 0)
        self.assertTrue(manifest["completed"])
        self.assertEqual(len(records), len(self.rows))
        self.assertEqual(attempts, len(self.rows))

    def test_checkout_discovery_supports_git_directories_and_worktrees(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            script = root / "sample" / "evaluate_relations.py"
            with patch.object(evaluation, "__file__", str(script)):
                # The host's temporary-directory ancestors may themselves be checkouts.
                with patch.object(evaluation.Path, "exists", return_value=False):
                    self.assertIsNone(evaluation.executing_checkout())
                (root / ".git").mkdir()
                self.assertEqual(evaluation.executing_checkout(), root)
                (root / ".git").rmdir()
                (root / ".git").write_text("gitdir: /synthetic/worktree")
                self.assertEqual(evaluation.executing_checkout(), root)

    def test_private_path_guard_resolves_symlinks_and_allows_copied_scripts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            checkout, outside = root / "repo", root / "outside"
            checkout.mkdir()
            outside.mkdir()
            (outside / "link").symlink_to(checkout, target_is_directory=True)
            with patch.object(evaluation, "executing_checkout", return_value=checkout):
                evaluation.require_external_private_paths(outside / "data", outside / "results")
                for path in (checkout, checkout / "data", outside / "link" / "results"):
                    with self.subTest(path=path), self.assertRaises(ValueError):
                        evaluation.require_external_private_paths(path)
            with patch.object(evaluation, "executing_checkout", return_value=None):
                evaluation.require_external_private_paths(checkout / "data")

    def test_live_run_rejects_checkout_paths_before_key_network_or_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            checkout = root / "repo"
            checkout.mkdir()
            for name in ("dataset", "choices", "output"):
                args = argparse.Namespace(execute=True, reserve_budget_usd=1, price_multiplier=1,
                    limit=1, language="ja", output=root / "out.jsonl", timeout=1, prompt_key=True,
                    dataset=root / "rows.jsonl", choices=root / "choices.json")
                setattr(args, name, checkout / "private.jsonl")
                with self.subTest(path=name), \
                     patch.object(evaluation, "executing_checkout", return_value=checkout), \
                     patch.object(evaluation.getpass, "getpass", side_effect=AssertionError("Key prompted")), \
                     patch.object(evaluation.os.environ, "get", side_effect=AssertionError("Environment read")), \
                     patch.object(evaluation.http.client, "HTTPSConnection", side_effect=AssertionError("Network")):
                    with self.assertRaisesRegex(ValueError, "outside the executing repository"):
                        evaluation.run(args, self.rows, self.relations)
                self.assertFalse(args.output.exists())
                self.assertFalse(args.output.with_suffix(".manifest.json").exists())

    def test_budget_denial_does_not_read_key_or_network(self):
        args = argparse.Namespace(execute=True, reserve_budget_usd=.0000001, price_multiplier=1,
                                  limit=None, language="ja")
        with patch.object(evaluation.os.environ, "get", side_effect=AssertionError("Key read")), \
             patch.object(evaluation.http.client, "HTTPSConnection", side_effect=AssertionError("Network")):
            with self.assertRaises(ValueError):
                evaluation.run(args, self.rows, self.relations)

    def test_long_context_is_blocked_before_key_or_network(self):
        self.rows[0]["articles"]["ja"]["text"] = "x" * 300000
        args = argparse.Namespace(execute=True, reserve_budget_usd=100, price_multiplier=1,
                                  limit=None, language="ja")
        with patch.object(evaluation.os.environ, "get", side_effect=AssertionError("Key read")), \
             patch.object(evaluation.http.client, "HTTPSConnection", side_effect=AssertionError("Network")):
            with self.assertRaises(ValueError):
                evaluation.run(args, self.rows, self.relations)

    def test_input_validation_and_plan_are_offline(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, choices = Path(tmp) / "rows.jsonl", Path(tmp) / "choices.json"
            data.write_text("\n".join(json.dumps(r) for r in self.rows), encoding="utf-8")
            choices.write_text(json.dumps({"relations": self.relations}), encoding="utf-8")
            with patch.object(evaluation.http.client, "HTTPSConnection", side_effect=AssertionError("Network")):
                rows, rels = evaluation.load_inputs(data, choices)
                plan = evaluation.plan(rows, rels, data, choices)
            self.assertEqual(plan["status"], "offline_only")
            self.assertEqual(plan["records_per_language"], 3)
            self.assertEqual(plan["languages"]["en"]["requests"], 3)


if __name__ == "__main__":
    unittest.main()
