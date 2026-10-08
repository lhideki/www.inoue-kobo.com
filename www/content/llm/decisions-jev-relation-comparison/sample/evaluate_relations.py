#!/usr/bin/env python3
"""Plan, run, and score a matched Decisions/Jev relation-choice evaluation.

Standard library only. `plan` and `score` are offline. `run` requires an
explicit --execute flag, a cost guard, and a user-provided OPENAI_API_KEY.
No API key, article text, or raw response/error body is written to results.
"""
from __future__ import annotations

import argparse
import getpass
import hashlib
import http.client
import json
import math
import os
import platform
import re
import ssl
import statistics
import sys
import time
import warnings
from collections import Counter
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

MODEL = "gpt-6-luna"
ENDPOINT = "https://api.openai.com/v1/decisions"
PRICE_USD_PER_MILLION = 0.10  # Decisions list price checked 2026-10-08.
RESERVE_OVERHEAD = 4096  # Conservative estimate, NOT a provider billing guarantee.


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def executing_checkout():
    """Find the nearest Git checkout containing this script, including worktrees."""
    for directory in Path(__file__).resolve().parents:
        if (directory / ".git").exists():
            return directory
    return None


def require_external_private_paths(*paths):
    """Prevent local website builds from publishing private evaluation files."""
    checkout = executing_checkout()
    if checkout is not None:
        for path in paths:
            if Path(path).resolve().is_relative_to(checkout):
                raise ValueError("Private inputs and results must be outside the executing repository checkout")


def unique_by_id(rows):
    result = {}
    for row in rows:
        if row["id"] in result:
            raise ValueError("Duplicate record ID")
        result[row["id"]] = row
    return result


def load_inputs(dataset, choices):
    rows = read_jsonl(dataset)
    relations = json.loads(Path(choices).read_text(encoding="utf-8"))["relations"]
    if not rows or not 2 <= len(relations) <= 255:
        raise ValueError("Nonempty dataset and 2–255 relations required")
    unique_by_id(rows)
    ids = {r["id"] for r in relations}
    if len(ids) != len(relations):
        raise ValueError("Duplicate relation ID")
    for lang in ("ja", "en"):
        labels = [r["labels"][lang] for r in relations]
        if len(set(labels)) != len(labels) or any(not isinstance(x, str) or not x for x in labels):
            raise ValueError("Relation labels must be unique nonempty strings")
        for relation in relations:
            if not isinstance(relation["descriptions"][lang], str):
                raise ValueError("Relation description must be a string")
        for row in rows:
            if row["expected_relation"]["id"] not in ids:
                raise ValueError("Expected relation not in candidate set")
            if not isinstance(row["articles"][lang]["text"], str) or not row["articles"][lang]["text"]:
                raise ValueError("Missing article text")
    return rows, relations


def instructions(row, language):
    # Identical wording to the original run_jev.py. Do not tune on test labels.
    subject = row["subject"][language]["label"]
    obj = row["object"][language]["label"]
    if language == "ja":
        return ("日本語Wikipedia本文に基づき、"
                f"主語「{subject}」と目的語「{obj}」の間に"
                "成立する関係を候補から選んでください。")
    return ("Using the English Wikipedia extract, choose the relation that holds between "
            f"the subject “{subject}” and object “{obj}”.")


def payload(row, relations, language):
    # Explicit allowlist: gold labels, dataset IDs, Wikipedia URLs, and metadata
    # never become model input. Preserve the original candidate order.
    return {"model": MODEL, "input": row["articles"][language]["text"], "questions": [{
        "type": "choice", "name": "relation", "instructions": instructions(row, language),
        "choices": [{"value": r["labels"][language], "description": r["descriptions"][language]}
                    for r in relations]}]}


def encoded_payload(row, relations, language):
    return json.dumps(payload(row, relations, language), ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def reserve_tokens(body):
    # UTF-8 bytes + a fixed overhead is deliberately more conservative than
    # a character/4 heuristic, especially for Japanese. Provider accounting
    # can differ: this reserve is an operational guard, not a hard spend cap.
    return len(body) + RESERVE_OVERHEAD


def plan(rows, relations, dataset, choices):
    out = {"status": "offline_only", "model": MODEL, "endpoint": ENDPOINT,
           "dataset_sha256": sha256(dataset), "choices_sha256": sha256(choices),
           "records_per_language": len(rows), "relation_count": len(relations),
           "list_price_usd_per_million_input_tokens": PRICE_USD_PER_MILLION,
           "estimate_method": "UTF-8 request bytes + 4096 per request; not an invoice or guaranteed cap",
           "languages": {}}
    for lang in ("ja", "en"):
        sizes = [len(encoded_payload(r, relations, lang)) for r in rows]
        reserve = sum(sizes) + len(rows) * RESERVE_OVERHEAD
        out["languages"][lang] = {"requests": len(rows), "request_bytes": sum(sizes),
            "largest_request_bytes": max(sizes), "reserved_input_tokens": reserve,
            "reserved_list_price_usd": reserve / 1e6 * PRICE_USD_PER_MILLION}
    return out


def percentile(values, fraction):
    return sorted(values)[math.ceil(len(values) * fraction) - 1] if values else None


def valid_tokens(value):
    return type(value) is int and value >= 0


def summarize(rows, relations, results, language, historical_jev=False):
    expected = unique_by_id(rows)
    records = unique_by_id(results)
    if set(records) - set(expected):
        raise ValueError("Result contains ID outside selected dataset")
    label_ids = {r["labels"][language]: r["id"] for r in relations}
    predicted, states, durations, all_durations, tokens, models = {}, Counter(), [], [], [], Counter()
    for key, record in records.items():
        if record["expected"]["id"] != expected[key]["expected_relation"]["id"]:
            raise ValueError("Gold label mismatch between result and dataset")
        if not historical_jev:
            request_hash = hashlib.sha256(encoded_payload(expected[key], relations, language)).hexdigest()
            if record.get("language") != language or record.get("request_sha256") != request_hash:
                raise ValueError("Result language/request hash does not match current inputs")
        status = "choice" if historical_jev else record["status"]
        if not historical_jev and status in ("choice", "refusal") and record.get("model_returned") != MODEL:
            raise ValueError("Result model does not match the verified Decisions model")
        if status not in ("choice", "refusal", "http_error", "transport_error", "invalid_response", "interrupted"):
            raise ValueError("Unknown result status")
        states[status] += 1
        value = record.get("elapsed_ms")
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("Missing/invalid elapsed time")
        all_durations.append(value)
        if status == "choice":
            if record.get("predicted") not in label_ids:
                raise ValueError("Prediction is outside candidate set")
            predicted[key] = label_ids[record["predicted"]]
            durations.append(value)
        if record.get("model_returned"):
            models[record["model_returned"]] += 1
        usage = record.get("usage", {})
        if isinstance(usage, dict) and valid_tokens(usage.get("input_tokens")):
            tokens.append(usage["input_tokens"])
    pairs = [(row["expected_relation"]["id"], predicted.get(row["id"])) for row in rows]
    correct = sum(e == p for e, p in pairs)
    per_relation = {}
    for relation in relations:
        rid = relation["id"]
        tp = sum(e == rid and p == rid for e, p in pairs)
        fp = sum(e != rid and p == rid for e, p in pairs)
        fn = sum(e == rid and p != rid for e, p in pairs)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        per_relation[rid] = {"label": relation["labels"][language], "support": tp + fn,
                            "precision": precision, "recall": recall,
                            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}
    return {"records": len(rows), "attempted": len(records), "missing": len(rows) - len(records),
            "valid_choices": len(predicted), "status_counts": dict(states), "correct": correct,
            "accuracy_all_records": correct / len(rows),
            "accuracy_valid_choices": correct / len(predicted) if predicted else None,
            "macro_f1_all_candidates": statistics.mean(x["f1"] for x in per_relation.values()),
            "latency_all_attempts": {"count": len(all_durations), "p50_ms": percentile(all_durations, .5),
                                     "p95_ms": percentile(all_durations, .95)},
            "latency_valid_choices": {"count": len(durations), "p50_ms": percentile(durations, .5),
                                      "p95_ms": percentile(durations, .95),
                                      "mean_ms": statistics.mean(durations) if durations else None},
            "input_tokens_reported": sum(tokens) if tokens else None,
            "usage_records": len(tokens), "model_returned_counts": dict(models), "per_relation": per_relation,
            "confusions": [{"expected": a, "predicted": b, "count": n}
                for (a, b), n in Counter((e, p) for e, p in pairs if e != p).most_common()]}


def paired_comparison(rows, relations, decisions, jev, language):
    d = unique_by_id(decisions)
    j = unique_by_id(jev)
    labels = {r["labels"][language]: r["id"] for r in relations}
    counts = Counter()
    for row in rows:
        key, gold = row["id"], row["expected_relation"]["id"]
        if key not in d or key not in j or d[key].get("status") != "choice":
            counts["not_paired_valid"] += 1
            continue
        dc = labels[d[key]["predicted"]] == gold
        jc = labels[j[key]["predicted"]] == gold
        counts[("both_correct" if dc and jc else "decisions_only_correct" if dc else
                "jev_only_correct" if jc else "both_wrong")] += 1
    for k in ("both_correct", "decisions_only_correct", "jev_only_correct", "both_wrong", "not_paired_valid"):
        counts.setdefault(k, 0)
    return dict(counts)


def normalize_response(response, labels):
    if not isinstance(response, dict):
        raise ValueError("Response must be an object")
    answers = response.get("answers")
    if not isinstance(answers, list) or len(answers) != 1 or not isinstance(answers[0], dict) or answers[0].get("name") != "relation":
        raise ValueError("Unexpected answer schema")
    answer = answers[0]
    if answer.get("type") == "refusal":
        return {"status": "refusal", "predicted": None}
    if answer.get("type") != "choice" or not isinstance(answer.get("choice"), str) or answer.get("choice") not in labels:
        raise ValueError("Unexpected choice")
    probabilities = answer.get("probabilities")
    if not isinstance(probabilities, list) or len(probabilities) != len(labels):
        raise ValueError("Incomplete probability distribution")
    distribution = {}
    for item in probabilities:
        if not isinstance(item, dict):
            raise ValueError("Probability must be an object")
        value, prob = item.get("value"), item.get("probability")
        if not isinstance(value, str) or value not in labels or value in distribution or type(prob) not in (float, int) or not math.isfinite(prob) or not 0 <= prob <= 1:
            raise ValueError("Invalid probability")
        distribution[value] = prob
    if abs(sum(distribution.values()) - 1) > .001:
        raise ValueError("Probabilities do not sum to one")
    confidence = answer.get("confidence")
    if type(confidence) not in (int, float) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("Invalid confidence")
    return {"status": "choice", "predicted": answer["choice"], "probabilities": distribution,
            "confidence": confidence}


ERROR_TYPES = frozenset({"insufficient_quota", "rate_limit_error", "tokens", "requests",
    "invalid_request_error", "authentication_error", "permission_error", "not_found_error",
    "server_error", "api_error", "billing_error"})
ERROR_CODES = frozenset({"insufficient_quota", "rate_limit_exceeded", "invalid_api_key",
    "invalid_request_error", "model_not_found", "context_length_exceeded", "permission_denied",
    "billing_hard_limit_reached", "billing_not_active", "account_deactivated", "server_error",
    "credit_balance_exhausted", "organization_usage_limit_exceeded",
    "organization_spend_limit_exceeded", "project_spend_limit_exceeded", "slow_down"})
ERROR_HINTS = {
    "insufficient_quota": "API reported insufficient quota; inspect account quota before any retry.",
    "rate_limit_exceeded": "API reported a rate limit; no automatic retry was made.",
    "invalid_api_key": "API rejected authentication; no automatic retry was made.",
    "credit_balance_exhausted": "API reported exhausted credits; no automatic retry was made.",
    "organization_usage_limit_exceeded": "API reported an organization usage limit; no automatic retry was made.",
    "organization_spend_limit_exceeded": "API reported an organization spend limit; no automatic retry was made.",
    "project_spend_limit_exceeded": "API reported a project spend limit; no automatic retry was made.",
    "slow_down": "API requested slower requests; no automatic retry was made.",
}


def excludes_key(value, api_key):
    return isinstance(value, str) and bool(value) and api_key not in value


def safe_request_id(value, api_key):
    # Recognized request-ID forms only; never retain arbitrary header strings.
    if excludes_key(value, api_key) and re.fullmatch(
            r"req_[A-Za-z0-9]{16,64}|[0-9a-fA-F]{32}|[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", value):
        return value
    return None


def safe_http_error(raw, retry_after, api_key):
    # No response message/param, raw body, request data, or arbitrary headers.
    details = {}
    try:
        response = json.loads(raw)
        error = response.get("error") if isinstance(response, dict) else None
        if isinstance(error, dict):
            for name, allowed in (("type", ERROR_TYPES), ("code", ERROR_CODES)):
                value = error.get(name)
                if excludes_key(value, api_key) and value in allowed:
                    details[name] = value
                elif value is not None:
                    details[name + "_omitted"] = True
        else:
            details["body_unrecognized"] = True
    except (ValueError, UnicodeDecodeError, TypeError):
        details["body_unrecognized"] = True
    if excludes_key(retry_after, api_key):
        if re.fullmatch(r"[0-9]{1,5}", retry_after) and int(retry_after) <= 86400:
            details["retry_after_seconds"] = int(retry_after)
        else:
            try:
                # Strict IMF-fixdate; no arbitrary value is echoed or saved.
                if not re.fullmatch(r"(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun), [0-9]{2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) [0-9]{4} [0-9]{2}:[0-9]{2}:[0-9]{2} GMT", retry_after):
                    raise ValueError("Unrecognized retry time")
                details["retry_after_utc"] = parsedate_to_datetime(retry_after).astimezone(timezone.utc).isoformat()
            except (ValueError, TypeError, OverflowError):
                details["retry_after_omitted"] = True
    elif retry_after is not None:
        details["retry_after_omitted"] = True
    return details


def run(args, rows, relations, api_key=None):
    # Validate everything before reading the key, creating output, or networking.
    if not args.execute:
        raise ValueError("No API call: run requires --execute after authorizing cost and data transmission")
    if args.reserve_budget_usd <= 0 or not math.isfinite(args.reserve_budget_usd):
        raise ValueError("A positive finite reserve budget is required")
    if args.price_multiplier < 1 or not math.isfinite(args.price_multiplier):
        raise ValueError("price-multiplier must be finite and >= 1")
    if getattr(args, "timeout", 60) <= 0 or not math.isfinite(getattr(args, "timeout", 60)):
        raise ValueError("timeout must be finite and positive")
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("limit must be positive")
        rows = rows[:args.limit]
    requests = [encoded_payload(r, relations, args.language) for r in rows]
    if any(reserve_tokens(body) > 272_000 for body in requests):
        raise ValueError("No API call: long-context input is outside this frozen short-text evaluation; recheck pricing")
    reserved = sum(reserve_tokens(b) for b in requests) / 1e6 * PRICE_USD_PER_MILLION * args.price_multiplier
    if reserved > args.reserve_budget_usd:
        raise ValueError(f"No API call: conservative reserve ${reserved:.4f} exceeds approved guard")
    if args.output.exists() or args.output.with_suffix(".manifest.json").exists():
        raise ValueError("Output already exists; use a new path to preserve previous evidence")
    require_external_private_paths(args.dataset, args.choices, args.output,
                                   args.output.with_suffix(".manifest.json"))
    if api_key is not None:
        key = api_key
    elif getattr(args, "prompt_key", False):
        if not sys.stdin.isatty():
            raise ValueError("Masked input requires your own interactive terminal")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", getpass.GetPassWarning)
                key = getpass.getpass("OpenAI API key (hidden; kept in this process only): ")
        except getpass.GetPassWarning:
            raise ValueError("Masked input unavailable; no key entered. Use a supported private terminal") from None
    else:
        key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError("OPENAI_API_KEY is missing; enter it privately in your own terminal")
    # Verified TLS defaults without create_default_context's SSLKEYLOGFILE hook.
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.load_default_certs()
    connection = http.client.HTTPSConnection("api.openai.com", timeout=args.timeout, context=context)
    labels = {r["labels"][args.language] for r in relations}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"model_requested": MODEL, "endpoint": ENDPOINT, "language": args.language,
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
        "platform": platform.platform(), "dataset_sha256": sha256(args.dataset), "choices_sha256": sha256(args.choices),
        "script_sha256": sha256(__file__), "planned_records": len(rows), "concurrency": 1,
        "automatic_retries": 0, "warmup_requests": 0, "reserve_budget_usd": args.reserve_budget_usd,
        "reserved_cost_usd": reserved, "price_multiplier": args.price_multiplier,
        "connection_policy": "one HTTPSConnection, HTTP keep-alive when server permits; first request includes TCP/TLS; no redirects",
        "timing": "HTTPS request through full response read/JSON decoding; includes client/network; excludes local input construction",
        "completed": False}
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    started = time.perf_counter()
    try:
        with args.output.open("x", encoding="utf-8") as target:
            for n, (row, body) in enumerate(zip(rows, requests), 1):
                record = {"id": row["id"], "expected": {"id": row["expected_relation"]["id"]},
                          "language": args.language, "request_sha256": hashlib.sha256(body).hexdigest()}
                call_started = time.perf_counter()
                try:
                    # Fixed host/path only. http.client never follows redirects.
                    connection.request("POST", "/v1/decisions", body, {"Authorization": "Bearer " + key, "Content-Type": "application/json"})
                    result = connection.getresponse()
                    record["request_id"] = safe_request_id(result.getheader("x-request-id"), key)
                    raw = result.read()
                    if result.status != 200:
                        record.update(status="http_error", http_status=result.status, predicted=None)
                        record["api_error"] = safe_http_error(raw, result.getheader("Retry-After"), key)
                    else:
                        response = json.loads(raw)
                        if isinstance(response, dict):
                            model, usage = response.get("model"), response.get("usage")
                            if model == MODEL and excludes_key(model, key):
                                record["model_returned"] = model
                            elif model is not None:
                                record["model_returned_omitted"] = True
                            if isinstance(usage, dict) and valid_tokens(usage.get("input_tokens")):
                                # Only retain known, non-sensitive billing counters.
                                record["usage"] = {k: v for k, v in usage.items()
                                    if k in ("input_tokens", "output_tokens", "total_tokens") and valid_tokens(v)}
                        record["elapsed_ms"] = (time.perf_counter() - call_started) * 1000
                        if record.get("model_returned") != MODEL:
                            raise ValueError("Unverified model returned; quarantine this response")
                        record.update(normalize_response(response, labels))
                except (http.client.HTTPException, TimeoutError, OSError):
                    record.update(status="transport_error", predicted=None)
                except (ValueError, TypeError, KeyError):
                    record.update(status="invalid_response", predicted=None)
                except KeyboardInterrupt:
                    record.update(status="interrupted", predicted=None)
                record.setdefault("elapsed_ms", (time.perf_counter() - call_started) * 1000)
                target.write(json.dumps(record, ensure_ascii=False) + "\n")
                target.flush()
                print(f"{n}/{len(rows)}: {record['status']}", flush=True)
                if record["status"] == "http_error":
                    # Fixed descriptions selected by allowlisted codes, never API message text.
                    print(ERROR_HINTS.get(record["api_error"].get("code"),
                        "API error; inspect saved diagnostics and charges before any retry."), flush=True)
                usage = record.get("usage", {}).get("input_tokens")
                # Stop rather than silently incur uncertain duplicate/retry charges.
                if record["status"] not in ("choice", "refusal") or not valid_tokens(usage) or usage > reserve_tokens(body) or not record.get("model_returned"):
                    manifest["stop_reason"] = "error_or_unverified_usage_or_model; inspect before resuming"
                    break
            else:
                manifest["completed"] = True
    finally:
        connection.close()
        manifest["wall_clock_seconds"] = time.perf_counter() - started
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest["completed"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "run", "score"))
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--choices", type=Path, required=True)
    parser.add_argument("--language", choices=("ja", "en"), default="ja")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decisions", type=Path)
    parser.add_argument("--jev", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--prompt-key", action="store_true", help="Read a hidden key in your own interactive terminal; do not save it")
    parser.add_argument("--reserve-budget-usd", type=float, default=0)
    parser.add_argument("--price-multiplier", type=float, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout", type=float, default=60)
    args = parser.parse_args()
    rows, relations = load_inputs(args.dataset, args.choices)
    if args.command == "run":
        return 0 if run(args, rows, relations) else 2
    if args.command == "plan":
        result = plan(rows, relations, args.dataset, args.choices)
    else:
        result = {"language": args.language, "dataset_sha256": sha256(args.dataset),
                  "choices_sha256": sha256(args.choices)}
        if not args.decisions and not args.jev:
            parser.error("score requires --decisions and/or --jev")
        if args.decisions:
            d = read_jsonl(args.decisions)
            result["decisions"] = summarize(rows, relations, d, args.language)
            result["decisions_results_sha256"] = sha256(args.decisions)
        if args.jev:
            j = read_jsonl(args.jev)
            result["jev_historical"] = summarize(rows, relations, j, args.language, historical_jev=True)
            result["jev_results_sha256"] = sha256(args.jev)
        if args.decisions and args.jev:
            result["paired_valid_choices"] = paired_comparison(rows, relations, d, j, args.language)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as target:
        json.dump(result, target, ensure_ascii=False, indent=2)
        target.write("\n")
    print(json.dumps({"output": str(args.output), "mode": "offline"}))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, FileExistsError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)
