#!/usr/bin/env python3
"""Run the approved bilingual experiment once with one masked key prompt.

Default is offline preflight only. No keys are read until --execute is given.
The $10 guard applies to this experiment, not unrelated account usage.
"""
import argparse
import getpass
import json
import math
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import evaluate_relations as evaluation

APPROVED_MAX_USD = 10.0
EXPECTED_HASHES = {
    "dataset.jsonl": "49295af0df7c14a408771b749924011f35902a82996126727cb44be911f18b91",
    "relation_choices.json": "e5e00d938d20f8b32966fff7b17dc12d26fde687f11afb893009992586c4b70c",
    "jev-results-ja.jsonl": "a41625f4ea8ce4355ae7e5074d9203312b4b1660f189489e3ebb726e0614c183",
    "jev-results-en.jsonl": "23c573654e42ba9b2597101cdeef5c36dab282a06162c569ab3ae307d949c2fc",
}


def preflight(data_dir, output, budget, multiplier):
    if not math.isfinite(budget) or not 0 < budget <= APPROVED_MAX_USD:
        raise ValueError("Budget must be positive and at most the approved USD 10")
    if not math.isfinite(multiplier) or multiplier < 1.1:
        raise ValueError("Keep at least the 10% pricing-premium allowance")
    if output.exists():
        raise ValueError("Experiment directory already exists. Inspect prior charges/results; do not automatically rerun")
    for name, digest in EXPECTED_HASHES.items():
        if evaluation.sha256(data_dir / name) != digest:
            raise ValueError("Frozen input hash mismatch: " + name)
    rows, relations = evaluation.load_inputs(data_dir / "dataset.jsonl", data_dir / "relation_choices.json")
    if len(rows) != 1418 or len(relations) != 36:
        raise ValueError("Unexpected frozen evaluation size")
    report = evaluation.plan(rows, relations, data_dir / "dataset.jsonl", data_dir / "relation_choices.json")
    allocations = {lang: math.ceil(details["reserved_list_price_usd"] * multiplier * 1e6) / 1e6
                   for lang, details in report["languages"].items()}
    if sum(allocations.values()) > budget:
        raise ValueError("Combined conservative reservation exceeds approved experiment budget")
    return rows, relations, report, allocations


def private_key():
    if not sys.stdin.isatty():
        raise ValueError("Use your own interactive terminal for masked key entry")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", getpass.GetPassWarning)
            key = getpass.getpass("OpenAI API key (hidden; this process only): ")
    except getpass.GetPassWarning:
        raise ValueError("Masked key entry unavailable; no fallback to echoed input") from None
    if not key:
        raise ValueError("Empty key; no API call")
    return key


def execute(args):
    rows, relations, plan, allocations = preflight(args.data, args.output, args.budget_usd, args.price_multiplier)
    print(json.dumps({"mode": "paid execution requested" if args.execute else "offline preflight",
        "requests": 2836, "budget_usd": args.budget_usd, "reserved_usd": sum(allocations.values()),
        "allocations_usd": allocations, "price_multiplier": args.price_multiplier}, indent=2), flush=True)
    if not args.execute:
        return 0
    key = private_key()
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    ledger_path = args.output / "experiment.json"
    ledger = {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "budget_usd": args.budget_usd, "budget_scope": "this experiment only; unrelated account usage is excluded",
        "reserve_method": plan["estimate_method"], "reserved_usd": sum(allocations.values()),
        "price_multiplier": args.price_multiplier, "allocations_usd": allocations,
        "automatic_retries": 0, "languages": {}}
    def save():
        ledger_path.write_text(json.dumps(ledger, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    save()
    try:
        for lang in ("ja", "en"):
            output = args.output / f"decisions-{lang}.jsonl"
            print(f"Starting {lang}: 1418 requests; no automatic retry", flush=True)
            run_args = argparse.Namespace(execute=True, prompt_key=False, reserve_budget_usd=allocations[lang],
                price_multiplier=args.price_multiplier, limit=None, language=lang, output=output,
                dataset=args.data / "dataset.jsonl", choices=args.data / "relation_choices.json", timeout=60)
            evaluation.run(run_args, rows, relations, api_key=key)
            manifest = json.loads(output.with_suffix(".manifest.json").read_text(encoding="utf-8"))
            results = evaluation.read_jsonl(output)
            metered = [r.get("usage", {}).get("input_tokens") for r in results]
            verified = [v for v in metered if evaluation.valid_tokens(v)]
            info = {"completed": manifest["completed"], "attempts": len(results),
                "metered_input_tokens": sum(verified), "unverified_usage_attempts": len(results) - len(verified),
                "metered_cost_proxy_usd": sum(verified) / 1e6 * evaluation.PRICE_USD_PER_MILLION * args.price_multiplier}
            ledger["languages"][lang] = info
            save()
            if not manifest["completed"] or info["unverified_usage_attempts"]:
                ledger["status"] = "stopped; inspect prior attempts and charges before any retry"
                return 2
            baseline = evaluation.read_jsonl(args.data / f"jev-results-{lang}.jsonl")
            summary = {"language": lang, "dataset_sha256": plan["dataset_sha256"],
                "choices_sha256": plan["choices_sha256"], "decisions_results_sha256": evaluation.sha256(output),
                "decisions": evaluation.summarize(rows, relations, results, lang),
                "jev_historical": evaluation.summarize(rows, relations, baseline, lang, historical_jev=True),
                "paired_valid_choices": evaluation.paired_comparison(rows, relations, results, baseline, lang)}
            (args.output / f"comparison-{lang}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        ledger["status"] = "completed"
        return 0
    except KeyboardInterrupt:
        ledger["status"] = "interrupted; do not automatically retry"
        return 130
    except Exception:
        ledger["status"] = "stopped by local error; inspect existing files before any retry"
        # No raw exception text, environment, key, HTTP headers, or response body.
        print("Local error. Preserved results/ledger; do not automatically retry.", file=sys.stderr)
        return 2
    finally:
        key = None
        ledger["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        save()


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=root / "data")
    parser.add_argument("--output", type=Path, default=root / "results")
    parser.add_argument("--budget-usd", type=float, default=10)
    parser.add_argument("--price-multiplier", type=float, default=1.1)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    try:
        return execute(args)
    except (ValueError, FileExistsError, FileNotFoundError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
