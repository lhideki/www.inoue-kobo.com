#!/usr/bin/env python3
"""Evaluate relation choices. Without --execute, print the request/cost plan only."""
import argparse
from collections import Counter
import getpass
import http.client
import json
import math
from pathlib import Path
import ssl
import sys
import time
import warnings

MODEL = "gpt-6-luna"
PRICE = 0.10  # USD per million input tokens, checked 2026-10-08.


def instructions(row, language):
    subject, obj = row["subject"][language]["label"], row["object"][language]["label"]
    if language == "ja":
        return ("日本語Wikipedia本文に基づき、"
                f"主語「{subject}」と目的語「{obj}」の間に"
                "成立する関係を候補から選んでください。")
    return ("Using the English Wikipedia extract, choose the relation that holds between "
            f"the subject “{subject}” and object “{obj}”.")


def payload(row, relations, language):
    return {"model": MODEL, "input": row["articles"][language]["text"], "questions": [{
        "type": "choice", "name": "relation", "instructions": instructions(row, language),
        "choices": [{"value": r["labels"][language], "description": r["descriptions"][language]}
                    for r in relations]}]}


def summarize(rows, records, relations, language):
    label_ids = {r["labels"][language]: r["id"] for r in relations}
    outcomes = [(row["expected_relation"]["id"], label_ids.get(record.get("predicted")))
                for row, record in zip(rows, records)]
    f1 = []
    for relation in relations:
        rid = relation["id"]
        tp = sum(gold == rid and predicted == rid for gold, predicted in outcomes)
        fp = sum(gold != rid and predicted == rid for gold, predicted in outcomes)
        fn = sum(gold == rid and predicted != rid for gold, predicted in outcomes)
        f1.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    times = sorted(record["elapsed_ms"] for record in records)
    return {"records": len(rows), "status_counts": dict(Counter(r["status"] for r in records)),
            "accuracy": sum(gold == predicted for gold, predicted in outcomes) / len(rows),
            "macro_f1": sum(f1) / len(f1),
            "p50_ms": times[math.ceil(len(times) * .50) - 1],
            "p95_ms": times[math.ceil(len(times) * .95) - 1],
            "input_tokens": sum(r["input_tokens"] for r in records)}


def evaluate(rows, relations, language, bodies, output, key):
    labels = {r["labels"][language] for r in relations}
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.load_default_certs()
    records = []
    with output.open("x", encoding="utf-8") as target:
        connection = http.client.HTTPSConnection("api.openai.com", timeout=60, context=context)
        try:
            for row, body in zip(rows, bodies):
                record = {"id": row["id"], "status": "error", "predicted": None}
                started = time.perf_counter()
                try:
                    connection.request("POST", "/v1/decisions", body,
                                       {"Authorization": "Bearer " + key, "Content-Type": "application/json"})
                    response = connection.getresponse()
                    raw = response.read()
                    if response.status != 200:
                        record["http_status"] = response.status
                        raise ValueError("HTTP error")
                    result = json.loads(raw)
                    if result.get("model") != MODEL or len(result["answers"]) != 1:
                        raise ValueError("Unexpected response")
                    answer, tokens = result["answers"][0], result["usage"]["input_tokens"]
                    if type(tokens) is not int or not 0 <= tokens <= len(body) + 4096:
                        raise ValueError("Unverified token usage")
                    if answer.get("name") != "relation":
                        raise ValueError("Unexpected question")
                    if answer["type"] == "choice" and answer["choice"] in labels:
                        record.update(status="choice", predicted=answer["choice"])
                    elif answer["type"] == "refusal":
                        record["status"] = "refusal"
                    else:
                        raise ValueError("Unexpected answer")
                    record["input_tokens"] = tokens
                except (Exception, KeyboardInterrupt):
                    record["elapsed_ms"] = (time.perf_counter() - started) * 1000
                    target.write(json.dumps(record, ensure_ascii=False) + "\n")
                    raise RuntimeError("Evaluation stopped. Inspect the saved output before restarting.") from None
                record["elapsed_ms"] = (time.perf_counter() - started) * 1000
                target.write(json.dumps(record, ensure_ascii=False) + "\n")
                target.flush()
                records.append(record)
        finally:
            connection.close()
    return summarize(rows, records, relations, language)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Directory with dataset.jsonl and relation_choices.json")
    parser.add_argument("--language", choices=("ja", "en"), required=True)
    parser.add_argument("--output", type=Path, required=True, help="New JSONL result file outside this checkout")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--budget-usd", type=float, default=0, help="Required for execution; UTF-8-based reservation, not an invoice limit")
    args = parser.parse_args()
    rows = [json.loads(line) for line in (args.data / "dataset.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    relations = json.loads((args.data / "relation_choices.json").read_text(encoding="utf-8"))["relations"]
    labels = [r["labels"][args.language] for r in relations]
    if not rows or not 2 <= len(relations) <= 255 or len(set(labels)) != len(labels):
        parser.error("Expected nonempty records and unique candidate labels")
    if len({row["id"] for row in rows}) != len(rows):
        parser.error("Duplicate record IDs")
    relation_ids = {r["id"] for r in relations}
    if (len(relation_ids) != len(relations) or any(not isinstance(label, str) or not label for label in labels)
            or any(not isinstance(r["id"], str) or not r["id"] for r in relations)
            or any(not isinstance(r["descriptions"][args.language], str) for r in relations)
            or any(row["expected_relation"]["id"] not in relation_ids
                   or not isinstance(row["articles"][args.language]["text"], str) for row in rows)):
        parser.error("Invalid text, candidate definitions or gold relation IDs")
    bodies = [json.dumps(payload(row, relations, args.language), ensure_ascii=False,
                         separators=(",", ":")).encode("utf-8") for row in rows]
    reserved_tokens = [len(body) + 4096 for body in bodies]
    if max(reserved_tokens) > 272000:
        parser.error("Long-context inputs require a separate price estimate")
    reserve = sum(reserved_tokens) * PRICE / 1e6 * 1.1
    print(json.dumps({"requests": len(rows), "reserved_cost_usd": reserve}))
    if not args.execute:
        return 0
    if not math.isfinite(args.budget_usd) or args.budget_usd <= 0 or reserve > args.budget_usd:
        parser.error("The supplied budget must cover the conservative reservation")
    checkout = next((p for p in Path(__file__).resolve().parents if (p / ".git").exists()), None)
    private_paths = [args.output, args.data / "dataset.jsonl", args.data / "relation_choices.json"]
    if checkout and any(p.resolve().is_relative_to(checkout) for p in private_paths):
        parser.error("Use input/output paths outside this checkout")
    if args.output.exists():
        parser.error("Output already exists")
    if not sys.stdin.isatty():
        parser.error("Interactive terminal required for key entry")
    with warnings.catch_warnings():
        warnings.simplefilter("error", getpass.GetPassWarning)
        key = getpass.getpass("OpenAI API key: ")
    if not key:
        parser.error("Empty API key")
    result = evaluate(rows, relations, args.language, bodies, args.output, key)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError, getpass.GetPassWarning) as error:
        print(str(error), file=sys.stderr)
        sys.exit(2)
