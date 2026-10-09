#!/usr/bin/env python3
"""Estimate Jev input cost with alternative tokenizers; this is not measured usage.

Requires tiktoken==0.14.0. Encoding assets may download on first use.
"""
import argparse
import json
from decimal import Decimal
from pathlib import Path

from evaluate_relations import instructions

PRICE_PER_MILLION = Decimal("0.042")


def input_fields(row, relations, language):
    criteria = {r["labels"][language]: r["descriptions"][language] for r in relations}
    return row["articles"][language]["text"], instructions(row, language), criteria


def cost_fields(state, prompt, criteria):
    return [state, prompt] + [text for pair in criteria.items() for text in pair]


def cost_json(state, prompt, criteria):
    content = {"state": state, "questions": {"relation": {
        "type": "choice", "criteria": criteria, "instructions": prompt}}}
    return json.dumps(content, ensure_ascii=False, separators=(",", ":"))


def estimate(fields, encoder, mode):
    if mode == "fieldwise":
        texts = cost_fields(*fields)
    elif mode == "compact_content_json":
        texts = [cost_json(*fields)]
    else:
        raise ValueError("Unknown serialization: " + mode)
    return sum(len(encoder.encode(text, disallowed_special=())) for text in texts)


def price(tokens):
    return str(Decimal(tokens) * PRICE_PER_MILLION / Decimal(1_000_000))


def build_estimates(rows, relations, encoders):
    if any(len({r["labels"][language] for r in relations}) != len(relations) for language in ("ja", "en")):
        raise ValueError("Candidate labels must be unique")
    result = {"note": "Alternative-tokenizer estimates, not measured Jev usage or billing bounds.",
              "requests_per_language": len(rows), "candidates_every_request": len(relations),
              "price_input_usd_per_million": str(PRICE_PER_MILLION), "scenarios": []}
    for name, encoder in encoders.items():
        for mode in ("fieldwise", "compact_content_json"):
            counts = {language: sum(estimate(input_fields(row, relations, language), encoder, mode)
                                    for row in rows) for language in ("ja", "en")}
            counts["both"] = sum(counts.values())
            result["scenarios"].append({"tokenizer": name, "serialization": mode,
                "proxy_input_tokens": counts, "proxy_cost_usd": {k: price(v) for k, v in counts.items()}})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Directory containing dataset.jsonl and relation_choices.json")
    parser.add_argument("--output", type=Path, help="Write a new JSON file; defaults to stdout")
    args = parser.parse_args()
    import tiktoken
    if tiktoken.__version__ != "0.14.0":
        parser.error("Requires tiktoken==0.14.0")
    rows = [json.loads(line) for line in (args.data / "dataset.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    relations = json.loads((args.data / "relation_choices.json").read_text(encoding="utf-8"))["relations"]
    encoders = {name: tiktoken.get_encoding(name) for name in ("o200k_base", "cl100k_base")}
    output = json.dumps(build_estimates(rows, relations, encoders), ensure_ascii=False, indent=2)
    if args.output:
        with args.output.open("x", encoding="utf-8") as target:
            target.write(output + "\n")
    else:
        print(output)


if __name__ == "__main__":
    main()
