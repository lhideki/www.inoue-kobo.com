#!/usr/bin/env python3
"""Evaluate bilingual CoDEx-S relation choice and record Jev latency."""
from __future__ import annotations
import argparse, json, math, time
from collections import Counter
from pathlib import Path
from dotenv import load_dotenv
from typesafe_sdk import Choice, TypeSafeClient

ROOT = Path(__file__).resolve().parent


def percentile(values, fraction):
    return sorted(values)[math.ceil(len(values) * fraction) - 1] if values else None


def metrics(outcomes, relations, language):
    result = {}
    for relation in relations:
        relation_id = relation["id"]
        tp = sum(expected == relation_id and predicted == relation_id for expected, predicted in outcomes)
        fp = sum(expected != relation_id and predicted == relation_id for expected, predicted in outcomes)
        fn = sum(expected == relation_id and predicted != relation_id for expected, predicted in outcomes)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        result[relation_id] = {"label": relation["labels"][language], "support": sum(expected == relation_id for expected, _ in outcomes),
                               "precision": precision, "recall": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}
    return result


def main():
    load_dotenv(ROOT.parent / ".env")
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", choices=("ja", "en"), required=True)
    parser.add_argument("--dataset", type=Path, default=ROOT / "dataset" / "dataset.jsonl")
    parser.add_argument("--choices", type=Path, default=ROOT / "dataset" / "relation_choices.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--score", type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.output is None:
        args.output = ROOT / "dataset" / f"jev-results-{args.language}.jsonl"
    if args.score is None:
        args.score = ROOT / "dataset" / f"score-{args.language}.json"
    rows = [json.loads(line) for line in args.dataset.read_text().splitlines()]
    if args.limit:
        rows = rows[:args.limit]
    relations = json.loads(args.choices.read_text())["relations"]
    criteria = {row["labels"][args.language]: row["descriptions"][args.language] for row in relations}
    label_to_id = {row["labels"][args.language]: row["id"] for row in relations}
    if len(criteria) != len(relations):
        raise ValueError(f"Duplicate {args.language} relation labels")
    outcomes, durations, correct = [], [], 0
    started = time.perf_counter()
    with TypeSafeClient() as client, args.output.open("w") as output:
        for index, row in enumerate(rows, 1):
            article = row["articles"][args.language]
            expected = row["expected_relation"]
            if args.language == "ja":
                state = article["text"]
                instructions = (
                    "日本語Wikipedia本文に基づき、"
                    f"主語「{row['subject']['ja']['label']}」と目的語「{row['object']['ja']['label']}」の間に"
                    "成立する関係を候補から選んでください。"
                )
            else:
                state = article["text"]
                instructions = (
                    "Using the English Wikipedia extract, choose the relation that holds between "
                    f"the subject “{row['subject']['en']['label']}” and object “{row['object']['en']['label']}”."
                )
            call_started = time.perf_counter()
            response = client.system_one(state=state, questions={"relation": Choice(instructions=instructions, criteria=criteria)})
            elapsed_ms = (time.perf_counter() - call_started) * 1000
            answer = response.choices["relation"]
            predicted_id = label_to_id[answer.choice]
            is_correct = predicted_id == expected["id"]
            correct += is_correct
            outcomes.append((expected["id"], predicted_id)); durations.append(elapsed_ms)
            output.write(json.dumps({"id": row["id"], "expected": expected, "predicted": answer.choice, "correct": is_correct,
                                     "elapsed_ms": elapsed_ms, "probabilities": answer.probabilities}, ensure_ascii=False) + "\n")
            print(f"{index}/{len(rows)}: {'OK' if is_correct else 'NG'}")
    elapsed = time.perf_counter() - started
    per_relation = metrics(outcomes, relations, args.language)
    score = {"task": "relation_choice", "language": args.language, "dataset": args.dataset.name, "results": args.output.name,
             "records": len(rows), "relation_choice_count": len(relations), "correct": correct, "accuracy": correct / len(rows) if rows else None,
             "macro_f1": sum(row["f1"] for row in per_relation.values()) / len(per_relation) if per_relation else None,
             "timing": {"wall_clock_seconds": elapsed, "jev_call_total_seconds": sum(durations) / 1000,
                        "jev_call_mean_ms": sum(durations) / len(durations) if durations else None, "jev_call_p50_ms": percentile(durations, .5),
                        "jev_call_p95_ms": percentile(durations, .95), "records_per_second": len(rows) / elapsed if elapsed else None},
             "per_relation": per_relation, "expected_relation_counts": dict(sorted(Counter(expected for expected, _ in outcomes).items()))}
    args.score.write_text(json.dumps(score, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(score, ensure_ascii=False))


if __name__ == "__main__":
    main()
