#!/usr/bin/env python3
"""Offline-only aggregate report and figures; never reads an API key or calls APIs."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics

import evaluate_relations as evaluation


def report(dataset_dir, decisions_dir):
    data = dataset_dir / "dataset.jsonl"
    choices = dataset_dir / "relation_choices.json"
    rows, relations = evaluation.load_inputs(data, choices)
    label_ids = {lang: {r["labels"][lang]: r["id"] for r in relations} for lang in ("ja", "en")}
    expected_ids = {r["id"] for r in rows}
    output = {
        "task": "closed-set relation choice; known subject/object and saved Wikipedia text",
        "records_per_language": len(rows), "relations": len(relations),
        "dataset_sha256": evaluation.sha256(data), "choices_sha256": evaluation.sha256(choices),
        "refusal_policy": "primary accuracy denominator is every dataset row; refusals count as incorrect and false negatives",
        "latency_caveat": "Jev historical client/network/server timings; not a controlled same-environment comparison",
        "prices_checked": "2026-10-08", "decisions_standard_usd_per_million_input_tokens": 0.10,
        "jev_current_usd_per_million_input_tokens": 0.042,
        "historical_jev_model": "not recorded", "languages": {},
        "dataset_profile": {
            "unique_subjects": len({r["subject"]["id"] for r in rows}),
            "language_texts_are_parallel_translations": False,
            "text_lengths": {lang: {
                "mean_characters": statistics.mean(len(r["articles"][lang]["text"]) for r in rows),
                "at_10000_characters": sum(len(r["articles"][lang]["text"]) == 10000 for r in rows),
            } for lang in ("ja", "en")},
        },
    }
    for lang in ("ja", "en"):
        result_file = decisions_dir / f"decisions-{lang}.jsonl"
        d_rows = evaluation.read_jsonl(result_file)
        j_file = dataset_dir / f"jev-results-{lang}.jsonl"
        j_rows = evaluation.read_jsonl(j_file)
        d, j = evaluation.unique_by_id(d_rows), evaluation.unique_by_id(j_rows)
        manifest = json.loads(result_file.with_suffix(".manifest.json").read_text(encoding="utf-8"))
        if set(d) != expected_ids or set(j) != expected_ids:
            raise ValueError("Incomplete or mismatched result IDs")
        if not manifest.get("completed") or manifest.get("planned_records") != len(rows):
            raise ValueError("Run is not complete")
        if manifest.get("dataset_sha256") != output["dataset_sha256"] or manifest.get("choices_sha256") != output["choices_sha256"]:
            raise ValueError("Run manifest inputs differ")
        if any(r.get("status") not in ("choice", "refusal") or r.get("model_returned") != evaluation.MODEL for r in d_rows):
            raise ValueError("Unexpected status or model")
        d_score = evaluation.summarize(rows, relations, d_rows, lang)
        j_score = evaluation.summarize(rows, relations, j_rows, lang, historical_jev=True)
        if d_score["usage_records"] != len(rows):
            raise ValueError("Usage missing for some requests")
        paired, refusals = Counter(), Counter()
        for row in rows:
            key, gold = row["id"], row["expected_relation"]["id"]
            dp = label_ids[lang].get(d[key].get("predicted")) if d[key]["status"] == "choice" else None
            jp = label_ids[lang][j[key]["predicted"]]
            dc, jc = dp == gold, jp == gold
            paired["both_correct" if dc and jc else "decisions_only_correct" if dc else "jev_only_correct" if jc else "both_wrong"] += 1
            if d[key]["status"] == "refusal":
                refusals[gold] += 1
        for name in ("both_correct", "decisions_only_correct", "jev_only_correct", "both_wrong"):
            paired.setdefault(name, 0)
        per_relation = []
        for relation in relations:
            rid = relation["id"]
            ds, js = d_score["per_relation"][rid], j_score["per_relation"][rid]
            per_relation.append({"id": rid, "label_ja": relation["labels"]["ja"],
                "label_en": relation["labels"]["en"], "support": ds["support"],
                "decisions_f1": ds["f1"], "jev_f1": js["f1"],
                "f1_difference": ds["f1"] - js["f1"], "decisions_refusals": refusals[rid]})
        tokens = d_score["input_tokens_reported"]
        safe_manifest_keys = ("started_at_utc", "finished_at_utc", "python", "platform",
            "script_sha256", "model_requested", "concurrency", "automatic_retries",
            "warmup_requests", "wall_clock_seconds", "connection_policy", "timing")
        output["languages"][lang] = {
            "decisions": d_score, "jev_historical": j_score,
            "coverage": d_score["valid_choices"] / len(rows),
            "paired_all_records": dict(paired),
            "paired_valid_choices_only": evaluation.paired_comparison(rows, relations, d_rows, j_rows, lang),
            "accuracy_difference_percentage_points": 100 * (d_score["accuracy_all_records"] - j_score["accuracy_all_records"]),
            "macro_f1_difference": d_score["macro_f1_all_candidates"] - j_score["macro_f1_all_candidates"],
            "per_relation_comparison": per_relation,
            "cost": {"metered_input_tokens": tokens, "standard_rate_estimate_usd": tokens / 1e6 * .1,
                "with_10_percent_premium_estimate_usd": tokens / 1e6 * .1 * 1.1,
                "invoice_verified": False},
            "decisions_run": {k: manifest[k] for k in safe_manifest_keys if k in manifest},
            "result_hashes": {"decisions": evaluation.sha256(result_file), "jev": evaluation.sha256(j_file)},
        }
    output["decisions_total_input_tokens"] = sum(r["cost"]["metered_input_tokens"] for r in output["languages"].values())
    output["decisions_total_standard_estimate_usd"] = output["decisions_total_input_tokens"] / 1e6 * .1
    output["decisions_total_premium_estimate_usd"] = output["decisions_total_standard_estimate_usd"] * 1.1
    return output


def figures(data, output):
    # Plotting is offline. Matplotlib is an optional reporting dependency only.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    import numpy as np
    font_file = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    font = FontProperties(fname=str(font_file)) if font_file.exists() else FontProperties(family="sans-serif")
    plt.rcParams.update({"font.family": font.get_name(), "font.size": 12, "axes.spines.top": False,
                        "axes.spines.right": False, "axes.titlepad": 16, "savefig.facecolor": "white"})
    if font_file.exists():
        from matplotlib import font_manager
        font_manager.fontManager.addfont(str(font_file))
    output.mkdir(parents=True, exist_ok=True)
    colors = {"decisions": "#267f85", "jev_historical": "#506b9b"}
    langs = ("ja", "en")
    positions = np.arange(2)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    fig.subplots_adjust(left=.065, right=.985, bottom=.19, top=.79, wspace=.17)
    for ax, metric, title, scale in ((axes[0], "accuracy_all_records", "Accuracy（拒否を含む全件）", 100),
                                    (axes[1], "macro_f1_all_candidates", "Macro F1（36関係）", 1)):
        for i, provider in enumerate(("jev_historical", "decisions")):
            values = [data["languages"][lang][provider][metric] * scale for lang in langs]
            bars = ax.bar(positions + (i - .5) * .34, values, width=.30, color=colors[provider],
                          label="Jev（保存結果）" if provider == "jev_historical" else "Decisions")
            ax.bar_label(bars, labels=[f"{x:.2f}%" if scale == 100 else f"{x:.4f}" for x in values], padding=5, fontsize=11)
        ax.set_xticks(positions, ["日本語", "英語"])
        ax.set_ylim(0, 108 if scale == 100 else 1.08)
        ax.set_title(title, fontsize=14)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=.16)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .015), ncol=2, frameon=False)
    fig.suptitle("同一1,418件・36候補での関係選択", fontsize=17, y=.97)
    fig.savefig(output / "quality-comparison.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 4.8), layout="constrained")
    labels, correct, wrong, refused = [], [], [], []
    for lang, japanese in (("ja", "日本語"), ("en", "英語")):
        for provider, name in (("jev_historical", "Jev"), ("decisions", "Decisions")):
            score = data["languages"][lang][provider]
            labels.append(f"{japanese} / {name}")
            correct.append(score["correct"])
            wrong.append(score["valid_choices"] - score["correct"])
            refused.append(score["records"] - score["valid_choices"])
    y = np.arange(len(labels))
    ax.barh(y, correct, color="#267f85", label="正解")
    ax.barh(y, wrong, left=correct, color="#c47d38", label="誤選択")
    ax.barh(y, refused, left=np.array(correct) + np.array(wrong), color="#697078", label="拒否")
    for i, (a, b, c) in enumerate(zip(correct, wrong, refused)):
        ax.text(a / 2, i, str(a), ha="center", va="center", color="white", weight="bold")
        if b > 0: ax.text(a + b / 2, i, str(b), ha="center", va="center", color="white")
        ax.text(1440, i, f"拒否 {c}件", va="center", fontsize=11, color="#444444")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1590)
    ax.set_xticks([0, 400, 800, 1200, 1418])
    ax.set_xlabel("件数（各1,418件）")
    ax.set_title("正解・誤選択・拒否の内訳", fontsize=16)
    ax.legend(loc="lower center", bbox_to_anchor=(.5, -.28), ncol=3, frameon=False)
    fig.savefig(output / "answer-breakdown.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--decisions-dir", type=Path)
    parser.add_argument("--from-aggregate", type=Path, help="Replot the published aggregate JSON without private inputs")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--figure-dir", type=Path)
    args = parser.parse_args()
    if args.from_aggregate:
        data = json.loads(args.from_aggregate.read_text(encoding="utf-8"))
    else:
        if not args.dataset_dir or not args.decisions_dir or not args.output:
            parser.error("Raw-result aggregation requires --dataset-dir, --decisions-dir, and --output")
        data = report(args.dataset_dir, args.decisions_dir)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.figure_dir:
        figures(data, args.figure_dir)
    print("Offline report/figure generation completed")


if __name__ == "__main__":
    main()
