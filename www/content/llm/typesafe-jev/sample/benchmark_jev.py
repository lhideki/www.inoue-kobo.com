"""Measure real Jev API latency. Defaults: 106 requests, no automatic retries.

Run from a directory with .env containing TYPESAFE_API_KEY:
    python benchmark_jev.py --rounds 20 --output benchmark-results.json
Times include SDK serialization, network and response decoding, not client creation.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import statistics
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

from dotenv import dotenv_values
from typesafe_sdk import Choice, Noul, Score, RetryPolicy, TypeSafeClient

STATE = (
    "昔々、あるところにおじいさんとおばあさんが住んでいました。おばあさんが川で洗濯をしていると、"
    "大きな桃がどんぶらこと流れてきました。家に持ち帰り桃を割ると、中から元気な男の子が生まれ、"
    "二人は桃太郎と名付けて育てました。桃太郎はすくすくと成長し、ある日、鬼ヶ島の鬼たちが村人から"
    "金銀財宝を奪い、乱暴を働いていることを知ります。桃太郎は鬼退治を決意し、おばあさんが作ってくれた"
    "きび団子を持って旅に出ました。道中、犬・猿・雉に出会い、きび団子を分け与えることで仲間にします。"
    "桃太郎たちは力を合わせて鬼ヶ島に上陸し、鬼の大将と激しく戦った末に鬼たちを降伏させ、奪われていた"
    "財宝を取り戻しました。桃太郎たちは財宝を持って村に帰り、おじいさんとおばあさんや村人たちと共に"
    "平和な暮らしを取り戻しました。"
)

QUESTIONS = {
    "succeeded_with_animal_help": Noul(
        instructions="桃太郎は犬・猿・雉など動物たちの協力を得て鬼退治に成功しましたか？",
        criteria={
            "true": "動物たちが仲間になり、鬼退治の成功に貢献している",
            "false": "動物の協力がない、または鬼退治に失敗している",
        },
    ),
    "genre": Choice(
        instructions="この物語のジャンルとして最も適切なものはどれですか？",
        criteria={
            "昔話・おとぎ話": "口承で伝わる伝統的な民話、不思議な出来事や教訓を含む",
            "恋愛小説": "登場人物の恋愛関係が主題の物語",
            "推理小説": "謎解きや犯人捜しが主題の物語",
            "伝記": "実在した人物の生涯を記録した書物",
        },
    ),
    "oni_threat_level": Score(
        instructions="鬼ヶ島の鬼が村にもたらしていた脅威はどの程度深刻でしたか？",
        criteria=[
            {
                # スコア値 0
                "what": "村人に軽いいたずらをする程度",
                "examples": ["作物を少し荒らす", "夜中に物音を立てて驚かせる"],
            },
            {
                # スコア値 1
                "what": "村人の生活や財産に大きな被害を与えている",
                "examples": ["金銀財宝を略奪する", "家屋を破壊する"],
            },
            {
                # スコア値 2
                "what": "村人の命や安全を脅かすほど深刻な脅威となっている",
                "examples": ["村人に暴力を振るう", "村を占拠し支配する"],
            },
        ],
    ),
    "is_happy_ending": Noul(
        instructions="この物語はハッピーエンドで終わりますか？"
    ),
    "protagonist_trait": Choice(
        instructions="主人公の桃太郎の性格として最も近いものはどれですか？",
        criteria={"勇敢": None, "臆病": None, "狡猾": None, "怠惰": None},
    ),
    "moral_lesson_strength": Score(
        instructions="この物語の教訓性（子供向けの教育的価値）はどの程度強いですか？",
        criteria=[
            {
                # スコア値 0
                "what": "教訓性はほとんどない",
                "examples": ["単なる出来事の羅列で教訓的な結論がない"],
            },
            {
                # スコア値 1
                "what": "ある程度の教訓が含まれている",
                "examples": ["努力や協力の大切さが部分的に読み取れる"],
            },
            {
                # スコア値 2
                "what": "勧善懲悪など明確な教訓が中心テーマとなっている",
                "examples": ["悪が裁かれ、善が報われる結末が明示されている"],
            },
        ],
    ),
}


def stats(values):
    ordered = sorted(values)
    return {
        "n": len(values),
        "min_ms": min(values),
        "median_ms": statistics.median(values),
        "mean_ms": statistics.mean(values),
        "p95_ms": ordered[math.ceil(len(values) * 0.95) - 1],
        "max_ms": max(values),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--output", type=Path, default=Path("benchmark-results.json"))
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--model", default="jev-1.13.0")
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    if args.output.exists():
        parser.error("output already exists; choose a new path")
    api_key = os.environ.get("TYPESAFE_API_KEY") or dotenv_values(args.env_file).get("TYPESAFE_API_KEY")
    if not api_key:
        parser.error("Set TYPESAFE_API_KEY or provide --env-file")
    names = list(QUESTIONS)
    first_three = {name: QUESTIONS[name] for name in names[:3]}
    groups = {
        "sequential_3": [{name: QUESTIONS[name]} for name in names[:3]],
        "batch_3": [first_three],
        "batch_6": [QUESTIONS],
    }
    data = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "os": platform.system(),
            "os_release": platform.release(),
            "machine": platform.machine(),
            "typesafe_sdk": version("typesafe-sdk"),
            "python_dotenv": version("python-dotenv"),
        },
        "model_requested": args.model,
        "rounds": args.rounds,
        "random_seed": 20260919,
        "state_characters": len(STATE),
        "timing": "perf_counter around synchronous system_one; includes network and SDK; reused client; retries disabled",
        "p95_method": "nearest-rank",
        "requests": [],
        "groups": [],
    }

    def save():
        args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")

    def call(client, questions, phase, condition, round_index):
        start = perf_counter()
        try:
            response = client.system_one(model=args.model, state=STATE, questions=questions)
        except Exception as exc:
            # Store no exception body, headers or credentials.
            data["requests"].append({"phase": phase, "condition": condition, "round": round_index,
                "elapsed_ms": (perf_counter() - start) * 1000, "error_type": type(exc).__name__})
            save()
            raise SystemExit("API call failed: " + type(exc).__name__ + "; see output file") from None
        elapsed_ms = (perf_counter() - start) * 1000
        answers = {}
        for name, answer in response.answers.items():
            answers[name] = {key: getattr(answer, key) for key in
                ["noul", "choice", "score", "probabilities", "confidence", "legend"] if hasattr(answer, key)}
        data["requests"].append({
            "phase": phase, "condition": condition, "round": round_index,
            "question_ids": list(questions), "elapsed_ms": elapsed_ms,
            "model": response.model, "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens, "answers": answers,
        })

    def measure_group(client, condition, phase, round_index):
        first = len(data["requests"])
        for questions in groups[condition]:
            call(client, questions, phase, condition, round_index)
        # Total API-call time: excludes local logging between calls.
        elapsed_ms = sum(r["elapsed_ms"] for r in data["requests"][first:])
        data["groups"].append({"phase": phase, "condition": condition,
            "round": round_index, "elapsed_ms": elapsed_ms})
        return elapsed_ms

    # Pin the endpoint; do not inherit a custom endpoint from the environment.
    with TypeSafeClient(api_key=api_key, base_url="https://api.typesafe.ai",
                        retry=RetryPolicy(max_retries=0), timeout=30.0) as client:
        call(client, first_three, "cold", "batch_3", -1)
        save()
        print("Initial batch_3: %.1f ms" % data["requests"][-1]["elapsed_ms"], flush=True)
        for condition in groups:
            measure_group(client, condition, "warmup", -1)
        rng = random.Random(data["random_seed"])
        for round_index in range(args.rounds):
            order = list(groups)
            rng.shuffle(order)
            for condition in order:
                measure_group(client, condition, "measured", round_index)
            save()
            if (round_index + 1) % 5 == 0:
                print("Completed %d/%d rounds" % (round_index + 1, args.rounds), flush=True)
    data["summary"] = {condition: stats([g["elapsed_ms"] for g in data["groups"]
        if g["phase"] == "measured" and g["condition"] == condition]) for condition in groups}
    for name in names[:3]:
        data["summary"][name] = stats([r["elapsed_ms"] for r in data["requests"]
            if r["phase"] == "measured" and r["question_ids"] == [name]])
    data["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    save()
    print(json.dumps(data["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
