Decisions / Jev relation comparison, 2026-10-08
=============================================

Published aggregate-results.json describes a completed bilingual evaluation:
1,418 rows in each language, 36 fixed choices. Decisions returned 1,414 choices
and 4 refusals in JA, 1,391 choices and 27 refusals in EN; no full-run API errors
or automatic retries. The preliminary one-record diagnostic and original 429
attempt are excluded from benchmark metrics/time/tokens. Jev is a historical
baseline, with unrecorded model/SDK/region/connection conditions.

Privacy and provenance
----------------------
No raw Wikipedia text, per-record Decisions results, request IDs or credentials
are distributed here. Input hashes are in aggregate-results.json. Matching
frozen inputs are required for exact repetition; rebuilding from current
Wikipedia/Wikidata can change the experiment. The original dataset-building
script is linked from the earlier Jev article. Wikipedia prose has separate
reuse requirements from the CoDEx software license.

The full Decisions IDs, input hashes, model coverage, usage and metrics were
validated on the execution computer. Public aggregate arithmetic was checked
separately, and historical Jev scores were independently recomputed from the
saved baseline. Do not treat rounded display values as full-precision data.

Offline tests and figures
-------------------------
Python 3.10+ and standard library suffice for evaluation/tests. Run in sample/:
  python3 -m unittest discover -s tests -v

Figure generation optionally needs matplotlib and numpy. Japanese labels need
a Japanese-capable font (the published figures use Noto Sans CJK):
  python3 build_comparison.py --from-aggregate aggregate-results.json \
    --figure-dir ../images

With access to private frozen inputs and saved full results, recompute offline:
  python3 build_comparison.py --dataset-dir /path/to/frozen-data \
    --decisions-dir /path/to/full-results --output /path/to/new-aggregate.json

This produces additional per-relation statistics when raw records are available.
The public aggregate intentionally contains only the fields used in the article.
Neither offline command reads an API key or calls the API.

Preparing an explicitly authorized new paid run
----------------------------------------------
Never rerun a paid benchmark just to inspect its results. Confirm the current
API price/access, permission to transmit the specific data, and spending budget
before any live run. Some public biographies contain sensitive information.
Keep inputs and results outside a repository. The frozen input filenames are:
  dataset.jsonl, relation_choices.json, jev-results-ja.jsonl, jev-results-en.jsonl

The launcher requires explicit --data and --output paths. When run from a Git
checkout, launcher preflight and the standalone live runner reject private
input/result paths within that checkout, including symlinks resolving into it.
Git ignore rules alone do not prevent a local website build from copying files.
Copied scripts outside a checkout still work; offline scoring and figure/report
generation are unchanged.

Offline preflight:
  python3 run_experiment.py --data /path/to/frozen-data \
    --output /path/to/new-private-results

Only after authorization, add --execute. The launcher asks for the API key once
in your own interactive terminal with echo disabled; it aborts if masking fails.
No key is saved. Never put a key in chat, command-line arguments, source files,
plaintext credential files, trace output or logs. Do not enable shell/debug/TLS
key logging. Alternatively, evaluate_relations.py supports a privately supplied
OPENAI_API_KEY environment variable.

This example launcher has a USD 10 maximum experiment guard and a default 1.1
price multiplier. The multiplier is a conservative allowance, not evidence that
a premium applies to your account. Its UTF-8-byte-plus-overhead reservation is
not a tokenizer measurement or guaranteed invoice cap. This guard covers only
this new experiment, not other API activity or previous attempts; those costs
must also fit any overall approved budget. The historical run used a separate
continuation wrapper to reserve its prior failed/diagnostic attempts.

Preserve existing results. The launcher refuses existing output, performs no
automatic retries, and stops before EN if JA fails. Failed/uncertain calls can
still have billing implications. Inspect evidence and remaining authorized
budget before any continuation; do not delete files or change output paths just
to evade the guard. Safe error diagnostics use fixed allowlists, not raw messages.
The standalone live CLI exits nonzero on incomplete runs while preserving logs.

Metric definitions
------------------
Primary accuracy is correct / 1418 per language. Refusals remain in that
denominator. Conditional accuracy is correct / valid choices, shown separately.
Macro F1 averages all 36 fixed classes; refusals are false negatives for their
gold classes and undefined class F1 is zero. Quantiles use nearest rank, matching
the original Jev runner. Full-pair counts include refusals; valid-choice-only
pairs are separately labeled. Confidence is distinct from maximum probability,
and neither is a calibrated correctness guarantee. Rounded historical Jev
probabilities were not used for comparative calibration claims.

Accuracy error bars
-------------------
Only the Accuracy panel shows reference 95% Wilson score intervals. They are
derived by build_comparison.wilson_interval from unrounded correct counts and
all 1,418 records (including refusals as incorrect), using z=1.95996398454.
Formula: https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm
The intervals assume independent binary trials; repeated subjects/articles mean
independence is not guaranteed. This fixed dataset does not justify a general
population guarantee or a repeated-API-run variance estimate. Interval overlap
is not a paired significance test. Macro F1 and fixed prices have no error bars;
latency p50/p95 remain response-time quantiles, not confidence intervals.

Costs
-----
The full run used 11,304,340 input tokens. At USD 0.10/M input tokens its estimate
is USD 1.130434. A hypothetical 10% premium makes USD 1.2434774; premium applicability
and actual invoicing were not verified. These are full-run-only price estimates.
