# Selected evaluation metrics

The original evaluator exports the following **ten metrics**, in this order. The interactive
Chinese guide in [site/index.html](../site/index.html) includes worked examples,
source correspondence judgment, a Q calculator, and a Joint F1 calculator.
The active results site adds **Global–Local Instruction Consistency F1** as its
eleventh metric. Removal PQ is omitted from the active viewer and CSV.

The scoring protocol remains `editing-caption-eval477-luna-low-v1-20261008`.
Reporting selection is versioned as `selected-ten-v1`; it changes the reported
metrics and paired aggregation, not the frozen scoring formulas or judge calls.

## Matching and notation

Target boxes use a Hungarian assignment maximizing total IoU, with edges below
0.50 gated before assignment. Matching uses geometry, not labels or descriptions.
Source-reference boxes are matched only within matched target pairs and the same
source frame, also with IoU >= 0.50. Every unmatched or duplicate region/edge
remains in its respective denominator.

`Np, Ng`: all predicted/GT target regions, including edit, keep and generated.
`Ep, Eg`: all predicted/GT source-reference edges attached to target regions.
`M, L`: matched target pairs and matched reference edges, respectively.
`w = min(target IoU, source-reference IoU)` for each matched edge.
`C`: judge's boolean `correspondence_correct`, based on the actual source entity.
`O`: judge's boolean `origin_correct`, based on visible edit/keep/generated provenance.

The judge supplies precision and recall independently from `{0,.25,.5,.75,1}`.
`Q = 2*p*r/(p+r)`, or zero if `critical_error=true` or `p+r=0`. These are holistic
rubric ratings, not counted fractions of atomic facts. The same formula is used
separately for region description, local instruction, global description, and
global instruction; each has its own precision, recall and critical-error flag.

## The eleven displayed metrics

| Metric | Per-case definition | Scope |
| --- | --- | --- |
| Target RegionPQ | `2 * sum(target IoU over M) / (Np + Ng)` | Target localization and coverage |
| Target Desc F1 | `2 * sum(Q_desc over M) / (Np + Ng)` | Target-visible facts in region descriptions |
| Caption Embedding F1 | `2 * sum(clipped cosine similarity over M) / (Np + Ng)` | Matched region descriptions only |
| Reference GroundedPQ | `2 * sum(w * C over L) / (Ep + Eg)` | Both endpoints and correct source correspondence |
| Local Edit Instruction F1 | `2 * sum(Q_instruction * O over pairs both marked edit) / (Np_edit + Ng_edit)` | Local changes and their edit provenance |
| Editing Edge Joint F1 | `2 * sum(w * C * O * J over L) / (Ep + Eg)` | Full source-to-target relation and local operation |
| Global HLD F1 | `Q_global_description` | Entire target scene |
| Global Instruction F1 | `Q_global_instruction` | Entire source-to-target transformation |
| Global HLD critical-error rate | Mean of `global_description.critical_error` | Severe errors in target description; lower is better |
| Global Instruction critical-error rate | Mean of `global_instruction.critical_error` | Severe errors in edit direction/content; lower is better |
| Global–Local Instruction Consistency F1 | `2M / (G_count + L_count)` | Agreement of atomic global/local actions within a caption |

For Joint F1, `J = Q_instruction` for a GT edit and `J = O` for a GT keep.
For a keep, `O*O = O` because O is boolean. Local Edit Instruction F1 does not
explicitly multiply source-reference IoU or correspondence C. Generated regions
without references and standalone removal records are not part of edge metrics.
Additions and removals are considered by the global instruction judge.

The three target-region metrics return zero when `Np + Ng = 0`. Reference
metrics are N/A when `Ep + Eg = 0`; local instruction is N/A when both sides
have zero edit regions. Q values are separate per scope: an error in a local
description does not automatically set the global instruction's error flag.

## Judge and embedding

The default judge is `gpt-6-luna`, reasoning effort `low`. It sees the original
source and target images, region/edge geometry and descriptions, global texts,
and frozen model-assisted GT context. Image evidence is the final authority.
Candidate variant identity and raw original user instruction are withheld.
The exact prompt and strict schema are in `benchmark/bench_contracts.py`.

Source correspondence is a model visual judgment: the reference must identify
the source entity/component that actually corresponds to the target, not merely
overlapping background or another similar instance. It returns a boolean and
text evidence. There is no independently verified instance-ID assignment or
separate identity confidence value; repeated objects and large changes can be ambiguous.

Embedding uses `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`,
revision `e8f8c211226b894fcb81acc59f3b34ba3efd5f42`, max length 512 tokens,
attention-mask mean pooling and L2 normalization. Cosine similarities are
clipped to [0,1]; an empty description contributes zero. Model settings and
truncation counts are audited. Cached vectors require matching text hashes and
model settings. `--model-path` must refer to that pinned snapshot.

## Original two-way report aggregation

For each metric, both displayed variant means and their delta use **the same
cases where both variants have a non-null value**. Per-case scores are averaged
equally; regions are not pooled across cases. Precision/recall are converted to
Q before averaging. Each row reports its own paired N. Static metrics preserve
their coverage even when semantic judge tasks fail.

`summary.json`, `REPORT.md` and `per-case.csv` contain only the selected ten metric
columns (plus metadata). `by_variant` summaries also provide weighted means
renormalized over applicable rows; these are distinct from the paired macro
table. Raw per-case judgments and score JSON retain the frozen diagnostic fields
for verification, without another paid evaluation.

Native invalid predictions are scored as empty, with zero applicable quality
scores. Their critical-error flags are false in this protocol, so the two
critical-error rates are not total failure rates. Provider/validation failures
remain missing semantic scores and are not converted to zero. A complete run
requires all 954 semantic and 954 static results for the frozen 477-case dataset.

## Global–Local Instruction Consistency F1

This diagnostic compares the global and local instructions within one caption.
It requires a separate judge pass; the ten existing scores cannot be converted
into it. Its frozen protocol is `global-local-instruction-consistency-v2-20261009`,
using `gpt-6.1-sol` with reasoning effort `medium`. The prompt and strict schema
are in `benchmark/consistency/contracts.py`; validation and matching are in
`benchmark/consistency/metric.py`.

1. Extract G, the global actions that should be expressed by region edits, and
   L, the actions from all region edit instructions, retaining region IDs. Split
   compound actions, expand multiple objects, and merge repeated descriptions.
2. Match actions one-to-one when the source entity, operation, direction and
   target state agree and explicit before-state/preservation constraints do not
   conflict. Synonyms can match; vague claims do not establish a match.
3. Let M be the count of consistent matches. Compute `P = M / |L|`,
   `R = M / |G|`, and `F1 = 2M / (|G| + |L|)`.

Missing, extra and contradictory actions remain in the denominators. Independent
additions/removals are excluded; an in-scope global action with no local instruction
remains in G. Global "car blue, sign right" versus local "car blue, sign left"
has two actions on each side and one match, so F1 = 0.5.

Each judge response returns normalized actions, compatibility edges and verbatim
evidence. Code validates evidence quotes, local-instruction coverage and action
indices, then computes maximum-cardinality one-to-one matching and F1. Parent
and child descriptions of the same physical edit are merged; instructions
affecting multiple distinct entities use consistent object scope on both sides.

The judge receives only that caption's global instruction, local instructions
and region context for entity/hierarchy resolution. It receives no images, GT,
original user instruction, candidate identity or other candidates. Region context
cannot supply missing actions or target states.

Before scoring, freeze eligibility using the independent GT predicate: at least
one GT `elements.layout` record has an `edit` field. This selects 444 of 477
cases for the five-way evaluation. Average case scores equally over the same
eligible, native-valid, successfully scored intersection for all five candidates.
An eligible caption with one or both sides missing scores 0. Cases confirmed to
require no region edits are N/A; API failures stay missing and coverage must be
reported. Missing instructions must not remove a case from the eligible cohort.
Native inference failures are excluded and never assigned zero.

The first structurally valid judge reply is retained. Up to five attempts may
repair a recorded invalid reply, with the validation diagnostic preserved; an
unknown transport outcome is not replayed automatically. Requests, responses,
evidence, input hashes and the frozen protocol are archived for audit. The 15
v1 Luna pilot captions were discarded as a group because hierarchy normalization
was inconsistent; no v1 bulk results are mixed into the v2 evaluation.
Recorded HTTP 5xx gateway failures with no judge response may receive one
additional delivery in a separate recovery namespace. The original error and
request hashes are preserved in `recovery-policy.json`; the judge request and
metric protocol stay identical. A request with an unknown transport outcome is
excluded from this recovery.

To recompute one saved valid judgment without making an API request:

```bash
python -m benchmark.consistency --payload payload.json --judgment judgment.json
```

The payload is the candidate's `payload` from the archived `inputs.jsonl`, or the
JSON in the first user text item of its saved `judge-N/request.json`. Use the
accepted `judgment.json` beside that request. The cohort-level export includes
per-case scores, G/L/match counts, statuses, caption hashes and paired ranks.

This measures internal consistency, not visual correctness: global and local
instructions can agree while both being wrong. Use it alongside instruction
quality scores grounded in source/target image evidence.

## Current five-way summary

The [published table](results.md) includes original R21-I2, 11k, 11k Init. + 4k,
6k and 9k. For each metric, select the same native-valid cases with non-null
scores in all five variants, then average cases equally. Native inference
failures and missing judge scores are excluded, never replaced by zero. This
publication policy differs from the original two-way CLI's frozen empty-output
policy described above; it does not rewrite the original per-case artifacts.

The consistency cohort has 444 eligible cases and 2,203 scored captions. The
last unscored 4k case 406 was excluded by user request. It was already outside
the native-valid five-way intersection, so its exclusion leaves the paired
consistency N at 431. The other ten metrics retain their own paired cohorts.

Geometry matching maximizes the sum of eligible IoUs; consistency matching
maximizes the number of compatible action pairs. Both enforce one-to-one
matching. The latter uses instruction evidence, not bounding-box IoU.
