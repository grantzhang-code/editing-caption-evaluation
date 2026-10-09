# Selected evaluation metrics

Final reports retain the following **ten metrics**, in this order. The interactive
Chinese guide in [site/index.html](../site/index.html) includes worked examples,
source correspondence judgment, a Q calculator, and a Joint F1 calculator.
It also documents Removal GroundedPQ as a proposed eleventh metric, described below.

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

## The ten metrics

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

## Final report aggregation

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

## Proposed eleventh metric: Removal GroundedPQ

**Design only.** This metric is not implemented in the evaluator or included in
the existing 477-case results. The ten implemented metric formulas and report
selection remain unchanged.

Each source reference in an independent removal record is one deletion item;
multiple references are expanded separately. The removal item's operation is already
explicitly deletion.
Within the same source frame, match predicted and GT deletion boxes one-to-one
using threshold-gated Hungarian assignment at source IoU >= 0.50. No target box
or target IoU is required. Unmatched and duplicate items stay in the denominator.

For a case with Dp predicted and Dg GT deletion items:

```text
Removal GroundedPQ = 2 * sum(source_IoU * C) / (Dp + Dg)
```

C is a model's boolean judgment that the source item is the correct entity or
component and is actually removed in the target, rather than merely moved or
modified. C still requires a new model-assisted visual judgment; existing global
instruction scores cannot substitute for it.

Q_remove is fixed to 1 and omitted from the formula. Local deletion instruction
precision, recall and text quality are not evaluated. Incorrect or missing local
instruction text is therefore not independently penalized by this metric; missing
or incorrect structured deletion items are still penalized through matching, the
denominator and C. Global Instruction F1 continues to evaluate the global text,
but does not automatically validate every local removal instruction. An additional
edit/keep/generated origin factor is not used for explicit removal items.

If Dp+Dg=0, the per-case metric is N/A. If exactly one side has deletion items,
there are no matches and the score is zero. With Dp=Dg=1, source IoU=.8 and C=1,
the score is .8. If one additional false deletion is predicted (Dp=2, Dg=1),
it becomes 2*.8/3, approximately .533. If C=0, that match contributes zero.

### Proposed aggregation: micro over a fixed common case set

For each variant, accumulate the numerator and denominator across the same case
set S before division:

```text
Removal GroundedPQ micro =
  2 * sum_over_cases_and_matches(source_IoU * C)
  / sum_over_cases(Dp + Dg)
```

This includes a GT-negative case where one variant falsely predicts a removal,
even if the other variant correctly predicts none. Do not filter to cases where
both per-case values are non-null: that would discard the false-positive penalty.
Both-empty cases contribute zero to both totals. If a variant's total denominator
is zero its score is N/A. Missing required judge results make the comparison
incomplete; do not silently treat provider failures as zero.

This proposed deletion-item micro average is distinct from the existing ten
metrics' paired case-level macro averages. Report the common case count,
predicted and GT deletion counts, matched counts, unmatched predicted/GT counts,
and false deletion counts on GT-negative cases alongside it.
