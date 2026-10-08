# Frozen metric protocol

`benchmark/metrics.py` implements geometry and aggregation;
`benchmark/bench_contracts.py` contains the complete judge prompt and strict JSON
schema. The protocol ID is `editing-caption-eval477-luna-low-v1-20261008`.

## Matching

Target regions are matched with a one-to-one Hungarian assignment maximizing
total IoU. Edges below IoU 0.50 are gated before assignment. Matching uses only
geometry, never language, region labels or provenance class. Source-reference
boxes are matched inside each matched target pair and within the same frame ID,
also at IoU 0.50. Unmatched and duplicate predictions count in denominators.
RegionPQ sensitivity at 0.30 and 0.75 is saved in each per-case score.

Let `P,G` be target-region counts, `EP,EG` source-edge counts, and `M` matched
target pairs. The judge returns precision and recall from `{0,.25,.5,.75,1}`.
Quality `Q` is their harmonic mean, or zero for a critical error. A critical
error includes a wrong direction, identity, count, salient exact text, negation,
source frame or major hallucination.

| Metric | Definition |
| --- | --- |
| Target RegionPQ | `2 * sum(target IoU) / (P + G)` |
| Target Match F1 | `2 * len(M) / (P + G)` |
| Target Desc F1 | `2 * sum(matched description Q) / (P + G)` |
| Caption Embedding F1 | `2 * sum(matched cosine similarity clipped to [0,1]) / (P + G)` |
| Reference PairPQ | `2 * sum(min(target IoU, reference IoU)) / (EP + EG)` |
| Reference conditional IoU | Mean IoU of matched source edges only |
| Reference Link F1 | `2 * sum(correct source correspondences) / (EP + EG)` |
| Reference GroundedPQ | Reference PairPQ with incorrect correspondences contributing zero |
| Local Edit Instruction F1 | `2 * sum(instruction Q * origin correctness)` over pairs both marked edit, divided by all predicted plus GT edit-region counts |
| Editing Edge Joint F1 | `2 * sum(min IoUs * correspondence correctness * origin correctness * local quality) / (EP + EG)` |
| Global HLD / Instruction F1 | Judge Q for target description / source-to-target edit instruction |
| Event Recall | Fraction of frozen GT events covered and not contradicted by the global instruction |

For a keep edge, local quality is origin correctness. Per-origin edit, keep and
generated RegionPQ, description and origin metrics are also emitted. Removal
content is included in the global instruction/event evaluation; this protocol
does not claim a separate removal-box detection metric. Generated local
instructions are not counted as local edits. Conditional metrics do not measure
missed regions or edges and should be read alongside unconditional F1/PQ.

## Judge and embedding

The default judge is `gpt-6-luna`, reasoning effort `low`, with original source
and target images and frozen GT events. Candidate variant and checkpoint identity
are withheld from the prompt. Region/edge assignments are fixed; the judge must
return each assigned pair and each GT event exactly once. It flags suspect GT
facts without modifying the frozen reference. Provider status, model, effort,
JSON schema and assignment coverage are checked before accepting a judgment.

Embedding uses `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`,
revision `e8f8c211226b894fcb81acc59f3b34ba3efd5f42`, max length 512,
attention-mask mean pooling, and L2 normalization. It uses matched region
descriptions only. Truncation counts and model settings are saved in the
embedding audit. Cached vectors are reused only with matching text hashes and
model settings. `--model-path` must point to that pinned snapshot; it permits
offline loading, not a different model. Floating-point values can vary slightly
across runtime versions and hardware.

## Reporting

Macro means exclude inapplicable null values. Weighted means renormalize the
selected cases' positive weights over applicable rows. Every metric reports
`n_applicable` and `applicable_weight`. Paired deltas use only cases where both
variants have an applicable score and report a separate paired N. Partial
variant macro means may cover different case sets and are not paired estimates.

Native prediction failures receive empty-prediction scores, while provider and
validation failures remain failed/pending tasks. A complete result needs all
954 semantic scores and all 954 static scores. The public workflow does not
change the metric formulas from the frozen run.
