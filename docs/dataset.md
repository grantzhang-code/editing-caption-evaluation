# Dataset contract

All five JSONL views contain exactly the same ordered 477 `(sample_rank,
record_uid)` identities. Ranks retain their original selection values and are
not renumbered. `paired477.parquet` has exactly the same rows as `paired477.jsonl`.
See the release's `validation-report.json` for verification against upstream
GT and prediction files, and for downloaded S3 artifact hashes and version IDs.

The S3 release also retains the frozen selection, model and embedding metadata,
and checkpoint binding audit. Raw candidate outputs are never rewritten in the
caption exports. The checkpoint's `invalid_schema` and `length_truncated` rows
retain their error and raw response even when no valid caption is available.

## Evaluation JSONL

Each row has these fields:

| Field | Meaning |
| --- | --- |
| `sample_rank`, `record_uid` | Unique case identifiers |
| `cohort`, `benchmark_group`, `weight` | Subset reporting and positive analysis weight |
| `images.source`, `images.target` | Original scoring pixels, MIME type and SHA-256 |
| `gt.caption`, `gt.caption_sha256` | Native GT object and exact compact-JSON digest |
| `gt.events` | Frozen semantic edit events with unique IDs |
| `gt.model`, `gt.reasoning_effort` | GT generation provenance |
| `predictions.original`, `predictions.checkpoint_11000` | Both candidates, including native failures |

A prediction specifies `caption`, `schema`, `bbox_order`, `status`, `error`,
`original_caption_text`, and `raw_prediction_sha256`. Truncated rows also retain
`raw_response`. The hash binds the original caption text, or the raw response
when no parsed caption exists. GT hashes use UTF-8 JSON with
`ensure_ascii=False`, `separators=(',', ':')`, and preserved key order.

Image specs use either `kind: image` and `uri`, or `kind: parquet`, `uri`,
`row_group`, `record_uid`, `column`, and optional array `index`. Both require
`sha256` and `mime`. The selected image bytes must exactly match SHA-256. S3
credentials must cover the referenced image locations as well as the release.
The release preserves image locators; image bytes are not duplicated into it.
Local relative image paths resolve from the process working directory.

## Coordinate and schema conversion

GT and original captions use `elements.layout` with **YXYX coordinates on
0–1000**. Checkpoint captions use `compositional_deconstruction.elements` and
`reference_edits` with **XYXY coordinates on 0–1000**.

`benchmark.prediction_adapter` deep-copies the checkpoint caption and swaps
coordinates to YXYX for target, source-reference and removal boxes. It converts
reference actions to native `edit` or `keep`; generated regions have neither.
Text is preserved and coordinates are not rescaled. Generated-region local
instructions are recorded by the adapter audit and excluded from the local-edit
metric. The paired export additionally includes the adapted evaluation caption.

The evaluator currently supports one source image, `frame_0`, and exactly the
two named variants. It does not silently reinterpret an unknown coordinate order.
Both native failure rows remain in evaluation as empty predictions with zero
applicable quality scores; N/A metrics remain null.
