# Editing Caption Evaluation

Reproduce a paired comparison of **477 frozen cases**: model-assisted GT,
original JSON captions, and checkpoint-11000 JSON captions. This repository
contains the scoring implementation, fixed judge prompt and schema, tests,
and a synthetic example. The complete dataset is stored in S3.

477 个 case 按 `record_uid` 和 `sample_rank` 对齐。GT 使用 Astra medium；
语义评分使用 Luna low。checkpoint-11k 的 475 条成功输出、1 条 schema 错误、
1 条截断输出全部保留。失败输出按空预测评分，不从分母中删除。

## Final metric selection

Read the [interactive Chinese metric guide](https://grantzhang-alpha-document.highway.canva-internal.dev/)
(Canva WARP access), including formulas, matching rules, model judgment criteria,
examples, a Q calculator, and an Editing Edge Joint F1 calculator.
The self-contained page is also in [site/index.html](site/index.html).

Final reports retain these **ten metrics**, in this order:

1. Target RegionPQ
2. Target Desc F1
3. Caption Embedding F1
4. Reference GroundedPQ
5. Local Edit Instruction F1
6. Editing Edge Joint F1
7. Global HLD F1
8. Global Instruction F1
9. Global HLD critical-error rate
10. Global Instruction critical-error rate

The guide additionally documents **11. Removal Joint F1** as a
[new metric design](https://grantzhang-alpha-document.highway.canva-internal.dev/#removal-joint).
It evaluates source deletion boxes, actual deletion correspondence and local
removal instructions. Its proposed micro aggregation includes false deletions
on GT-negative cases. This metric is not implemented in the evaluator and has
no score in the existing 477-case results.

Both displayed means and their delta use the same applicable paired cases per
metric. Static scores retain full coverage when judge tasks fail. The selection
applies to final summaries and CSV metric columns; frozen raw score artifacts
remain available for audit. See [the metric contract](docs/metrics.md).

## Dataset

```text
s3://core-cn-oss.canva.com/usr/grantzhang/editing-caption-benchmark/20261008-paired477-astra-medium-v1/
```

| Artifact | Contents |
| --- | --- |
| `gt.jsonl` | 477 GT captions |
| `original.jsonl` | 477 original captions |
| `checkpoint_11000.jsonl` | 477 checkpoint outputs, including native status, errors and raw responses |
| `paired477.jsonl` / `paired477.parquet` | One row per case containing all three captions and provenance |
| `evaluation-inputs477.jsonl` | Evaluator input: captions, GT events, weights, image locators and hashes |
| `selection-lock.json` | Frozen case identities and GT hashes |
| `prediction-binding-audit.json` | Checkpoint provenance and conversion contract |
| `validation-report.json` | Row alignment, source equality and remote SHA-256 verification |

The [release manifest](data/paired477-manifest.json) contains exact sizes and
SHA-256 checksums. The repository is public; S3 data and referenced images
require authorized AWS access. Image bytes are verified against their hashes,
including images stored within Parquet row groups.

## Install and download

```bash
git clone https://github.com/grantzhang-code/editing-caption-evaluation.git
cd editing-caption-evaluation
python -m venv .venv
source .venv/bin/activate
pip install -e '.[embedding,dev]'

# Configure an AWS credential chain that can read the release and image objects.
# If using profiles: export AWS_PROFILE=your-profile
python -m benchmark.download \
  --manifest data/paired477-manifest.json \
  --output datasets/paired477

python -m benchmark.evaluate \
  --dataset datasets/paired477/evaluation-inputs477.jsonl --validate-only
```

Validation reports `477` cases and `954` prediction tasks. This command validates
caption bindings and schemas without fetching images or calling a judge.
It intentionally preserves the two native checkpoint failures.

## Run evaluation

```bash
# Geometry plus pinned multilingual MiniLM embedding metrics; no judge API call.
python -m benchmark.static_metrics \
  --dataset datasets/paired477/evaluation-inputs477.jsonl --results results/paired477

# Supply OPENAI_API_KEY securely through your environment.
export EVAL_OUTPUT_DIR=results/paired477
export JUDGE_MODEL=gpt-6-luna
export JUDGE_EFFORT=low
python -m benchmark.evaluate \
  --dataset datasets/paired477/evaluation-inputs477.jsonl --concurrency 8

python -m benchmark.summarize \
  --dataset datasets/paired477/evaluation-inputs477.jsonl --results results/paired477
```

Semantic scoring makes paid Responses API calls and requires access to the
selected judge model. Set `JUDGE_MODEL` and `JUDGE_EFFORT` to another supported
combination if necessary, use a fresh output directory, and report the change;
the frozen comparison uses the defaults above. Available concurrency is 1–256.
GT generation and checkpoint inference are upstream steps, outside this workflow.

Results include `REPORT.md`, `summary.json`, `per-case.csv`, `gt-concerns.json`,
and per-case geometry, judgments, request hashes, provider responses and scores.
`summary.json` explicitly distinguishes scored, failed, and pending tasks.
Only `complete: true` represents a finished comparison. No partial historical
score table is presented as the final 477-case result.

Optional durable S3 output, resume behavior and result publication are documented
in [Running and resuming](docs/running.md). See [Metric protocol](docs/metrics.md)
and [Dataset format](docs/dataset.md) for the scoring and coordinate conventions.

## Offline checks

```bash
pip install -e '.[dev]'
python -m pytest -q
python -m benchmark.evaluate --dataset examples/synthetic.jsonl --validate-only
```

The test suite exercises duplicate penalties, mismatched source frames, wrong
edit direction, empty and generated predictions, exact coordinate conversion,
caption/image integrity, failure denominators and an offline reporting flow.
It does not require AWS credentials, an API key or model downloads.

## Interpretation

The GT is a frozen model-assisted reference, not human-certified truth. These
477 cases are the verified GT-success subset of an original 500-case request,
selected before candidate scoring. Results do not estimate performance on the
entire 5,000-case inference population. Shared upstream parses may introduce
correlated errors; flagged GT concerns remain in the report. The original
raw user instruction is withheld from the judge.
