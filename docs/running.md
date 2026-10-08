# Running, resuming and publishing

Run commands from the repository root. Set `EVAL_OUTPUT_DIR` for semantic
evaluation and pass the same directory as `--results` to static metrics and
summarization. Static and semantic scoring can run independently. Summarization
revalidates geometry, judgments, provider responses, identities and scores.

## Local and S3 outputs

```bash
export EVAL_OUTPUT_DIR=results/paired477
export EVAL_OUTPUT_S3=s3://YOUR-BUCKET/YOUR-RUN-PREFIX
python -m benchmark.evaluate \
  --dataset datasets/paired477/evaluation-inputs477.jsonl --concurrency 8

python -m benchmark.summarize \
  --dataset datasets/paired477/evaluation-inputs477.jsonl \
  --results results/paired477 --s3-output "$EVAL_OUTPUT_S3"
```

Set `EVAL_OUTPUT_S3` before starting the process to persist each semantic artifact
both locally and in S3. Static embedding outputs remain local. Summarization
with `--s3-output` downloads semantic case artifacts; `--skip-pull` uses the local
copy. Summarization without `--s3-output` makes no AWS calls.

Use `--publish` together with `--s3-output` to upload a finished report and
checksummed metrics manifest under that run's `exports/`. Publication fails if
any score or embedding metric is missing. This does not make S3 objects public.

## Resume and failures

Rerun the same command with the same output directory/prefix. The run contract
binds dataset, images, candidate hashes, judge prompt/schema, model, effort and
protocol. Completed scores are reused after identity checks. A changed contract
requires a new output directory and S3 prefix. Do not run simultaneous evaluators
against the same local output directory.

Before each paid request, the transport stores a durable intent containing its
hash. A recorded response is reused. HTTP 429 is retried with bounded backoff;
other HTTP and transport errors are recorded. A prior intent with no recorded
response causes the task to fail instead of silently making another paid call.
Inspect the attempt's `transport-unknown.json`, `api-error.json` and provider
records before retrying uncertain requests. Recover a known response when
possible; otherwise use a new output location only after deciding to incur
another call. Automatic validation repair is limited to two recorded attempts.

Invalid native checkpoint predictions receive deterministic zero-quality
judgments without an API request. Provider failures are distinct from native
prediction failures and are never silently converted to zero-quality predictions.

`--validate-only` checks local input contracts and performs no image preflight.
A full evaluation verifies GT hashes and original image bytes before semantic
requests. Credentials are taken from `OPENAI_API_KEY` and the standard AWS
credential chain; they are not stored in the repository.
