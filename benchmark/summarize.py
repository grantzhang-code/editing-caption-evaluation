"""Independently validate judgments, combine static metrics, report paired results."""

import argparse, collections, concurrent.futures, csv, hashlib, json, time
from pathlib import Path
import boto3
from botocore.config import Config
from jsonschema import Draft202012Validator
import numpy as np
from . import metrics
from .data import load_dataset
from .bench_contracts import MODEL, EFFORT, JUDGE_SCHEMA

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
BUCKET = ""
PREFIX = ""
COUNTS = {
    "pred_regions",
    "gt_regions",
    "matched_regions",
    "pred_edges",
    "gt_edges",
    "matched_edges",
    "gt_concern_count",
}
MAIN = [
    ("target_region_pq", "Target RegionPQ"),
    ("target_match_f1", "Target Match F1"),
    ("target_caption_embedding_f1", "Caption Embedding F1"),
    ("target_desc_f1", "Target Desc F1"),
    ("reference_pair_pq", "Reference PairPQ"),
    ("reference_iou_conditional", "Reference conditional IoU"),
    ("reference_link_f1", "Reference Link F1"),
    ("reference_grounded_pq", "Reference GroundedPQ"),
    ("local_edit_instruction_f1", "Local Edit Instruction F1"),
    ("editing_edge_joint_f1", "Editing Edge Joint F1"),
    ("global_description_semantic_f1", "Global HLD F1"),
    ("global_instruction_semantic_f1", "Global Instruction F1"),
    ("event_recall", "Event Recall"),
    ("global_description_critical_error", "Global HLD critical-error rate"),
    ("global_instruction_critical_error", "Global Instruction critical-error rate"),
]


def dump(v):
    return json.dumps(v, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def save(name, v):
    (OUT / name).write_text(dump(v) + "\n")


def pull(s3):
    objects = [
        o
        for page in s3.get_paginator("list_objects_v2").paginate(
            Bucket=BUCKET, Prefix=PREFIX + "cases/"
        )
        for o in page.get("Contents", [])
    ]

    def one(o):
        key = o["Key"]
        p = OUT / key[len(PREFIX) :]
        if p.exists() and p.stat().st_size == o["Size"]:
            return
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(s3.get_object(Bucket=BUCKET, Key=key)["Body"].read())

    with concurrent.futures.ThreadPoolExecutor(max_workers=24) as pool:
        list(pool.map(one, objects))
    for name in ("progress.json", "run-status.json"):
        try:
            (OUT / name).write_bytes(s3.get_object(Bucket=BUCKET, Key=PREFIX + name)["Body"].read())
        except s3.exceptions.NoSuchKey:
            pass


def aggregate(rows):
    result = {}
    for key in sorted({k for r in rows for k in r["metrics"]} - COUNTS):
        eligible = [
            (r["metrics"][key], r["analysis_weight"])
            for r in rows
            if r["metrics"].get(key) is not None
        ]
        if eligible:
            assert all(0 <= v <= 1 for v, w in eligible), (key, eligible)
            result[key] = dict(
                macro=float(np.mean([v for v, w in eligible])),
                selected_subset_weighted=sum(v * w for v, w in eligible)
                / sum(w for v, w in eligible),
                n_applicable=len(eligible),
                applicable_weight=sum(w for v, w in eligible),
            )
    return result


def main():
    global OUT, BUCKET, PREFIX
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--s3-output")
    ap.add_argument("--skip-pull", action="store_true")
    ap.add_argument("--publish", action="store_true")
    args = ap.parse_args()
    OUT = args.results
    OUT.mkdir(parents=True, exist_ok=True)
    if args.s3_output:
        from urllib.parse import urlparse

        uri = urlparse(args.s3_output)
        BUCKET = uri.netloc
        PREFIX = uri.path.lstrip("/").rstrip("/") + "/"
    s3 = (
        boto3.Session().client("s3", config=Config(max_pool_connections=32))
        if args.s3_output
        else None
    )
    if args.s3_output and not args.skip_pull:
        pull(s3)
    items, gts, samples, preds = load_dataset(args.dataset)
    task_count = len(preds)
    case_count = len(items)
    variants = sorted({p["variant"] for p in preds})
    static_path = OUT / "static-scores.json"
    static = (
        {(x["variant"], x["rank"]): x for x in json.loads(static_path.read_text())}
        if static_path.exists()
        else {}
    )
    scored = []
    failures = []
    pending = []
    concerns = []
    events = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
    usage = collections.defaultdict(collections.Counter)
    for pred in preds:
        variant = pred["variant"]
        rank = pred["sample_rank"]
        path = OUT / f"cases/{variant}/{rank:03d}"
        if not (path / "score.json").exists():
            if (path / "failure.json").exists():
                failures.append(json.loads((path / "failure.json").read_text()))
            else:
                pending.append(dict(variant=variant, rank=rank))
            continue
        result = json.loads((path / "score.json").read_text())
        gt = gts[rank]
        assert (
            result["record_uid"] == pred["record_uid"]
            and result["gt_sha256"] == gt["caption_sha256"]
        )
        assert result["prediction_sha256"] == pred["original_prediction_sha256"]
        assert result["judge_model"] == MODEL and result["judge_reasoning_effort"] == EFFORT
        p = (
            json.loads(pred["evaluation_caption"])
            if not result["invalid_prediction"]
            else {
                "elements": {"layout": []},
                "high_level_description": "",
                "normalized_edit_instruction": "",
            }
        )
        geo = metrics.prepare(p, gt["caption"])
        assert geo == json.loads((path / "geometry.json").read_text())
        judge = json.loads((path / "judgment.json").read_text())
        Draft202012Validator(JUDGE_SCHEMA).validate(judge)
        metrics.validate_judge(judge, geo, gt["raw_ir"]["events"])
        verified = metrics.score(p, gt["caption"], geo, judge)
        for k, v in verified.items():
            observed = result["metrics"][k]
            # Python 3.12+ improves float summation versus the 3.11 UTP image.
            assert (
                v == observed if v is None or isinstance(v, bool) else abs(v - observed) <= 1e-12
            ), (variant, rank, k)
        if result["judge_called"]:
            providers = [
                json.loads(p.read_text()) for p in path.glob("judge-attempt-*/response.json")
            ]
            provider = next(p for p in providers if p["id"] == result["response_id"])
            assert provider["model"] == MODEL or provider["model"].startswith(MODEL + "-")
            assert provider["reasoning"]["effort"] == EFFORT and provider["status"] == "completed"
            texts = [
                b["text"]
                for x in provider["output"]
                if x["type"] == "message"
                for b in x["content"]
                if b["type"] == "output_text"
            ]
            assert len(texts) == 1 and json.loads(texts[0]) == judge
        if (variant, rank) in static:
            fixed = static[(variant, rank)]
            for k in (
                "target_region_pq",
                "target_match_f1",
                "reference_pair_pq",
                "reference_iou_conditional",
            ):
                assert (
                    fixed[k] == verified[k]
                    if fixed[k] is None
                    else abs(fixed[k] - verified[k]) <= 1e-12
                ), (variant, rank, k)
            result["metrics"]["target_caption_embedding_f1"] = fixed["target_caption_embedding_f1"]
        for kind in ("edit", "keep"):
            regions_p = p["elements"]["layout"]
            regions_g = gt["caption"]["elements"]["layout"]
            den = sum(len(metrics.refs(r)) for r in regions_p if metrics.origin(r) == kind) + sum(
                len(metrics.refs(r)) for r in regions_g if metrics.origin(r) == kind
            )
            accepted = [
                e
                for e in geo["edges"]
                if metrics.origin(regions_p[e["pred_index"]]) == kind
                and metrics.origin(regions_g[e["gt_index"]]) == kind
            ]
            result["metrics"][kind + "_reference_pair_pq"] = (
                2 * sum(min(e["target_iou"], e["reference_iou"]) for e in accepted) / den
                if den
                else None
            )
        scored.append(result)
        for issue in judge["gt_concerns"]:
            concerns.append(dict(variant=variant, rank=rank, **issue))
        kinds = {e["id"]: e["kind"] for e in gt["raw_ir"]["events"]}
        for e in judge["events"]:
            x = events[variant][kinds[e["gt_event_id"]]]
            x[0] += bool(e["covered"] and not e["contradicted"])
            x[1] += 1
    for p in OUT.glob("cases/*/*/judge-attempt-*/response.json"):
        variant = p.parents[2].name
        u = json.loads(p.read_text()).get("usage") or {}
        usage[variant]["responses"] += 1
        for k in ("input_tokens", "output_tokens", "total_tokens"):
            usage[variant][k] += u.get(k, 0)
    summary = dict(
        requested_cases=case_count,
        requested_tasks=task_count,
        scored=len(scored),
        failed=len(failures),
        pending=len(pending),
        static_complete=len(static) == task_count,
        judge_model=MODEL,
        judge_reasoning_effort=EFFORT,
        gt_source="externally_frozen",
        complete=len(scored) == task_count and len(static) == task_count,
        by_variant={},
        failures=failures,
        pending_tasks=pending,
        gt_concerns=concerns,
        usage={k: dict(v) for k, v in usage.items()},
        timestamp=time.time(),
    )
    for variant in variants:
        rows = [r for r in scored if r["variant"] == variant]
        summary["by_variant"][variant] = dict(
            scored=len(rows),
            native_invalid_predictions=sum(bool(r["invalid_prediction"]) for r in rows),
            overall=aggregate(rows),
            by_cohort={
                c: aggregate([r for r in rows if r["test_cohort"] == c])
                for c in sorted({r["test_cohort"] for r in rows})
            },
            gt_concern_cases=len({x["rank"] for x in concerns if x["variant"] == variant}),
            event_recall_by_kind={
                k: dict(covered=v[0], total=v[1], recall=v[0] / v[1])
                for k, v in events[variant].items()
            },
        )
    paired = {}
    indexed = {(r["variant"], r["rank"]): r for r in scored}
    for metric, label in MAIN:
        pairs = []
        for rank in gts:
            a = indexed.get(("original", rank))
            b = indexed.get(("checkpoint_11000", rank))
            if (
                a
                and b
                and a["metrics"].get(metric) is not None
                and b["metrics"].get(metric) is not None
            ):
                pairs.append(float(b["metrics"][metric]) - float(a["metrics"][metric]))
        if pairs:
            paired[metric] = dict(
                n=len(pairs), mean_checkpoint_minus_original=float(np.mean(pairs))
            )
    summary["paired_comparison"] = paired
    save("summary.json", summary)
    save("gt-concerns.json", concerns)
    print(
        dump(
            {k: summary[k] for k in ("scored", "failed", "pending", "static_complete", "complete")}
        ),
        flush=True,
    )
    if scored:
        flat = [
            dict(
                variant=r["variant"],
                rank=r["rank"],
                record_uid=r["record_uid"],
                cohort=r["test_cohort"],
                benchmark_group=r["benchmark_group"],
                invalid_prediction=r["invalid_prediction"],
                **r["metrics"],
            )
            for r in scored
        ]
        with (OUT / "per-case.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(flat[0]))
            writer.writeheader()
            writer.writerows(flat)
    lines = [
        f"# Paired editing-caption evaluation: {case_count} frozen cases",
        "",
        f"{MODEL} ({EFFORT}) scored {len(scored)}/{task_count} tasks; failed {len(failures)}; pending {len(pending)}; embedding complete: {len(static) == task_count}.",
        "",
        "GT: Astra medium, frozen before evaluation. Original and checkpoint-11000 use the same frozen identities. Native checkpoint failures remain in the denominator with zero scores. Shared upstream parses and the selected GT-success subset limit interpretation as human accuracy or full population performance.",
        "",
        "| Metric | Original macro | Checkpoint-11000 macro | Paired delta (checkpoint − original) | Paired N |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric, label in MAIN:
        a = summary["by_variant"]["original"]["overall"].get(metric)
        b = summary["by_variant"]["checkpoint_11000"]["overall"].get(metric)
        d = paired.get(metric)
        if a and b and d:
            lines.append(
                f"| {label} | {a['macro']:.4f} | {b['macro']:.4f} | {d['mean_checkpoint_minus_original']:+.4f} | {d['n']} |"
            )
    lines += [
        "",
        "Higher is better except critical-error rates. Conditional reference IoU excludes unmatched edges. The weighted summaries renormalize frame weights over the selected applicable subset; see summary.json for denominators, cohort results, events, concerns and usage.",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")
    if args.publish:
        assert args.s3_output, "--s3-output required to publish"
        if not summary["complete"]:
            raise RuntimeError(
                "Cannot publish final metrics before all requested scores and embedding metrics are complete"
            )
        names = [
            "summary.json",
            "per-case.csv",
            "REPORT.md",
            "gt-concerns.json",
            "embedding-audit.json",
        ]
        manifest = dict(
            cases=case_count,
            tasks=task_count,
            judge_model=MODEL,
            judge_reasoning_effort=EFFORT,
            artifacts=[],
        )
        for name in names:
            body = (OUT / name).read_bytes()
            key = PREFIX + "exports/" + name
            s3.put_object(Bucket=BUCKET, Key=key, Body=body)
            manifest["artifacts"].append(
                dict(
                    name=name,
                    uri="s3://" + BUCKET + "/" + key,
                    sha256=hashlib.sha256(body).hexdigest(),
                    bytes=len(body),
                )
            )
        save("metrics-manifest.json", manifest)
        s3.put_object(
            Bucket=BUCKET,
            Key=PREFIX + "exports/metrics-manifest.json",
            Body=(OUT / "metrics-manifest.json").read_bytes(),
        )


if __name__ == "__main__":
    main()
