"""Paired original/checkpoint-11000 evaluation against frozen model-assisted GTs."""

import argparse, base64, concurrent.futures, json, threading, time, traceback
from pathlib import Path
from jsonschema import Draft202012Validator, ValidationError
from . import metrics
from . import runtime as rt
from .bench_contracts import MODEL, EFFORT, JUDGE_SCHEMA, JUDGE_PROMPT, VERSION
from .data import load_dataset, image_bytes

ROOT = Path(__file__).resolve().parents[1]
ITEMS = {}
GTS = {}
SAMPLES = {}
PREDS = []
IMAGES = {}


def valid_prediction(p):
    assert (
        isinstance(p, dict)
        and isinstance(p["high_level_description"], str)
        and isinstance(p["normalized_edit_instruction"], str)
    )
    regions = p["elements"]["layout"]
    assert isinstance(regions, list)
    labels = [r["label"] for r in regions]
    assert len(labels) == len(set(labels))
    for r in regions:
        assert metrics.valid_box(r["bbox"]) and isinstance(r["desc"], str)
        assert not ("keep" in r and "edit" in r)
        for ref in metrics.refs(r):
            assert metrics.valid_box(ref["reference_bbox"]) and ref["reference_id"] == "frame_0"


def preflight(rank):
    item = ITEMS[rank]
    g = GTS[rank]
    assert g["caption_sha256"] == rt.digest(rt.ordered(g["caption"]))
    parts = []
    for role in ("source", "target"):
        spec = item["images"][role]
        body = image_bytes(spec)
        parts.extend(
            [
                dict(type="input_text", text="SOURCE frame_0" if role == "source" else "TARGET"),
                dict(
                    type="input_image",
                    detail="high",
                    image_url="data:" + spec["mime"] + ";base64," + base64.b64encode(body).decode(),
                ),
            ]
        )
    IMAGES[rank] = parts
    rt.write(
        f"inputs-verified/{rank:03d}.json",
        dict(record_uid=item["record_uid"], gt_sha256=g["caption_sha256"]),
    )
    with rt.LOCK:
        rt.STATE["preflight_ok"] += 1


def decode(provider):
    assert provider.get("status") == "completed", str(provider.get("incomplete_details"))
    assert provider.get("model") == MODEL or provider.get("model", "").startswith(MODEL + "-")
    assert provider.get("reasoning", {}).get("effort") == EFFORT
    parts = [
        b
        for x in provider.get("output", [])
        if x.get("type") == "message"
        for b in x.get("content", [])
    ]
    if any(b.get("type") == "refusal" for b in parts):
        raise RuntimeError("provider_refusal")
    texts = [b["text"] for b in parts if b.get("type") == "output_text"]
    assert len(texts) == 1
    value = json.loads(texts[0])
    Draft202012Validator(JUDGE_SCHEMA).validate(value)
    return value


def one(pred, key, start_attempt=0):
    rank = pred["sample_rank"]
    variant = pred["variant"]
    name = f"cases/{variant}/{rank:03d}"
    gt = GTS[rank]
    sample = SAMPLES[rank]
    try:
        previous = rt.read(name + "/score.json")
        if previous:
            assert (
                previous["gt_sha256"] == gt["caption_sha256"]
                and previous["prediction_sha256"] == pred["original_prediction_sha256"]
            )
            assert previous["record_uid"] == pred["record_uid"]
            assert previous["judge_model"] == MODEL and previous["judge_reasoning_effort"] == EFFORT
            assert previous.get("protocol_version") == VERSION, (
                "Protocol changed; use a new output directory"
            )
            with rt.LOCK:
                rt.STATE["scored"] += 1
            return dict(variant=variant, rank=rank, status="scored", reused=True)
        invalid = pred["invalid_reason"]
        try:
            assert invalid is None, invalid
            p = json.loads(pred["evaluation_caption"])
            valid_prediction(p)
        except Exception as e:
            invalid = invalid or type(e).__name__ + ": " + str(e)
            p = {
                "elements": {"layout": []},
                "high_level_description": "",
                "normalized_edit_instruction": "",
            }
        geo = metrics.prepare(p, gt["caption"])
        events = gt["raw_ir"]["events"]
        rt.write(name + "/geometry.json", geo)
        provider = None
        if invalid:
            zero = dict(
                precision=0,
                recall=0,
                critical_error=False,
                evidence="Invalid prediction counted as empty by the frozen protocol.",
            )
            judgment = dict(
                gt_concerns=[],
                regions=[],
                edges=[],
                global_description=zero,
                global_instruction=zero,
                events=[
                    dict(
                        gt_event_id=e["id"],
                        covered=False,
                        contradicted=False,
                        evidence="Invalid prediction",
                    )
                    for e in events
                ],
                error_tags=["invalid_prediction"],
            )
        else:
            context = dict(
                geometry=geo,
                prediction_global={
                    k: p[k] for k in ("high_level_description", "normalized_edit_instruction")
                },
                gt_global={
                    k: gt["caption"][k]
                    for k in ("high_level_description", "normalized_edit_instruction")
                },
                gt_events=events,
                pred_removals=p.get("elements", {}).get("remove", []),
                prediction_is_invalid=False,
            )
            body = dict(
                model=MODEL,
                reasoning={"effort": EFFORT},
                max_output_tokens=32768,
                store=False,
                instructions=JUDGE_PROMPT,
                input=[
                    dict(
                        role="user",
                        content=IMAGES[rank]
                        + [
                            dict(
                                type="input_text",
                                text=json.dumps(context, ensure_ascii=False, separators=(",", ":")),
                            )
                        ],
                    )
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "editing_benchmark_judge",
                        "strict": True,
                        "schema": JUDGE_SCHEMA,
                    }
                },
            )
            rt.write(
                name + "/request-audit.json",
                dict(
                    model=MODEL,
                    reasoning_effort=EFFORT,
                    request_sha256=rt.digest(rt.canonical(body)),
                    gt_sha256=gt["caption_sha256"],
                    prediction_sha256=pred["original_prediction_sha256"],
                    evaluation_caption_sha256=rt.digest(pred["evaluation_caption"].encode()),
                    judge_prompt_sha256=rt.digest(JUDGE_PROMPT.encode()),
                    judge_schema_sha256=rt.digest(rt.canonical(JUDGE_SCHEMA)),
                    candidate_model_identity_visible=False,
                ),
            )
            for attempt in range(start_attempt, start_attempt + 2):
                provider = rt.call(name + f"/judge-attempt-{attempt}", body, key)
                try:
                    judgment = decode(provider)
                    metrics.validate_judge(judgment, geo, events)
                    break
                except (ValueError, AssertionError, KeyError, TypeError, ValidationError) as e:
                    rt.write(
                        name + f"/judge-validation-error-{attempt}.json",
                        dict(error=type(e).__name__ + ": " + str(e)),
                    )
                    if attempt == start_attempt + 1:
                        raise
        metrics.validate_judge(judgment, geo, events)
        rt.write(name + "/judgment.json", judgment)
        scores = metrics.score(p, gt["caption"], geo, judgment)
        for threshold in (0.3, 0.75):
            z = metrics.prepare(p, gt["caption"], threshold)
            d = z["np"] + z["ng"]
            scores[f"target_region_pq_at_{threshold}"] = (
                2 * sum(x["target_iou"] for x in z["pairs"]) / d if d else 0.0
            )
        result = dict(
            rank=rank,
            variant=variant,
            record_uid=pred["record_uid"],
            test_cohort=sample["test_cohort"],
            benchmark_group=sample["benchmark_group"],
            analysis_weight=sample["analysis_weight"],
            status="scored",
            invalid_prediction=invalid,
            gt_sha256=gt["caption_sha256"],
            prediction_sha256=pred["original_prediction_sha256"],
            protocol_version=VERSION,
            judge_model=MODEL,
            judge_reasoning_effort=EFFORT,
            judge_called=provider is not None,
            response_id=provider.get("id") if provider else None,
            usage=provider.get("usage") if provider else None,
            metrics=scores,
        )
        rt.write(name + "/score.json", result)
        with rt.LOCK:
            rt.STATE["scored"] += 1
        print(
            json.dumps(dict(rank=rank, variant=variant, status="scored", invalid=bool(invalid))),
            flush=True,
        )
        return dict(rank=rank, variant=variant, status="scored")
    except Exception as e:
        result = dict(
            rank=rank,
            variant=variant,
            status="failed",
            error=type(e).__name__ + ": " + str(e),
            traceback=traceback.format_exc(),
        )
        rt.write(name + "/failure.json", result)
        with rt.LOCK:
            rt.STATE["failed"] += 1
        print(json.dumps({k: v for k, v in result.items() if k != "traceback"}), flush=True)
        return result


def main():
    global ITEMS, GTS, SAMPLES, PREDS
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--validate-only", action="store_true")
    args = ap.parse_args()
    ITEMS, GTS, SAMPLES, PREDS = load_dataset(args.dataset)
    if args.validate_only:
        for pred in PREDS:
            if not pred["invalid_reason"]:
                valid_prediction(json.loads(pred["evaluation_caption"]))
        print(json.dumps(dict(cases=len(ITEMS), tasks=len(PREDS))))
        return 0
    assert 1 <= args.concurrency <= 256
    assert len(ITEMS) == len(GTS) == len(SAMPLES)
    identity = dict(
        protocol_version=VERSION,
        judge_model=MODEL,
        judge_reasoning_effort=EFFORT,
        prompt_sha256=rt.digest(JUDGE_PROMPT.encode()),
        schema_sha256=rt.digest(rt.canonical(JUDGE_SCHEMA)),
        dataset_sha256=rt.digest(rt.canonical(dict(items=ITEMS, predictions=PREDS))),
    )
    previous = rt.read("run-contract.json")
    if previous is not None and previous != identity:
        raise ValueError("Output belongs to a different dataset or protocol")
    rt.write("run-contract.json", identity)
    done = threading.Event()

    def heartbeat():
        while not done.wait(30):
            with rt.LOCK:
                snapshot = dict(rt.STATE, requested=len(PREDS), timestamp=time.time())
            rt.write_run("progress.json", snapshot)
            print(json.dumps(snapshot), flush=True)

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        rt.STATE["stage"] = "preflight"
        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
            list(pool.map(preflight, ITEMS))
        rt.STATE["stage"] = "semantic_scoring"
        key = rt.api_key()
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            results = list(pool.map(lambda p: one(p, key), PREDS))
        rt.STATE["stage"] = (
            "complete"
            if all(x["status"] == "scored" for x in results)
            else "finished_with_failures"
        )
        rt.write_run("run-status.json", dict(rt.STATE, requested=len(PREDS), results=results))
        return 0 if rt.STATE["stage"] == "complete" else 1
    finally:
        done.set()
        thread.join(timeout=5)
        rt.write_run("progress.json", dict(rt.STATE, requested=len(PREDS), timestamp=time.time()))


if __name__ == "__main__":
    raise SystemExit(main())
