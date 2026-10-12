#!/usr/bin/env python3
"""Add verified Global–Local F1 exports to a staged results viewer."""

import argparse
import copy
import hashlib
import json
import math
import shutil
import statistics
from pathlib import Path

METRIC = "global_local_instruction_consistency_f1"


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(data, final, output):
    assert not output.exists(), "Use a new staging directory"
    manifest = read(final / "manifest.json")
    assert manifest["status"] == "PASS"
    names = set()
    for artifact in manifest["artifacts"]:
        name = artifact["name"]
        assert Path(name).name == name and name not in names
        names.add(name)
        path = final / name
        assert path.stat().st_size == artifact["bytes"] and digest(path) == artifact["sha256"]
    assert {"summary.json", "scores-verified.json", "protocol.json"} <= names
    summary, protocol = read(final / "summary.json"), read(final / "protocol.json")
    assert digest(data / "index.json") == protocol["source_index_sha256"]
    index = read(data / "index.json")
    keys = [v["key"] for v in index["variants"]]
    assert keys == [v["key"] for v in protocol["variants"]]
    rows = read(final / "scores-verified.json")
    by_key = {(r["variant"], r["rank"]): r for r in rows}
    assert len(by_key) == len(rows) == len(index["cases"]) * len(keys)
    old_metrics = copy.deepcopy(index["metrics"])
    assert not any(m["key"] == METRIC for m in old_metrics)
    paired = []
    scored = 0
    for case in index["cases"]:
        details = read(data / "cases" / f"{case['rank']:03d}.json")
        assert details["uid"] == case["uid"]
        for key in keys:
            row = by_key[key, case["rank"]]
            variant, detail = case["variants"][key], details["variants"][key]
            assert row["record_uid"] == case["uid"]
            assert row["gt_sha256"] == details["gt_sha256"]
            assert row["prediction_sha256"] == detail["sha256"]
            assert hashlib.sha256(detail["raw"].encode()).hexdigest() == detail["sha256"]
            assert row["protocol_version"] == protocol["version"]
            assert row["judge_model"] == protocol["judge_model"]
            assert row["judge_reasoning_effort"] == protocol["judge_reasoning_effort"]
            status, value = row["status"], row[METRIC]
            assert status in (
                "scored",
                "native_invalid",
                "not_applicable",
                "failed",
                "user_excluded",
            )
            assert bool(variant["invalid"]) == (status == "native_invalid")
            if status == "scored":
                assert row["eligible"] and isinstance(value, (int, float))
                assert math.isfinite(value) and 0 <= value <= 1
                ng, nl, matched = (
                    row[k]
                    for k in ("global_action_count", "local_action_count", "matched_action_count")
                )
                assert 0 <= matched <= min(ng, nl)
                assert value == (2 * matched / (ng + nl) if ng + nl else 0)
                scored += 1
            else:
                assert value is None
                if status == "not_applicable":
                    assert not row["eligible"]
                if status == "failed":
                    assert row["eligible"] and row["failure_class"] in (
                        "semantic_validation",
                        "transport_or_api",
                    )
                if status == "user_excluded":
                    assert row["eligible"] and row["exclusion_reason"]
            variant["metrics"][METRIC] = value
            variant["consistency"] = {
                field: row.get(field)
                for field in (
                    "status",
                    "eligible",
                    "global_action_count",
                    "local_action_count",
                    "matched_action_count",
                    "accepted_attempt",
                )
            }
        if all(by_key[key, case["rank"]]["status"] == "scored" for key in keys):
            paired.append(case["rank"])
    assert scored == summary["scored_captions"]
    assert sum(r["status"] == "failed" for r in rows) == summary["failed_captions"]
    assert sum(r["status"] == "user_excluded" for r in rows) == summary.get("excluded_captions", 0)
    assert sorted(paired) == summary["paired_ranks"] and len(paired) == summary["paired_n"]
    means = {key: statistics.mean(by_key[key, rank][METRIC] for rank in paired) for key in keys}
    assert means == summary["paired_means"]
    index["metrics"].append(
        {
            "key": METRIC,
            "label": "Global–Local Instruction Consistency F1",
            "anchor": "global-local-consistency",
            "lower_is_better": False,
            "aggregation": "macro",
            "status": "computed",
            "n": len(paired),
            "values": means,
        }
    )
    assert index["metrics"][:-1] == old_metrics
    index["audit"]["displayed_metrics"] = len(index["metrics"])
    index["audit"]["pending_metrics"] = []
    index["audit"]["consistency"] = {
        k: summary[k]
        for k in (
            "protocol_version",
            "judge_model",
            "judge_reasoning_effort",
            "eligible_cases",
            "not_applicable_cases",
            "scored_captions",
            "failed_captions",
            "fully_scored",
            "coverage",
            "paired_n",
            "source_input_sha256",
        )
    }
    index["audit"]["consistency"].update(
        {
            k: summary[k]
            for k in ("excluded_captions", "excluded_cases", "execution_closed", "closure_reason")
            if k in summary
        }
    )
    output.mkdir(parents=True)
    for name, value in (
        ("index.json", index),
        ("metrics.json", index["metrics"]),
        ("audit.json", index["audit"]),
    ):
        (output / name).write_text(
            json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        )
    (output / "cases").symlink_to((data / "cases").resolve(), target_is_directory=True)
    (output / "consistency").mkdir()
    for name in sorted(names | {"manifest.json"}):
        shutil.copy2(final / name, output / "consistency" / name)
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--final", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    index = build(args.data, args.final, args.output)
    print(json.dumps(index["audit"]["consistency"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
