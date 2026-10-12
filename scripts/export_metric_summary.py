#!/usr/bin/env python3
"""Verify a private viewer index and export only the public aggregate table."""

import argparse
import csv
import hashlib
import html
import json
import math
import statistics
from pathlib import Path

CONSISTENCY = "global_local_instruction_consistency_f1"
KEYS = ("original", "ckpt11k", "ckpt4k", "ckpt6k", "ckpt9k")
METRICS = (
    "target_region_pq",
    "target_desc_f1",
    "target_caption_embedding_f1",
    "reference_grounded_pq",
    "local_edit_instruction_f1",
    "editing_edge_joint_f1",
    "global_description_semantic_f1",
    "global_instruction_semantic_f1",
    "global_description_critical_error",
    "global_instruction_critical_error",
    CONSISTENCY,
)
STEM = "five-way-metrics-20261009"


def aggregate(index):
    """Recompute the actual five-way cohort; copy only allowlisted public fields."""
    assert tuple(v["key"] for v in index["variants"]) == KEYS
    assert tuple(m["key"] for m in index["metrics"]) == METRICS
    cases = index["cases"]
    assert len({c["rank"] for c in cases}) == len(cases)
    metrics = []
    for metric in index["metrics"]:
        key = metric["key"]
        shared = [
            c
            for c in cases
            if all(
                not c["variants"][k]["invalid"] and c["variants"][k]["metrics"][key] is not None
                for k in KEYS
            )
        ]
        assert shared and len(shared) == metric["n"], (key, "paired count mismatch")
        values = {}
        for variant in KEYS:
            scores = [c["variants"][variant]["metrics"][key] for c in shared]
            assert all(
                isinstance(v, (int, float)) and math.isfinite(v) and 0 <= v <= 1 for v in scores
            )
            if key == CONSISTENCY:
                assert all(
                    c["variants"][variant]["consistency"]["status"] == "scored" for c in shared
                )
            value = statistics.mean(scores)
            assert math.isclose(value, metric["values"][variant], rel_tol=0, abs_tol=1e-12), (
                key,
                variant,
                "mean mismatch",
            )
            # Retain the original published precision after independent verification.
            values[variant] = metric["values"][variant]
        metrics.append(
            {
                "key": key,
                "label": metric["label"],
                "anchor": metric["anchor"],
                "lower_is_better": metric["lower_is_better"],
                "aggregation": "macro",
                "n": len(shared),
                "values": values,
            }
        )
    audit = index["audit"]["consistency"]
    coverage = {
        key: {
            status: sum(c["variants"][key]["consistency"]["status"] == status for c in cases)
            for status in ("scored", "native_invalid", "not_applicable", "failed", "user_excluded")
        }
        for key in KEYS
    }
    assert sum(c["scored"] for c in coverage.values()) == audit["scored_captions"]
    assert sum(c["user_excluded"] for c in coverage.values()) == audit["excluded_captions"]
    assert metrics[-1]["n"] == audit["paired_n"]
    return {
        "version": "five-way-eleven-metrics-20261009",
        "cases": len(cases),
        "variants": [{"key": v["key"], "label": v["label"]} for v in index["variants"]],
        "metrics": metrics,
        "aggregation": "Per-metric five-way native-valid, non-null intersection; equal case weights.",
        "judges": {"base": "gpt-6-luna / low", "consistency": "gpt-6.1-sol / medium"},
        "consistency": {
            **{
                k: audit[k]
                for k in (
                    "protocol_version",
                    "eligible_cases",
                    "not_applicable_cases",
                    "scored_captions",
                    "excluded_captions",
                    "paired_n",
                )
            },
            "coverage": coverage,
        },
    }


def displayed(value, lower):
    return f"{100 * value:.2f}%" if lower else f"{value:.4f}"


def export(index_path, output):
    raw = index_path.read_bytes()
    summary = aggregate(json.loads(raw))
    summary["source_index_sha256"] = hashlib.sha256(raw).hexdigest()
    for directory in ("data", "docs", "site"):
        (output / directory).mkdir(parents=True, exist_ok=True)
    (output / "data" / f"{STEM}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    with (output / "data" / f"{STEM}.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["metric", "n", *KEYS, "lower_is_better", "aggregation"])
        for metric in summary["metrics"]:
            writer.writerow(
                [
                    metric["key"],
                    metric["n"],
                    *(metric["values"][k] for k in KEYS),
                    metric["lower_is_better"],
                    metric["aggregation"],
                ]
            )
    labels = [v["label"] for v in summary["variants"]]
    headers = ["Metric", "N", *labels]
    rows = [
        [
            m["label"] + (" ↓" if m["lower_is_better"] else " ↑"),
            str(m["n"]),
            *(displayed(m["values"][k], m["lower_is_better"]) for k in KEYS),
        ]
        for m in summary["metrics"]
    ]
    notes = [
        (
            "Snapshot of the 477-case Highway summary with five variants and eleven metrics. "
            "Each row uses the same native-valid, non-null case intersection across all five variants. "
            "Cases receive equal weight; inference failures and missing scores are never filled with zero."
        ),
        (
            "Quality scores use [0, 1]; critical-error rates are percentages. The CSV and JSON store "
            "all values in [0, 1] at their original precision."
        ),
        (
            "Base semantic judge: gpt-6-luna / low. Consistency judge: gpt-6.1-sol / medium. "
            f"Consistency: {summary['consistency']['eligible_cases']} eligible cases, "
            f"{summary['consistency']['scored_captions']:,} scored captions, "
            f"{summary['consistency']['excluded_captions']} user-excluded caption, and "
            f"{summary['consistency']['paired_n']} common scored cases. "
            "The exclusion does not change the shared cohort. Removal PQ and checkpoint-2k "
            "are absent from the active table."
        ),
        (
            "6k and 9k use a different inference prompt from 11k Init. + 4k. "
            "6k inference used H100; 9k used the devbox. Comparisons include these differences. "
            "Instruction consistency measures agreement within a caption, not image-grounded correctness."
        ),
    ]
    markdown = [
        "# Five-way metric summary",
        "",
        notes[0],
        "",
        "| " + " | ".join(headers) + " |",
        "|---|---:" + "|---:" * len(KEYS) + "|",
    ]
    markdown += ["| " + " | ".join(row) + " |" for row in rows]
    markdown += [
        "",
        *[note + "\n" for note in notes[1:]],
        (
            f"[CSV](../data/{STEM}.csv) · [JSON](../data/{STEM}.json) · "
            "[Metric definitions](metrics.md) · "
            "[Live cases](https://grantzhang-alpha-document.highway.canva-internal.dev/#results)"
        ),
        "",
    ]
    (output / "docs/results.md").write_text("\n".join(markdown))
    table = "<thead><tr>" + "".join("<th>" + html.escape(v) + "</th>" for v in headers)
    table += "</tr></thead><tbody>"
    for row in rows:
        table += "<tr>" + "".join("<td>" + html.escape(v) + "</td>" for v in row) + "</tr>"
    table += "</tbody>"
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Editing Caption · Five-way metric summary</title><link rel="icon" href="data:,">
<style>body{{font:16px/1.6 system-ui,sans-serif;color:#193730;background:#f7faf8;margin:0;padding:24px}}
main{{max-width:1500px;margin:auto}}a{{color:#08744f}}.table-wrap{{overflow-x:auto}}
table{{border-collapse:collapse;background:white;width:100%;font-size:14px}}
th,td{{padding:12px;border-bottom:1px solid #dce5df;text-align:right;white-space:nowrap}}
th:first-child,td:first-child{{text-align:left}}thead{{background:#eaf3ed}}p{{max-width:1100px}}</style>
</head><body><main><a href="index.html">Metric definitions</a><h1>Five-way metric summary</h1>
<p>{html.escape(notes[0])}</p><div class="table-wrap"><table>{table}</table></div>
{"".join("<p>" + html.escape(note) + "</p>" for note in notes[1:])}
<p><a href="../data/{STEM}.csv" download>Download CSV</a> · <a href="../data/{STEM}.json">JSON</a> ·
<a href="https://grantzhang-alpha-document.highway.canva-internal.dev/#results">Live cases (WARP)</a></p>
</main></body></html>
"""
    (output / "site/summary.html").write_text(page)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    summary = export(args.index, args.output)
    print(f"Verified and exported {len(summary['metrics'])} metrics × {len(KEYS)} variants.")


if __name__ == "__main__":
    main()
