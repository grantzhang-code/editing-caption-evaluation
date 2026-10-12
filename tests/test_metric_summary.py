"""Protect paired coverage and aggregate-only publication."""

import copy
import json

import pytest

from scripts.export_metric_summary import CONSISTENCY, KEYS, METRICS, aggregate, export


def fixture_index():
    values = dict(zip(KEYS, [0.2, 0.4, 0.6, 0.8, 1.0]))
    cases = []
    for rank in range(3):
        variants = {
            k: {
                "invalid": False,
                "raw": "PRIVATE_CAPTION",
                "metrics": dict.fromkeys(METRICS, v),
                "consistency": {"status": "scored"},
            }
            for k, v in values.items()
        }
        cases.append({"rank": rank, "uid": "PRIVATE_CASE_ID", "variants": variants})
    cases[1]["variants"]["ckpt11k"].update(invalid=True, consistency={"status": "native_invalid"})
    cases[2]["variants"]["ckpt4k"].update(
        metrics=dict.fromkeys(METRICS), consistency={"status": "user_excluded"}
    )
    # These high scores must not enter any five-way paired mean.
    for case in cases[1:]:
        case["variants"]["original"]["metrics"] = dict.fromkeys(METRICS, 1.0)
    return {
        "variants": [{"key": k, "label": k, "private": "PRIVATE_VARIANT"} for k in KEYS],
        "cases": cases,
        "metrics": [
            {
                "key": m,
                "label": m,
                "anchor": m,
                "lower_is_better": "critical_error" in m,
                "n": 1,
                "values": values.copy(),
            }
            for m in METRICS
        ],
        "audit": {
            "consistency": {
                "protocol_version": "test",
                "eligible_cases": 3,
                "not_applicable_cases": 0,
                "scored_captions": 13,
                "excluded_captions": 1,
                "paired_n": 1,
                "private": "PRIVATE_AUDIT",
            }
        },
    }


def test_invalid_and_missing_cases_are_excluded_without_zero_fill():
    result = aggregate(fixture_index())
    assert all(m["n"] == 1 and m["values"]["original"] == 0.2 for m in result["metrics"])
    assert result["consistency"]["coverage"]["ckpt4k"]["user_excluded"] == 1


@pytest.mark.parametrize("field", ["mean", "n", "status"])
def test_stale_or_unscored_summary_is_rejected(field):
    index = fixture_index()
    if field == "mean":
        index["metrics"][0]["values"]["original"] = 0.9
    elif field == "n":
        index["metrics"][0]["n"] = 2
    else:
        index["cases"][0]["variants"]["original"]["consistency"]["status"] = "failed"
    with pytest.raises(AssertionError):
        aggregate(index)


def test_export_does_not_include_private_case_or_audit_fields(tmp_path):
    index = fixture_index()
    original = copy.deepcopy(index)
    path = tmp_path / "input.json"
    path.write_text(json.dumps(index))
    output = tmp_path / "public"
    result = export(path, output)
    assert index == original and result["metrics"][-1]["key"] == CONSISTENCY
    assert len(list(output.rglob("*.*"))) == 4
    assert all("PRIVATE_" not in p.read_text() for p in output.rglob("*.*"))
