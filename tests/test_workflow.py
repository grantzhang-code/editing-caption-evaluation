"""Offline integration checks for release binding, invalid rows and summaries."""

import copy
import json
import sys
from pathlib import Path

import pytest

from benchmark import evaluate, runtime, static_metrics, summarize
from benchmark.data import image_bytes, load_dataset

FIXTURE = Path(__file__).resolve().parents[1] / "examples/synthetic.jsonl"


def dataset(tmp_path, mutate=None):
    row = json.loads(FIXTURE.read_text())
    for image in row["images"].values():
        image["uri"] = str(FIXTURE.parent / "pixel.png")
    if mutate:
        mutate(row)
    path = tmp_path / "input.jsonl"
    path.write_text(json.dumps(row) + "\n")
    return path


def test_rejects_changed_gt(tmp_path):
    path = dataset(
        tmp_path, lambda row: row["gt"]["caption"].update(high_level_description="tampered")
    )
    with pytest.raises(ValueError, match="GT caption SHA256"):
        load_dataset(str(path))


def test_rejects_missing_variant(tmp_path):
    path = dataset(tmp_path, lambda row: row["predictions"].pop("original"))
    with pytest.raises(ValueError, match="both frozen variants"):
        load_dataset(str(path))


def test_rejects_changed_prediction_with_unchanged_raw_hash(tmp_path):
    path = dataset(
        tmp_path,
        lambda row: row["predictions"]["original"]["caption"].update(
            high_level_description="tampered"
        ),
    )
    with pytest.raises(ValueError, match="bound raw text"):
        load_dataset(str(path))


def test_rejects_changed_image(tmp_path):
    row = json.loads(dataset(tmp_path).read_text())
    image = copy.deepcopy(row["images"]["source"])
    assert image_bytes(image)
    image["sha256"] = "0" * 64
    with pytest.raises(AssertionError, match="image SHA256"):
        image_bytes(image)


def test_invalid_predictions_complete_without_api_or_aws(tmp_path, monkeypatch):
    def invalidate(row):
        for prediction in row["predictions"].values():
            prediction.update(status="length_truncated", error="truncated")

    path = dataset(tmp_path, invalidate)
    items, gts, samples, preds = load_dataset(str(path))
    monkeypatch.setattr(evaluate, "ITEMS", items)
    monkeypatch.setattr(evaluate, "GTS", gts)
    monkeypatch.setattr(evaluate, "SAMPLES", samples)
    monkeypatch.setattr(runtime, "OUT", tmp_path / "results")
    monkeypatch.setattr(runtime, "S3", None)
    monkeypatch.setattr(runtime, "STATE", dict(scored=0, failed=0))

    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid rows must not call a provider")

    monkeypatch.setattr(runtime, "call", forbidden)
    for prediction in preds:
        assert evaluate.one(prediction, "unused")["status"] == "scored"
        assert evaluate.one(prediction, "unused")["reused"]
    monkeypatch.setattr(
        sys, "argv", ["static", "--dataset", str(path), "--results", str(runtime.OUT)]
    )
    static_metrics.main()
    monkeypatch.setattr(
        sys, "argv", ["summary", "--dataset", str(path), "--results", str(runtime.OUT)]
    )
    summarize.main()
    summary = json.loads((runtime.OUT / "summary.json").read_text())
    assert summary["complete"] and summary["scored"] == 2 and summary["requested_cases"] == 1
    for variant in summary["by_variant"].values():
        assert variant["native_invalid_predictions"] == 1
        assert variant["overall"]["global_instruction_semantic_f1"]["macro"] == 0
    monkeypatch.setattr(evaluate, "MODEL", "different-judge")
    assert evaluate.one(preds[0], "unused")["status"] == "failed"
