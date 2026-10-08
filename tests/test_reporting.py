from benchmark.reporting import SELECTED_METRICS, paired_comparison, reporting_rows


def test_static_metrics_keep_full_coverage_when_one_judge_fails():
    predictions = [
        dict(variant=v, sample_rank=r, record_uid=str(r), invalid_reason=None)
        for r in (1, 2)
        for v in ("original", "checkpoint_11000")
    ]
    samples = {
        r: dict(test_cohort="test", benchmark_group="edit", analysis_weight=1) for r in (1, 2)
    }
    static = {
        (p["variant"], p["sample_rank"]): dict(
            target_region_pq=1 if p["variant"] == "original" else 0.5,
            target_caption_embedding_f1=0.8,
        )
        for p in predictions
    }
    scored = [
        dict(
            variant=p["variant"],
            rank=p["sample_rank"],
            metrics=dict(
                target_desc_f1=0.9 if p["variant"] == "original" else 0.6, event_recall=0.99
            ),
        )
        for p in predictions
        if (p["variant"], p["sample_rank"]) != ("checkpoint_11000", 2)
    ]
    rows = reporting_rows(
        predictions, samples, scored, static, [dict(variant="checkpoint_11000", rank=2)]
    )
    assert len(SELECTED_METRICS) == 10
    assert all(list(r["metrics"]) == [k for k, _ in SELECTED_METRICS] for r in rows)
    assert rows[-1]["status"] == "failed"
    assert rows[-1]["metrics"]["target_desc_f1"] is None
    pairs = paired_comparison(rows)
    assert pairs["target_region_pq"]["n"] == 2
    assert pairs["target_desc_f1"]["n"] == 1
    assert pairs["target_region_pq"]["original_macro"] == 1
    assert pairs["target_region_pq"]["checkpoint_11000_macro"] == 0.5
    assert pairs["target_region_pq"]["mean_checkpoint_minus_original"] == -0.5
    assert "event_recall" not in pairs


def test_paired_means_exclude_unpaired_values_and_keep_zero_scores():
    rows = [
        dict(variant="original", rank=1, metrics={"target_desc_f1": 0.8}),
        dict(variant="checkpoint_11000", rank=1, metrics={"target_desc_f1": 0}),
        dict(variant="original", rank=2, metrics={"target_desc_f1": 1}),
        dict(variant="checkpoint_11000", rank=2, metrics={"target_desc_f1": None}),
    ]
    score = paired_comparison(rows)["target_desc_f1"]
    assert score == dict(
        n=1, original_macro=0.8, checkpoint_11000_macro=0, mean_checkpoint_minus_original=-0.8
    )
