"""The selected ten metrics and paired reporting independent of judge coverage."""

import statistics

REPORT_VERSION = "selected-ten-v1"
SELECTED_METRICS = [
    ("target_region_pq", "Target RegionPQ"),
    ("target_desc_f1", "Target Desc F1"),
    ("target_caption_embedding_f1", "Caption Embedding F1"),
    ("reference_grounded_pq", "Reference GroundedPQ"),
    ("local_edit_instruction_f1", "Local Edit Instruction F1"),
    ("editing_edge_joint_f1", "Editing Edge Joint F1"),
    ("global_description_semantic_f1", "Global HLD F1"),
    ("global_instruction_semantic_f1", "Global Instruction F1"),
    ("global_description_critical_error", "Global HLD critical-error rate"),
    ("global_instruction_critical_error", "Global Instruction critical-error rate"),
]


def reporting_rows(predictions, samples, scored, static, failures):
    indexed = {(r["variant"], r["rank"]): r for r in scored}
    failed = {(r["variant"], r["rank"]) for r in failures}
    rows = []
    for pred in predictions:
        key = pred["variant"], pred["sample_rank"]
        score = indexed.get(key)
        values = dict(score["metrics"]) if score else {}
        # Static coverage is independent of judge failures.
        for metric in ("target_region_pq", "target_caption_embedding_f1"):
            if metric in static.get(key, {}):
                values[metric] = static[key][metric]
        rows.append(
            dict(
                variant=key[0],
                rank=key[1],
                record_uid=pred["record_uid"],
                **samples[key[1]],
                invalid_prediction=pred["invalid_reason"],
                status="scored" if score else "failed" if key in failed else "pending",
                metrics={name: values.get(name) for name, _ in SELECTED_METRICS},
            )
        )
    return rows


def paired_comparison(rows):
    indexed = {(r["variant"], r["rank"]): r["metrics"] for r in rows}
    ranks = sorted({r["rank"] for r in rows})
    result = {}
    for metric, _ in SELECTED_METRICS:
        pairs = []
        for rank in ranks:
            a = indexed.get(("original", rank), {}).get(metric)
            b = indexed.get(("checkpoint_11000", rank), {}).get(metric)
            if a is not None and b is not None:
                pairs.append((float(a), float(b)))
        if pairs:
            result[metric] = dict(
                n=len(pairs),
                original_macro=statistics.mean(a for a, b in pairs),
                checkpoint_11000_macro=statistics.mean(b for a, b in pairs),
                mean_checkpoint_minus_original=statistics.mean(b - a for a, b in pairs),
            )
    return result
