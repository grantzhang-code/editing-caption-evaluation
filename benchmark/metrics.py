"""Geometry-only matching and editing extension of ImageParser section 4.3."""

import re
import numpy as np
from scipy.optimize import linear_sum_assignment


def valid_box(b):
    return (
        isinstance(b, list)
        and len(b) == 4
        and all(type(x) is int for x in b)
        and 0 <= b[0] < b[2] <= 1000
        and 0 <= b[1] < b[3] <= 1000
    )


def iou(a, b):
    if not valid_box(a) or not valid_box(b):
        return 0.0
    inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def match(a, b, threshold=0.5):
    if not a or not b:
        return []
    scores = np.array([[iou(x, y) for y in b] for x in a])
    # Pre-gate forbidden edges; dummy assignments have zero reward. No language/type in matching.
    rr, cc = linear_sum_assignment(-np.where(scores >= threshold, scores, 0.0))
    return [
        (int(i), int(j), float(scores[i, j])) for i, j in zip(rr, cc) if scores[i, j] >= threshold
    ]


def origin(r):
    return "edit" if "edit" in r else "keep" if "keep" in r else "generated"


def refs(r):
    return r.get("edit", r.get("keep", {})).get("references", [])


def instruction(r):
    return r.get("edit", {}).get("instruction", "")


def quality(x):
    p, r = x["precision"], x["recall"]
    return 0.0 if x["critical_error"] or p + r == 0 else 2 * p * r / (p + r)


def plain(text):
    text = re.sub(r"</?(?:text|br)>", " ", text or "")
    text = re.sub(r"<frame>\d+</frame>", " ", text)
    return re.sub(r"<([^<>]+?)\s+\d+>", r"\1", text).strip()


def prepare(p, g, threshold=0.5):
    pr = p.get("elements", {}).get("layout", [])
    gr = g["elements"]["layout"]
    matched = match([r.get("bbox") for r in pr], [r["bbox"] for r in gr], threshold)
    pairs = []
    edges = []
    for i, j, q in matched:
        pairs.append(dict(pred_index=i, gt_index=j, target_iou=q, pred=pr[i], gt=gr[j]))
        a, b = refs(pr[i]), refs(gr[j])
        for frame in sorted({r.get("reference_id") for r in a + b}):
            ai = [k for k, r in enumerate(a) if r.get("reference_id") == frame]
            bj = [k for k, r in enumerate(b) if r.get("reference_id") == frame]
            for k, l, s in match(
                [a[z].get("reference_bbox") for z in ai],
                [b[z].get("reference_bbox") for z in bj],
                threshold,
            ):
                edges.append(
                    dict(
                        pred_index=i,
                        gt_index=j,
                        pred_ref_index=ai[k],
                        gt_ref_index=bj[l],
                        target_iou=q,
                        reference_iou=s,
                        pred_reference=a[ai[k]],
                        gt_reference=b[bj[l]],
                    )
                )
    return dict(
        pairs=pairs,
        edges=edges,
        np=len(pr),
        ng=len(gr),
        ep=sum(len(refs(r)) for r in pr),
        eg=sum(len(refs(r)) for r in gr),
    )


def validate_judge(j, geo, events):
    expected = {(x["pred_index"], x["gt_index"]) for x in geo["pairs"]}
    observed = [(x["pred_index"], x["gt_index"]) for x in j["regions"]]
    assert len(observed) == len(set(observed)) and set(observed) == expected, (
        "incomplete/duplicate region judgments"
    )
    keys = ("pred_index", "gt_index", "pred_ref_index", "gt_ref_index")
    expected = {tuple(x[k] for k in keys) for x in geo["edges"]}
    observed = [tuple(x[k] for k in keys) for x in j["edges"]]
    assert len(observed) == len(set(observed)) and set(observed) == expected, (
        "incomplete/duplicate edge judgments"
    )
    observed = [x["gt_event_id"] for x in j["events"]]
    assert len(observed) == len(set(observed)) and set(observed) == {e["id"] for e in events}, (
        "incomplete/duplicate event judgments"
    )


def score(p, g, geo, j, similarities=None):
    pr = p.get("elements", {}).get("layout", [])
    gr = g["elements"]["layout"]
    d = geo["np"] + geo["ng"]
    rj = {(x["pred_index"], x["gt_index"]): x for x in j["regions"]}
    ej = {
        (x["pred_index"], x["gt_index"], x["pred_ref_index"], x["gt_ref_index"]): x
        for x in j["edges"]
    }
    result = {
        "target_region_pq": 2 * sum(x["target_iou"] for x in geo["pairs"]) / d if d else 0.0,
        "target_match_f1": 2 * len(geo["pairs"]) / d if d else 0.0,
        "target_desc_f1": 2 * sum(quality(x["desc"]) for x in j["regions"]) / d if d else 0.0,
        "target_caption_embedding_f1": 2 * sum(similarities) / d
        if similarities is not None and d
        else None,
        "target_desc_conditional": float(np.mean([quality(x["desc"]) for x in j["regions"]]))
        if j["regions"]
        else None,
        "global_description_semantic_f1": quality(j["global_description"]),
        "global_instruction_semantic_f1": quality(j["global_instruction"]),
        "global_description_precision": j["global_description"]["precision"],
        "global_description_recall": j["global_description"]["recall"],
        "global_instruction_precision": j["global_instruction"]["precision"],
        "global_instruction_recall": j["global_instruction"]["recall"],
        "global_description_critical_error": j["global_description"]["critical_error"],
        "global_instruction_critical_error": j["global_instruction"]["critical_error"],
        "event_recall": sum(x["covered"] and not x["contradicted"] for x in j["events"])
        / len(j["events"])
        if j["events"]
        else None,
    }
    ed = geo["ep"] + geo["eg"]
    weights = []
    joint = []
    for x in geo["edges"]:
        a = ej[tuple(x[k] for k in ("pred_index", "gt_index", "pred_ref_index", "gt_ref_index"))]
        correct = a["correspondence_correct"]
        q = min(x["target_iou"], x["reference_iou"])
        weights.append(q * correct)
        pair = rj[(x["pred_index"], x["gt_index"])]
        r = gr[x["gt_index"]]
        iq = quality(pair["instruction"]) if origin(r) == "edit" else float(pair["origin_correct"])
        joint.append(q * correct * pair["origin_correct"] * iq)
    result.update(
        reference_link_f1=2
        * sum(
            ej[tuple(x[k] for k in ("pred_index", "gt_index", "pred_ref_index", "gt_ref_index"))][
                "correspondence_correct"
            ]
            for x in geo["edges"]
        )
        / ed
        if ed
        else None,
        reference_pair_pq=2
        * sum(min(x["target_iou"], x["reference_iou"]) for x in geo["edges"])
        / ed
        if ed
        else None,
        reference_grounded_pq=2 * sum(weights) / ed if ed else None,
        editing_edge_joint_f1=2 * sum(joint) / ed if ed else None,
        reference_iou_conditional=float(np.mean([x["reference_iou"] for x in geo["edges"]]))
        if geo["edges"]
        else None,
    )
    editden = sum(origin(r) == "edit" for r in pr) + sum(origin(r) == "edit" for r in gr)
    elig = [x for x in geo["pairs"] if origin(x["gt"]) == "edit" and origin(x["pred"]) == "edit"]
    result["local_edit_instruction_f1"] = (
        2
        * sum(
            quality(rj[(x["pred_index"], x["gt_index"])]["instruction"])
            * rj[(x["pred_index"], x["gt_index"])]["origin_correct"]
            for x in elig
        )
        / editden
        if editden
        else None
    )
    for category in ["edit", "keep", "generated"]:
        denominator = sum(origin(r) == category for r in pr) + sum(
            origin(r) == category for r in gr
        )
        accepted = [
            x for x in geo["pairs"] if origin(x["gt"]) == category and origin(x["pred"]) == category
        ]
        result[category + "_region_pq"] = (
            2 * sum(x["target_iou"] for x in accepted) / denominator if denominator else None
        )
        result[category + "_desc_f1"] = (
            2
            * sum(quality(rj[(x["pred_index"], x["gt_index"])]["desc"]) for x in accepted)
            / denominator
            if denominator
            else None
        )
        result[category + "_origin_f1"] = (
            2
            * sum(rj[(x["pred_index"], x["gt_index"])]["origin_correct"] for x in accepted)
            / denominator
            if denominator
            else None
        )
    result.update(
        pred_regions=len(pr),
        gt_regions=len(gr),
        matched_regions=len(geo["pairs"]),
        pred_edges=geo["ep"],
        gt_edges=geo["eg"],
        matched_edges=len(geo["edges"]),
        gt_concern_count=len(j["gt_concerns"]),
    )
    return result
