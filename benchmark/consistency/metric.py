"""Validate evidence bindings, then compute one-to-one action-set F1."""

import re

from jsonschema import Draft202012Validator

from .contracts import SCHEMA


def normalized(text):
    return re.sub(r"\s+", " ", text).strip()


def maximum_matching(pairs, ng, nl):
    adjacency = {g: [] for g in range(ng)}
    for g, l in sorted(set(pairs)):
        assert 0 <= g < ng and 0 <= l < nl
        adjacency[g].append(l)
    owner = {}

    def augment(g, seen):
        for l in adjacency[g]:
            if l in seen:
                continue
            seen.add(l)
            if l not in owner or augment(owner[l], seen):
                owner[l] = g
                return True
        return False

    for g in range(ng):
        augment(g, set())
    return sorted((g, l) for l, g in owner.items())


def score(payload, judgment):
    Draft202012Validator(SCHEMA).validate(judgment)
    texts = {r["instruction_id"]: r["instruction"] for r in payload["local_instructions"]}
    assert len(texts) == len(payload["local_instructions"])
    cited = set()
    for side in ["global", "local"]:
        signatures = set()
        for action in judgment[side + "_actions"]:
            assert action["subject"].strip() and action["operation"].strip(), (
                "Empty action identity"
            )
            assert action["evidence"], "Action lacks evidence"
            signature = tuple(
                normalized(action[k]).lower()
                for k in ["subject", "operation", "target_state", "constraints"]
            )
            assert signature not in signatures, "Duplicate normalized action"
            signatures.add(signature)
            for ev in action["evidence"]:
                iid = ev["instruction_id"]
                assert (iid == "global") if side == "global" else iid in texts, (
                    "Evidence crosses sides or uses an unknown instruction"
                )
                source = payload["global_instruction"] if side == "global" else texts[iid]
                quote = normalized(ev["quote"])
                assert quote and quote in normalized(source), (
                    f"Non-verbatim evidence: {iid}: {quote[:120]}"
                )
                if side == "local":
                    cited.add(iid)
    coverage = judgment["local_coverage"]
    assert len(coverage) == len(texts) and {c["instruction_id"] for c in coverage} == set(texts), (
        "Incomplete/duplicate local coverage"
    )
    assert {c["instruction_id"] for c in coverage if c["disposition"] == "actions"} == cited, (
        "Local coverage and cited actions disagree"
    )
    for ev in judgment["global_exclusions"]:
        assert normalized(ev["quote"]) and normalized(ev["quote"]) in normalized(
            payload["global_instruction"]
        ), "Non-verbatim global exclusion"
    ng, nl = len(judgment["global_actions"]), len(judgment["local_actions"])
    pairs = [(p["global_index"], p["local_index"]) for p in judgment["compatible_pairs"]]
    assert len(pairs) == len(set(pairs)), "Duplicate compatible pair"
    matches = maximum_matching(pairs, ng, nl)
    m = len(matches)
    return {
        "global_action_count": ng,
        "local_action_count": nl,
        "matched_action_count": m,
        "precision": m / nl if nl else 0.0,
        "recall": m / ng if ng else 0.0,
        "global_local_instruction_consistency_f1": 2 * m / (ng + nl) if ng + nl else 0.0,
        "matched_pairs": matches,
    }
