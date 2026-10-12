"""Guard action-set matching and judge evidence validation."""

import copy

import pytest

from benchmark.consistency.metric import maximum_matching, score


def action(subject, state, instruction_id, quote):
    return {
        "subject": subject,
        "operation": "change",
        "target_state": state,
        "constraints": "",
        "evidence": [{"instruction_id": instruction_id, "quote": quote}],
    }


def example():
    payload = {
        "global_instruction": "car blue; sign right",
        "local_instructions": [
            {"instruction_id": "l0", "instruction": "car blue"},
            {"instruction_id": "l1", "instruction": "sign left"},
        ],
    }
    judgment = {
        "global_actions": [
            action("car", "blue", "global", "car blue"),
            action("sign", "right", "global", "sign right"),
        ],
        "local_actions": [
            action("car", "blue", "l0", "car blue"),
            action("sign", "left", "l1", "sign left"),
        ],
        "compatible_pairs": [{"global_index": 0, "local_index": 0, "reason": "same car color"}],
        "local_coverage": [
            {"instruction_id": "l0", "disposition": "actions"},
            {"instruction_id": "l1", "disposition": "actions"},
        ],
        "global_exclusions": [],
    }
    return payload, judgment


def test_opposite_direction_keeps_unmatched_actions_in_denominator():
    payload, judgment = example()
    assert score(payload, judgment)["global_local_instruction_consistency_f1"] == 0.5


def test_matching_uses_augmenting_paths_without_duplicate_credit():
    assert len(maximum_matching([(0, 0), (0, 1), (1, 0)], 2, 2)) == 2
    assert len(maximum_matching([(0, 0), (1, 0), (2, 0)], 3, 1)) == 1


@pytest.mark.parametrize("one_side", [False, True])
def test_missing_actions_on_eligible_case_score_zero(one_side):
    judgment = {
        "global_actions": [],
        "local_actions": [],
        "compatible_pairs": [],
        "local_coverage": [],
        "global_exclusions": [],
    }
    if one_side:
        judgment["global_actions"] = [action("car", "blue", "global", "car blue")]
    payload = {"global_instruction": "car blue", "local_instructions": []}
    assert score(payload, judgment)["global_local_instruction_consistency_f1"] == 0


@pytest.mark.parametrize("invalid", ["quote", "coverage", "index", "duplicate"])
def test_invalid_evidence_or_action_graph_is_rejected(invalid):
    payload, judgment = example()
    if invalid == "quote":
        judgment["local_actions"][0]["evidence"][0]["quote"] = "car red"
    elif invalid == "coverage":
        judgment["local_coverage"].pop()
    elif invalid == "index":
        judgment["compatible_pairs"][0]["local_index"] = 100
    else:
        judgment["global_actions"].append(copy.deepcopy(judgment["global_actions"][0]))
    with pytest.raises(AssertionError):
        score(payload, judgment)
