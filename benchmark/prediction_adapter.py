"""Mechanical checkpoint assistant XYXY -> native evaluation YXYX adapter."""

import copy


def swap(box):
    assert isinstance(box, list) and len(box) == 4 and all(type(v) is int for v in box)
    assert 0 <= box[0] < box[2] <= 1000 and 0 <= box[1] < box[3] <= 1000
    return [box[1], box[0], box[3], box[2]]


def checkpoint_to_native(caption):
    original = copy.deepcopy(caption)
    comp = original.pop("compositional_deconstruction")
    regions = comp["elements"]
    groups = comp["reference_edits"]
    assert len(groups) == 1 and groups[0]["reference_id"] == "frame_0"
    actions = groups[0]["edit"]
    assert len(regions) == len(actions)
    generated_instructions = {}
    for region, action in zip(regions, actions):
        assert region["label"] == action["label"]
        assert "edit" not in region and "keep" not in region
        region["bbox"] = swap(region["bbox"])
        instruction = action["instruction"]
        assert isinstance(instruction, str)
        if "reference_bbox" in action:
            refs = [dict(reference_id="frame_0", reference_bbox=swap(action["reference_bbox"]))]
            if instruction == "keep":
                region["keep"] = dict(references=refs)
            else:
                region["edit"] = dict(instruction=instruction, references=refs)
        else:
            assert instruction != "keep", "keep without a reference"
            generated_instructions[region["label"]] = instruction
    removals = [
        dict(
            instruction=a["instruction"],
            references=[dict(reference_id="frame_0", reference_bbox=swap(a["reference_bbox"]))],
        )
        for a in groups[0]["remove"]
    ]
    original["elements"] = dict(background=comp["background"], layout=regions, remove=removals)
    return original, dict(
        source_bbox_order="xyxy_1000",
        evaluation_bbox_order="yxyx_1000",
        target_count=len(regions),
        generated_instructions_not_scored_as_local_edits=generated_instructions,
        text_rewritten=False,
        geometry_rescaled=False,
    )
