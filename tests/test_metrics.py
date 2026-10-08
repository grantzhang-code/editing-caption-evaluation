import unittest
from benchmark.metrics import match, iou, prepare, score, valid_box


def region(box, desc="red cup", frame="frame_0", kind="edit"):
    r = dict(label="<cup 1>", bbox=box, desc=desc)
    if kind in ("edit", "keep"):
        r[kind] = {"references": [dict(reference_id=frame, reference_bbox=box)]}
        if kind == "edit":
            r[kind]["instruction"] = "Recolor blue cup red."
    return r


def cap(regions):
    return dict(elements=dict(layout=regions))


def quality(critical=False):
    return dict(precision=1.0, recall=1.0, critical_error=critical, evidence="test")


def judgments(geo, critical=False):
    return dict(
        regions=[
            dict(
                pred_index=x["pred_index"],
                gt_index=x["gt_index"],
                desc=quality(),
                instruction=quality(critical),
                origin_correct=True,
            )
            for x in geo["pairs"]
        ],
        edges=[
            {
                **{k: x[k] for k in ("pred_index", "gt_index", "pred_ref_index", "gt_ref_index")},
                "correspondence_correct": True,
            }
            for x in geo["edges"]
        ],
        global_description=quality(),
        global_instruction=quality(),
        events=[],
        gt_concerns=[],
    )


class MetricsTest(unittest.TestCase):
    def test_bad_boxes(self):
        self.assertFalse(valid_box([0, 0, 0, 2]))
        self.assertFalse(valid_box([False, 0, 1, 2]))
        self.assertEqual(iou([0, 0, 1000, 1000], [0, 0, 1000, 500]), 0.5)

    def test_duplicate_penalty(self):
        r = region([0, 0, 500, 500])
        p = cap([r, r])
        g = cap([r])
        geo = prepare(p, g)
        s = score(p, g, geo, judgments(geo))
        self.assertAlmostEqual(s["target_region_pq"], 2 / 3)
        self.assertAlmostEqual(s["local_edit_instruction_f1"], 2 / 3)

    def test_wrong_source_frame(self):
        p = cap([region([0, 0, 500, 500], frame="frame_1")])
        g = cap([region([0, 0, 500, 500])])
        geo = prepare(p, g)
        s = score(p, g, geo, judgments(geo))
        self.assertEqual(s["target_region_pq"], 1)
        self.assertEqual(s["reference_pair_pq"], 0)

    def test_direction_error_zero(self):
        p = cap([region([0, 0, 500, 500])])
        geo = prepare(p, p)
        s = score(p, p, geo, judgments(geo, True))
        self.assertEqual(s["local_edit_instruction_f1"], 0)
        self.assertEqual(s["editing_edge_joint_f1"], 0)

    def test_empty_prediction(self):
        p = cap([])
        g = cap([region([0, 0, 500, 500])])
        geo = prepare(p, g)
        s = score(p, g, geo, judgments(geo))
        self.assertEqual(s["target_region_pq"], 0)
        self.assertEqual(s["reference_pair_pq"], 0)

    def test_generated_has_no_fake_reference(self):
        p = cap([region([0, 0, 500, 500], kind="generated")])
        geo = prepare(p, p)
        s = score(p, p, geo, judgments(geo))
        self.assertEqual(s["target_region_pq"], 1)
        self.assertIsNone(s["reference_pair_pq"])
        self.assertIsNone(s["local_edit_instruction_f1"])

    def test_threshold_gated_matching(self):
        self.assertEqual(match([[0, 0, 100, 100]], [[500, 500, 600, 600]]), [])

    def test_geometry_does_not_use_text(self):
        p = cap([region([0, 0, 500, 500], desc="wrong")])
        g = cap([region([0, 0, 500, 500])])
        self.assertEqual(prepare(p, g)["pairs"][0]["target_iou"], 1)


if __name__ == "__main__":
    unittest.main()
