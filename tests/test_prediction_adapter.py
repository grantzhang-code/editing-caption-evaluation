import unittest
from benchmark.prediction_adapter import checkpoint_to_native


class AdapterTest(unittest.TestCase):
    def test_coordinate_origin_and_instruction_preservation(self):
        regions = [
            dict(label=f"<r {i}>", bbox=[10, 20, 30, 40], desc="unchanged text") for i in (1, 2, 3)
        ]
        actions = [
            dict(label="<r 1>", instruction="Turn blue to red.", reference_bbox=[1, 2, 7, 9]),
            dict(label="<r 2>", instruction="keep", reference_bbox=[11, 12, 17, 19]),
            dict(label="<r 3>", instruction="Add this object."),
        ]
        c = dict(
            high_level_description="H",
            normalized_edit_instruction="N",
            compositional_deconstruction=dict(
                background="B",
                elements=regions,
                reference_edits=[
                    dict(
                        reference_id="frame_0",
                        edit=actions,
                        remove=[dict(instruction="Remove it.", reference_bbox=[3, 4, 8, 9])],
                    )
                ],
            ),
        )
        out, audit = checkpoint_to_native(c)
        rs = out["elements"]["layout"]
        self.assertEqual(rs[0]["bbox"], [20, 10, 40, 30])
        self.assertEqual(rs[0]["edit"]["references"][0]["reference_bbox"], [2, 1, 9, 7])
        self.assertEqual(rs[0]["edit"]["instruction"], "Turn blue to red.")
        self.assertIn("keep", rs[1])
        self.assertNotIn("edit", rs[2])
        self.assertNotIn("keep", rs[2])
        self.assertEqual(
            out["elements"]["remove"][0]["references"][0]["reference_bbox"], [4, 3, 9, 8]
        )
        self.assertEqual(c["compositional_deconstruction"]["elements"][0]["bbox"], [10, 20, 30, 40])
        self.assertEqual(
            audit["generated_instructions_not_scored_as_local_edits"]["<r 3>"], "Add this object."
        )


if __name__ == "__main__":
    unittest.main()
