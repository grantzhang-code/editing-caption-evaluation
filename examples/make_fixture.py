"""Create one wholly synthetic case; no benchmark content or credentials."""

import base64
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    pixel = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a3ioAAAAASUVORK5CYII="
    )
    (root / "pixel.png").write_bytes(pixel)
    caption = {
        "high_level_description": "A blank canvas.",
        "normalized_edit_instruction": "Keep the canvas unchanged.",
        "elements": {"background": "Blank.", "layout": [], "remove": []},
    }
    text = json.dumps(caption, ensure_ascii=False, separators=(",", ":"))
    image = {
        "kind": "image",
        "uri": "examples/pixel.png",
        "sha256": hashlib.sha256(pixel).hexdigest(),
        "mime": "image/png",
    }
    prediction = {
        "caption": caption,
        "status": "success",
        "error": None,
        "bbox_order": "yxyx_1000",
        "schema": "native",
        "original_caption_text": text,
        "raw_prediction_sha256": hashlib.sha256(text.encode()).hexdigest(),
    }
    row = {
        "sample_rank": 1,
        "record_uid": "synthetic-blank-canvas",
        "cohort": "synthetic",
        "benchmark_group": "keep",
        "weight": 1,
        "images": {"source": image, "target": image},
        "gt": {
            "caption": caption,
            "events": [],
            "caption_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "model": "synthetic",
            "reasoning_effort": "none",
        },
        "predictions": {"original": prediction, "checkpoint_11000": prediction},
    }
    (root / "synthetic.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
