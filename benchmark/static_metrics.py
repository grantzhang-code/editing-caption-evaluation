"""Deterministic geometry and pinned local MiniLM metrics for both frozen variants."""

import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
from . import metrics
from .data import load_dataset

OUT = Path("results")


def save(name, v):
    (OUT / name).write_text(json.dumps(v, ensure_ascii=False, separators=(",", ":")) + "\n")


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--model-path")
    args = ap.parse_args()
    OUT = args.results
    OUT.mkdir(parents=True, exist_ok=True)
    items, gts, samples, preds = load_dataset(args.dataset)
    geometries = {}
    texts = set()
    for p in preds:
        caption = (
            json.loads(p["evaluation_caption"])
            if not p["invalid_reason"]
            else {"elements": {"layout": []}}
        )
        geo = metrics.prepare(caption, gts[p["sample_rank"]]["caption"])
        geometries[(p["variant"], p["sample_rank"])] = geo
        for pair in geo["pairs"]:
            texts.update((pair["pred"]["desc"], pair["gt"]["desc"]))
    texts = sorted(texts)
    hashes = [hashlib.sha256(t.encode()).hexdigest() for t in texts]
    assert len(set(hashes)) == len(texts)
    cache = OUT / "embedding-cache.npz"
    vectors = {}
    lock = dict(
        model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        revision="e8f8c211226b894fcb81acc59f3b34ba3efd5f42",
        max_length=512,
        pooling="attention_mask_mean",
        normalize="l2",
    )
    audit_path = OUT / "embedding-audit.json"
    if cache.exists() and audit_path.exists():
        audit = json.loads(audit_path.read_text())
        with np.load(cache) as data:
            if data["hashes"].tolist() == hashes and all(
                audit.get(k) == v for k, v in lock.items()
            ):
                vectors = dict(zip(texts, data["vectors"]))
    if texts and not vectors:
        import torch
        from transformers import AutoTokenizer, AutoModel

        torch.set_num_threads(4)
        tok = AutoTokenizer.from_pretrained(
            args.model_path or lock["model"],
            revision=lock["revision"],
            local_files_only=bool(args.model_path),
        )
        model = AutoModel.from_pretrained(
            args.model_path or lock["model"],
            revision=lock["revision"],
            local_files_only=bool(args.model_path),
        ).eval()
        truncated = 0
        for start in range(0, len(texts), 24):
            chunk = texts[start : start + 24]
            truncated += sum(
                len(tok(t, add_special_tokens=True, truncation=False)["input_ids"]) > 512
                for t in chunk
            )
            batch = tok(chunk, padding=True, truncation=True, max_length=512, return_tensors="pt")
            with torch.inference_mode():
                hidden = model(**batch).last_hidden_state
                mask = batch["attention_mask"].unsqueeze(-1)
                values = torch.nn.functional.normalize(
                    (hidden * mask).sum(1) / mask.sum(1).clamp(min=1), p=2, dim=1
                ).numpy()
            vectors.update(zip(chunk, values))
            if start % 480 == 0:
                progress = dict(
                    stage="embedding",
                    completed=min(start + 24, len(texts)),
                    total=len(texts),
                    timestamp=time.time(),
                )
                save("static-progress.json", progress)
                print(json.dumps(progress), flush=True)
        np.savez_compressed(
            cache, hashes=np.array(hashes), vectors=np.stack([vectors[t] for t in texts])
        )
        save(
            "embedding-audit.json",
            dict(**lock, unique_descriptions=len(texts), truncated_descriptions=truncated),
        )
    if not texts:
        save("embedding-audit.json", dict(**lock, unique_descriptions=0, truncated_descriptions=0))
    scores = []
    for p in preds:
        geo = geometries[(p["variant"], p["sample_rank"])]
        den = geo["np"] + geo["ng"]
        ed = geo["ep"] + geo["eg"]
        similarity = [
            float(np.clip(np.dot(vectors[x["pred"]["desc"]], vectors[x["gt"]["desc"]]), 0, 1))
            if x["pred"]["desc"].strip() and x["gt"]["desc"].strip()
            else 0
            for x in geo["pairs"]
        ]
        scores.append(
            dict(
                variant=p["variant"],
                rank=p["sample_rank"],
                target_region_pq=2 * sum(x["target_iou"] for x in geo["pairs"]) / den
                if den
                else 0.0,
                target_match_f1=2 * len(geo["pairs"]) / den if den else 0.0,
                target_caption_embedding_f1=2 * sum(similarity) / den if den else 0.0,
                reference_pair_pq=2
                * sum(min(x["target_iou"], x["reference_iou"]) for x in geo["edges"])
                / ed
                if ed
                else None,
                reference_iou_conditional=float(np.mean([x["reference_iou"] for x in geo["edges"]]))
                if geo["edges"]
                else None,
            )
        )
    save("static-scores.json", scores)
    save("static-progress.json", dict(stage="complete", tasks=len(scores), timestamp=time.time()))
    print(json.dumps({"static_tasks": len(scores), "unique_descriptions": len(texts)}), flush=True)


if __name__ == "__main__":
    main()
