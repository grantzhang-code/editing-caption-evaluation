"""Generic local/S3 input and SHA-bound image loading; no dataset-specific code."""

import hashlib, json, math
from pathlib import Path
from urllib.parse import urlparse
import boto3
import pyarrow.fs as fs
import pyarrow.parquet as pq
from .prediction_adapter import checkpoint_to_native


def read_bytes(uri):
    if uri.startswith("s3://"):
        p = urlparse(uri)
        return (
            boto3.Session()
            .client("s3")
            .get_object(Bucket=p.netloc, Key=p.path.lstrip("/"))["Body"]
            .read()
        )
    return Path(uri).read_bytes()


def image_bytes(spec):
    uri = spec["uri"]
    if spec.get("kind", "image") == "image":
        value = read_bytes(uri)
    else:
        if uri.startswith("s3://"):
            session = boto3.Session()
            credentials = session.get_credentials().get_frozen_credentials()
            filesystem = fs.S3FileSystem(
                access_key=credentials.access_key,
                secret_key=credentials.secret_key,
                session_token=credentials.token,
                region=session.region_name or "us-east-1",
            )
            file = filesystem.open_input_file(uri[5:])
        else:
            file = Path(uri).open("rb")
        try:
            table = pq.ParquetFile(file).read_row_group(
                spec["row_group"], columns=["record_uid", spec["column"]], use_threads=False
            )
            matches = [r for r in table.to_pylist() if r["record_uid"] == spec["record_uid"]]
            assert len(matches) == 1, "ambiguous image row"
            value = matches[0][spec["column"]]
            if "index" in spec:
                value = value[spec["index"]]
        finally:
            file.close()
    assert hashlib.sha256(value).hexdigest() == spec["sha256"], "image SHA256 mismatch"
    return value


def load_dataset(uri):
    rows = [json.loads(line) for line in read_bytes(uri).decode().splitlines() if line.strip()]
    if not rows:
        raise ValueError("Dataset is empty")
    assert len({r["record_uid"] for r in rows}) == len(rows), "duplicate record_uid"
    assert len({r["sample_rank"] for r in rows}) == len(rows), "duplicate sample_rank"
    items = {}
    gts = {}
    samples = {}
    predictions = []
    for r in rows:
        rank = r["sample_rank"]
        if type(rank) is not int or rank < 0:
            raise ValueError("Invalid sample_rank")
        weight = r.get("weight", 1.0)
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("Weight must be finite and positive")
        if (
            hashlib.sha256(
                json.dumps(
                    r["gt"]["caption"], ensure_ascii=False, separators=(",", ":"), allow_nan=False
                ).encode()
            ).hexdigest()
            != r["gt"]["caption_sha256"]
        ):
            raise ValueError("GT caption SHA256 mismatch")
        if set(r["predictions"]) != {"original", "checkpoint_11000"}:
            raise ValueError("Expected both frozen variants")
        items[rank] = r
        gts[rank] = dict(
            caption=r["gt"]["caption"],
            raw_ir={"events": r["gt"]["events"]},
            caption_sha256=r["gt"]["caption_sha256"],
        )
        samples[rank] = dict(
            test_cohort=r.get("cohort", "unspecified"),
            benchmark_group=r.get("benchmark_group"),
            analysis_weight=r.get("weight", 1.0),
        )
        for variant, p in r["predictions"].items():
            if p.get("schema", "native") not in ("native", "assistant_reference_edits"):
                raise ValueError("Unknown prediction schema")
            native = p["caption"]
            invalid = p.get("error") if p.get("status", "success") != "success" else None
            if p.get("status", "success") != "success":
                invalid = invalid or p["status"]
            if invalid is None and p.get("schema", "native") == "assistant_reference_edits":
                assert p["bbox_order"] == "xyxy_1000"
                native, _ = checkpoint_to_native(native)
            elif invalid is None:
                assert p["bbox_order"] == "yxyx_1000"
            raw = (
                p.get("original_caption_text")
                or p.get("raw_response")
                or json.dumps(p["caption"], ensure_ascii=False, separators=(",", ":"))
            )
            raw_sha = hashlib.sha256(raw.encode()).hexdigest()
            if p.get("raw_prediction_sha256"):
                assert raw_sha == p["raw_prediction_sha256"]
            if invalid is None and json.loads(raw) != p["caption"]:
                raise ValueError("Prediction caption differs from its bound raw text")
            predictions.append(
                dict(
                    variant=variant,
                    sample_rank=rank,
                    record_uid=r["record_uid"],
                    evaluation_caption=json.dumps(
                        native, ensure_ascii=False, separators=(",", ":")
                    ),
                    original_prediction_sha256=raw_sha,
                    invalid_reason=invalid,
                )
            )
    return items, gts, samples, predictions
