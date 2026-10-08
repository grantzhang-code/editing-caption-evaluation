"""Durable Responses transport. Credentials are supplied by the environment."""

import hashlib, json, os, random, threading, time
from pathlib import Path
from urllib.parse import urlparse
import boto3, requests
from botocore.config import Config
from botocore.exceptions import ClientError
from .bench_contracts import MODEL, EFFORT

OUT = Path(os.environ.get("EVAL_OUTPUT_DIR", "results"))
OUT.mkdir(parents=True, exist_ok=True)
destination = urlparse(os.environ.get("EVAL_OUTPUT_S3", ""))
BUCKET = destination.netloc
PREFIX = destination.path.lstrip("/").rstrip("/") + "/" if BUCKET else ""
S3 = boto3.Session().client("s3", config=Config(max_pool_connections=320)) if BUCKET else None
RUN_RECORD_PREFIX = os.environ.get("EVAL_RUN_PREFIX", "")
LOCK = threading.Lock()
STATE = dict(
    model=MODEL,
    reasoning_effort=EFFORT,
    stage="initializing",
    preflight_ok=0,
    scored=0,
    failed=0,
    active_api=0,
    peak_active_api=0,
    api_responses=0,
    rate_limit_rejections=0,
)


def ordered(v):
    return json.dumps(v, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def canonical(v):
    return json.dumps(
        v, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def digest(b):
    return hashlib.sha256(b).hexdigest()


def write(name, value):
    body = ordered(value) + b"\n"
    p = OUT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix(p.suffix + ".tmp")
    temp.write_bytes(body)
    temp.replace(p)
    if S3:
        S3.put_object(Bucket=BUCKET, Key=PREFIX + name, Body=body, ContentType="application/json")
    return value


def write_run(name, value):
    return write(RUN_RECORD_PREFIX + name, value)


def read(name):
    p = OUT / name
    if p.exists():
        return json.loads(p.read_bytes())
    if not S3:
        return None
    try:
        body = S3.get_object(Bucket=BUCKET, Key=PREFIX + name)["Body"].read()
    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404", "NotFound"):
            return None
        raise
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(body)
    return json.loads(body)


def api_key():
    return os.environ["OPENAI_API_KEY"]


def call(name, body, key):
    h = digest(canonical(body))
    intent = read(name + "/intent.json")
    response = read(name + "/response.json")
    if intent:
        assert intent["request_sha256"] == h, "changed request on resume"
    if response is not None:
        return response
    if intent:
        raise RuntimeError("Prior request has no recorded response; inspect before retrying")
    intent = dict(request_sha256=h, model=MODEL, reasoning_effort=EFFORT, created_at=time.time())
    if S3:
        S3.put_object(
            Bucket=BUCKET, Key=PREFIX + name + "/intent.json", Body=ordered(intent), IfNoneMatch="*"
        )
    else:
        p = OUT / name / "intent.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("x") as f:
            json.dump(intent, f)
    for attempt in range(16):
        with LOCK:
            STATE["active_api"] += 1
            STATE["peak_active_api"] = max(STATE["peak_active_api"], STATE["active_api"])
        try:
            r = requests.post(
                "https://api.openai.com/v1/responses",
                headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
                json=body,
                timeout=(30, 1800),
            )
        except requests.RequestException as e:
            write(name + "/transport-unknown.json", dict(error=type(e).__name__, request_sha256=h))
            raise
        finally:
            with LOCK:
                STATE["active_api"] -= 1
        if r.status_code == 429:
            with LOCK:
                STATE["rate_limit_rejections"] += 1
            write(name + f"/rate-limit-{attempt}.json", dict(http_status=429, body=r.text[:3000]))
            try:
                delay = float(r.headers.get("Retry-After", 0))
            except ValueError:
                delay = 0
            time.sleep(min(120, max(delay, 2**attempt)) + random.random())
            continue
        if r.status_code != 200:
            write(name + "/api-error.json", dict(http_status=r.status_code, body=r.text[:4000]))
            raise RuntimeError("HTTP " + str(r.status_code))
        value = r.json()
        write(name + "/response.json", value)
        with LOCK:
            STATE["api_responses"] += 1
        return value
    raise RuntimeError("HTTP 429 retry limit reached")
