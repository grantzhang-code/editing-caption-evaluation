"""Fetch a release manifest and verify every downloaded artifact by SHA-256."""

import argparse
import hashlib
import json
from pathlib import Path

from .data import read_bytes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Local path or s3:// URI")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest_bytes = read_bytes(args.manifest)
    manifest = json.loads(manifest_bytes)
    args.output.mkdir(parents=True, exist_ok=True)
    seen = set()
    for artifact in manifest["artifacts"]:
        name = artifact["name"]
        if Path(name).name != name or name in (".", "..", "manifest.json") or name in seen:
            raise ValueError(f"Unsafe or duplicate artifact name: {name}")
        seen.add(name)
        target = args.output / name
        body = target.read_bytes() if target.exists() else b""
        if hashlib.sha256(body).hexdigest() != artifact["sha256"]:
            body = read_bytes(artifact["uri"])
        if len(body) != artifact["bytes"] or hashlib.sha256(body).hexdigest() != artifact["sha256"]:
            raise ValueError(f"Artifact verification failed: {name}")
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_bytes(body)
        temporary.replace(target)
        print(f"Verified {name} ({len(body)} bytes)")
    (args.output / "manifest.json").write_bytes(manifest_bytes)


if __name__ == "__main__":
    main()
