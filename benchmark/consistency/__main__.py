"""Replay one saved consistency judgment without an API request."""

import argparse
import json
from pathlib import Path

from .metric import score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--judgment", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.payload.read_text())
    judgment = json.loads(args.judgment.read_text())
    print(json.dumps(score(payload, judgment), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
