#!/usr/bin/env python3
"""Send a tokenizer-formatted Dream request to an SGLang HTTP server."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))


DEFAULT_QUESTION = (
    "Natalia sold clips to 48 of her friends in April, and then she sold half "
    "as many clips in May. How many clips did Natalia sell altogether in April "
    "and May? Answer with the final number."
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.getenv("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "30000")))
    parser.add_argument(
        "--model-path",
        default=os.getenv("MODEL_PATH", "Dream-org/Dream-v0-Instruct-7B"),
    )
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=int(os.getenv("MAX_NEW_TOKENS", "128")),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.getenv("REQUEST_TIMEOUT", "600")),
    )
    parser.add_argument("--expect-substring", default=None)
    args = parser.parse_args()

    import requests
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.question}],
        tokenize=False,
        add_generation_prompt=True,
    )
    url = f"http://{args.host}:{args.port}/generate"
    payload = {
        "text": prompt,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": args.max_new_tokens,
        },
    }
    response = requests.post(url, json=payload, timeout=args.timeout)
    print(f"HTTP {response.status_code} from {url}")
    response.raise_for_status()
    data = response.json()
    print(json.dumps(data, indent=2, ensure_ascii=False))

    text = data.get("text", "")
    if not text.strip():
        print("Empty generation text.", file=sys.stderr)
        return 2
    if args.expect_substring is not None and args.expect_substring not in text:
        print(
            f"Expected substring {args.expect_substring!r} was not found.",
            file=sys.stderr,
        )
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
