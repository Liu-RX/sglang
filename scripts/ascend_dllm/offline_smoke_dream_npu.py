#!/usr/bin/env python3
"""Run a one-prompt Dream dLLM speculative-decoding smoke test on Ascend NPU."""

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


def default_config_path() -> str:
    return os.getenv(
        "DLLM_ALGORITHM_CONFIG",
        "examples/dllm_low_confidence_spec_vbs1.yaml",
    )


def configure_ascend_env(npu_devices: str) -> None:
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = npu_devices
    os.environ.setdefault("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT", "1")
    os.environ.setdefault("SGLANG_SET_CPU_AFFINITY", "1")
    os.environ.setdefault("HCCL_BUFFSIZE", "200")
    os.environ.setdefault("HCCL_EXEC_TIMEOUT", "200")
    os.environ.setdefault("STREAMS_PER_DEVICE", "32")
    os.environ.setdefault("AUTO_USE_UC_MEMORY", "0")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        default=os.getenv("MODEL_PATH", "Dream-org/Dream-v0-Instruct-7B"),
    )
    parser.add_argument(
        "--dllm-algorithm-config",
        default=default_config_path(),
        help="YAML config for LowConfidence. Use an empty string for baseline.",
    )
    parser.add_argument(
        "--npu-devices",
        default=os.getenv(
            "ASCEND_RT_VISIBLE_DEVICES",
            os.getenv("NPU_DEVICES", "0"),
        ),
    )
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=int(os.getenv("MAX_NEW_TOKENS", "128")),
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=int(os.getenv("CONTEXT_LENGTH", "1024")),
    )
    parser.add_argument(
        "--max-running-requests",
        type=int,
        default=int(os.getenv("MAX_RUNNING_REQUESTS", "1")),
    )
    parser.add_argument(
        "--mem-fraction-static",
        type=float,
        default=float(os.getenv("MEM_FRACTION_STATIC", "0.75")),
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        default=int(os.getenv("TP_SIZE", "1")),
    )
    parser.add_argument(
        "--base-gpu-id",
        type=int,
        default=int(os.getenv("BASE_GPU_ID", "0")),
        help="SGLang keeps this argument name for all accelerators.",
    )
    parser.add_argument("--dtype", default=os.getenv("DTYPE", "auto"))
    parser.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "info"))
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    configure_ascend_env(args.npu_devices)

    import sglang as sgl
    from transformers import AutoTokenizer

    dllm_config = args.dllm_algorithm_config or None
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.question}],
        tokenize=False,
        add_generation_prompt=True,
    )

    print("Starting SGLang Engine on Ascend NPU:")
    print(
        json.dumps(
            {
                "model_path": args.model_path,
                "dllm_algorithm_config": dllm_config,
                "device": "npu",
                "attention_backend": "ascend",
                "ASCEND_RT_VISIBLE_DEVICES": os.getenv("ASCEND_RT_VISIBLE_DEVICES"),
                "tp_size": args.tp_size,
                "max_new_tokens": args.max_new_tokens,
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    llm = sgl.Engine(
        model_path=args.model_path,
        trust_remote_code=True,
        device="npu",
        attention_backend="ascend",
        dllm_algorithm="LowConfidence",
        dllm_algorithm_config=dllm_config,
        max_running_requests=args.max_running_requests,
        context_length=args.context_length,
        disable_cuda_graph=True,
        disable_radix_cache=True,
        mem_fraction_static=args.mem_fraction_static,
        tp_size=args.tp_size,
        base_gpu_id=args.base_gpu_id,
        dtype=args.dtype,
        log_level=args.log_level,
    )
    try:
        output = llm.generate(
            prompt,
            {
                "temperature": 0.0,
                "max_new_tokens": args.max_new_tokens,
            },
        )
    finally:
        llm.shutdown()

    print("Generation output:")
    print(json.dumps(output, indent=2, ensure_ascii=False))
    if args.output_json:
        Path(args.output_json).write_text(
            json.dumps(output, indent=2, ensure_ascii=False) + "\n"
        )
    text = output.get("text", "") if isinstance(output, dict) else ""
    return 0 if text.strip() else 2


if __name__ == "__main__":
    raise SystemExit(main())
