#!/usr/bin/env python3
"""Check whether the current Python environment can run SGLang on Ascend NPU."""

from __future__ import annotations

import argparse
import importlib
import os
import platform
import shutil
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def print_kv(key: str, value) -> None:
    print(f"{key}: {value}")


def import_required(module_name: str, failures: list[str]):
    try:
        module = importlib.import_module(module_name)
        print_kv(module_name, getattr(module, "__version__", "import ok"))
        return module
    except Exception as exc:
        failures.append(f"failed to import {module_name}: {exc}")
        print_kv(module_name, f"FAILED ({exc})")
        return None


def run_npu_smi() -> None:
    exe = shutil.which("npu-smi")
    if exe is None:
        print_kv("npu-smi", "not found in PATH")
        return

    try:
        output = subprocess.check_output(
            [exe, "info"],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=15,
        )
    except Exception as exc:
        print_kv("npu-smi info", f"FAILED ({exc})")
        return

    print("npu-smi info:")
    for line in output.splitlines()[:40]:
        print(line)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=os.getenv("MODEL_PATH"))
    parser.add_argument(
        "--skip-tokenizer",
        action="store_true",
        help="Skip AutoTokenizer.from_pretrained(model_path).",
    )
    args = parser.parse_args()

    failures: list[str] = []
    print_kv("repo_root", REPO_ROOT)
    print_kv("python", sys.version.replace("\n", " "))
    print_kv("platform", platform.platform())
    print_kv("ASCEND_RT_VISIBLE_DEVICES", os.getenv("ASCEND_RT_VISIBLE_DEVICES"))
    print_kv("PYTHONPATH", os.getenv("PYTHONPATH"))
    print_kv("torch package", package_version("torch"))
    print_kv("torch_npu package", package_version("torch-npu"))
    print_kv("sglang package", package_version("sglang"))
    print_kv("sgl-kernel-npu package", package_version("sgl-kernel-npu"))
    print_kv("triton-ascend package", package_version("triton-ascend"))

    torch = import_required("torch", failures)
    import_required("torch_npu", failures)
    import_required("sgl_kernel_npu", failures)
    import_required("sglang", failures)

    if torch is not None:
        has_npu = hasattr(torch, "npu")
        print_kv("torch has npu", has_npu)
        if not has_npu:
            failures.append("torch.npu is missing; torch_npu is not active")
        else:
            try:
                available = torch.npu.is_available()
                print_kv("torch.npu.is_available", available)
                if not available:
                    failures.append("torch.npu.is_available() returned False")
                print_kv("torch.npu.device_count", torch.npu.device_count())
                if torch.npu.device_count() > 0:
                    free_mem, total_mem = torch.npu.mem_get_info()
                    print_kv("npu memory free MB", free_mem // 1024 // 1024)
                    print_kv("npu memory total MB", total_mem // 1024 // 1024)
            except Exception as exc:
                failures.append(f"failed to query torch.npu: {exc}")
                print_kv("torch.npu query", f"FAILED ({exc})")

    run_npu_smi()

    if args.model_path and not args.skip_tokenizer:
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                args.model_path,
                trust_remote_code=True,
            )
            print_kv("tokenizer", tokenizer.__class__.__name__)
            print_kv("tokenizer vocab_size", getattr(tokenizer, "vocab_size", "N/A"))
            print_kv(
                "tokenizer mask_token_id",
                getattr(tokenizer, "mask_token_id", "N/A"),
            )
        except Exception as exc:
            failures.append(f"failed to load tokenizer from {args.model_path}: {exc}")
            print_kv("tokenizer", f"FAILED ({exc})")

    if failures:
        print("\nFAILED checks:")
        for item in failures:
            print(f"- {item}")
        return 1

    print("\nAll required Ascend NPU checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
