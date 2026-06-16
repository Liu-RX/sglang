#!/usr/bin/env python3
"""Offline Dream dLLM eval runner for GSM8K, MBPP, and HumanEval."""

from __future__ import annotations

import argparse
import ast
import asyncio
import gzip
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

INVALID = -9999999


@dataclass
class EvalExample:
    task_id: str
    prompt: str
    label: Any = None
    tests: str | None = None
    entry_point: str | None = None
    raw: dict[str, Any] | None = None


def read_records(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("data", "test", "train", "validation"):
                if isinstance(data.get(key), list):
                    return data[key]
        raise ValueError(f"Unsupported JSON dataset shape: {path}")

    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fin:
        return [json.loads(line) for line in fin if line.strip()]


def get_answer_value(answer_str: str) -> int:
    answer_str = answer_str.replace(",", "")
    numbers = re.findall(r"-?\d+", answer_str)
    if not numbers:
        return INVALID
    try:
        return ast.literal_eval(numbers[-1])
    except (SyntaxError, ValueError):
        return INVALID


def load_gsm8k(path: str | None, num_examples: int | None) -> list[EvalExample]:
    if path is None:
        from sglang.utils import download_and_cache_file

        url = (
            "https://raw.githubusercontent.com/openai/grade-school-math/master/"
            "grade_school_math/data/test.jsonl"
        )
        path = download_and_cache_file(url)
    rows = read_records(path)
    if num_examples:
        rows = rows[:num_examples]
    examples = []
    for i, row in enumerate(rows):
        prompt = (
            row["question"]
            + "\nSolve the problem and answer with the final number."
        )
        examples.append(
            EvalExample(
                task_id=str(row.get("task_id", i)),
                prompt=prompt,
                label=get_answer_value(row["answer"]),
                raw=row,
            )
        )
    if any(ex.label == INVALID for ex in examples):
        raise ValueError("Found GSM8K examples with invalid numeric labels.")
    return examples


def normalize_test_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("["):
            try:
                parsed = ast.literal_eval(stripped)
                if isinstance(parsed, list):
                    return [str(x) for x in parsed]
            except Exception:
                pass
        return [line for line in stripped.splitlines() if line.strip()]
    return [str(value)]


def load_mbpp(path: str | None, num_examples: int | None) -> list[EvalExample]:
    if path is not None:
        rows = read_records(path)
    else:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "Install datasets or pass --data-path for MBPP."
            ) from exc

        dataset = None
        errors = []
        for args in (
            ("google-research-datasets/mbpp", "sanitized"),
            ("mbpp", "sanitized"),
            ("google-research-datasets/mbpp", None),
            ("mbpp", None),
        ):
            try:
                if args[1] is None:
                    dataset = load_dataset(args[0], split="test")
                else:
                    dataset = load_dataset(args[0], args[1], split="test")
                break
            except Exception as exc:
                errors.append(f"{args}: {exc}")
        if dataset is None:
            raise RuntimeError("Failed to load MBPP dataset:\n" + "\n".join(errors))
        rows = list(dataset)

    if num_examples:
        rows = rows[:num_examples]

    examples = []
    for i, row in enumerate(rows):
        tests = normalize_test_list(row.get("test_list") or row.get("tests"))
        test_imports = normalize_test_list(row.get("test_imports"))
        tests_src = "\n".join(test_imports + tests)
        prompt_text = row.get("prompt") or row.get("text") or row.get("question")
        if not prompt_text:
            raise ValueError(f"MBPP row {i} does not contain prompt/text.")
        prompt = (
            "Write Python code that solves the following problem. "
            "Return only Python code, no markdown.\n\n"
            f"Problem:\n{prompt_text}\n"
        )
        if tests:
            prompt += "\nThe code should pass these tests:\n" + "\n".join(tests)
        examples.append(
            EvalExample(
                task_id=str(row.get("task_id", i)),
                prompt=prompt,
                label=row.get("code"),
                tests=tests_src,
                raw=dict(row),
            )
        )
    return examples


def load_humaneval(path: str | None, num_examples: int | None) -> list[EvalExample]:
    if path is not None:
        rows = read_records(path)
    else:
        try:
            from human_eval.data import read_problems

            rows = list(read_problems().values())
        except Exception:
            try:
                from datasets import load_dataset
            except ImportError as exc:
                raise ImportError(
                    "Install human-eval/datasets or pass --data-path for HumanEval."
                ) from exc
            rows = list(load_dataset("openai/openai_humaneval", split="test"))

    if num_examples:
        rows = rows[:num_examples]

    examples = []
    for i, row in enumerate(rows):
        prompt_src = row["prompt"]
        entry_point = row.get("entry_point")
        test_src = row.get("test", "")
        if entry_point and "check(" in test_src:
            test_src = test_src + f"\ncheck({entry_point})\n"
        prompt = (
            "Read the following Python function signature and docstring, then "
            "fully implement the function. Return only Python code, no markdown.\n\n"
            + prompt_src
        )
        examples.append(
            EvalExample(
                task_id=str(row.get("task_id", i)),
                prompt=prompt,
                label=row.get("canonical_solution"),
                tests=test_src,
                entry_point=entry_point,
                raw=dict(row),
            )
        )
    return examples


def load_examples(
    task: str,
    data_path: str | None,
    num_examples: int | None,
) -> list[EvalExample]:
    if num_examples is not None and num_examples <= 0:
        num_examples = None
    if task == "gsm8k":
        return load_gsm8k(data_path, num_examples)
    if task == "mbpp":
        return load_mbpp(data_path, num_examples)
    if task == "humaneval":
        return load_humaneval(data_path, num_examples)
    raise ValueError(f"Unknown task: {task}")


def extract_code(text: str) -> str:
    text = text or ""
    fenced = re.findall(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL)
    if fenced:
        return fenced[0].strip()
    return text.strip()


def build_code_program(example: EvalExample, completion: str, task: str) -> str:
    code = extract_code(completion)
    tests = example.tests or ""
    if task == "humaneval":
        has_full_def = bool(
            example.entry_point
            and re.search(rf"\bdef\s+{re.escape(example.entry_point)}\s*\(", code)
        )
        if has_full_def:
            return code + "\n\n" + tests
        raw_prompt = example.raw.get("prompt", "") if example.raw else ""
        return raw_prompt + code + "\n\n" + tests
    return code + "\n\n" + tests


def run_python_program(program: str, timeout: float) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory(prefix="dllm_eval_") as tmpdir:
        path = Path(tmpdir) / "candidate.py"
        path.write_text(program, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, str(path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return False, "timeout"
        if proc.returncode == 0:
            return True, ""
        err = (proc.stderr or proc.stdout or "").strip()
        return False, err[-2000:]


def judge_code_outputs(
    task: str,
    examples: list[EvalExample],
    texts: list[str],
    timeout: float,
) -> list[dict[str, Any]]:
    rows = []
    for example, text in zip(examples, texts):
        program = build_code_program(example, text, task)
        passed, error = run_python_program(program, timeout)
        rows.append(
            {
                "passed": passed,
                "error": error,
                "extracted_code": extract_code(text),
            }
        )
    return rows


async def run_all(engine, prompts, sampling_params, parallel: int):
    semaphore = asyncio.Semaphore(parallel)
    completed = 0

    async def run_one(index: int, prompt: str):
        nonlocal completed
        async with semaphore:
            tic = time.perf_counter()
            output = await engine.async_generate(prompt, sampling_params)
            output["request_latency"] = time.perf_counter() - tic
            output["index"] = index
            completed += 1
            if completed % 10 == 0 or completed == len(prompts):
                print(f"completed {completed}/{len(prompts)}", flush=True)
            return output

    tasks = [asyncio.create_task(run_one(i, prompt)) for i, prompt in enumerate(prompts)]
    return await asyncio.gather(*tasks)


def to_chat_prompts(model_path: str, examples: Iterable[EvalExample]) -> list[str]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    prompts = []
    for example in examples:
        prompts.append(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": example.prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    return prompts


def build_engine(args):
    import sglang as sgl

    kwargs = dict(
        model_path=args.model_path,
        trust_remote_code=True,
        device=args.device,
        attention_backend=args.attention_backend,
        dllm_algorithm="LowConfidence",
        dllm_algorithm_config=args.dllm_algorithm_config or None,
        max_running_requests=args.max_running_requests,
        context_length=args.context_length,
        disable_cuda_graph=args.disable_cuda_graph,
        disable_radix_cache=args.disable_radix_cache,
        mem_fraction_static=args.mem_fraction_static,
        tp_size=args.tp_size,
        base_gpu_id=args.base_gpu_id,
        dtype=args.dtype,
        log_level=args.log_level,
    )
    return sgl.Engine(**kwargs)


def compute_metrics(task: str, examples, outputs, judge_rows, latency, judge_latency):
    texts = [out.get("text", "") for out in outputs]
    prompt_tokens = sum(
        out.get("meta_info", {}).get("prompt_tokens", 0) for out in outputs
    )
    completion_tokens = sum(
        out.get("meta_info", {}).get("completion_tokens", 0) for out in outputs
    )
    request_latencies = [out.get("request_latency", 0.0) for out in outputs]

    if task == "gsm8k":
        preds = [get_answer_value(text) for text in texts]
        labels = [ex.label for ex in examples]
        correct = [pred == label for pred, label in zip(preds, labels)]
        invalid = [pred == INVALID for pred in preds]
        score = sum(correct) / len(correct) if correct else 0.0
        invalid_rate = sum(invalid) / len(invalid) if invalid else 0.0
        per_sample_extra = [
            {"pred": pred, "label": label, "correct": ok}
            for pred, label, ok in zip(preds, labels, correct)
        ]
    else:
        correct = [bool(row["passed"]) for row in judge_rows]
        score = sum(correct) / len(correct) if correct else 0.0
        invalid_rate = 0.0
        per_sample_extra = judge_rows

    metrics = {
        "task": task,
        "num_examples": len(examples),
        "score": score,
        "accuracy": score,
        "pass_at_1": score if task in ("mbpp", "humaneval") else None,
        "invalid": invalid_rate,
        "generation_latency_sec": latency,
        "judge_latency_sec": judge_latency,
        "total_latency_sec": latency + judge_latency,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "output_throughput_tok_s": completion_tokens / latency if latency > 0 else 0.0,
        "request_throughput_req_s": len(examples) / latency if latency > 0 else 0.0,
        "mean_request_latency_sec": (
            sum(request_latencies) / len(request_latencies) if request_latencies else 0.0
        ),
        "max_request_latency_sec": max(request_latencies) if request_latencies else 0.0,
    }
    return metrics, per_sample_extra


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["gsm8k", "mbpp", "humaneval"], required=True)
    parser.add_argument(
        "--model-path",
        default=os.getenv("MODEL_PATH", "Dream-org/Dream-v0-Instruct-7B"),
    )
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--num-examples", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--max-running-requests", type=int, default=1)
    parser.add_argument("--mem-fraction-static", type=float, default=0.75)
    parser.add_argument("--device", default="npu")
    parser.add_argument("--attention-backend", default="ascend")
    parser.add_argument("--disable-cuda-graph", action="store_true", default=True)
    parser.add_argument("--enable-cuda-graph", dest="disable_cuda_graph", action="store_false")
    parser.add_argument("--disable-radix-cache", action="store_true", default=True)
    parser.add_argument("--enable-radix-cache", dest="disable_radix_cache", action="store_false")
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--base-gpu-id", type=int, default=0)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--log-level", default="info")
    parser.add_argument("--dllm-algorithm-config", default=None)
    parser.add_argument("--mode-name", default="baseline")
    parser.add_argument("--output-dir", default="benchmark/dllm_eval/results")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--code-timeout", type=float, default=10.0)
    args = parser.parse_args()

    examples = load_examples(args.task, args.data_path, args.num_examples)
    prompts = to_chat_prompts(args.model_path, examples)

    sampling_params = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
    }

    engine = build_engine(args)
    tic = time.perf_counter()
    try:
        outputs = asyncio.run(run_all(engine, prompts, sampling_params, args.parallel))
    finally:
        engine.shutdown()
    generation_latency = time.perf_counter() - tic

    judge_tic = time.perf_counter()
    if args.task in ("mbpp", "humaneval"):
        judge_rows = judge_code_outputs(
            args.task,
            examples,
            [out.get("text", "") for out in outputs],
            args.code_timeout,
        )
    else:
        judge_rows = []
    judge_latency = time.perf_counter() - judge_tic

    metrics, per_sample_extra = compute_metrics(
        args.task,
        examples,
        outputs,
        judge_rows,
        generation_latency,
        judge_latency,
    )
    metrics.update(
        {
            "mode": args.mode_name,
            "model_path": args.model_path,
            "data_path": args.data_path,
            "max_new_tokens": args.max_new_tokens,
            "context_length": args.context_length,
            "parallel": args.parallel,
            "max_running_requests": args.max_running_requests,
            "mem_fraction_static": args.mem_fraction_static,
            "device": args.device,
            "attention_backend": args.attention_backend,
            "tp_size": args.tp_size,
            "base_gpu_id": args.base_gpu_id,
            "dtype": args.dtype,
            "dllm_algorithm_config": args.dllm_algorithm_config,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "code_timeout": args.code_timeout,
        }
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / f"{args.task}_{args.mode_name}_metrics.json"
    outputs_path = output_dir / f"{args.task}_{args.mode_name}_outputs.jsonl"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
    with outputs_path.open("w", encoding="utf-8") as fout:
        for example, output, extra in zip(examples, outputs, per_sample_extra):
            row = {
                "task": args.task,
                "mode": args.mode_name,
                "task_id": example.task_id,
                "prompt": example.prompt,
                "text": output.get("text", ""),
                "prompt_tokens": output.get("meta_info", {}).get("prompt_tokens"),
                "completion_tokens": output.get("meta_info", {}).get("completion_tokens"),
                "request_latency": output.get("request_latency"),
                **extra,
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
