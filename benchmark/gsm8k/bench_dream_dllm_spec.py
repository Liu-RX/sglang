import argparse
import ast
import asyncio
import json
import re
import time
from pathlib import Path


INVALID = -9999999


def get_answer_value(answer_str: str) -> int:
    answer_str = answer_str.replace(",", "")
    numbers = re.findall(r"\d+", answer_str)
    if len(numbers) < 1:
        return INVALID
    try:
        return ast.literal_eval(numbers[-1])
    except SyntaxError:
        return INVALID


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
            if completed % 25 == 0 or completed == len(prompts):
                print(f"completed {completed}/{len(prompts)}", flush=True)
            return output

    tasks = [asyncio.create_task(run_one(i, prompt)) for i, prompt in enumerate(prompts)]
    return await asyncio.gather(*tasks)


def load_gsm8k(path: str | None):
    from sglang.utils import download_and_cache_file, read_jsonl

    if path is None:
        url = (
            "https://raw.githubusercontent.com/openai/grade-school-math/master/"
            "grade_school_math/data/test.jsonl"
        )
        path = download_and_cache_file(url)
    return list(read_jsonl(path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="Dream-org/Dream-v0-Instruct-7B")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--num-questions", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--max-running-requests", type=int, default=4)
    parser.add_argument("--mem-fraction-static", type=float, default=0.75)
    parser.add_argument("--device", default=None)
    parser.add_argument("--attention-backend", default="flashinfer")
    parser.add_argument(
        "--disable-cuda-graph",
        dest="disable_cuda_graph",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--enable-cuda-graph",
        dest="disable_cuda_graph",
        action="store_false",
    )
    parser.add_argument("--disable-radix-cache", action="store_true")
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--base-gpu-id", type=int, default=0)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--dllm-algorithm-config", default=None)
    parser.add_argument("--mode-name", default="baseline")
    parser.add_argument("--output-dir", default="benchmark/gsm8k/results_dllm_spec")
    args = parser.parse_args()

    import numpy as np
    import sglang as sgl
    from transformers import AutoTokenizer

    data = load_gsm8k(args.data_path)
    if args.end_index is None:
        args.end_index = len(data)
    data = data[args.start_index : args.end_index]
    if args.num_questions is not None:
        data = data[: args.num_questions]

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    prompts = [
        tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": (
                        row["question"]
                        + "\nSolve the problem and answer with the final number."
                    ),
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        for row in data
    ]
    labels = [get_answer_value(row["answer"]) for row in data]
    assert all(label != INVALID for label in labels)

    engine_kwargs = dict(
        model_path=args.model_path,
        trust_remote_code=True,
        dllm_algorithm="LowConfidence",
        dllm_algorithm_config=args.dllm_algorithm_config,
        max_running_requests=args.max_running_requests,
        context_length=args.context_length,
        disable_cuda_graph=args.disable_cuda_graph,
        disable_radix_cache=args.disable_radix_cache,
        attention_backend=args.attention_backend,
        mem_fraction_static=args.mem_fraction_static,
        tp_size=args.tp_size,
        base_gpu_id=args.base_gpu_id,
        dtype=args.dtype,
    )
    if args.device is not None:
        engine_kwargs["device"] = args.device
    engine = sgl.Engine(**engine_kwargs)

    sampling_params = {
        "temperature": 0.0,
        "max_new_tokens": args.max_new_tokens,
        "context_length": args.context_length,
    }

    tic = time.perf_counter()
    try:
        outputs = asyncio.run(run_all(engine, prompts, sampling_params, args.parallel))
    finally:
        engine.shutdown()
    latency = time.perf_counter() - tic

    preds = [get_answer_value(output["text"]) for output in outputs]
    labels_np = np.array(labels)
    preds_np = np.array(preds)
    accuracy = float(np.mean(preds_np == labels_np))
    invalid = float(np.mean(preds_np == INVALID))
    completion_tokens = sum(
        output["meta_info"]["completion_tokens"] for output in outputs
    )
    prompt_tokens = sum(output["meta_info"]["prompt_tokens"] for output in outputs)
    output_throughput = completion_tokens / latency

    result_rows = []
    for row, output, label, pred in zip(data, outputs, labels, preds):
        result_rows.append(
            {
                "question": row["question"],
                "label": label,
                "pred": pred,
                "correct": pred == label,
                "text": output["text"],
                "prompt_tokens": output["meta_info"]["prompt_tokens"],
                "completion_tokens": output["meta_info"]["completion_tokens"],
                "request_latency": output["request_latency"],
            }
        )

    metrics = {
        "mode": args.mode_name,
        "start_index": args.start_index,
        "end_index": args.start_index + len(data),
        "num_questions": len(data),
        "accuracy": accuracy,
        "invalid": invalid,
        "latency_sec": latency,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "output_throughput_tok_s": output_throughput,
        "parallel": args.parallel,
        "max_running_requests": args.max_running_requests,
        "mem_fraction_static": args.mem_fraction_static,
        "max_new_tokens": args.max_new_tokens,
        "device": args.device,
        "attention_backend": args.attention_backend,
        "disable_cuda_graph": args.disable_cuda_graph,
        "disable_radix_cache": args.disable_radix_cache,
        "tp_size": args.tp_size,
        "base_gpu_id": args.base_gpu_id,
        "dtype": args.dtype,
        "dllm_algorithm_config": args.dllm_algorithm_config,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{args.start_index:04d}_{args.start_index + len(data):04d}"
    metrics_path = output_dir / f"{args.mode_name}_{suffix}_metrics.json"
    rows_path = output_dir / f"{args.mode_name}_{suffix}_outputs.jsonl"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    with rows_path.open("w") as f:
        for row in result_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
