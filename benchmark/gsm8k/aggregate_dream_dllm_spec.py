import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="benchmark/gsm8k/results_dllm_spec_full")
    parser.add_argument("--mode-name", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    metric_files = sorted(input_dir.glob(f"{args.mode_name}_*_metrics.json"))
    output_files = sorted(input_dir.glob(f"{args.mode_name}_*_outputs.jsonl"))
    if not metric_files or not output_files:
        raise RuntimeError(f"No chunk files found for mode {args.mode_name}")

    total_latency = 0.0
    prompt_tokens = 0
    completion_tokens = 0
    chunk_ranges = []
    for path in metric_files:
        metrics = json.loads(path.read_text())
        total_latency += metrics["latency_sec"]
        prompt_tokens += metrics["prompt_tokens"]
        completion_tokens += metrics["completion_tokens"]
        chunk_ranges.append([metrics["start_index"], metrics["end_index"]])

    num_questions = 0
    num_correct = 0
    num_invalid = 0
    seen = set()
    for path in output_files:
        with path.open() as f:
            for line in f:
                row = json.loads(line)
                key = (row["question"], row["label"])
                if key in seen:
                    raise RuntimeError(f"Duplicate row found in {path}: {key[0][:80]}")
                seen.add(key)
                num_questions += 1
                num_correct += int(row["correct"])
                num_invalid += int(row["pred"] == -9999999)

    aggregate = {
        "mode": args.mode_name,
        "num_chunks": len(metric_files),
        "chunk_ranges": chunk_ranges,
        "num_questions": num_questions,
        "accuracy": num_correct / num_questions,
        "invalid": num_invalid / num_questions,
        "latency_sec": total_latency,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "output_throughput_tok_s": completion_tokens / total_latency,
    }
    out_path = input_dir / f"{args.mode_name}_aggregate_metrics.json"
    out_path.write_text(json.dumps(aggregate, indent=2) + "\n")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
