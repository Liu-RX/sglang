#!/usr/bin/env python3
"""Summarize per-task/per-mode metrics into a compact JSON and Markdown table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    rows = []
    for path in sorted(input_dir.glob("*_metrics.json")):
        metrics = json.loads(path.read_text())
        rows.append(metrics)
    if not rows:
        raise RuntimeError(f"No metrics files found in {input_dir}")

    summary_path = input_dir / "summary.json"
    summary_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")

    headers = [
        "task",
        "mode",
        "score",
        "gen_latency_s",
        "total_latency_s",
        "tok/s",
        "req/s",
        "completion_tokens",
        "spec_accept",
        "spec_proposed",
        "blocked_unmask",
        "inactive_steps",
        "normal_fw",
    ]
    table = ["| " + " | ".join(headers) + " |"]
    table.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        table.append(
            "| "
            + " | ".join(
                [
                    str(row.get("task")),
                    str(row.get("mode")),
                    f"{row.get('score', 0):.4f}",
                    f"{row.get('generation_latency_sec', 0):.2f}",
                    f"{row.get('total_latency_sec', 0):.2f}",
                    f"{row.get('output_throughput_tok_s', 0):.2f}",
                    f"{row.get('request_throughput_req_s', 0):.2f}",
                    str(row.get("completion_tokens", 0)),
                    ""
                    if row.get("dllm_spec_accept_rate") is None
                    else f"{row.get('dllm_spec_accept_rate', 0):.2%}",
                    str(row.get("dllm_spec_proposed_positions", "")),
                    str(row.get("dllm_spec_blocked_normal_unmask_steps", "")),
                    ""
                    if row.get("dllm_inactive_sample_step_rate") is None
                    else f"{row.get('dllm_inactive_sample_step_rate', 0):.2%}",
                    str(row.get("dllm_normal_forward_calls", "")),
                ]
            )
            + " |"
        )
    md_path = input_dir / "summary.md"
    md_path.write_text("\n".join(table) + "\n")
    print("\n".join(table))
    print(f"\nWrote {summary_path} and {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
