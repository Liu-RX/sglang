#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/liurenxi/sglang-for-dllm}"
CONDA_ENV="${CONDA_ENV:-/home/liurenxi/anaconda3/envs/lrx-dflash}"
GPU="${CUDA_VISIBLE_DEVICES:-1}"
DATA_PATH="${DATA_PATH:-/tmp/test.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-benchmark/gsm8k/results_dllm_spec_full}"
CHUNK_SIZE="${CHUNK_SIZE:-75}"
PARALLEL="${PARALLEL:-2}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-2}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.75}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export CUDA_HOME="${CUDA_HOME:-${CONDA_ENV}}"
export PATH="${CONDA_ENV}/nvvm/bin:${CONDA_ENV}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${CONDA_ENV}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export CPATH="${CONDA_ENV}/targets/x86_64-linux/include:${CONDA_ENV}/lib/python3.11/site-packages/nvidia/cuda_runtime/include:${CPATH:-}"
export CPLUS_INCLUDE_PATH="${CONDA_ENV}/targets/x86_64-linux/include:${CONDA_ENV}/lib/python3.11/site-packages/nvidia/cuda_runtime/include:${CPLUS_INCLUDE_PATH:-}"
export SGLANG_ENABLE_JIT_DEEPGEMM="${SGLANG_ENABLE_JIT_DEEPGEMM:-0}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_DIR}"

if [[ ! -f "${DATA_PATH}" ]]; then
  mkdir -p "$(dirname "${DATA_PATH}")"
  "${CONDA_ENV}/bin/python" - "${DATA_PATH}" <<'PY'
import shutil
import sys
from pathlib import Path

from sglang.utils import download_and_cache_file

URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/master/"
    "grade_school_math/data/test.jsonl"
)
dst = Path(sys.argv[1])
src = Path(download_and_cache_file(URL))
if src.resolve() != dst.resolve():
    shutil.copyfile(src, dst)
PY
fi

NUM_QUESTIONS="$(wc -l < "${DATA_PATH}")"

run_mode() {
  local mode="$1"
  local cfg="${2:-}"
  local start end suffix metrics_path

  for ((start=0; start<NUM_QUESTIONS; start+=CHUNK_SIZE)); do
    end=$((start + CHUNK_SIZE))
    if (( end > NUM_QUESTIONS )); then
      end="${NUM_QUESTIONS}"
    fi
    suffix="$(printf "%04d_%04d" "${start}" "${end}")"
    metrics_path="${OUTPUT_DIR}/${mode}_${suffix}_metrics.json"
    if [[ -f "${metrics_path}" ]]; then
      echo "[skip] ${mode} ${start}-${end}"
      continue
    fi

    echo "[run] ${mode} ${start}-${end}"
    local args=(
      benchmark/gsm8k/bench_dream_dllm_spec.py
      --data-path "${DATA_PATH}"
      --start-index "${start}"
      --end-index "${end}"
      --parallel "${PARALLEL}"
      --max-running-requests "${MAX_RUNNING_REQUESTS}"
      --mem-fraction-static "${MEM_FRACTION_STATIC}"
      --max-new-tokens "${MAX_NEW_TOKENS}"
      --mode-name "${mode}"
      --output-dir "${OUTPUT_DIR}"
    )
    if [[ -n "${cfg}" ]]; then
      args+=(--dllm-algorithm-config "${cfg}")
    fi
    "${CONDA_ENV}/bin/python" "${args[@]}"
  done

  "${CONDA_ENV}/bin/python" benchmark/gsm8k/aggregate_dream_dllm_spec.py \
    --input-dir "${OUTPUT_DIR}" \
    --mode-name "${mode}"
}

run_mode baseline
run_mode vbs1 examples/dllm_low_confidence_spec_vbs1.yaml
run_mode vbs2 examples/dllm_low_confidence_spec_vbs2.yaml
run_mode spiffy examples/dllm_low_confidence_spec_spiffy.yaml
