#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

MODEL_PATH="${MODEL_PATH:-Dream-org/Dream-v0-Instruct-7B}"
NPU_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-${NPU_DEVICES:-0}}"
DATA_PATH="${DATA_PATH:-/tmp/gsm8k_test.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-benchmark/gsm8k/results_dllm_spec_npu_smoke}"
NUM_QUESTIONS="${NUM_QUESTIONS:-20}"
CHUNK_SIZE="${CHUNK_SIZE:-20}"
PARALLEL="${PARALLEL:-1}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-1}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.75}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-1024}"
TP_SIZE="${TP_SIZE:-1}"
BASE_GPU_ID="${BASE_GPU_ID:-0}"
DTYPE="${DTYPE:-auto}"
MODES="${MODES:-baseline vbs1}"

export ASCEND_RT_VISIBLE_DEVICES="${NPU_DEVICES}"
export PYTHONPATH="${REPO_ROOT}/python:${PYTHONPATH:-}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT="${SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT:-1}"
export SGLANG_SET_CPU_AFFINITY="${SGLANG_SET_CPU_AFFINITY:-1}"
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-200}"
export HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-200}"
export STREAMS_PER_DEVICE="${STREAMS_PER_DEVICE:-32}"
export AUTO_USE_UC_MEMORY="${AUTO_USE_UC_MEMORY:-0}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_DIR}"

if [[ ! -f "${DATA_PATH}" ]]; then
  mkdir -p "$(dirname "${DATA_PATH}")"
  "${PYTHON_BIN}" - "${DATA_PATH}" <<'PY'
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

TOTAL_LINES="$(wc -l < "${DATA_PATH}")"
if (( NUM_QUESTIONS > 0 && NUM_QUESTIONS < TOTAL_LINES )); then
  TOTAL_QUESTIONS="${NUM_QUESTIONS}"
else
  TOTAL_QUESTIONS="${TOTAL_LINES}"
fi

declare -A MODE_CONFIGS=(
  [baseline]=""
  [vbs1]="examples/dllm_low_confidence_spec_vbs1.yaml"
  [vbs2]="examples/dllm_low_confidence_spec_vbs2.yaml"
  [spiffy]="examples/dllm_low_confidence_spec_spiffy.yaml"
  [vbs1_confidence]="examples/dllm_low_confidence_spec_vbs1_confidence.yaml"
  [vbs1_bs32_confidence]="benchmark/gsm8k/dllm_low_confidence_spec_vbs1_bs32_confidence.yaml"
  [vbs2_bs32_confidence]="benchmark/gsm8k/dllm_low_confidence_spec_vbs2_bs32_confidence.yaml"
)

run_mode() {
  local mode="$1"
  local cfg="${MODE_CONFIGS[${mode}]:-__missing__}"
  local start end suffix metrics_path
  if [[ "${cfg}" == "__missing__" ]]; then
    echo "Unknown mode: ${mode}" >&2
    echo "Known modes: ${!MODE_CONFIGS[*]}" >&2
    exit 2
  fi

  for ((start=0; start<TOTAL_QUESTIONS; start+=CHUNK_SIZE)); do
    end=$((start + CHUNK_SIZE))
    if (( end > TOTAL_QUESTIONS )); then
      end="${TOTAL_QUESTIONS}"
    fi
    suffix="$(printf "%04d_%04d" "${start}" "${end}")"
    metrics_path="${OUTPUT_DIR}/${mode}_${suffix}_metrics.json"
    if [[ -f "${metrics_path}" ]]; then
      echo "[skip] ${mode} ${start}-${end}"
      continue
    fi

    echo "[run] ${mode} ${start}-${end}"
    args=(
      benchmark/gsm8k/bench_dream_dllm_spec.py
      --model-path "${MODEL_PATH}"
      --data-path "${DATA_PATH}"
      --start-index "${start}"
      --end-index "${end}"
      --parallel "${PARALLEL}"
      --max-running-requests "${MAX_RUNNING_REQUESTS}"
      --mem-fraction-static "${MEM_FRACTION_STATIC}"
      --max-new-tokens "${MAX_NEW_TOKENS}"
      --context-length "${CONTEXT_LENGTH}"
      --mode-name "${mode}"
      --output-dir "${OUTPUT_DIR}"
      --device npu
      --attention-backend ascend
      --disable-cuda-graph
      --disable-radix-cache
      --tp-size "${TP_SIZE}"
      --base-gpu-id "${BASE_GPU_ID}"
      --dtype "${DTYPE}"
    )
    if [[ -n "${cfg}" ]]; then
      args+=(--dllm-algorithm-config "${cfg}")
    fi
    "${PYTHON_BIN}" "${args[@]}"
  done

  "${PYTHON_BIN}" benchmark/gsm8k/aggregate_dream_dllm_spec.py \
    --input-dir "${OUTPUT_DIR}" \
    --mode-name "${mode}"
}

echo "Running Dream dLLM GSM8K on Ascend NPU"
echo "  MODEL_PATH=${MODEL_PATH}"
echo "  ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}"
echo "  TOTAL_QUESTIONS=${TOTAL_QUESTIONS}"
echo "  MODES=${MODES}"
echo "  OUTPUT_DIR=${OUTPUT_DIR}"

for mode in ${MODES}; do
  run_mode "${mode}"
done
