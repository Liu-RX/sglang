#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

MODEL_PATH="${MODEL_PATH:-Dream-org/Dream-v0-Instruct-7B}"
DLLM_ALGORITHM_CONFIG="${DLLM_ALGORITHM_CONFIG:-examples/dllm_low_confidence_spec_vbs1.yaml}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30000}"
NPU_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-${NPU_DEVICES:-0}}"
TP_SIZE="${TP_SIZE:-1}"
BASE_GPU_ID="${BASE_GPU_ID:-0}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-1}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-1024}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.75}"
DTYPE="${DTYPE:-auto}"

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

DLLM_CONFIG_ARGS=()
case "${DLLM_ALGORITHM_CONFIG}" in
  ""|"none"|"None"|"null")
    ;;
  *)
    DLLM_CONFIG_ARGS=(--dllm-algorithm-config "${DLLM_ALGORITHM_CONFIG}")
    ;;
esac

cd "${REPO_ROOT}"

echo "Starting Dream dLLM server on Ascend NPU"
echo "  MODEL_PATH=${MODEL_PATH}"
echo "  DLLM_ALGORITHM_CONFIG=${DLLM_ALGORITHM_CONFIG}"
echo "  ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}"
echo "  TP_SIZE=${TP_SIZE}"
echo "  HOST=${HOST}"
echo "  PORT=${PORT}"

exec "${PYTHON_BIN}" -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --trust-remote-code \
  --device npu \
  --attention-backend ascend \
  --dllm-algorithm LowConfidence \
  "${DLLM_CONFIG_ARGS[@]}" \
  --max-running-requests "${MAX_RUNNING_REQUESTS}" \
  --context-length "${CONTEXT_LENGTH}" \
  --mem-fraction-static "${MEM_FRACTION_STATIC}" \
  --tp-size "${TP_SIZE}" \
  --base-gpu-id "${BASE_GPU_ID}" \
  --dtype "${DTYPE}" \
  --disable-cuda-graph \
  --disable-radix-cache \
  --host "${HOST}" \
  --port "${PORT}" \
  "$@"
