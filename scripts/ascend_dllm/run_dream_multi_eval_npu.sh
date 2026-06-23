#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

MODEL_PATH="${MODEL_PATH:-Dream-org/Dream-v0-Instruct-7B}"
NPU_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-${NPU_DEVICES:-0}}"
OUTPUT_DIR="${OUTPUT_DIR:-benchmark/dllm_eval/results_npu_$(date +%Y%m%d_%H%M%S)}"
TASKS="${TASKS:-gsm8k}"
MODES="${MODES:-baseline vbs1}"

GSM8K_DATA_PATH="${GSM8K_DATA_PATH:-}"
MBPP_DATA_PATH="${MBPP_DATA_PATH:-}"
HUMANEVAL_DATA_PATH="${HUMANEVAL_DATA_PATH:-}"

GSM8K_NUM_EXAMPLES="${GSM8K_NUM_EXAMPLES:-200}"
MBPP_NUM_EXAMPLES="${MBPP_NUM_EXAMPLES:-0}"
HUMANEVAL_NUM_EXAMPLES="${HUMANEVAL_NUM_EXAMPLES:-0}"

GSM8K_MAX_NEW_TOKENS="${GSM8K_MAX_NEW_TOKENS:-512}"
MBPP_MAX_NEW_TOKENS="${MBPP_MAX_NEW_TOKENS:-512}"
HUMANEVAL_MAX_NEW_TOKENS="${HUMANEVAL_MAX_NEW_TOKENS:-512}"

CONTEXT_LENGTH="${CONTEXT_LENGTH:-1024}"
PARALLEL="${PARALLEL:-1}"
SUBMIT_BATCH_SIZE="${SUBMIT_BATCH_SIZE:-0}"
SLEEP_BETWEEN_BATCHES="${SLEEP_BETWEEN_BATCHES:-0}"
FLUSH_CACHE_BETWEEN_BATCHES="${FLUSH_CACHE_BETWEEN_BATCHES:-0}"
LOG_EACH_REQUEST="${LOG_EACH_REQUEST:-0}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-1}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.75}"
TP_SIZE="${TP_SIZE:-1}"
BASE_GPU_ID="${BASE_GPU_ID:-0}"
DTYPE="${DTYPE:-auto}"
CODE_TIMEOUT="${CODE_TIMEOUT:-10}"
STRICT_MEM_CHECK_IDLE="${STRICT_MEM_CHECK_IDLE:-0}"
DLLM_PROFILE="${DLLM_PROFILE:-0}"

BASELINE_CONFIG="${BASELINE_CONFIG:-}"
VBS1_CONFIG="${VBS1_CONFIG:-}"
VBS2_CONFIG="${VBS2_CONFIG:-}"
if [[ -z "${BASELINE_CONFIG}" ]]; then
  BASELINE_CONFIG="benchmark/dllm_eval/configs/dream_baseline_bs32_confidence.yaml"
fi
if [[ -z "${VBS1_CONFIG}" ]]; then
  VBS1_CONFIG="benchmark/dllm_eval/configs/dream_vbs1_bs32_confidence.yaml"
fi
if [[ -z "${VBS2_CONFIG}" ]]; then
  VBS2_CONFIG="benchmark/dllm_eval/configs/dream_vbs2_bs32_confidence.yaml"
fi

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
export SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE="${STRICT_MEM_CHECK_IDLE}"

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_DIR}"

config_for_mode() {
  case "$1" in
    baseline) echo "${BASELINE_CONFIG}" ;;
    vbs1) echo "${VBS1_CONFIG}" ;;
    vbs2) echo "${VBS2_CONFIG}" ;;
    *)
      echo "Unknown mode: $1" >&2
      return 2
      ;;
  esac
}

data_path_for_task() {
  case "$1" in
    gsm8k) echo "${GSM8K_DATA_PATH}" ;;
    mbpp) echo "${MBPP_DATA_PATH}" ;;
    humaneval) echo "${HUMANEVAL_DATA_PATH}" ;;
    *)
      echo "Unknown task: $1" >&2
      return 2
      ;;
  esac
}

num_examples_for_task() {
  case "$1" in
    gsm8k) echo "${GSM8K_NUM_EXAMPLES}" ;;
    mbpp) echo "${MBPP_NUM_EXAMPLES}" ;;
    humaneval) echo "${HUMANEVAL_NUM_EXAMPLES}" ;;
  esac
}

max_tokens_for_task() {
  case "$1" in
    gsm8k) echo "${GSM8K_MAX_NEW_TOKENS}" ;;
    mbpp) echo "${MBPP_MAX_NEW_TOKENS}" ;;
    humaneval) echo "${HUMANEVAL_MAX_NEW_TOKENS}" ;;
  esac
}

is_truthy() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

echo "Running Dream dLLM multi-eval on Ascend NPU"
echo "  MODEL_PATH=${MODEL_PATH}"
echo "  ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}"
echo "  TASKS=${TASKS}"
echo "  MODES=${MODES}"
echo "  OUTPUT_DIR=${OUTPUT_DIR}"
echo "  PARALLEL=${PARALLEL}"
echo "  SUBMIT_BATCH_SIZE=${SUBMIT_BATCH_SIZE}"
echo "  MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS}"
echo "  STRICT_MEM_CHECK_IDLE=${STRICT_MEM_CHECK_IDLE}"
echo "  DLLM_PROFILE=${DLLM_PROFILE}"

for task in ${TASKS}; do
  for mode in ${MODES}; do
    metrics_path="${OUTPUT_DIR}/${task}_${mode}_metrics.json"
    profile_path="${OUTPUT_DIR}/${task}_${mode}_dllm_profile.jsonl"
    if [[ -f "${metrics_path}" ]]; then
      echo "[skip] ${task}/${mode}"
      continue
    fi

    cfg="$(config_for_mode "${mode}")"
    data_path="$(data_path_for_task "${task}")"
    num_examples="$(num_examples_for_task "${task}")"
    max_new_tokens="$(max_tokens_for_task "${task}")"

    args=(
      benchmark/dllm_eval/bench_dream_dllm_tasks.py
      --task "${task}"
      --mode-name "${mode}"
      --model-path "${MODEL_PATH}"
      --dllm-algorithm-config "${cfg}"
      --output-dir "${OUTPUT_DIR}"
      --num-examples "${num_examples}"
      --max-new-tokens "${max_new_tokens}"
      --context-length "${CONTEXT_LENGTH}"
      --parallel "${PARALLEL}"
      --submit-batch-size "${SUBMIT_BATCH_SIZE}"
      --sleep-between-batches "${SLEEP_BETWEEN_BATCHES}"
      --max-running-requests "${MAX_RUNNING_REQUESTS}"
      --mem-fraction-static "${MEM_FRACTION_STATIC}"
      --device npu
      --attention-backend ascend
      --disable-cuda-graph
      --disable-radix-cache
      --tp-size "${TP_SIZE}"
      --base-gpu-id "${BASE_GPU_ID}"
      --dtype "${DTYPE}"
      --code-timeout "${CODE_TIMEOUT}"
    )
    if [[ -n "${data_path}" ]]; then
      args+=(--data-path "${data_path}")
    fi
    if is_truthy "${FLUSH_CACHE_BETWEEN_BATCHES}"; then
      args+=(--flush-cache-between-batches)
    fi
    if is_truthy "${LOG_EACH_REQUEST}"; then
      args+=(--log-each-request)
    fi
    if is_truthy "${DLLM_PROFILE}"; then
      args+=(--dllm-profile-path "${profile_path}")
    fi

    echo "[run] ${task}/${mode}"
    "${PYTHON_BIN}" "${args[@]}"
  done
done

"${PYTHON_BIN}" benchmark/dllm_eval/summarize_results.py --input-dir "${OUTPUT_DIR}"
