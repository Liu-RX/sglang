#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/liurenxi/sglang-for-dllm}"
CONDA_ENV="${CONDA_ENV:-/home/liurenxi/anaconda3/envs/lrx-dflash}"
MODEL_PATH="${MODEL_PATH:-Dream-org/Dream-v0-Instruct-7B}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30000}"
GPU="${CUDA_VISIBLE_DEVICES:-1}"
DLLM_CONFIG_ARGS=()
if [[ -n "${DLLM_ALGORITHM_CONFIG:-}" ]]; then
  DLLM_CONFIG_ARGS=(--dllm-algorithm-config "${DLLM_ALGORITHM_CONFIG}")
fi

export CUDA_VISIBLE_DEVICES="${GPU}"
export CUDA_HOME="${CUDA_HOME:-${CONDA_ENV}}"
export PATH="${CONDA_ENV}/nvvm/bin:${CONDA_ENV}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${CONDA_ENV}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export CPATH="${CONDA_ENV}/targets/x86_64-linux/include:${CONDA_ENV}/lib/python3.11/site-packages/nvidia/cuda_runtime/include:${CPATH:-}"
export CPLUS_INCLUDE_PATH="${CONDA_ENV}/targets/x86_64-linux/include:${CONDA_ENV}/lib/python3.11/site-packages/nvidia/cuda_runtime/include:${CPLUS_INCLUDE_PATH:-}"
export SGLANG_ENABLE_JIT_DEEPGEMM="${SGLANG_ENABLE_JIT_DEEPGEMM:-0}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"

cd "${REPO_ROOT}"

exec "${CONDA_ENV}/bin/python" -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --trust-remote-code \
  --dllm-algorithm LowConfidence \
  "${DLLM_CONFIG_ARGS[@]}" \
  --max-running-requests 1 \
  --context-length 1024 \
  --disable-cuda-graph \
  --attention-backend flashinfer \
  --host "${HOST}" \
  --port "${PORT}"
