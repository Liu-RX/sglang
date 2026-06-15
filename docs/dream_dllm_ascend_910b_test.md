# Dream dLLM 在昇腾 910B 上的投机解码测试说明

## 可行性结论

可以测试，但需要走 SGLang 的 NPU 运行路径：

- 启动参数使用 `--device npu --attention-backend ascend`。
- 不要直接用现有 CUDA/FlashInfer 脚本；那些脚本里写了 CUDA 环境变量和 `flashinfer`。
- 建议先跑 `baseline` 和 `vbs1`。`vbs2`、`spiffy` 会额外构造验证候选，显存压力更大。
- Dream-7B 需要能放进所选 NPU 显存。单卡 910B 如果 OOM，先降低 `MAX_NEW_TOKENS`、`CONTEXT_LENGTH`，再尝试 `TP_SIZE=2` 双卡。

判断依据：

- 本仓库已有 Ascend NPU 后端，并在文档中支持 Atlas 800I A2/910B。
- `ServerArgs` 在 NPU + dLLM 推理时会使用 `ascend` attention backend。
- 仓库里已有 `--dllm-algorithm LowConfidence` 的 NPU CI 测试。
- 迁移后的 `LowConfidence` 投机解码主体是 PyTorch 张量操作，不直接依赖 CUDA-only 自定义 kernel。

当前机器没有昇腾硬件，所以这里无法本地实测 NPU 运行结果；脚本已经做过 Python 语法和 bash 语法检查。

## 环境要求

版本建议跟随仓库 Ascend 文档：

- Python 3.11
- CANN 8.5.0
- PyTorch 2.8.0
- `torch_npu==2.8.0.post2`
- `triton-ascend`
- 与 CANN 8.5.0、910B 匹配的 `sgl-kernel-npu`

如果用容器，910B / Atlas 800I A2 对应 `main-cann8.5.0-910b` 镜像标签。源码安装时，安装 SGLang 前请使用 `python/pyproject_npu.toml` 作为 Python project 文件。

## 通用设置

把仓库复制或 clone 到昇腾服务器后：

```bash
cd /path/to/sglang-for-dllm
export PYTHONPATH="$PWD/python:${PYTHONPATH:-}"

export ASCEND_RT_VISIBLE_DEVICES=0
export MODEL_PATH=/path/to/Dream-v0-Instruct-7B
export DLLM_ALGORITHM_CONFIG=examples/dllm_low_confidence_spec_vbs1.yaml
```

如果服务器能访问 Hugging Face，也可以：

```bash
export MODEL_PATH=Dream-org/Dream-v0-Instruct-7B
```

## 1. 环境自检

```bash
python3 scripts/ascend_dllm/check_ascend_env.py --model-path "$MODEL_PATH"
```

期望结果：

- `torch.npu.is_available: True`
- 能 import `torch_npu`
- 能 import `sgl_kernel_npu`
- 能 import `sglang`
- 脚本退出码为 0

## 2. 离线 smoke 测试

这个脚本会启动一个进程内 SGLang Engine，跑一道 GSM8K 风格题目，然后自动关闭 Engine：

```bash
python3 scripts/ascend_dllm/offline_smoke_dream_npu.py \
  --model-path "$MODEL_PATH" \
  --dllm-algorithm-config "$DLLM_ALGORITHM_CONFIG"
```

不启用投机解码、只测 baseline：

```bash
python3 scripts/ascend_dllm/offline_smoke_dream_npu.py \
  --model-path "$MODEL_PATH" \
  --dllm-algorithm-config ""
```

## 3. 启动 HTTP 服务

```bash
PORT=30000 \
MODEL_PATH="$MODEL_PATH" \
DLLM_ALGORITHM_CONFIG=examples/dllm_low_confidence_spec_vbs1.yaml \
bash scripts/ascend_dllm/start_dream_sglang_server_npu.sh
```

该脚本实际会启动：

```bash
python3 -m sglang.launch_server \
  --device npu \
  --attention-backend ascend \
  --dllm-algorithm LowConfidence \
  --disable-cuda-graph \
  --disable-radix-cache
```

另开一个 shell 测试服务：

```bash
python3 scripts/ascend_dllm/request_dream_generate.py \
  --model-path "$MODEL_PATH" \
  --host 127.0.0.1 \
  --port 30000
```

## 4. 小规模 GSM8K 投机解码测试

默认跑 20 道题，模式为 `baseline` 和 `vbs1`：

```bash
MODEL_PATH="$MODEL_PATH" \
NUM_QUESTIONS=20 \
MODES="baseline vbs1" \
bash scripts/ascend_dllm/run_dream_gsm8k_dllm_spec_npu.sh
```

结果目录：

```text
benchmark/gsm8k/results_dllm_spec_npu_smoke/
```

加入更多投机模式：

```bash
MODEL_PATH="$MODEL_PATH" \
NUM_QUESTIONS=50 \
MODES="baseline vbs1 vbs2 spiffy" \
bash scripts/ascend_dllm/run_dream_gsm8k_dllm_spec_npu.sh
```

完整 GSM8K：

```bash
MODEL_PATH="$MODEL_PATH" \
NUM_QUESTIONS=1319 \
CHUNK_SIZE=75 \
MODES="baseline vbs1 vbs2" \
OUTPUT_DIR=benchmark/gsm8k/results_dllm_spec_npu_full \
bash scripts/ascend_dllm/run_dream_gsm8k_dllm_spec_npu.sh
```

## 常用调参

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
export TP_SIZE=1
export MEM_FRACTION_STATIC=0.75
export CONTEXT_LENGTH=1024
export MAX_NEW_TOKENS=128
export MAX_RUNNING_REQUESTS=1
export PARALLEL=1
```

显存紧张时：

```bash
export MEM_FRACTION_STATIC=0.65
export CONTEXT_LENGTH=768
export MAX_NEW_TOKENS=64
export MODES="baseline vbs1"
```

双卡张量并行：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1
export TP_SIZE=2
```

## 常见问题

- `torch_npu detected, but NPU device is not available`：检查容器 device 映射和 `ASCEND_RT_VISIBLE_DEVICES`。
- 出现 FlashInfer、CUDA、`cuda_runtime.h` 或 `nvidia-smi` 错误：大概率用了 CUDA 脚本，请改用 `scripts/ascend_dllm/*_npu.*`。
- `vbs2` 或 `spiffy` OOM：先确认 `baseline`、`vbs1` 能跑，再降低 `MAX_NEW_TOKENS`、降低 `MEM_FRACTION_STATIC`，或使用张量并行。
- 服务器不能联网：`MODEL_PATH` 指到本地 Dream 权重目录，GSM8K wrapper 额外设置 `DATA_PATH=/path/to/gsm8k/test.jsonl`。
- 昇腾服务器不需要 Codex；以上都是普通 shell/Python 命令。
