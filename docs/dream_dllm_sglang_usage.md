# Dream-7B dLLM SGLang 使用说明

本文档说明本仓库中 Dream-7B dLLM 适配版 SGLang 的运行环境、服务启动方式和测试方法。

## 代码与环境

- 代码目录：`/home/liurenxi/sglang-for-dllm`
- Conda 根目录：`/home/liurenxi/anaconda3`
- 推荐环境：`/home/liurenxi/anaconda3/envs/lrx-dflash`
- Python：`3.11`
- Torch：`2.9.1+cu128`
- Dream 模型：`Dream-org/Dream-v0-Instruct-7B`
- 模型缓存：`/root/.cache/huggingface/hub/models--Dream-org--Dream-v0-Instruct-7B`
- 本地源码优先加载文件：
  `/home/liurenxi/anaconda3/envs/lrx-dflash/lib/python3.11/site-packages/00-local-sglang-for-dllm.pth`

确认当前环境加载的是本地可编辑源码：

```bash
/home/liurenxi/anaconda3/envs/lrx-dflash/bin/python - <<'PY'
import sglang, torch
print(sglang.__file__)
print(torch.__version__)
PY
```

期望看到：

```text
/home/liurenxi/sglang-for-dllm/python/sglang/__init__.py
2.9.1+cu128
```

## 关键环境变量

Dream-7B 首次运行会触发 FlashInfer/Triton JIT。建议统一使用以下环境变量：

```bash
export CONDA_ENV=/home/liurenxi/anaconda3/envs/lrx-dflash
export CUDA_VISIBLE_DEVICES=1
export CUDA_HOME=$CONDA_ENV
export PATH=$CONDA_ENV/nvvm/bin:$CONDA_ENV/bin:$PATH
export LD_LIBRARY_PATH=$CONDA_ENV/lib:$CONDA_ENV/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
export CPATH=$CONDA_ENV/targets/x86_64-linux/include:$CONDA_ENV/lib/python3.11/site-packages/nvidia/cuda_runtime/include:${CPATH:-}
export CPLUS_INCLUDE_PATH=$CONDA_ENV/targets/x86_64-linux/include:$CONDA_ENV/lib/python3.11/site-packages/nvidia/cuda_runtime/include:${CPLUS_INCLUDE_PATH:-}
export SGLANG_ENABLE_JIT_DEEPGEMM=0
export TRANSFORMERS_VERBOSITY=error
```

`CUDA_VISIBLE_DEVICES=1` 是本机验证时使用的空闲 GPU。换机器或换卡时可以改成其他 GPU。

## 启动推理服务

推荐使用仓库内脚本：

```bash
cd /home/liurenxi/sglang-for-dllm
CUDA_VISIBLE_DEVICES=1 HOST=0.0.0.0 PORT=30000 \
  bash scripts/start_dream_sglang_server.sh
```

脚本默认参数：

```text
MODEL_PATH=Dream-org/Dream-v0-Instruct-7B
dllm_algorithm=LowConfidence
max_running_requests=1
context_length=1024
attention_backend=flashinfer
disable_cuda_graph=true
SGLANG_ENABLE_JIT_DEEPGEMM=0
```

可以通过环境变量覆盖：

```bash
CUDA_VISIBLE_DEVICES=3 PORT=31000 MODEL_PATH=/path/to/Dream-v0-Instruct-7B \
  bash scripts/start_dream_sglang_server.sh
```

## 启动 dLLM 投机解码

本仓库的 `LowConfidence` dLLM 算法支持可选自投机解码。当前移植了三种验证方式：

- `verify_batch_size: 1`：单候选验证，低于 `threshold` 的投机 token 会被打回 mask。
- `verify_batch_size: 2`：两候选验证，在“保留全部投机 token”和“全部打回 mask”的候选之间选择。
- `verify_batch_size: spiffy`：按 spiffy DAG 构造最多 8 个候选节点，并按父子结构逐层接受。

未移植 pangu 版本中的前缀/后缀匹配，也未启用 `verify_batch_size=4`。

示例配置位于：

```text
examples/dllm_low_confidence_spec_vbs1.yaml
examples/dllm_low_confidence_spec_vbs2.yaml
examples/dllm_low_confidence_spec_spiffy.yaml
```

HTTP 服务启动方式：

```bash
cd /home/liurenxi/sglang-for-dllm
CUDA_VISIBLE_DEVICES=1 \
DLLM_ALGORITHM_CONFIG=examples/dllm_low_confidence_spec_spiffy.yaml \
HOST=0.0.0.0 PORT=30000 \
  bash scripts/start_dream_sglang_server.sh
```

也可以直接传给 `launch_server`：

```bash
python -m sglang.launch_server \
  --model-path Dream-org/Dream-v0-Instruct-7B \
  --trust-remote-code \
  --dllm-algorithm LowConfidence \
  --dllm-algorithm-config examples/dllm_low_confidence_spec_vbs2.yaml \
  --max-running-requests 1 \
  --context-length 1024 \
  --disable-cuda-graph \
  --attention-backend flashinfer \
  --host 0.0.0.0 \
  --port 30000
```

关键配置项：

```yaml
speculative_decoding: true
alg: confidence_threshold
speculative_decoding_mode: greedy  # greedy 或 confidence
verify_batch_size: spiffy  # 也可以是 1 或 2
threshold: 0.95
speculative_confidence_threshold: 0.5
self_speculative_confidence_threshold: 0.7
confidence_speculative_threshold: 0.9
num_speculate_tokens: 3
```

`alg: confidence_threshold` 对齐 pangu 的 confidence 解码方式：每轮至少解码一个最高置信度 mask token，并额外解码所有置信度超过 `threshold` 的 mask token。投机解码通常需要在该模式下开启才有明显收益。

`speculative_decoding_mode: greedy` 保持原有投机选点行为：每轮从超过
`self_speculative_confidence_threshold` 的 mask token 中取最多
`num_speculate_tokens` 个最高置信度位置进行投机填充。

`speculative_decoding_mode: confidence` 会解码所有置信度超过
`confidence_speculative_threshold` 的 mask token，默认阈值为 `0.9`，不受
`num_speculate_tokens` 限制。

`num_speculate_tokens` 控制 greedy 模式下每轮从模型自身分布中预填的 token 数量。调高通常更激进，但候选验证开销也会增加。

服务启动后可检查健康状态：

```bash
curl http://127.0.0.1:30000/health
curl http://127.0.0.1:30000/get_model_info
```

## 离线 Smoke 测试

不启动 HTTP 服务，直接用 `sgl.Engine` 跑一个 GSM8K 样例：

```bash
cd /home/liurenxi/sglang-for-dllm

CUDA_VISIBLE_DEVICES=1 \
CUDA_HOME=/home/liurenxi/anaconda3/envs/lrx-dflash \
PATH=/home/liurenxi/anaconda3/envs/lrx-dflash/nvvm/bin:/home/liurenxi/anaconda3/envs/lrx-dflash/bin:$PATH \
LD_LIBRARY_PATH=/home/liurenxi/anaconda3/envs/lrx-dflash/lib:/home/liurenxi/anaconda3/envs/lrx-dflash/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-} \
CPATH=/home/liurenxi/anaconda3/envs/lrx-dflash/targets/x86_64-linux/include:/home/liurenxi/anaconda3/envs/lrx-dflash/lib/python3.11/site-packages/nvidia/cuda_runtime/include:${CPATH:-} \
CPLUS_INCLUDE_PATH=/home/liurenxi/anaconda3/envs/lrx-dflash/targets/x86_64-linux/include:/home/liurenxi/anaconda3/envs/lrx-dflash/lib/python3.11/site-packages/nvidia/cuda_runtime/include:${CPLUS_INCLUDE_PATH:-} \
SGLANG_ENABLE_JIT_DEEPGEMM=0 \
TRANSFORMERS_VERBOSITY=error \
/home/liurenxi/anaconda3/envs/lrx-dflash/bin/python run_dream_gsm8k_smoke.py
```

启用投机解码的 smoke 测试：

```bash
cd /home/liurenxi/sglang-for-dllm

CUDA_VISIBLE_DEVICES=1 \
DLLM_ALGORITHM_CONFIG=examples/dllm_low_confidence_spec_spiffy.yaml \
CUDA_HOME=/home/liurenxi/anaconda3/envs/lrx-dflash \
PATH=/home/liurenxi/anaconda3/envs/lrx-dflash/nvvm/bin:/home/liurenxi/anaconda3/envs/lrx-dflash/bin:$PATH \
LD_LIBRARY_PATH=/home/liurenxi/anaconda3/envs/lrx-dflash/lib:/home/liurenxi/anaconda3/envs/lrx-dflash/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-} \
CPATH=/home/liurenxi/anaconda3/envs/lrx-dflash/targets/x86_64-linux/include:/home/liurenxi/anaconda3/envs/lrx-dflash/lib/python3.11/site-packages/nvidia/cuda_runtime/include:${CPATH:-} \
CPLUS_INCLUDE_PATH=/home/liurenxi/anaconda3/envs/lrx-dflash/targets/x86_64-linux/include:/home/liurenxi/anaconda3/envs/lrx-dflash/lib/python3.11/site-packages/nvidia/cuda_runtime/include:${CPLUS_INCLUDE_PATH:-} \
SGLANG_ENABLE_JIT_DEEPGEMM=0 \
TRANSFORMERS_VERBOSITY=error \
/home/liurenxi/anaconda3/envs/lrx-dflash/bin/python run_dream_gsm8k_smoke.py
```

验证通过时，输出文本应包含类似内容：

```text
Natalia sold 48 clips in April.
In May, she sold half as many clips, so she sold 48/2 = 24 clips.
...
The answer is: 72
```

## HTTP 服务测试

服务启动后，可以用 OpenAI 兼容接口测试：

```bash
curl http://127.0.0.1:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Dream-org/Dream-v0-Instruct-7B",
    "messages": [
      {
        "role": "user",
        "content": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? Answer with the final number."
      }
    ],
    "temperature": 0,
    "max_tokens": 128
  }'
```

也可以用原生 `/generate` 接口，但需要自己用 Dream tokenizer 的 chat template 生成 prompt。`run_dream_gsm8k_smoke.py` 展示了模板化方式。

## 重要实现参数

Dream-7B 在本仓库中的关键适配点：

- `DreamModel` 映射到 Qwen2 权重结构：`python/sglang/srt/models/dream.py`
- Dream 使用非 causal/full attention，以匹配官方 `attention_mask="full"` 行为。
- Dream dLLM 使用 full-sequence diffusion 生成，而不是只对当前 32-token cache block 解码。
- Dream 默认 `block_size=128`，用于一次覆盖 `max_new_tokens=128` 的 GSM8K smoke。
- full-sequence 分支使用 entropy remasking 风格的 token transfer。

## 常见问题

如果看到 `CUDA_HOME`、`cuda_runtime.h`、`cicc` 或 FlashInfer JIT 相关错误，优先确认环境变量是否完整设置。

如果 `import sglang` 加载到了 site-packages 而不是 `/home/liurenxi/sglang-for-dllm/python/sglang`，检查 `.pth` 文件是否存在。

如果显存不足，先用 `nvidia-smi` 找空闲卡，并修改 `CUDA_VISIBLE_DEVICES`。Dream-7B bf16 推理建议至少预留约 20GB 以上显存，首跑 JIT 还会有额外开销。

如果 HTTP 服务启动但生成为空或只输出 EOS，确认当前代码包含 Dream 适配补丁，并且启动参数包含：

```text
--trust-remote-code
--dllm-algorithm LowConfidence
--disable-cuda-graph
--attention-backend flashinfer
```
