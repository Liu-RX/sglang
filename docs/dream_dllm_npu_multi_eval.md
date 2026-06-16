# Dream dLLM NPU 多任务评测脚本

这套脚本用于在已部署好的昇腾 NPU SGLang 环境中，评测 Dream dLLM 的三种模式：

- `baseline`：LowConfidence dLLM，关闭自投机解码。
- `vbs1`：开启自投机解码，`verify_batch_size: 1`。
- `vbs2`：开启自投机解码，`verify_batch_size: 2`。

默认评测任务：

- `gsm8k`：数值准确率。
- `mbpp`：代码执行 `pass@1`。
- `humaneval`：代码执行 `pass@1`。

每个任务/模式都会输出：

- `score` / `accuracy` / `pass_at_1`
- `generation_latency_sec`
- `judge_latency_sec`
- `total_latency_sec`
- `output_throughput_tok_s`
- `request_throughput_req_s`
- `mean_request_latency_sec`

## 快速运行

```bash
cd /path/to/sglang-for-dllm
export PYTHONPATH="$PWD/python:${PYTHONPATH:-}"
export ASCEND_RT_VISIBLE_DEVICES=0
export MODEL_PATH=/path/to/Dream-v0-Instruct-7B

bash scripts/ascend_dllm/run_dream_multi_eval_npu.sh
```

默认会跑完整数据集。如果想先 smoke：

```bash
GSM8K_NUM_EXAMPLES=20 \
MBPP_NUM_EXAMPLES=20 \
HUMANEVAL_NUM_EXAMPLES=20 \
bash scripts/ascend_dllm/run_dream_multi_eval_npu.sh
```

结果目录默认类似：

```text
benchmark/dllm_eval/results_npu_20260615_120000/
```

目录中会包含每个任务/模式的：

- `*_metrics.json`
- `*_outputs.jsonl`
- `summary.json`
- `summary.md`

## 常用参数

```bash
export TASKS="gsm8k mbpp humaneval"
export MODES="baseline vbs1 vbs2"
export PARALLEL=1
export SUBMIT_BATCH_SIZE=1
export MAX_RUNNING_REQUESTS=1
export MEM_FRACTION_STATIC=0.75
export CONTEXT_LENGTH=2048
export TP_SIZE=1
export CODE_TIMEOUT=10
```

不同任务的输出长度：

```bash
export GSM8K_MAX_NEW_TOKENS=256
export MBPP_MAX_NEW_TOKENS=512
export HUMANEVAL_MAX_NEW_TOKENS=512
```

如果数据集不能联网下载，可以指定本地路径：

```bash
export GSM8K_DATA_PATH=/path/to/gsm8k_test.jsonl
export MBPP_DATA_PATH=/path/to/mbpp_test.jsonl
export HUMANEVAL_DATA_PATH=/path/to/HumanEval.jsonl.gz
```

本地数据格式：

- GSM8K JSONL/JSON：每条包含 `question` 和 `answer`。
- MBPP JSONL/JSON：每条包含 `prompt` 或 `text`，以及 `test_list`。
- HumanEval JSONL/JSON/JSONL.GZ：每条包含 `task_id`、`prompt`、`test`、`entry_point`。

## NPU 稳定性和定位参数

默认脚本按最保守方式提交请求：

```bash
export SUBMIT_BATCH_SIZE=1
export PARALLEL=1
export MAX_RUNNING_REQUESTS=1
export LOG_EACH_REQUEST=1
export STRICT_MEM_CHECK_IDLE=1
```

- `SUBMIT_BATCH_SIZE=1`：每次只向 offline Engine 提交一个请求，便于定位哪条样本触发调度器内存池检查。
- `LOG_EACH_REQUEST=1`：打印 `starting i/N` 和 `completed i/N`。
- `STRICT_MEM_CHECK_IDLE=1`：保持 SGLang 空闲期内存池严格检查。若报 `pool memory leak detected`，日志中的 `failed i/N` 可用于定位样本。

如果小样本严格检查可以通过，但完整数据集仍在空闲期报内存池检查错误，可先确认是否为检查器/账本误报：

```bash
TASKS=gsm8k MODES=baseline \
STRICT_MEM_CHECK_IDLE=0 \
SUBMIT_BATCH_SIZE=1 \
PARALLEL=1 \
MAX_RUNNING_REQUESTS=1 \
bash scripts/ascend_dllm/run_dream_multi_eval_npu.sh
```

`STRICT_MEM_CHECK_IDLE=0` 会把该检查从抛异常改成 warning，只建议用于确认和临时完整跑分；如果随后出现可用 KV 持续下降或 OOM，说明仍是真实释放问题。

需要强制每批之间清空 cache 时可加：

```bash
export FLUSH_CACHE_BETWEEN_BATCHES=1
```

## 单独跑一个任务

```bash
python3 benchmark/dllm_eval/bench_dream_dllm_tasks.py \
  --task humaneval \
  --mode-name vbs1 \
  --model-path "$MODEL_PATH" \
  --dllm-algorithm-config benchmark/dllm_eval/configs/dream_vbs1_bs32_confidence.yaml \
  --output-dir benchmark/dllm_eval/results_debug \
  --num-examples 20 \
  --max-new-tokens 512 \
  --device npu \
  --attention-backend ascend \
  --disable-cuda-graph \
  --disable-radix-cache \
  --submit-batch-size 1 \
  --log-each-request
```

## 配置文件

默认配置位于：

```text
benchmark/dllm_eval/configs/dream_baseline_bs32_confidence.yaml
benchmark/dllm_eval/configs/dream_vbs1_bs32_confidence.yaml
benchmark/dllm_eval/configs/dream_vbs2_bs32_confidence.yaml
```

可以通过环境变量替换：

```bash
export BASELINE_CONFIG=/path/to/baseline.yaml
export VBS1_CONFIG=/path/to/vbs1.yaml
export VBS2_CONFIG=/path/to/vbs2.yaml
```

## 注意

MBPP 和 HumanEval 会执行模型生成的 Python 代码来计算 `pass@1`。脚本会把每个样本放在临时目录中用子进程执行，并设置超时，但它仍然属于执行生成代码的评测流程，请只在隔离评测环境中运行。
