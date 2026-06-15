# Dream dLLM Speculative Decoding Report

## 落地工作

- 在 SGLang 上完成 Dream-7B dLLM 推理适配：Dream 模型映射、full attention、full-sequence diffusion 解码，以及 `LowConfidence` 配置化入口。
- 在 `LowConfidence` 中加入自投机解码：支持 `verify_batch_size=1/2/spiffy`，支持 greedy/confidence 两种投机选点，并暴露阈值、投机 token 数等 YAML 配置。
- 实现 vbs=1 单候选校验；实现 vbs=2 双候选校验，并把两个候选合并为一个 expanded `ForwardBatch` 做 batched verify，减少串行验证开销。
- 补齐运行链路：示例配置、启动脚本、smoke 脚本、GSM8K benchmark/aggregate 脚本、轻量 profile 统计。

## 测试设置

模型为 `Dream-org/Dream-v0-Instruct-7B`，数据为 GSM8K，`block_size=32`，`max_new_tokens=512`，baseline 使用 `alg=confidence_threshold`，投机配置使用 `threshold=0.9`、`confidence_speculative_threshold=0.8`。

## 结果摘要

| 设置 | 样本 | Acc | Latency | Tok/s | 相对 baseline |
|---|---:|---:|---:|---:|---:|
| baseline, parallel=1 | 1319 | 37.83% | 362.41s | 171.7 | - |
| vbs=1, parallel=1 | 1319 | 36.92% | 316.45s | 193.7 | +12.8% tok/s |
| baseline, parallel=2 | 1319 | 37.83% | 358.47s | 173.7 | - |
| vbs=1, parallel=2 | 1319 | 36.92% | 315.25s | 194.5 | +12.0% tok/s |
| baseline, parallel=4 | 1319 | 37.30% | 225.79s | 272.6 | - |
| vbs=1, parallel=4 | 1319 | 36.69% | 202.67s | 299.7 | +9.9% tok/s |
| baseline, parallel=1 | 100 | 43.00% | 29.26s | 149.7 | - |
| vbs=2, parallel=1 | 100 | 44.00% | 27.60s | 158.8 | +6.1% tok/s |

vbs=1 已完成全量 GSM8K 对比，在 parallel=1/2/4 下吞吐分别提升约 12.8%/12.0%/9.9%，精度相对 baseline 小幅下降约 0.6-0.9 pp。vbs=2 当前完成 100 题切片对比，batched verify 后吞吐提升约 6%-10%，精度未下降；profile 显示普通 forward 从 2594 次降到 1704 次，另有 541 次 batched candidate forward。

原始结果位于 `benchmark/gsm8k/results_conf090_spec080_bs32_mnt512_full_concurrency_20260611_094254/`、`benchmark/gsm8k/results_baseline_vs_vbs2_after_softmax_opt_20260612_155311/` 和 `benchmark/gsm8k/results_forward_count_baseline_vs_vbs2_20260612_071459/`。
