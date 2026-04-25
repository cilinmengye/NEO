# NEO vs vLLM：Avg token latency / TTFT / TPOT 异常结果分析

## 1. 结论摘要

针对 `evaluation/reproduce-fig6b.py` 生成的三张图，当前结果可以用一句话概括：

> NEO 通过把一部分 decode attention 与 KV cache 放到 CPU 侧，扩大了可接纳的并发请求集合，显著降低高负载下的新请求排队时间，因此 TTFT 和端到端平均 token latency 更低；但 CPU decode、GPU/CPU 协同、KV swap 与 pipeline 失衡会拉长 decode 阶段的 inter-token interval，因此 TPOT 可能高于 vLLM。vLLM 在本实验中启用了 chunked prefill，decode 侧调度较强，所以高负载下 TPOT 可以优于 NEO，但它仍受 GPU KV cache 容量与 preemption/recompute/排队影响，TTFT 会快速恶化。

对用户原解释的判断：

| 解释点 | 判断 | 修正/补充 |
|---|---|---|
| NEO 因 CPU/KV offload 减轻 GPU KV cache 压力，从而接收更多请求、降低 TTFT | 基本成立 | 更准确地说，这是 **admission capacity 与 queueing delay** 的改善：NEO 不只是“让 GPU batch 更大”，还把一部分后续 decode 常驻集转移到 CPU decode 队列。 |
| vLLM 单 GPU KV cache 压力导致请求排队，TTFT 变高 | 成立 | 但不能简单说 vLLM 是 prefill-first。本实验显式启用了 `--enable-chunked-prefill`，此时 vLLM scheduler 是 decode-prioritized；TTFT 高主要来自 GPU KV 容量不足、prefill admission 受限、preemption/recompute 与等待队列堆积。 |
| NEO TPOT 高来自 CPU decode、GPU/CPU 同步、PCIe/KV 迁移 | 基本成立 | 还要补充：NEO 调度目标更偏 throughput / average latency，不一定最小化每个请求的 inter-token latency；CPU path 一旦无法完全被 GPU work hide，就会直接体现为 TPOT 上升。 |
| vLLM TPOT 随 request rate 增大后低于 NEO | 成立且合理 | 因为 vLLM 的 decode 仍在 GPU 上执行，并且 chunked prefill 下优先调度 running decode；它牺牲的是新请求 TTFT，而不是已进入 decode 的请求 TPOT。 |
| NEO Avg token latency 始终更低但 TPOT 更高是否矛盾 | 不矛盾 | `fig6b.pdf` 的指标是 `(end-start)/output_len`，包含 TTFT/排队时间；高负载下 vLLM TTFT 可达几十到一百多秒，足以主导平均 token latency。 |

另外有一个重要测量 caveat：当前 `api_client.py` 对 TPOT 的口径依赖服务端 stream payload。实际结果文件中，`vllm-ac-lat-1_5.json` 使用 `stream_observation: "chunk"` / `chunk_offsets`，`ours-ac-lat-1_5.json` 使用 `stream_observation: "token"` / `token_offsets`。虽然样例里 event 数与 output_len 一致，但严格比较 TPOT 前最好统一为同一种 token-level 输出口径。

## 2. 实验脚本与指标口径

### 2.1 workload 与 request rate

`reproduce-fig6b.py` 使用 Azure trace 的 code 与 conv 两份 CSV，分别取前 `NUM_THRESHOLD = 2000` 条，因此总请求数约为 4000：

- `ContextTokens` 转成 prompt token 数；
- `GeneratedTokens` 转成输出 token 数；
- `vllm_rates = [1.5, 2.5, 3.1, 3.5]`；
- `ours_rates = [1.5, 2.5, 3.1, 3.5]`；
- 每个 rate 下调用 `run_test(..., collect_stream_metrics=True)`。

关键代码：

- workload 构造：`/home/yxlin/github/swift/NEO/evaluation/reproduce-fig6b.py:39-63`
- rate 设置：`/home/yxlin/github/swift/NEO/evaluation/reproduce-fig6b.py:71-72`
- 测试调用：`/home/yxlin/github/swift/NEO/evaluation/reproduce-fig6b.py:82-100`

`benchmark.py` 中请求按指数分布间隔发出：`np.random.exponential(1 / rate, len(prompts))`，因此这是 arrival-rate latency test，而不是固定并发吞吐测试：`/home/yxlin/github/swift/NEO/evaluation/benchmark.py:101-131`。

### 2.2 server 启动参数

vLLM 启动参数来自 `server.py`：

- `--enable-chunked-prefill`
- `--preemption-mode recompute`
- `--num-gpu-blocks-override 1650`
- `--block-size 16`
- `--max-num-batched-tokens 26400`
- `--max-num-seqs 1024`
- `--swap-space 60`

对应代码：`/home/yxlin/github/swift/NEO/evaluation/server.py:26-49`。

NEO 的 `ours` 启动参数：

- 没有 `--always-use-gpu`；
- 启用 `--extra-layer-for-cprf`；
- 使用完整 `swap_space = config["swap_space"]`。

对应代码：`/home/yxlin/github/swift/NEO/evaluation/server.py:55-87`。

这意味着本实验不是“普通 vLLM vs 普通 SwiftLLM”，而是：

- vLLM：GPU-only decode + chunked prefill + recompute preemption；
- NEO：GPU/CPU partial offloading + load-aware scheduling + CPU swap space。

### 2.3 三个图的指标定义

`illustrator.py` 对三类指标的定义如下。

1. `Avg token latency`：

```python
(end - start) / output_len
```

见 `/home/yxlin/github/swift/NEO/evaluation/illustrator.py:53-58`。

它不是纯 decode TPOT，也不是除以 input+output token；它包含：

- 客户端发出请求到服务端接纳的等待；
- prefill 排队与执行；
- TTFT；
- 后续 decode；
- streaming/HTTP 开销。

因此它容易被 TTFT/queueing delay 主导。

2. `TTFT`：

使用 stream 中第一个有输出事件相对 `start` 的 offset：`first_token_offset`，见 `/home/yxlin/github/swift/NEO/evaluation/illustrator.py:59-71` 与 `/home/yxlin/github/swift/NEO/evaluation/api_client.py:126-130`。

3. `TPOT`：

优先使用 `token_offsets` 的相邻差值；如果没有 token offsets，则使用 `chunk_offsets`；最后才 fallback 到 `(end-start-first_token_offset)/(events_received-1)`，见 `/home/yxlin/github/swift/NEO/evaluation/illustrator.py:23-50`。

这带来一个口径风险：若某个系统 stream payload 暴露 token id，另一个只暴露文本 chunk，则 TPOT 的语义可能分别是 token interval 与 chunk interval。

### 2.4 当前结果文件中的数值

按 `illustrator.py` 的同一逻辑重新汇总现有 JSON 文件，得到：

| System | Rate | Avg token latency (s) | TTFT (s) | TPOT (s) | stream observation |
|---|---:|---:|---:|---:|---|
| vLLM | 1.5 | 0.0454 | 0.7917 | 0.0327 | chunk |
| vLLM | 2.5 | 0.4450 | 47.8128 | 0.0424 | chunk |
| vLLM | 3.1 | 0.8518 | 96.1686 | 0.0475 | chunk |
| vLLM | 3.5 | 1.0914 | 123.9229 | 0.0524 | chunk |
| NEO | 1.5 | 0.0411 | 0.2151 | 0.0329 | token |
| NEO | 2.5 | 0.2145 | 0.2866 | 0.2068 | token |
| NEO | 3.1 | 0.6881 | 14.5230 | 0.5719 | token |
| NEO | 3.5 | 0.8896 | 30.4145 | 0.6487 | token |

这张表说明：

- vLLM 的 TPOT 确实显著低于高负载下 NEO 的 TPOT；
- 但 vLLM 的 TTFT 在 2.5 req/s 后迅速上升到几十秒甚至一百秒以上；
- NEO 的 TTFT 也会在高负载下上升，但远小于 vLLM；
- 因为 `Avg token latency` 包含 TTFT，所以 NEO 即使 TPOT 较高，仍能保持更低的平均 token latency。

## 3. NEO 论文证据

NEO 论文的核心动机是 GPU memory crisis：现代 inference engine 如 vLLM 把 KV cache 放在 GPU memory 中，KV cache 随 prompt/output length 线性增长，导致可达到的 batch size 受 GPU 显存限制。论文在摘要中明确说，NEO 将一部分 attention compute 和 KV cache states 从 GPU offload 到本地 CPU，从而有效增大 GPU batch size，提高吞吐；并提出 asymmetric GPU-CPU pipelining 与 load-aware scheduling。

关键论文摘录来自 `/tmp/neo-paper.txt`：

- NEO offload 部分 attention compute 和 KV cache 到 CPU，以增大 GPU batch size：`/tmp/neo-paper.txt:13-16`
- vLLM 等系统将 KV cache 存在 GPU，KV cache 随 prompt/output length 线性增长：`/tmp/neo-paper.txt:41-45`
- throughput 受 GPU memory bounded batch size 影响：`/tmp/neo-paper.txt:200-204`
- CPU memory/offloading 可增加 batch size，但传统 layer-by-layer KV swap 会让 PCIe bandwidth 成为瓶颈：`/tmp/neo-paper.txt:63-78`
- NEO 的 asymmetric pipelining：一个 sub-batch 将部分 decoding attention 与 KV cache 放在 CPU，另一个 sub-batch 在 GPU 上运行，并尝试 overlap：`/tmp/neo-paper.txt:114-129`

论文层面的含义：

1. 用户关于“NEO 通过 CPU offload 缓解 GPU KV cache 压力”的解释与论文目标一致。
2. 用户关于“CPU decode / KV 迁移 / PCIe 可能导致 TPOT 变差”的解释也与论文对 offloading 风险的讨论一致。
3. 但论文强调的是通过 partial offloading 和 load-aware scheduling **避免 CPU 过载**，不是无条件把所有 decode 放到 CPU。因此分析 NEO TPOT 时必须看实际 scheduler 是否成功 hide CPU time。

## 4. NEO 代码证据

### 4.1 三类队列：waiting / GPU decode / CPU decode

NEO scheduler 维护：

- `waiting_q`：尚未进入 forward 的新请求；
- `gpu_decoding_q`：KV cache 驻留 GPU、在 GPU decode 的请求；
- `cpu_decoding_q`：被换出到 CPU 侧 decode 的请求。

代码：`/home/yxlin/github/swift/NEO/swiftllm/server/scheduler.py:143-149`。

新请求先进入 `waiting_q`：`/home/yxlin/github/swift/NEO/swiftllm/server/scheduler.py:173-179`。

这直接支持“NEO 有 CPU decode/offload 常驻路径”，不是单纯 GPU-only continuous batching。

### 4.2 GPU block 压力下 swap out 到 CPU

在 `_get_next_batch_new()` 中，scheduler 先统计当前 GPU decoding 请求所需 KV blocks，并扣减 batch/token budget：`/home/yxlin/github/swift/NEO/swiftllm/server/scheduler.py:360-367`。

如果 budget 透支或 GPU block 超过阈值，就从 `gpu_decoding_q` 队尾取 victim，放到 `cpu_decoding_q`，并记录 `swpout_reqs`：

- `/home/yxlin/github/swift/NEO/swiftllm/server/scheduler.py:368-379`

这说明高负载或长上下文下，NEO 会把部分 decode 请求转移到 CPU 侧，减少 GPU KV 常驻压力。

### 4.3 GPU 有余量时 swap in

如果本轮没有 swap out，scheduler 尝试从 `cpu_decoding_q` 取请求 swap back 到 GPU，只要不超过保守的 `swap_in_threshold` 与 batch budget：

- `/home/yxlin/github/swift/NEO/swiftllm/server/scheduler.py:380-394`

这说明 NEO 并不是“CPU decode 一去不回”，而是在 GPU capacity 允许时拉回 GPU。

### 4.4 新 prefill admission 可进入 GPU 或 CPU 路径

对 `waiting_q` 中的新请求，scheduler 检查：

- intermediate blocks 不超限；
- CPU blocks 不超限；
- request id 不耗尽；
- batch/token budget 足够。

见 `/home/yxlin/github/swift/NEO/swiftllm/server/scheduler.py:396-415`。

如果 GPU KV 容量还能容纳，则 `pref_to_gpu`；否则进入 `pref_to_cpu`：`/home/yxlin/github/swift/NEO/swiftllm/server/scheduler.py:416-425`。

需要注意：代码注释明确指出 `pref_to_cpu` 不是说 prefill compute 全在 CPU，而是 prefill 计算仍经过 GPU，生成出的 KV 会放到 CPU/中间区：`/home/yxlin/github/swift/NEO/swiftllm/server/scheduler.py:346-348`。

这点修正了一个容易过度简化的说法：NEO 的 CPU 主要承接 decode attention/KV cache，而不是把 prefill 的 dense compute 迁到 CPU。

### 4.5 load-aware scheduling：只在 CPU decode 可被 hide 时加入更多 CPU decode

`_get_remains()` 计算 CPU decode 是否能被另一侧 GPU work 遮蔽：

- 另一侧 linear time；
- 本侧 prefill time；
- 本侧 GPU decode time；
- 减去本侧 CPU time。

见 `/home/yxlin/github/swift/NEO/swiftllm/server/scheduler.py:181-205`。

在形成 pipeline candidate 时，scheduler 试探加入 CPU decode request；如果加入后 `min(remains) < 0`，说明 CPU decode 超出可遮蔽窗口，就跳过该请求：`/home/yxlin/github/swift/NEO/swiftllm/server/scheduler.py:263-288`。

最后比较 sequential 与 two-sub-batch pipelined 的预测吞吐，选择更高者：`/home/yxlin/github/swift/NEO/swiftllm/server/scheduler.py:312-323`。

这说明 NEO 的目标是 throughput-oriented load balancing，而不是严格最小化 TPOT。只要 pipeline candidate 的总体吞吐更高，它可能接受更长的 CPU-side token interval。

### 4.6 worker 侧确实存在 GPU/CPU swap 和 stream 同步

`block_swapper.py` 分配：

- GPU KV cache：`k_cache` / `v_cache` 在 CUDA 上；
- CPU KV swap space：`k_swap` / `v_swap` 在 CPU pinned memory；
- CPU Q/K/V/O buffers 也在 pinned memory。

见 `/home/yxlin/github/swift/NEO/swiftllm/worker/block_swapper.py:32-64`。

`model.py` 中，sequential forward 在进入 transformer layer 前等待 `cpu_communication_stream`：

- `/home/yxlin/github/swift/NEO/swiftllm/worker/model.py:338-352`

`do_one_iteration()` 会在 `cpu_communication_stream` 上逐层调用 `self.swapper.swap_blocks(...)`，然后再进入 `_forward_batches()`：

- `/home/yxlin/github/swift/NEO/swiftllm/worker/model.py:420-447`

这支持用户关于 TPOT 开销来源的判断：GPU/CPU KV swap、通信 stream 等只要不能完全 overlap，就会进入关键路径。

## 5. vLLM 调度与 KV cache 行为

### 5.1 本实验的 vLLM 不是普通 prefill-first 模式

`server.py` 明确传入 `--enable-chunked-prefill`：`/home/yxlin/github/swift/NEO/evaluation/server.py:46`。

本地 vLLM scheduler 源码中，默认调度 `_schedule_default()` 的注释和代码表示：默认策略先调度尽可能多的 prefill；如果已经调度了 prefill，就不调度 decode：`/home/yxlin/github/swift/NEO/.venv_vllm/lib/python3.12/site-packages/vllm/core/scheduler.py:1240-1247`。

但 chunked prefill 策略 `_schedule_chunked_prefill()` 明确写道：

- 先 schedule as many decoding requests as possible；
- 再 schedule chunked prefill；
- 再 swapped；
- 再 new prefill。

见 `/home/yxlin/github/swift/NEO/.venv_vllm/lib/python3.12/site-packages/vllm/core/scheduler.py:1310-1323`。

代码也直接说：

- “Decoding should be always scheduled first by fcfs”：`/home/yxlin/github/swift/NEO/.venv_vllm/lib/python3.12/site-packages/vllm/core/scheduler.py:1340-1346`
- “By default, vLLM scheduler prioritizes prefills. Once chunked prefill is enabled, the policy is changed to prioritize decode requests”：`/home/yxlin/github/swift/NEO/.venv_vllm/lib/python3.12/site-packages/vllm/core/scheduler.py:1365-1371`

因此，若解释为“vLLM prefill 优先导致 TPOT 差”，在本实验中是不准确的。相反，本实验 vLLM 的 TPOT 好，正符合 chunked prefill 的 decode-prioritized 设计。

### 5.2 vLLM 的 preemption mode 是 recompute

vLLM 源码定义两种 preemption：

- `SWAP`：把 blocks swap 到 CPU memory，恢复时再 swap back；
- `RECOMPUTE`：丢弃 blocks，恢复时重新计算，把序列当作新 prompt。

见 `/home/yxlin/github/swift/NEO/.venv_vllm/lib/python3.12/site-packages/vllm/core/scheduler.py:33-44`。

本实验启动参数为 `--preemption-mode recompute`：`/home/yxlin/github/swift/NEO/evaluation/server.py:47`。

这意味着当 GPU KV cache 不足时，vLLM 更倾向通过 recompute 处理被抢占请求，而不是像 NEO 那样把部分 decode attention 和 KV 常驻转到 CPU 侧继续服务。其结果是：

- 对已经处于 running decode 且成功保留在 GPU 的请求，TPOT 可以很低；
- 对还没进入或被 preempt/recompute 的请求，TTFT 和端到端 latency 会显著变差。

### 5.3 为什么 vLLM 高负载下 TTFT 会很高

vLLM 的核心限制仍是 GPU KV cache。高 request rate + 长 prompt/output 造成：

1. running decode 消耗 GPU KV blocks；
2. chunked prefill 优先保护 running decode；
3. 新 prefill 只能使用剩余 token/KV budget；
4. KV 空间不足时发生 preemption/recompute 或等待；
5. 新请求在 waiting 队列中积压，TTFT 随排队时间迅速上升。

所以 vLLM 的 TTFT 高不是因为“decode 不够快”，而是因为 admission 被 GPU KV capacity 限制，且 decode-prioritized 策略会把资源优先给已进入 decode 的请求。

## 6. 对用户解释的逐条判断

### 6.1 “NEO TTFT 更低，因为 CPU offload 缓解 GPU KV cache 压力”

**判断：成立。**

证据链：

- 论文目标就是 offload 部分 attention/KV 到 CPU，以扩大 effective batch size：`/tmp/neo-paper.txt:13-16`。
- scheduler 有 `gpu_decoding_q` 与 `cpu_decoding_q`：`scheduler.py:143-149`。
- GPU blocks 超限时把 decode request swap out 到 CPU：`scheduler.py:368-379`。
- 新 prefill 可以根据 GPU/CPU block capacity 进入 GPU 或 CPU 路径：`scheduler.py:396-425`。

更准确的表述：NEO 降低 TTFT 的核心不是单纯“GPU batch size 更大”，而是 **新请求更容易被 admission**。当 vLLM 的 waiting queue 因 GPU KV blocks 不足而堆积时，NEO 可以把部分 decode 常驻压力转移到 CPU，释放 GPU 侧 admission 空间，让新请求更早完成 prefill 并产生首 token。

### 6.2 “vLLM TTFT 高，因为单 GPU KV cache 压力导致排队”

**判断：成立，但需要修正 vLLM 调度表述。**

成立部分：

- vLLM 仍主要依赖 GPU KV cache；
- 本实验设置 `num_gpu_blocks_override=1650`、长上下文 Azure trace 与高 arrival rate，确实容易触发 KV capacity/admission bottleneck；
- 现有结果中 vLLM TTFT 从 1.5 req/s 的约 0.79s 上升到 3.5 req/s 的约 123.9s，说明 waiting/queueing 是主导因素。

需要修正：

- 本实验 vLLM 启用了 `--enable-chunked-prefill`；
- vLLM 源码确认 chunked prefill 下 decode 优先，而不是普通 prefill-first；
- 因此不能把 vLLM 的高 TTFT解释为“prefill 抢占 decode”，而应解释为“decode-prioritized + GPU KV 容量限制使新 prefill admission 变慢”。

### 6.3 “NEO TPOT 更高，因为 CPU decode 和 GPU/CPU 同步/传输”

**判断：基本成立。**

证据链：

- NEO 论文明确指出传统 KV swap 容易受 PCIe bandwidth 限制，NEO 通过 offload decode attention 避免反复完整 swap，但仍依赖 CPU/GPU 协同：`/tmp/neo-paper.txt:63-78`。
- NEO worker 分配 CPU pinned KV swap space 与 CPU Q/K/V/O buffers：`block_swapper.py:45-64`。
- `do_one_iteration()` 会在 communication stream 上逐层 swap KV blocks：`model.py:420-447`。
- sequential forward 会等待 communication stream：`model.py:338-352`。
- scheduler 只在预测上尝试 hide CPU decode；如果实际 CPU decode 或 copy 超出 hide window，就会反映为更长 TPOT：`scheduler.py:181-205`、`scheduler.py:263-288`。

补充：NEO 的 TPOT 变高不一定表示系统整体更差。它可能是在用较慢的 CPU decode path 换取更大的 admission capacity 和更低的 queueing delay。

### 6.4 “vLLM TPOT 随 rate 增大反而低于 NEO”

**判断：合理，且与源码行为一致。**

原因：

1. vLLM 的 running decode 仍在 GPU 上，decode step 的 raw execution latency 更低。
2. chunked prefill 下 vLLM 优先调度 decode，降低 inter-token latency。
3. 当 GPU KV 容量紧张时，vLLM 更倾向让新请求等或 recompute，而不是把 decode 放到 CPU 慢路径。
4. 因此 vLLM 可以呈现“TPOT 很好，但 TTFT 很差”的形态。

这是一种典型 trade-off：

- vLLM 保护已进入 decode 的请求，TPOT 低；
- NEO 保护 admission/throughput，TTFT 与平均 latency 低，但 CPU path 拉高 TPOT。

### 6.5 “NEO Avg token latency 始终更低”

**判断：与 TTFT 优势一致，不与 TPOT 劣势矛盾。**

`Avg token latency = (end-start)/output_len`，包含首 token 前等待。高负载下 vLLM 的 TTFT 是几十到一百多秒，远大于 TPOT 差异，因此即使 vLLM 每个后续 token 更快，整体平均 token latency 仍更差。

以 3.5 req/s 为例：

- vLLM TTFT ≈ 123.9s，TPOT ≈ 0.052s；
- NEO TTFT ≈ 30.4s，TPOT ≈ 0.649s；
- vLLM 的 TTFT 排队惩罚足以压倒其 TPOT 优势。

## 7. 更完整的解释模型

可以把端到端 latency 粗略分解为：

$$
L \approx W_{admission} + T_{prefill} + T_{first\ decode} + (N_{out}-1) \cdot T_{inter\ token}
$$

其中：

- `TTFT ≈ W_admission + T_prefill + T_first_decode`
- `TPOT ≈ T_inter_token`
- `Avg token latency ≈ L / N_out`

在本实验中：

### vLLM

- `T_inter_token` 小：GPU decode + chunked prefill decode-prioritized；
- `W_admission` 大：GPU KV cache 容量限制、waiting queue、recompute preemption；
- 因此：TPOT 低，但 TTFT 与 Avg token latency 高。

### NEO

- `W_admission` 小：CPU KV/offload 扩大有效并发与 admission capacity；
- `T_inter_token` 大：一部分请求走 CPU decode / CPU KV / swap / synchronization；
- 因此：TTFT 与 Avg token latency 低，但 TPOT 高。

这也解释了为什么三张图看起来“矛盾”，但其实对应不同阶段的 trade-off。

## 8. 需要特别注意的测量 caveat

`api_client.py` 根据 stream response 的字段决定观测类型：

- 有 `token_ids` 时，记录 `token_offsets`；
- 只有 `text` / `delta.content` 时，记录 `chunk_offsets`。

见 `/home/yxlin/github/swift/NEO/evaluation/api_client.py:24-55` 与 `/home/yxlin/github/swift/NEO/evaluation/api_client.py:131-155`。

当前样例：

- `vllm-ac-lat-1_5.json`：`stream_observation: "chunk"`，使用 `chunk_offsets`；
- `ours-ac-lat-1_5.json`：`stream_observation: "token"`，使用 `token_offsets`。

这说明 TPOT 比较可能混合了 chunk-level 与 token-level 事件。样例中 event 数等于 output_len，因此大概率每个 chunk 对应一个 token，但为了论文级严谨，建议进一步统一：

1. 让两个 server 都返回 `token_ids`；或
2. 都只用文本 chunk，并验证一个 chunk 是否严格等于一个 token；或
3. 不依赖 streaming event，改为 server 内部记录每轮 decode token 的 timestamp。

在未统一前，TPOT 结论应表述为“按当前 benchmark stream 事件口径，vLLM TPOT 低于 NEO”，不要过度声称绝对 token-level decode latency。

## 9. 建议补充验证实验

1. **统一 TPOT 观测口径**
   - 修改 NEO/vLLM 的 completion response，使两者都输出 `token_ids`；
   - 或统一按 server-side decode iteration timestamp 计算 TPOT。

2. **关闭 vLLM chunked prefill 对比**
   - 去掉 `--enable-chunked-prefill`；
   - 预期：vLLM TTFT 可能改善或变化，TPOT 可能变差；可验证“decode-prioritized”对 TPOT 的贡献。

3. **改变 vLLM preemption mode**
   - 对比 `--preemption-mode recompute` 与 `swap`；
   - 观察 TTFT、TPOT、preemption 次数。

4. **记录 vLLM preemption / waiting queue 长度**
   - 从 vLLM log 或 scheduler instrumentation 记录 `preempted`、waiting/running queue size、KV block usage；
   - 验证 TTFT 上升是否与 KV capacity bottleneck 同步发生。

5. **记录 NEO CPU decode 与 swap 时间**
   - 导出 `cdec_time`、swap time、pipeline remains、`gpu_decoding_q/cpu_decoding_q/waiting_q` 长度；
   - 验证 TPOT 上升是否来自 CPU decode 无法被 GPU work 完全 hide。

6. **分别画 prefill waiting time 与 decode interval**
   - 当前 `Avg token latency` 混合了所有阶段；
   - 建议拆成 queueing time、prefill time、first decode time、steady decode TPOT。

7. **按 output length 分桶**
   - 对短输出请求，TTFT 对 `(end-start)/output_len` 的影响更大；
   - 对长输出请求，TPOT 的影响更明显。分桶后可以更清晰地看到 NEO/vLLM trade-off。

## 10. 最终结论

用户的核心解释方向是正确的：NEO 的 CPU offloading 确实缓解了 GPU KV cache pressure，降低高负载下的 request admission/queueing delay，因此 TTFT 和平均 token latency 更好；NEO TPOT 更高也确实可以由 CPU decode、KV swap、PCIe/stream synchronization 与 pipeline imbalance 解释。

需要修正的是对 vLLM 的调度描述：本实验中的 vLLM 启用了 `--enable-chunked-prefill`，源码明确说明此时 decode 优先，而不是默认 prefill-first。因此 vLLM “TPOT 更低但 TTFT 更高”的现象并不反常，而是 decode-prioritized GPU-only 调度在 KV cache 压力下的典型表现：它优先保障已进入 decode 的请求，使 inter-token latency 低；但新请求 admission 被 GPU KV capacity 和 recompute/preemption 限制，TTFT 急剧上升。

因此，三张图的统一解释是：

- **vLLM 优先保 TPOT，代价是高负载下 TTFT/排队时间恶化。**
- **NEO 优先扩大有效并发与降低排队，代价是一部分请求 TPOT 因 CPU offload 路径变高。**
- **Avg token latency 包含 TTFT，所以在当前 workload 下由 TTFT 主导，NEO 因此整体更低。**
