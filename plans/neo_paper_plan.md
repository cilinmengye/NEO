# NEO 论文与代码实现对照笔记

> 论文：**NEO: Saving GPU Memory Crisis with CPU Offloading for Online LLM Inference**
> 仓库：`/home/yxlin/github/swift/NEO`

本文档的目标不是单纯复述论文，而是建立一条清晰链路：

- 论文到底在解决什么问题；
- 论文的三个核心创新点分别是什么；
- 这些创新点在仓库里由哪些模块实现；
- 一次在线推理 iteration 在系统里是怎么流动的；
- 复现实验时该看哪些脚本、哪些 flags、哪些代码路径。

---

## 1. 背景动机

### 1.1 论文要解决的核心问题

在线 LLM inference 的主要矛盾不是“算不动”，而是“**GPU memory 先满了，导致 batch 起不来**”。

在在线服务场景里，为了提高吞吐，系统会把不同用户请求放进同一个 batch 里做推理；但随着 batch 变大，**KV cache** 会快速吃掉 GPU 显存。结果是：

1. GPU 算力还有空闲；
2. 但显存已经限制了 batch size；
3. 吞吐上不去，单位 GPU 成本变高。

NEO 的核心判断是：

- 在线推理的瓶颈并不总是纯 GPU compute；
- 本地 CPU 内存容量远大于 GPU 显存；
- CPU 也不是完全不能参与推理，它至少能承担一部分 attention 相关状态与计算；
- 只要系统设计得好，就可以把 **“GPU 显存不足”** 变成 **“GPU+CPU 协同”** 问题。

因此论文提出：**把部分 attention compute 和部分 KV cache state 从 GPU offload 到本地 CPU**，从而扩大可服务 batch，提高吞吐。

### 1.2 为什么不是“全量 offload”

论文/代码的思路不是把模型主体搬到 CPU，而是只 offload 最吃显存、且相对可拆分的部分：

- KV cache 状态；
- 一部分 decoding attention。

模型主体的 dense linear / projection / FFN 仍然以 GPU 为主，这样才能保持高吞吐。

所以 NEO 本质上是 **selective / partial CPU offloading**，不是 CPU-only，也不是简单的 unified memory 方案。

---

## 2. 论文核心创新点

NEO 的核心创新可以概括为三条主线。

### 2.1 创新点一：Selective / Partial CPU Offloading

论文不是把所有请求统一搬去 CPU，而是：

- GPU 继续承担大部分 dense compute；
- CPU 接管部分 KV cache 与部分 decode attention；
- 请求可以分成 GPU decoding 和 CPU decoding；
- 部分 prefill 也可以走 CPU 对应的 swap / cache 流程。

直观理解：

- **GPU 负责“快但贵”的部分**；
- **CPU 提供“慢但容量大”的扩展空间**。

### 2.2 创新点二：Asymmetric GPU-CPU Pipelining

仅仅把请求分流到 CPU/GPU 还不够，因为 CPU 较慢，如果不做流水化，GPU 很容易在等 CPU，整体吞吐未必提升。

NEO 的关键是把两类工作交错起来：

- 某个 sub-batch 在做 attention；
- 另一个 sub-batch 同时在做 post-proj / pre-proj；
- CPU decode 与 GPU compute 尽量重叠。

这里的关键词是 **asymmetric**：

- 不是两个完全对称、同构的 stage；
- 而是 GPU 和 CPU 承担不同类型的工作，两个 sub-batch 的角色也不完全相同。

### 2.3 创新点三：Load-aware Scheduling

如果只是“能 offload 就 offload”，系统会很容易变成 CPU 过载，反而拖慢整体吞吐。

NEO 的调度器会根据性能预测：

- 判断当前该用 sequential mode 还是 pipelined mode；
- 决定哪些请求留在 GPU decode，哪些放到 CPU decode；
- 决定哪些新 prefill 请求可以进入本轮；
- 决定什么时候 swap out / swap in。

也就是说，NEO 不是静态规则，而是 **基于 profile / predictor 的负载感知调度**。

---

## 3. 论文摘要式中文总结

### 3.1 研究目标

论文目标是提升在线 LLM 服务系统的吞吐，尤其是在 GPU 显存不足、batch size 受限的情况下。

### 3.2 解决方案

论文提出 NEO：

1. 用 CPU 内存扩展 KV cache 容量；
2. 把部分 decode attention 放到 CPU 上做；
3. 通过 GPU-CPU 异构流水线隐藏 CPU 侧开销；
4. 用 load-aware scheduling 在不同 batch 结构和负载下做模式选择。

### 3.3 主要贡献

从仓库实现看，论文的贡献不是新网络结构，而是一个 **online serving system design**：

- 控制面：请求排队、分流、swap 决策、block 分配；
- 数据面：GPU KV cache / CPU swap space / CPU attention / layer pipeline；
- 性能面：profile table + predictor + scheduling heuristic；
- 实验面：对比 `ours / base / fsdc / vllm` 的系统级评测。

---

## 4. 仓库整体架构：控制面、数据面、实验面

从实现上，仓库可以按三层理解。

### 4.1 控制面（Control Plane）

负责“本轮该跑谁、谁在 GPU、谁在 CPU、哪些块要换进换出”。

关键文件：

- `swiftllm/server/api_server.py`
- `swiftllm/server/engine.py`
- `swiftllm/server/scheduler.py`
- `swiftllm/server/block_manager.py`
- `swiftllm/structs.py`

### 4.2 数据面（Data Plane）

负责“真正做 forward、做 swap、做 attention、做 pipeline”。

关键文件：

- `swiftllm/worker/model.py`
- `swiftllm/worker/layers/transformer_layer.py`
- `swiftllm/worker/block_swapper.py`

### 4.3 性能建模面（Performance Modeling Plane）

负责“调度器为什么做这个选择”。

关键文件：

- `swiftllm/perfpredictor.py`
- `swiftllm/server/profiler.py`

### 4.4 实验面（Evaluation Plane）

负责“论文图表如何在仓库里复现”。

关键文件：

- `evaluation/server.py`
- `evaluation/reproduce-fig6c.py`
- `evaluation/reproduce-fig10a.py`
- `README.md`
- `swiftllm/engine_config.py`

---

## 5. 系统入口：请求是如何进入 NEO 的

在线路径的入口在 `swiftllm/server/api_server.py`。

### 5.1 API server 做了什么

`/v1/completions` 接口读取用户 JSON，请求体被转成 `RawRequest`：

- `prompt`
- `max_tokens`
- `stream`

之后：

- streaming 模式调用 `engine.add_request_and_stream(...)`
- 非 streaming 模式调用 `engine.add_request_and_wait(...)`

这说明 API server 本身很薄，真正的系统逻辑都在 `AsyncEngine` 里。

### 5.2 AsyncEngine 的职责

`swiftllm/server/engine.py` 中的 `AsyncEngine` 是整个服务系统的总编排器，初始化时会依次做：

1. 创建 model executor；
2. profile 模型可用 block 数；
3. 创建 `BlockManager`；
4. 初始化 KV cache 和 swap 空间；
5. 初始化 performance table；
6. 创建 `Scheduler`；
7. 创建远程 tokenization engine。

这一步很关键：**scheduler 不是凭空工作的，它依赖 profiler 初始化出来的性能表。**

### 5.3 两个异步事件循环

`AsyncEngine` 里有两个主循环：

1. `_tokenize_raw_request_event_loop()`
   - 批量拿走待 tokenizer 的原始请求；
   - 调用 tokenization engine；
   - 把 tokenized request 送入 scheduler。

2. `_main_event_loop()`
   - 从 scheduler 取下一轮 batch；
   - 调用 block manager 准备 mapping / swapping；
   - 异步执行 model forward；
   - 更新输出 token，释放完成请求的 block。

因此在线系统的主骨架就是：

**request arrival → tokenize → schedule → prepare block/swap → forward → update output → reclaim resources**。

---

## 6. 核心数据结构：Request 与 SubBatch

要读懂调度器和 worker，必须先读懂 `swiftllm/structs.py`。

### 6.1 Request

`Request` 表示系统中的一个请求，包含：

- `prompt_token_ids`
- `prompt_len`
- `output_len`
- `max_output_len`
- `request_id`
- `output_token_ids`
- `output_q`
- `finished_event`

其中最关键的派生量是：

- `seq_len = prompt_len + output_len`

这决定了它当前需要多少 KV block。

### 6.2 Request 在 forward 中的输入 token 语义

`Request.get_input_tokens(reqs)` 的逻辑是：

- 对 prefill 请求，输入是整个 prompt；
- 对 decoding 请求，输入只需要最后一个 token。

这正好对应在线推理的两阶段：

- prefill：吃整段 prompt；
- decode：每轮只吃新生成的最后一个 token。

### 6.3 SubBatch 的四类请求划分

`SubBatch` 里把请求分成四类：

- `gprf_reqs`：GPU prefill
- `cprf_reqs`：CPU-prefill / 需要 CPU 参与的 prefill
- `gdec_reqs`：GPU decode
- `cdec_reqs`：CPU decode

这是全文最重要的抽象之一，因为它直接对应论文中的“异构分流”。

### 6.4 set_model_forward_args：把队列对象变成 forward 参数

`SubBatch.set_model_forward_args(...)` 会把上面四类请求变成 forward 所需的结构化字段，例如：

- `num_cprfs`
- `num_gprfs`
- `num_gdecs`
- `num_cdecs`
- `num_prefs`
- `num_prgds`
- `all_reqs`
- `seq_ids_list`
- `seq_lens_list`
- `sum_pref_toks`
- `sum_prgd_toks`
- `max_pref_toks`
- `seq_block_size`
- `num_seq_blocks`

其中最关键的是 `all_reqs` 的顺序：

**cprf + gprf + gdec + cdec**

后面 block allocation、attention 分流、CPU/GPU copy 都依赖这个顺序。

---

## 7. 创新点一：CPU Offloading 在代码中的实现

这一部分是论文中“Saving GPU Memory Crisis”的最直接实现。

---

### 7.1 控制面：BlockManager 管理 GPU / CPU block

`swiftllm/server/block_manager.py` 中：

- `DeviceBlockManager` 管一个设备（GPU 或 CPU）的 block 分配与回收；
- `BlockManager` 同时管理 GPU 和 CPU 两套 block manager。

#### 7.1.1 逻辑 block 与物理 block

核心思想是把 block 管理拆成两层：

- **VID（virtual block id）**：由 `(seq_id, block_index)` 隐式表示；
- **PID（physical block id）**：真实落在某个设备上的块编号。

`DeviceBlockManager.alloc(...)` 会：

1. 根据请求当前 `seq_len` 计算目标 block 数；
2. 找出还缺多少新 block；
3. 从 free list 里拿新的 physical block；
4. 把 `(seq_id, block_index) -> physical block id` 写入 block table。

这使得请求在 swap 前后逻辑位置不变，只是物理位置变化。

#### 7.1.2 free / alloc 是 swap 的基础

`DeviceBlockManager.free(...)` 负责：

- 取出某个请求占用的物理 block id；
- 把这些 block 标为 free；
- 把该请求的 `seq_num_blks` 归零。

`BlockManager._initiate_swap(...)` 则把这个 free/alloc 过程串起来：

- 如果是 swap-out：GPU free，CPU alloc；
- 如果是 swap-in：CPU free，GPU alloc。

因此，**swap 在控制面上首先是 block ownership 的转移**。

---

### 7.2 数据面：Swapper 同时持有 GPU cache、CPU swap、CPU QKV buffer

`swiftllm/worker/block_swapper.py` 中的 `Swapper` 是数据面的核心。

它初始化了四类重要存储：

#### 7.2.1 GPU KV cache

- `self.k_cache`
- `self.v_cache`

shape 大致是：

`[num_layers (+ extra_layer_for_cprf), num_gpu_blocks, num_kv_heads, block_size, head_dim]`

这说明 GPU 端缓存以 **layer × block** 组织。

#### 7.2.2 CPU KV swap space

- `self.k_swap`
- `self.v_swap`

它们在 CPU 上，并且 `pin_memory=True`，说明实现明确针对 host-device 异步传输优化。

#### 7.2.3 CPU 端 Q/K/V/O buffer

- `q_cpu`
- `k_cpu`
- `v_cpu`
- `o_cpu`

这些 buffer 用来承接 CPU decode attention 的数据流：

- GPU 上产生 q/k/v；
- 需要 CPU decode 的那部分被 copy 到 CPU pinned buffer；
- CPU kernel 计算 attention；
- 结果写到 `o_cpu`；
- 再 copy 回 GPU。

#### 7.2.4 GPU/CPU block table

- `gpu_block_table`
- `cpu_block_table`

它们和控制面的 `BlockManager` 一起构成“逻辑块 → 设备物理块”的完整映射体系。

---

### 7.3 swap 真正发生在哪里

真正的数据搬运发生在 `Swapper.swap_blocks(...)`。

这个函数调用 `swiftllm_c.swap_blocks(...)`，输入包括：

- 源 block id 列表
- 目标 block id 列表
- swap out / in 方向
- GPU layer / CPU layer
- GPU KV cache tensor
- CPU swap tensor

也就是说：

- `BlockManager` 决定该换哪些块、换到哪里；
- `Swapper` 负责真正把 KV 内容搬过去。

控制面与数据面在这里接上。

---

### 7.4 CPU decode attention 的真实实现

论文说“offload part of attention compute to CPU”，代码对应在 `swiftllm/worker/layers/transformer_layer.py`。

#### 7.4.1 _transfer_qkv：把 CPU decode 所需的 q/k/v 搬到 CPU

`_transfer_qkv(...)` 中：

- 如果 `batch.num_cdecs > 0`，就把最后 `num_cdecs` 个序列对应的 q/k/v 复制到：
  - `self.swapper.q_cpu`
  - `self.swapper.k_cpu`
  - `self.swapper.v_cpu`

这是一个非常关键的实现细节：

**CPU 并不是自己重新算 QKV，而是拿 GPU 已经算好的 QKV。**

因此 CPU 只承担 attention 部分，不承担前面的 dense projection。

#### 7.4.2 _attention：GPU decode 和 CPU decode 并行存在

`_attention(...)` 里有三条路径：

1. prefill attention
   - Ampere 及以上优先走 `flash_attn_cuda.varlen_fwd(...)`
   - 否则走自定义 `prefill_attention(...)`

2. GPU decode attention
   - 对 `num_gdecs > 0` 的部分调用 `paged_attention(...)`
   - 从 GPU KV cache 里取历史块

3. CPU decode attention
   - 等待 `qkvtr_e`，保证 QKV 传输完成；
   - 调用 `torch.ops.pacpu.paged_attention_cpu(...)`；
   - 输入是 CPU 上的 `q_cpu/k_cpu/v_cpu` 和 `k_swap/v_swap/cpu_block_table`；
   - 输出到 `o_cpu`；
   - 再异步 copy 回 GPU 的 `o` buffer。

这就是论文里“部分 attention compute offload 到 CPU”的最具体代码落点。

---

### 7.5 CPU prefill / partial offloading 是怎么接进去的

`SubBatch` 里有 `cprf_reqs`，而 `BlockManager.prepare(...)` 的最后一步专门做了 **cprf swaps**：

- 对 `batch.all_reqs[:batch.num_cprfs]` 调用 `_initiate_swap(...)`
- `is_swap_out=True`
- `omit_last=False`
- `use_itm=self.engine_config.extra_layer_for_cprf`

这意味着：

- CPU-prefill 请求也会触发额外的 KV block 转移；
- 这些请求在设计上不是“普通 prefill 全留 GPU”；
- 它们和论文中的 partial offloading 直接相关。

---

### 7.6 extra_layer_for_cprf 的含义

这是论文实现里的一个非常关键但容易忽略的开关。

在多个文件里都能看到它：

- `EngineConfig.extra_layer_for_cprf`
- `Swapper` 的 KV cache shape 会多一层
- `BlockManager` 的 split 数量会变化
- `transformer_layer.py` 中计算 `itm_layer` 时会用到
- `evaluation/server.py` 中 `ours` 和 `fsdc` 都会打开这个 flag

直观理解：

- 系统为 cprf 路径额外留出一层中间缓存空间；
- 这样在 swap / prefill / pipeline 交错时，能减少覆盖冲突。

论文的“partial offloading”在工程上不是一句话，而是和这个额外层位设计绑在一起的。

---

### 7.7 为什么这是“partial”而不是“full”

从整条数据流可以清楚看出：

- dense linear / RMSNorm / FFN 仍在 GPU；
- prefill attention 仍优先 GPU；
- GPU decode 仍然存在；
- CPU 只接管了部分 decode attention 与部分 KV state。

所以代码实现与论文术语是对齐的：

**不是把请求迁移到 CPU 上跑模型，而是把一部分 memory-heavy / attention-heavy 工作迁到 CPU。**

---

## 8. 创新点二：Asymmetric GPU-CPU Pipelining 在代码中的实现

这是 NEO 区别于“仅做 offloading”的第二个关键。

---

### 8.1 顶层 pipeline 在哪里发生

顶层 pipeline 逻辑在 `swiftllm/worker/model.py`。

有两种模式：

- `_forward_sequential(...)`
- `_forward_pipeline(...)`

当 `len(batches) == 1` 时走 sequential；
当 `len(batches) == 2` 时走 pipeline。

这和 scheduler 的模式选择直接对应。

---

### 8.2 _forward_pipeline 的基本思路

`_forward_pipeline(...)` 大致流程是：

1. 最后一层先执行 `forward_first_stage(...)`
2. 中间层循环执行 `forward_double(...)`
3. 最后一层执行 `forward_last_stage(...)`

这说明 pipeline 不是在 batch 维度简单并发，而是在 **layer 内部把两个 batch 的不同阶段交错起来**。

---

### 8.3 layer 级 pipeline：forward_double 的语义

`swiftllm/worker/layers/transformer_layer.py` 中的 `forward_double(...)` 是理解论文实现的关键。

注释里已经写得很清楚：

- batch 0：做 post-projection[i] -> pre-projection[i+1]
- batch 1：做 attention[i]
- 然后下一 stage 交换角色

也就是说，一个 layer 里同时交错两类工作：

- 一个 batch 在“attention 路径”；
- 另一个 batch 在“MLP/投影路径”。

这样可以尽量把 GPU 的不同类型算子拼接得更满。

---

### 8.4 为什么叫 asymmetric

它不是传统对称 pipeline，原因有三点。

#### 8.4.1 两个 sub-batch 的工作负载不同

`SubBatch` 可以混有：

- gprf
- cprf
- gdec
- cdec

因此两个 batch 的 attention / CPU 工作量不是天然对称的。

#### 8.4.2 CPU decode 是异构插入的

在 `_attention(...)` 里，CPU decode 不是单独 stage，而是插在 attention 过程中，并通过：

- `cpu_communication_stream`
- event 同步
- pinned memory copy

与 GPU 侧重叠。

#### 8.4.3 batch 之间不是简单一一对应

`Scheduler._decide_mode_and_gen_batch(...)` 会根据负载把 CPU decode 请求拆到 batch0/batch1，不保证两个 batch 工作量对称。

所以这里的 asymmetry 既来自：

- **硬件异构（GPU vs CPU）**
- 也来自 **batch 结构异构（不同数量的 pref/gdec/cdec）**

---

### 8.5 stream / event 是如何实现 overlap 的

`LlamaTransformerLayer` 中维护：

- 默认 CUDA stream
- `cpu_communication_stream`
- 一组 `TransformerEvents`

关键同步原语有：

- `_comm_wait_compute()`
- `_compute_wait_comm()`
- `qkvtr_e.record()`
- `qkvtr_e.synchronize()`

直观理解：

- 当 GPU 需要把 QKV 交给 CPU 时，用 communication stream 发起 copy；
- 当 CPU 结果还没回来时，不让依赖它的 GPU 计算越界使用；
- 但与此同时，另一个 batch 的 GPU 线性层、投影层可以继续推进。

这正是“用流水线隐藏 CPU 成本”的工程核心。

---

### 8.6 forward_first_stage / forward_last_stage 的作用

为了让双 batch pipeline 能首尾闭合，代码还拆出了：

- `forward_first_stage(...)`
- `forward_last_stage(...)`

这两个函数分别处理：

- pipeline 的启动阶段；
- pipeline 的收尾阶段。

说明这不是“随便塞两个 batch 并发执行”，而是认真实现了一个 layer-wise 的流水体系。

---

## 9. 创新点三：Load-aware Scheduling 在代码中的实现

如果说 CPU offloading 解决的是“容量”，pipeline 解决的是“重叠”，那 load-aware scheduling 解决的就是“什么时候该怎么组合这些手段”。

---

### 9.1 Scheduler 维护了哪几类队列

`swiftllm/server/scheduler.py` 中维护三类主要队列：

- `waiting_q`：等待进入系统的新请求
- `gpu_decoding_q`：当前在 GPU decode 路径上的请求
- `cpu_decoding_q`：当前在 CPU decode 路径上的请求

这三个队列就对应了系统里请求所处的主要状态。

---

### 9.2 ScheduleBudget：调度首先受 batch/tokens 约束

`ScheduleBudget` 跟踪两个预算：

- `remaining_batch_size`
- `remaining_tokens_in_batch`

这说明 scheduler 的第一层约束并不是“理论上能不能做”，而是“本轮能不能在不 OOM、不爆 batch 的前提下做”。

---

### 9.3 _get_next_batch_new：调度主流程

`Scheduler._get_next_batch_new(...)` 可以按 6 步理解：

#### 第 1 步：先把 GPU decoding 请求尽可能纳入本轮

它先统计当前 `gpu_decoding_q` 占用多少 GPU block 和 batch/token 预算。

#### 第 2 步：必要时 swap out

当出现两类情况时会触发 preemption：

- budget overspent；
- GPU block 超过 `swap_out_threshold`。

此时它把 `gpu_decoding_q` 尾部请求移到 `cpu_decoding_q`，并记录到 `swpout_reqs`。

这体现出它是严格按到达顺序近似 FCFS 的：

- 早来的请求尽量保留；
- 晚来的更容易被换出去。

#### 第 3 步：如果有余量，再 swap in

从 `cpu_decoding_q` 头部挑请求，看是否能重新放回 GPU。

#### 第 4 步：尝试加入新的 prefill 请求

遍历 `waiting_q`，检查：

- GPU intermediate blocks 是否足够；
- CPU blocks 是否足够；
- request id 是否足够；
- batch/token budget 是否足够。

同时用启发式决定：

- 优先放 GPU；
- 如果 GPU 压力大，就转去 CPU；
- 为保证公平性，如果前面的新请求已经进 CPU，后面的也不能随意插队进 GPU。

#### 第 5 步：形成 batch 并决定模式

调用 `_decide_mode_and_gen_batch(...)`。

#### 第 6 步：真实提交 prefill 请求

前面只是“试探上限”，这一步才真正从 `waiting_q` 取出请求并分配 request id。

---

### 9.4 _decide_mode_and_gen_batch：这是论文 claim 的核心落点

这是整个仓库里最值得反复读的函数之一。

它不是简单拼 batch，而是在做三件事：

1. 决定 CPU decode 请求如何拆到两个 sub-batch；
2. 必要时削减 prefill 数量，避免 CPU 闲等或超载；
3. 估算 sequential 与 pipelined 两种模式的吞吐，择优选择。

---

### 9.5 perfdata 是调度器的局部性能模型

`SubBatch` 内部有 `BatchPerfData`，它记录：

- `x`：batch 中请求数
- `s`：iteration width
- `n_g`：GPU decode token 总量
- `x_c`：CPU decode 请求数
- `n_c`：CPU decode token 总量

并通过 predictor 提供：

- `linr_T`
- `pref_T`
- `gdec_T`
- `cdec_T`
- `lnch_T`

因此 scheduler 不是依据固定阈值瞎猜，而是对当前 batch 组合有一个性能估算。

---

### 9.6 _get_remains：如何平衡两个 batch 的 CPU 容量

`_get_remains(...)` 估计每个 batch 的“CPU decoding capacity 是否还有余额”，形式上近似：

- 另一批的 linear 时间
- 加上当前批的 prefill/gdec 时间
- 减去当前批的 CPU 时间

直观含义是：

**在 GPU 侧推进这些工作的时候，CPU 侧有没有机会把本批 cdec 也做完。**

这正是“load-aware”比“纯规则调度”高明的地方：它真的在估计 overlap 是否成立。

---

### 9.7 sequential vs pipelined 的模式选择

函数最后会估计：

- `seqential_time`
- `pipelined_time`
- `seqential_rate`
- `pipelined_rate`

如果 pipelined 吞吐更高，就返回两个 batch；否则只返回一个 `gpu_only_batch`。

这就是论文中“load-aware scheduling”的最直接代码映射：

**不是默认永远 pipeline，而是根据估计吞吐做模式切换。**

---

## 10. 性能预测器为什么可信：Profiler + Predictor 的配合

这一层是理解 scheduler 的关键补充。

---

### 10.1 TablePerfPredictor：调度器使用的是插值表

`swiftllm/perfpredictor.py` 中的 `TablePerfPredictor` 维护多组 profile table：

- `linr_S_list / linr_T_list`
- `pref_S_list / pref_T_list`
- `gdec_N_list / gdec_T_list`
- `cdec_S_list / cdec_N_lists / cdec_T_lists`
- `lnch_T`

然后通过：

- 一维线性插值
- CPU decode 的二维/双线性插值

来估计未精确测过的点。

所以 predictor 不是常数表，而是 **profile table + interpolation**。

---

### 10.2 ModelProfiler：这些表是跑出来的，不是写死的

`swiftllm/server/profiler.py` 的 `init_profile_tables(...)` 会调用：

- `_profile_linr(...)`
- `_profile_pref(...)`
- `_profile_gdec(...)`
- `_profile_cdec(...)`

这些函数通过伪造 test case，真正跑模型，然后收集 `ModelPerfResult`。

### 10.3 CPU / GPU block 数也是 profile 出来的

`profile_num_blocks()` 会：

1. 根据 block size 和模型 KV slot 大小估算每个 block 占多少内存；
2. 构造一个最大 prefill batch；
3. 测 runtime peak memory；
4. 反推出可安全使用的 GPU blocks 数量；
5. 同时根据 swap space 算 CPU blocks 数量。

这意味着 NEO 的调度和内存管理都带有“先 profile 再运行”的特点。

---

## 11. 端到端心智模型：一次请求如何流过系统

这一节建议作为你之后读代码时的主心智模型。

---

### 11.1 请求进入

用户向 `api_server.py` 的 `/v1/completions` 发请求，构造 `RawRequest`。

### 11.2 请求对象化

`AsyncEngine.add_request_and_wait()` 或 `add_request_and_stream()` 把它变成 `Request`。

### 11.3 tokenizer 异步处理

`_tokenize_raw_request_event_loop()` 批量调用远程 tokenizer，把 `prompt` 变成 `prompt_token_ids`。

### 11.4 scheduler 决定本轮执行什么

`Scheduler.get_next_batch()` 返回三部分：

- `batches`
- `newly_swapped_out`
- `newly_swapped_in`

这一步已经做了：

- 队列选择
- CPU/GPU 分流
- 模式选择（sequential / pipeline）
- swap 决策

### 11.5 block manager 把高层决策翻译成底层 mapping

`BlockManager.prepare(...)` 会：

1. 做普通 swap 的 block 重映射；
2. 给 batch 里的请求分配新 block；
3. 给 cprf 做额外 swap；
4. 生成：
   - block table mappings
   - swappings
   - swap direction

### 11.6 worker 执行一次 iteration

`LlamaModel.do_one_iteration(...)` 里：

1. 先把 block tables 写到 `Swapper`；
2. 如果有 swap，就逐层调用 `self.swapper.swap_blocks(...)`；
3. 调用 `_forward_batches(...)`。

### 11.7 准备输入

`_prepare_inputs(...)` 生成 GPU 侧 forward 所需张量：

- `prgd_seq_ids`
- `prgd_seq_lens`
- `pref_st_locs_we`
- `position_cos`
- `position_sin`
- `attn_out_buf`
- `residual_buf`
- `last_token_indices`

### 11.8 pre-layer 产生 embedding

`pre_layer.forward(...)` 把 token ids 变成 embeddings。

### 11.9 transformer layers 执行

若单 batch：走 `_forward_sequential(...)`。
若双 batch：走 `_forward_pipeline(...)`。

在单层内部：

- `_preproj(...)` 做 RMSNorm + QKV + rotary + 可能的 KV 存储；
- `_transfer_qkv(...)` 把 cdec 的 QKV 拷到 CPU；
- `_attention(...)` 分别处理 prefill / gdec / cdec；
- `_postproj(...)` 做 o_proj + FFN。

### 11.10 post-layer 采样输出 token

`post_layer.forward(...)` 取出每个请求对应的最后 token logits，生成输出 token。

### 11.11 请求状态更新与资源回收

`BlockManager.update_and_free(...)`：

- 调用 `Request.update_output(...)` 更新每个请求的 output；
- 若请求结束，释放其 GPU / CPU blocks。

### 11.12 scheduler 移除完成请求

`scheduler.remove_finished_requests(...)` 把已完成请求从解码队列里移除，并释放 request id。

于是系统进入下一轮 iteration。

---

## 12. 单次 iteration 的更直观时间线

如果你想把论文里的系统图和代码串成一幅“时序图”，可以这样理解：

1. **上轮结束后**，系统里已经有一些请求在 GPU decode / CPU decode 队列中。
2. **scheduler** 根据当前队列、block 余量、perf predictor，决定：
   - 哪些继续在 GPU；
   - 哪些换到 CPU；
   - 哪些从 CPU 换回 GPU；
   - 是否接收新 prefill；
   - 是 1 个 batch 还是 2 个 batch。
3. **block manager** 把这个决策转成 block table 更新和 swap 参数。
4. **worker** 先做 swap，再进入 forward。
5. 对每个 layer：
   - GPU 做 pre-proj / post-proj / FFN；
   - GPU 做 prefill 和 GPU decode attention；
   - CPU 做 CPU decode attention；
   - CPU/GPU 之间通过 pinned memory + CUDA stream + events 同步。
6. **post layer** 产出新 token。
7. **engine** 更新请求状态、流式返回 token、释放完成请求资源。
8. 进入下一轮 iteration。

NEO 的核心就在于：

- 第 2 步不是盲目调度；
- 第 5 步不是顺序执行；
- 二者结合，才真正把 CPU 容量变成吞吐提升。

---

## 13. 实验与仓库脚本/参数映射

这一节对复现非常重要。

---

### 13.1 README 提供了论文与脚本的总体映射

`README.md` 已经明确说明：

- Figure 6c 是 **load-latency curve**；
- Figure 10a 是 **generation throughput**；
- 论文核心是 CPU offloading、asymmetric GPU-CPU pipelining、load-aware scheduling。

README 也给出直接的复现实验入口：

- `python evaluation/reproduce-fig6c.py`
- `python evaluation/reproduce-fig10a.py`

---

### 13.2 evaluation/server.py：不同系统名如何映射到不同模式

`evaluation/server.py` 是理解 `ours / base / fsdc / vllm` 的核心。

#### 13.2.1 vllm

当名字以 `vllm` 开头时，脚本会启动 vLLM server，并配置：

- `--enable-chunked-prefill`
- `--preemption-mode recompute`
- `--num-gpu-blocks-override`
- `--swap-space`

这对应论文里的 vLLM baseline。

#### 13.2.2 base

`base` 对应：

- `--always-use-gpu`

这意味着：

- 不使用 CPU offloading 的核心路径；
- 可以把它理解为论文里的 non-CPU-offloading baseline。

#### 13.2.3 ours

`ours` 对应：

- `--extra-layer-for-cprf`

同时会对 `num_gpu_blocks_override` 做 `(nl / (nl + 1))` 缩放。

这说明论文主方法默认包含：

- cprf 额外层位设计；
- 配合 CPU offloading 的 block 布局。

#### 13.2.4 fsdc

`fsdc` 对应：

- `--disable-partial-offl`
- `--extra-layer-for-cprf`

这通常可以理解为一个消融/对照：

- 保留某些 pipeline / cache 设计；
- 但关闭 partial offloading 相关行为。

---

### 13.3 EngineConfig 中哪些 flags 最关键

`swiftllm/engine_config.py` 中最值得关注的开关有：

- `extra_layer_for_cprf`
- `disable_partial_offl`
- `always_use_gpu`
- `monitor_performance`
- `tensor_parallel_degree`

其中和论文方法最相关的是前三个。

#### `--always-use-gpu`

把系统退化到 GPU-only 风格路径，是 baseline 的关键开关。

#### `--disable-partial-offl`

关闭 partial offloading 相关行为，用于做消融。

#### `--extra-layer-for-cprf`

启用 cprf 的额外 layer 空间，是 `ours` 路径的关键工程设计。

---

### 13.4 Figure 6c：仓库里实际怎么复现

`evaluation/reproduce-fig6c.py`：

- 读取 `config-t4-7b.json`
- 分别跑：
  - `vllm`
  - `ours`
- 使用真实 workload：`prepare_real_test("osc", ...)`
- 最后画出 `fig6c`

注意两个细节：

1. 当前默认只画两条线：`vllm` 和 `ours`；
2. 使用的是论文原始数据的子集（README 也说明了），所以 latency 会比论文原图低一些。

---

### 13.5 Figure 10a：仓库里实际怎么复现

`evaluation/reproduce-fig10a.py`：

- 读取 `config-a10-8b.json`
- 分别跑：
  - `base`
  - `ours`
- 使用 synthetic workload：
  - `input_len = 1000`
  - `output_lens = [50, 100, 200, 300, 400]`
- 最后画 throughput sensitivity 图。

这与 README 中 Figure 10a 的描述是对齐的。

---

### 13.6 example.py 不要误读

`examples/example.py` 的文件头已经明确写了：

> this script is for demonstration purposes only and uses symmetric pipelining. In evaluation, we use asymmetric pipelining instead.

这非常重要：

- 它适合帮助你理解 engine 的离线调用方式；
- 但**不能把它当成论文评测实现本身**；
- 论文中的核心实验路径还是 `evaluation/*` + `server/api_server.py`。

---

## 14. 实验设置与结果的中文理解

### 14.1 Figure 6c（负载-延迟曲线）

根据 README：

- 硬件：AWS g4.4xlarge，T4 GPU，8 核 CPU，64GB 内存；
- 模型：LLaMa-2-7B；
- 工作负载：OpenAI summarization comparison。

这个实验主要想说明：

- 在在线负载升高时，NEO 能比 baseline 支撑更高吞吐/更低延迟；
- CPU offloading + scheduling 不是只在离线场景有效。

### 14.2 Figure 10a（吞吐敏感性）

根据 README：

- 硬件：g5 系列，不同 CPU 容量；
- 模型：LLaMa-3-8B；
- 任务：固定输入长度、变化输出长度的 synthetic workload。

这个实验主要想说明：

- 当 CPU 资源更充足时，NEO 从 CPU offloading 中得到的收益更大；
- NEO 的收益与 output length、CPU capacity 都有关。

---

## 15. 与相关路线的关系

从系统设计路线看，NEO 不是在做以下事情：

- 不是修改 Transformer 架构；
- 不是压缩 KV cache；
- 不是 speculative decoding；
- 不是只做 paged attention。

它属于另一条路线：

**Serving-system co-design for online LLM inference**

更具体说，它补的是这样一个空白：

- 之前系统多把 CPU 当作“辅助控制器”或“swap 存储”；
- NEO 进一步把 CPU 变成 attention 计算路径的一部分；
- 同时再用 pipeline 和 scheduling 保证这种异构路径不会把 GPU 拖空。

所以它的创新是系统层的，而不是模型层的。

---

## 16. 论文方法的新颖性总结

如果要用最简洁的话概括论文方法的新颖性，我会总结成三句：

1. **它不是只把 KV cache 放到 CPU，而是把 CPU 纳入 attention 计算路径。**
2. **它不是只做 heterogeneous execution，而是明确用 asymmetric pipeline 去隐藏 CPU 代价。**
3. **它不是静态策略，而是用 profiler + predictor + scheduler 在运行时选模式。**

这三件事缺一不可：

- 只有 offload，没有 pipeline，CPU 很可能成为明显瓶颈；
- 只有 pipeline，没有 scheduling，可能根本选错 batch 组合；
- 只有 scheduling，没有 CPU compute path，也解决不了显存危机。

---

## 17. 局限性与实现取舍

从代码和实验设计可以看出，NEO 也有明显前提和取舍。

### 17.1 强依赖本地 CPU 与 NUMA 环境

`evaluation/server.py` 默认用 `numactl -N 0 -m 0` 启动服务，说明实现对 NUMA locality 是敏感的。

这意味着：

- 论文结果依赖 CPU/GPU 在同机、低延迟通信；
- 如果主机 NUMA 配置差，CPU offloading 收益可能下降。

### 17.2 强依赖硬件 profile

调度器的 predictor 来自 `ModelProfiler` 的 profile 结果，因此：

- 换 GPU / CPU / 模型 / block size 后，profile table 需要重新校准；
- 这不是一个完全零成本迁移的策略。

### 17.3 CPU decode 的收益依赖 workload

如果 workload 的 output 很短，或者系统几乎没有 decode 压力，那么：

- CPU offloading 的收益不一定大；
- pipeline 和 swapping 反而可能引入额外开销。

代码里通过 scheduler 的模式选择在缓解这个问题，但它不可能在所有负载下都稳赚。

### 17.4 工程复杂度明显提升

相比 GPU-only serving，NEO 额外引入：

- 双设备 block 管理；
- swap mapping；
- pinned memory；
- CPU kernel；
- performance profiler；
- 双 batch pipeline；
- 更复杂的调度器。

也就是说，NEO 的收益来自更复杂的系统设计，而不是“白拿”的优化。

---

## 18. 复现时最建议的阅读顺序

如果你接下来要继续读源码，我建议严格按这个顺序。

### 第一层：先建立系统骨架

1. `README.md`
2. `swiftllm/server/api_server.py`
3. `swiftllm/server/engine.py`

目标：看懂请求如何进入系统、谁是总编排器。

### 第二层：看调度与控制面

4. `swiftllm/structs.py`
5. `swiftllm/server/scheduler.py`
6. `swiftllm/server/block_manager.py`

目标：看懂请求如何被拆成 `gprf/cprf/gdec/cdec`，以及 block / swap 是如何准备的。

### 第三层：看数据面

7. `swiftllm/worker/block_swapper.py`
8. `swiftllm/worker/model.py`
9. `swiftllm/worker/layers/transformer_layer.py`

目标：看懂一轮 forward 如何真实执行，以及 CPU/GPU attention 怎样交错。

### 第四层：看性能建模

10. `swiftllm/perfpredictor.py`
11. `swiftllm/server/profiler.py`

目标：看懂 scheduler 为什么这么选。

### 第五层：看实验复现

12. `evaluation/server.py`
13. `evaluation/reproduce-fig6c.py`
14. `evaluation/reproduce-fig10a.py`
15. `examples/example.py`

目标：把论文图表、系统模式和实际启动命令对起来。

---

## 19. 如果你要把整套系统讲给别人听，可以这样概括

NEO 是一个在线 LLM serving system。它观察到 GPU 显存限制了 batch size，于是把一部分 KV cache 和 decode attention offload 到本地 CPU，用 CPU 内存和 CPU 计算扩展系统容量。为了不让 CPU 成为瓶颈，它把两个 sub-batch 组织成异构流水线：一个 batch 在做 attention，另一个 batch 在做 post-proj / pre-proj，同时 CPU decode 与 GPU compute 尽量重叠。最后，它再用 profile-based 的 load-aware scheduler 动态决定哪些请求该在 GPU、哪些该在 CPU，以及当前该走 sequential 还是 pipeline。论文的贡献本质上是 serving system design，而不是模型结构创新。

---

## 20. 最终结论

对于这篇论文和这份代码仓库，我认为最重要的理解不是某个 kernel 细节，而是下面这个总图景：

1. **容量问题**：GPU 显存放不下足够大的 batch，主要压力来自 KV cache。
2. **论文方案**：把部分 KV cache 和部分 decode attention 放到 CPU。
3. **工程难点**：CPU 慢，所以不能只 offload，必须 pipeline。
4. **系统关键**：pipeline 也不能静态开，需要 load-aware scheduling 决定何时、如何使用。
5. **代码映射**：
   - `scheduler.py` 负责决策；
   - `block_manager.py` 负责控制面的 block/swap 准备；
   - `block_swapper.py` 负责数据搬运；
   - `model.py + transformer_layer.py` 负责执行异构 forward；
   - `perfpredictor.py + profiler.py` 为调度器提供性能依据。
6. **实验映射**：
   - `evaluation/server.py` 定义 `ours/base/fsdc/vllm` 的运行模式；
   - `reproduce-fig6c.py` 和 `reproduce-fig10a.py` 对应论文关键图。

如果你之后继续深入源码，我建议一直带着这个问题去看：

> **这段代码是在解决“容量扩展”、 “异构流水线隐藏 CPU 开销”，还是“基于预测做模式选择”？**

只要始终用这三条主线去定位代码，整套 NEO 实现就会非常清晰。
