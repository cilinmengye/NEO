# NEO 相比 base-swiftLLM 的核心改动调研

> 目标：结合论文的两个核心创新点，解释 NEO 在代码层面做了哪些关键改动、这些改动各自解决什么问题，以及它们如何协同工作。
>
> 两个创新点：
> 1. **CPU Offloading**
> 2. **Asymmetric GPU-CPU Pipelining**

---

## 1. 先说结论：NEO 到底比 base-swiftLLM 多做了什么？

如果只用一句话概括：

**base-swiftLLM 主要还是一个“GPU 单批次推理 + CPU 作为 swap 空间”的系统；而 NEO 把 CPU 真正纳入了在线推理执行路径，并且在此基础上引入了预测驱动的异构双 sub-batch 流水线。**

更具体地说，NEO 的变化不是某一个 kernel 的小修小补，而是一个**跨控制平面（scheduler / block manager / profiler）和数据平面（model / transformer layer / swapper）**的系统性重构：

1. **请求抽象变了**：
   - base 主要关心 waiting / running / swapped。
   - NEO 进一步细分为 `gprf / cprf / gdec / cdec`，即显式区分“这个请求当前/接下来由 GPU 还是 CPU 继续承担什么工作”。

2. **调度目标变了**：
   - base 更像“GPU 不够了就 swap，一次只发一个 batch”。
   - NEO 则会根据预测的 GPU / CPU 时间，决定：
     - 哪些请求留在 GPU decode；
     - 哪些请求转成 CPU decode；
     - 新 prefill 请求先落到 GPU 还是 CPU 驻留路径；
     - 本轮到底走单 batch sequential，还是双 sub-batch pipeline。

3. **CPU 的角色变了**：
   - base 中 CPU 主要存放被换出的 KV；
   - NEO 中 CPU 不只存 KV，还实际参与了 **decode attention 计算**，即 `paged_attention_cpu(...)` 这条新路径。

4. **worker 的执行方式变了**：
   - base 的 worker 本质上是单 batch、逐层顺序 forward；
   - NEO 的 worker 能处理 1 个或 2 个 `SubBatch`，并把 `preproj / attention / postproj / QKV 传输 / swap` 交错起来执行。

5. **系统多了一整条“先 profile、再建模、再调度”的链路**：
   - `TransformerEvents` / `ModelEvents` / `ModelPerfResult`
   - `ModelProfiler`
   - `TablePerfPredictor`
   - `Scheduler`

这条链路是 NEO 相比 base 非常关键、但又最容易被忽略的新增主线。

---

## 2. 从一次迭代的主控制流看 NEO 的整体结构

NEO 最适合从 `engine -> scheduler -> block_manager -> model/layer` 这条链来理解。

### 2.1 AsyncEngine 的主循环

NEO 在 `swiftllm/server/engine.py:189` 开始的 `_main_event_loop()` 中，每轮做的事情是：

1. `self.scheduler.get_next_batch()` 取下一轮要执行的 batch 与 swap 决策（`swiftllm/server/engine.py:195`）
2. `self.block_manager.prepare(...)` 为这一轮准备 block 映射与 swap 参数（`swiftllm/server/engine.py:202`）
3. `self.executor.do_one_iteration(...)` 真正执行本轮 model forward（`swiftllm/server/engine.py:208`）
4. `self.block_manager.update_and_free(...)` 根据输出 token 更新请求状态并释放完成请求的 blocks（`swiftllm/server/engine.py:211`）
5. `self.scheduler.remove_finished_requests(...)` 从调度队列移除已完成请求（`swiftllm/server/engine.py:212`）

这五步对应的是一个非常清晰的职责分离：

- **Scheduler**：决定“这轮让谁算、在哪算、是否开 pipeline”
- **BlockManager**：决定“这轮 KV blocks 在 GPU/CPU 上怎么布置、哪些 block 要 swap”
- **Model / Layer**：决定“这轮具体怎么算、怎么 overlap GPU 与 CPU”

base-swiftLLM 也有 engine / scheduler / worker 的分工，但没有 NEO 这样显式的“异构调度 + 双 sub-batch + CPU decode”闭环。

---

## 3. 创新点一：CPU Offloading 在代码中的实现

这里最重要的一点是：

**NEO 的 CPU Offloading 不是“把 KV cache 暂时存到 CPU”这么简单，而是“把部分请求的 KV 状态与 decode attention 真正迁入 CPU 路径”。**

这件事分四层实现：

1. 请求抽象层：哪些请求属于 CPU 路径？
2. 调度层：什么时候把请求送去 CPU？
3. 存储/映射层：CPU 上怎么管理这些 blocks 和 buffers？
4. 执行层：CPU 上的 attention 到底在哪里算？

---

### 3.1 抽象层：`SubBatch` 是 NEO 的第一关键改动

NEO 的 `swiftllm/structs.py` 新增了 `BatchPerfData` 与 `SubBatch`。其中最关键的是 `SubBatch`（`swiftllm/structs.py:211`）。

`SubBatch` 把请求分成四类：

- `gprf_reqs`：GPU-prefill 请求
- `cprf_reqs`：CPU-prefill 路径请求
- `gdec_reqs`：GPU-decode 请求
- `cdec_reqs`：CPU-decode 请求

对应接口：
- `add_pref(req, is_gpu)`：`swiftllm/structs.py:226`
- `add_gdec(req)`：`swiftllm/structs.py:239`
- `add_cdec(req)`：`swiftllm/structs.py:243`
- `set_model_forward_args(...)`：`swiftllm/structs.py:254`

#### 这层改动的意义

base-swiftLLM 的调度抽象比较粗：主要是 waiting / running / swapped 三种状态，见 `base-swiftLLM/swiftllm/server/scheduler.py:44-47`。

而 NEO 把“请求处于哪个阶段、接下来由谁算”编码进了 batch 结构本身。

这很重要，因为后面 worker 的执行逻辑需要知道：

- 哪些 token 属于 prefill；
- 哪些 decoding 请求还在 GPU；
- 哪些 decoding 请求已经移到 CPU；
- 哪些 prefill 请求虽然本轮在 GPU 上完成 prompt attention，但其 KV 最终应转移到 CPU 驻留路径。

换句话说，**`SubBatch` 不是简单的数据容器，而是 NEO 整个异构执行模型的载体。**

---

### 3.2 调度层：Scheduler 决定哪些请求转到 CPU

NEO 的调度器在 `swiftllm/server/scheduler.py` 中。相比 base，最本质的新结构是：

- `waiting_q`：等待进入系统的请求
- `gpu_decoding_q`：当前由 GPU decode 的请求
- `cpu_decoding_q`：当前由 CPU decode 的请求

定义位置：`swiftllm/server/scheduler.py:107-109`。

而 base 只有：
- `waiting_q`
- `running_q`
- `swapped_q`

定义位置：`base-swiftLLM/swiftllm/server/scheduler.py:44-47`。

#### 这说明什么？

在 base 中，请求被 swap 到 CPU 后，本质上只是“暂时不在 GPU 上跑”，CPU 主要承担存储角色。

而在 NEO 中，`cpu_decoding_q` 是调度器的一等公民，意味着：

**有些请求不是“临时被放到 CPU 上等着”，而是明确被调度成“由 CPU 承担 decode attention 的请求”。**

#### NEO 如何做这件事？

在 `_get_next_batch_new()` 中（`swiftllm/server/scheduler.py:237`）：

1. 先根据当前 GPU block 使用情况决定是否把部分 GPU decoding 请求抢占到 CPU（`swiftllm/server/scheduler.py:264-272`）
2. 若 GPU 有余量，则尝试把一些 CPU decoding 请求 swap 回 GPU（`swiftllm/server/scheduler.py:273-284`）
3. 然后尝试接纳新的 prefill 请求，并决定这些 prefill 请求是：
   - `pref_to_gpu`
   - `pref_to_cpu`
   见 `swiftllm/server/scheduler.py:286-307`
4. 最后调用 `_decide_mode_and_gen_batch(...)` 组织成 1 个或 2 个 `SubBatch`（`swiftllm/server/scheduler.py:309`）

#### 这里的关键不是“有 swap”，而是“有 CPU 路径”

base 也有 swap，但它的核心问题是“GPU 不够就把一部分请求挪走”。

NEO 则进一步问：

- 哪些请求继续留在 GPU decode 更划算？
- 哪些请求放到 CPU decode 更划算？
- 哪些新请求可以先 prefill，再把其状态放到 CPU，以便后续由 CPU decode？

这已经不再是单纯的内存管理，而是**跨设备负载分配**。

---

### 3.3 `BatchPerfData` + `PerfPredictor`：CPU Offloading 不是静态规则，而是负载感知调度

NEO 的 `BatchPerfData` 在 `swiftllm/structs.py:148` 定义。它记录的不是日志，而是调度器需要的性能量：

- `linr_T`：线性层 / 其它非 attention 主体计算时间（`swiftllm/structs.py:193-196`）
- `pref_T`：GPU prefill 时间
- `gdec_T`：GPU decode attention 时间
- `cdec_T`：CPU decode attention 时间（`swiftllm/structs.py:197-199`）
- `lnch_T`：launch / Python 开销

这些值来自 `PerfPredictor` 接口：`swiftllm/perfpredictor.py:7-44`。

具体实现 `TablePerfPredictor` 在 `swiftllm/perfpredictor.py:70` 开始，支持：

- `get_linr_T(S)`：`swiftllm/perfpredictor.py:160`
- `get_pref_T(S)`：`swiftllm/perfpredictor.py:166`
- `get_gdec_T(N)`：`swiftllm/perfpredictor.py:172`
- `get_cdec_T(S, N)`：`swiftllm/perfpredictor.py:178`
- `get_lnch_T()`：`swiftllm/perfpredictor.py:195`

这里最值得注意的是：

### `get_cdec_T(S, N)` 是二维的

也就是说，CPU decode 时间不是只看 token 数 `N`，而是同时看：

- `S`：CPU decoding 的请求数 / iteration width 相关规模
- `N`：CPU decoding 的总 token 规模

这说明 NEO 对 CPU 路径不是粗略估计，而是当作一条真实可建模的执行路径来看待。

#### 调度器如何使用这些预测值？

NEO 用 `_get_remains()` 估计双 sub-batch 下 CPU 是否会成为瓶颈：

`swiftllm/server/scheduler.py:132-140`

```python
return [
    batches[j^1].perfdata.linr_T +
    batches[j].perfdata.pref_T +
    batches[j].perfdata.gdec_T -
    batches[j].perfdata.cpu_time
    for j in range(2)
]
```

它的含义可以理解为：

- GPU 这一侧的剩余可重叠时间是多少；
- CPU decode 是否能被这段时间掩蔽掉；
- 如果掩蔽不住，说明 CPU 会拖后腿。

因此，**NEO 的 CPU Offloading 不是“能 offload 就 offload”，而是“只有在预测上值得时，才把请求放进 CPU 路径”。**

---

### 3.4 BlockManager：为 CPU 路径准备 block 映射与 swap 编排

NEO 的 control-plane block manager 在 `swiftllm/server/block_manager.py`。

这里的关键变化是：

1. 它同时维护 GPU 和 CPU 的 `DeviceBlockManager`
2. `prepare(...)` 会在每次 forward 前把本轮所有 block 映射、swap 参数都先算好
3. worker 端不再像 base 那样自己主导 block 生命周期，而是接收 control plane 预先编排好的结果

#### `DeviceBlockManager`

定义于 `swiftllm/server/block_manager.py:14`。

其内部维护：
- `seq_num_blks`：每个请求当前分到多少 block（`swiftllm/server/block_manager.py:36-41`）
- `block_table`：虚拟 block id 到物理 block id 的映射（`swiftllm/server/block_manager.py:42-47`）
- `is_block_free`：每个 split 的空闲块表（`swiftllm/server/block_manager.py:48-54`）

注意：这里已经变成了**CPU 上的控制结构**，而不是 base 中那种偏 GPU 端 kernel 驱动的 block 表操作。

#### `BlockManager.prepare(...)`

核心函数在 `swiftllm/server/block_manager.py:195`。

它按三步组织一轮 forward：

1. **先做常规 swap**（`swiftllm/server/block_manager.py:224-230`）
2. **给本轮 batch 分配 blocks，并让 batch 生成 forward 所需参数**（`swiftllm/server/block_manager.py:232-248`）
3. **对 cprf 请求做额外的 swap 编排**（`swiftllm/server/block_manager.py:250-259`）

这第三步尤其关键，因为它说明：

**NEO 不只是处理已有 CPU decode 请求，还在本轮 prefill 完成后，就为后续 CPU 驻留路径准备好 KV 搬运。**

#### `cprf` 是什么意思？

这是一个很容易讲错的点。

`cprf` **不是**“prompt attention 在 CPU 上算”。

相反，`cprf` 请求在本轮仍然会参与 prefill attention，只是它们的 KV 最终会被安排到 CPU 路径，以便后续 decode 阶段可以走 CPU。

这一点从 `SubBatch.set_model_forward_args(...)` 可以看出来：
- `all_reqs = cprf + gprf + gdec + cdec`（`swiftllm/structs.py:272`）
- `num_prefs = num_cprfs + num_gprfs`（`swiftllm/structs.py:269`）

也就是说，`cprf` 仍然属于本轮 prefill 请求集合。

---

### 3.5 `Swapper`：CPU Offloading 的 worker 端承载层

NEO 的 `swiftllm/worker/block_swapper.py` 是 CPU Offloading 的另一个核心模块。

虽然用户点名的是 `model.py`，但如果不看 `Swapper`，很难真正理解 NEO 的 CPU 路径。

结合前面的阅读可知，`Swapper` 至少承担了四类对象：

1. GPU KV cache
2. CPU KV swap 空间
3. CPU 侧 Q/K/V/O pinned buffers
4. GPU / CPU block tables

从 `model.py` 看，`LlamaModel.init_kvcache_and_swap(...)` 会创建 `Swapper`（`swiftllm/worker/model.py:195-203`），并把它注入每一层（`swiftllm/worker/model.py:205-206`）。

这说明：

**NEO 的每一层 attention 都能访问同一套 GPU/CPU cache、CPU buffer 和 block table。**

这正是后面 `_transfer_qkv()`、`paged_attention_cpu(...)`、`swap_blocks(...)` 能协同工作的基础。

---

### 3.6 真正的 CPU decode attention 在哪里发生？

真正的 CPU 计算路径在 `swiftllm/worker/layers/transformer_layer.py`。

这一层里最关键的两个函数是：

- `_transfer_qkv(...)`：`swiftllm/worker/layers/transformer_layer.py:158`
- `_attention(...)`：`swiftllm/worker/layers/transformer_layer.py:258`

#### 第一步：GPU 先算出 Q/K/V

在 `_preproj(...)` 中：
- 做 RMSNorm
- 线性投影得到 `q/k/v`
- 做 rotary embedding

见 `swiftllm/worker/layers/transformer_layer.py:199-255`。

#### 第二步：把 CPU decode 对应的 Q/K/V 传到 CPU pinned buffer

`_transfer_qkv(...)` 里，如果 `batch.num_cdecs > 0`，会把最后这部分 CPU decode 请求对应的 `q/k/v` 复制到：

- `self.swapper.q_cpu`
- `self.swapper.k_cpu`
- `self.swapper.v_cpu`

对应代码：`swiftllm/worker/layers/transformer_layer.py:169-178`。

而且这一步是放在 `cpu_communication_stream` 上异步做的，不是阻塞默认计算流。

#### 第三步：CPU 上实际执行 attention

在 `_attention(...)` 中，如果 `batch.num_cdecs > 0`，会调用：

`torch.ops.pacpu.paged_attention_cpu(...)`

见 `swiftllm/worker/layers/transformer_layer.py:331-349`。

这一步用到的输入包括：

- CPU 上的 Q/K/V：`self.swapper.q_cpu / k_cpu / v_cpu`
- CPU 上的换出 KV：`self.swapper.k_swap / v_swap`
- CPU 侧 block table：`self.swapper.cpu_block_table`
- 输出缓冲区：`oc`

然后把 CPU attention 的输出 `oc` 异步拷回 GPU 上的 `o[-batch.num_cdecs:]`：

`swiftllm/worker/layers/transformer_layer.py:351-352`

#### 这就是 NEO CPU Offloading 的真正落点

也就是说，NEO 的 CPU Offloading 不是只“把 KV 挪到 CPU”；而是：

1. GPU 继续完成线性层和部分 attention 相关准备
2. CPU 为 `cdec` 请求执行 `paged_attention_cpu`
3. 输出再回到 GPU，继续后续投影和层间传递

这是一条**真实存在的跨设备分工执行路径**。

---

### 3.7 对照 base：为什么说 base 还没有 NEO 这种 CPU Offloading？

base-swiftLLM 当然也用了 CPU 内存。

例如在 `base-swiftLLM/swiftllm/worker/model.py` 中：
- `k_swap / v_swap` 在 CPU 上初始化（`base-swiftLLM/swiftllm/worker/model.py:150-159`）
- `swap_in_seqs()` / `swap_out_seqs()` 会调用 `swiftllm_c.swap_blocks(...)` 做 GPU 与 CPU 之间的块搬运（`base-swiftLLM/swiftllm/worker/model.py:361-399`）

但 base 的 CPU 路径本质上还是：

- 被换出的 KV 的存储位置
- 将来再换回 GPU 的中转空间

base 并没有像 NEO 那样：

- 显式维护 `cpu_decoding_q`
- 在 batch 中区分 `cdec`
- 将 CPU attention 建模为独立代价 `cdec_T`
- 在 layer 中调用 `paged_attention_cpu(...)`

所以一定要把这两件事分清楚：

- **base：CPU swap / CPU 存储**
- **NEO：CPU offloading = CPU 存储 + CPU decode attention 计算**

这正是论文里 CPU Offloading 的代码实质。

---

## 4. 创新点二：Asymmetric GPU-CPU Pipelining 在代码中的实现

第二个创新点更容易被误解成“NEO 就是一次跑两个 batch”。

其实并不是。

NEO 的关键不是“双 batch”本身，而是：

1. 这两个 batch 是**异构的**，里面混有不同类型请求（gprf/cprf/gdec/cdec）
2. 调度器不是固定返回两个 batch，而是预测后选择单 batch 或双 batch
3. worker 不是两个 batch 各跑一遍，而是在**层内部细粒度交错**

因此更准确地说，NEO 的创新点二应该理解为：

**由调度器构造异构 `SubBatch`，再由 worker 用分阶段层执行把 GPU 计算、CPU 通信和 CPU attention 重叠起来。**

---

### 4.1 pipeline 不是默认模式，而是调度器比较后再决定

NEO 的 `_decide_mode_and_gen_batch(...)` 在 `swiftllm/server/scheduler.py:142`。

它先构造：

- `batches = [SubBatch(...), SubBatch(...)]`
- `gpu_only_batch = SubBatch(...)`

见 `swiftllm/server/scheduler.py:158-159`。

然后：

1. 把所有 pref + gdec 先放进第一批（`swiftllm/server/scheduler.py:161-173`）
2. 基于预测结果，把 CPU decode 请求在两个 batch 之间拆分（`swiftllm/server/scheduler.py:184-204`）
3. 进一步调整 prefilling 数量，以避免 CPU 长时间空闲或过载（`swiftllm/server/scheduler.py:217-223`）
4. 最后比较 sequential 与 pipeline 的吞吐预测：
   - `seqential_time` / `seqential_rate`
   - `pipelined_time` / `pipelined_rate`
   见 `swiftllm/server/scheduler.py:224-234`

关键判断是：

```python
if seqential_rate < pipelined_rate:
    return batches
else:
    return [gpu_only_batch]
```

这说明：

**pipeline 不是硬编码执行模式，而是一个“只有在预测吞吐更高时才启用”的策略。**

这也解释了为什么 NEO 需要前面的 `ModelProfiler` / `TablePerfPredictor`：

没有这套性能建模，调度器就没法做 mode selection。

---

### 4.2 `LlamaModel`：worker 从单 batch forward 演进为支持双 sub-batch 的执行后端

NEO 的 `LlamaModel` 在 `swiftllm/worker/model.py`。

最关键的几个函数是：

- `do_one_iteration(...)`：`swiftllm/worker/model.py:333`
- `_forward_batches(...)`：`swiftllm/worker/model.py:297`
- `_forward_sequential(...)`：`swiftllm/worker/model.py:264`
- `_forward_pipeline(...)`：`swiftllm/worker/model.py:278`

#### `do_one_iteration(...)`

这是一轮 worker 执行的统一入口。它按顺序做：

1. 如果有 `swapper`，先设置 block tables（`swiftllm/worker/model.py:349-350`）
2. 如果这轮有 swap，就在 `cpu_communication_stream` 上发起 `swap_blocks(...)`（`swiftllm/worker/model.py:352-356`）
3. 最后调用 `_forward_batches(...)`（`swiftllm/worker/model.py:357`）

这说明 NEO 已经把“块映射更新 + swap + forward”统一到了每一轮迭代的单一接口里。

#### `_forward_batches(...)`

在 `swiftllm/worker/model.py:314-319` 中：

- `len(batches) == 1`：走 `_forward_sequential(...)`
- `len(batches) == 2`：走 `_forward_pipeline(...)`

这说明 NEO worker 已经天然支持两种执行模式。

而 base 的 worker 还是传统单 batch `forward(...)` 路径，见 `base-swiftLLM/swiftllm/worker/model.py:252-359`。它没有 `SubBatch`，没有 pipeline stage，也没有双 batch 执行接口。

#### `_forward_pipeline(...)`

NEO 的 pipeline 入口在 `swiftllm/worker/model.py:278`：

1. 先让最后一层对象调用 `forward_first_stage(...)`，拿到第二个 batch 后续要用的 `q1, k1, v1`（`swiftllm/worker/model.py:284`）
2. 中间层循环调用 `forward_double(...)`（`swiftllm/worker/model.py:289-290`）
3. 最后调用 `forward_last_stage(...)`，得到最终 embeddings（`swiftllm/worker/model.py:293`）

这不是两个 batch 各自执行完整 forward，而是**把两个 batch 在层间“穿插”起来**。

---

### 4.3 真正的流水化发生在单层内部：`TransformerLayer` 被拆成三段

NEO 的单层设计在 `swiftllm/worker/layers/transformer_layer.py` 中。

相比传统“layer.forward 一把梭”，NEO 实际把一层拆成了：

1. `_preproj(...)`：`swiftllm/worker/layers/transformer_layer.py:199`
2. `_attention(...)`：`swiftllm/worker/layers/transformer_layer.py:258`
3. `_postproj(...)`：`swiftllm/worker/layers/transformer_layer.py:358`

#### 为什么这很重要？

因为只有拆成这三段，系统才能把：

- batch A 的 `postproj`
- batch A 在下一层的 `preproj`
- batch B 的 `attention`

交错起来做。

这就是 NEO 所谓的 **asymmetric pipeline**：

- 不是两个 batch 对称地一层一层推进；
- 而是一个 batch 在做后投影/下一层预投影时，另一个 batch 在做 attention；
- 再加上 CPU 通信和 CPU decode attention 的异步执行，形成设备与阶段的重叠。

---

### 4.4 `next_layer_weight`：为什么 NEO 的层对象里会保存“下一层的权重”？

这是 NEO 一个非常不寻常、但非常关键的设计。

在 `LlamaTransformerLayer.__init__(...)` 中，除了当前层 `weight`，还额外接收了 `next_layer_weight`：

`swiftllm/worker/layers/transformer_layer.py:110-123`

而在 `LlamaModel` 构造层列表时，也确实给每层传入了当前层和“下一层”的权重：

`swiftllm/worker/model.py:168-178`

#### 这说明什么？

因为 pipeline 内部需要在“当前层对象”里同时做：

- 当前层的 `postproj`
- 下一层的 `preproj`

在 `_preproj(...)` 里就能看到：

```python
weight = self.weight if not layer_off else self.next_layer_weight
```

位置：`swiftllm/worker/layers/transformer_layer.py:208`

这正是为了支持 `_forward_pipeline_stage(...)` 中的交错：

- 对 batch0：做当前层的 `postproj[i]`，紧接着做下一层的 `preproj[i+1]`
- 对 batch1：同时做当前/偏移层的 `attention`

见 `_forward_pipeline_stage(...)`：`swiftllm/worker/layers/transformer_layer.py:397-427`。

所以，**`next_layer_weight` 不是多余缓存，而是 NEO 实现层间流水衔接的关键。**

---

### 4.5 双 batch 流水到底怎么交错？

NEO 在 layer 内有三个重要的 pipeline 函数：

- `forward_first_stage(...)`：`swiftllm/worker/layers/transformer_layer.py:451`
- `forward_double(...)`：`swiftllm/worker/layers/transformer_layer.py:430`
- `forward_last_stage(...)`：`swiftllm/worker/layers/transformer_layer.py:477`

#### `forward_first_stage(...)`

语义是：

- batch0：先做 `preproj -> attention`
- batch1：先做 `preproj`

函数注释写得很清楚：`swiftllm/worker/layers/transformer_layer.py:456-460`

也就是说，流水线启动时，先把第一个 batch 推进到 attention，再把第二个 batch 的 Q/K/V 准备好。

#### `forward_double(...)`

内部调用两次 `_forward_pipeline_stage(...)`，分别处理两个 stage：

`swiftllm/worker/layers/transformer_layer.py:445-446`

它完成的是一种交错：

- batch0：`postproj[i] -> preproj[i+1]`
- batch1：`attention[i]`

然后再交换角色执行一次。

这就是论文里所谓的“不对称”本质：

- 两个 batch 在同一时刻做的不是同一种工作；
- GPU/CPU 资源利用也不是对称划分；
- attention、QKV 传输、CPU attention、swap 都被插入到不同阶段。

#### `forward_last_stage(...)`

流水线尾声时：

- batch0：只剩 `postproj`
- batch1：执行 `attention -> postproj`

见注释：`swiftllm/worker/layers/transformer_layer.py:484-489`

最终再把两个 batch 的输出拼起来：

`swiftllm/worker/layers/transformer_layer.py:499-501`

---

### 4.6 为什么它叫 GPU-CPU pipeline，而不是纯 GPU pipeline？

因为 NEO 的 pipeline 不只 overlap 两个 batch 的 GPU 计算，还 overlap 了：

1. GPU 默认流上的计算
2. `cpu_communication_stream` 上的数据传输 / swap
3. CPU 上的 `paged_attention_cpu(...)`

例如：

- `_transfer_qkv(...)` 会在 `cpu_communication_stream` 上启动 GPU→CPU 的 QKV 传输（`swiftllm/worker/layers/transformer_layer.py:169-178`）
- `_attention(...)` 会在同步点之后执行 CPU attention，再把输出拷回 GPU（`swiftllm/worker/layers/transformer_layer.py:333-355`）
- `_swap_out_blocks(...)` 也会在通信流上启动块换出（`swiftllm/worker/layers/transformer_layer.py:181-196`）

因此这里的“流水线”不是只看神经网络层序，而是看：

- 不同 batch 的不同阶段
- GPU 计算与 CPU 通信
- GPU attention 与 CPU attention

如何在时间轴上尽量重叠。

这才是 **Asymmetric GPU-CPU Pipelining** 在代码中的真实含义。

---

## 5. `ModelEvents` 与 `ModelPerfResult` 到底是干什么的？

这是用户点名问到的地方，也是 NEO 最容易“看起来像附属类、其实是主线一部分”的地方。

我先给结论：

- `TransformerEvents`：**层级别**的性能分解
- `ModelEvents`：**整次 forward 级别**的阶段边界
- `ModelPerfResult`：把上面两者汇总成 profiling 结果

它们不是为了功能正确性，而是为了让系统能：

**profile → 建表 → 预测 → 调度**

也就是说，它们服务于 NEO 的“负载感知调度”和“模式选择”。

---

### 5.1 `TransformerEvents`：每一层内部的细粒度计时

`TransformerEvents` 定义于 `swiftllm/worker/layers/transformer_layer.py:33`。

它会记录：

- `stage_s`
- `linr_e`
- `pref_e`
- `gdec_e`
- `qkvtr_e`
- `lnch_s / lnch_m / lnch_e`
- `cdec_s / cdec_e`

这些被进一步暴露为：

- `linr_time`：`swiftllm/worker/layers/transformer_layer.py:50-55`
- `pref_time`：`swiftllm/worker/layers/transformer_layer.py:57-62`
- `gdec_time`：`swiftllm/worker/layers/transformer_layer.py:64-69`
- `cdec_time`：`swiftllm/worker/layers/transformer_layer.py:71-76`
- `lnch_time`：`swiftllm/worker/layers/transformer_layer.py:78-83`

#### 它为什么重要？

因为 NEO 不只是想知道“这一层总共跑了多久”，而是想知道：

- 非 attention 线性部分多久
- GPU prefill 多久
- GPU decode 多久
- CPU decode 多久
- launch / Python 开销多久

没有这种分解，调度器就无法知道：

- CPU decode 能不能被 GPU 侧时间掩蔽
- pipeline 模式到底值不值得开

---

### 5.2 `ModelEvents`：整次 forward 的宏观阶段边界

`ModelEvents` 定义于 `swiftllm/worker/model.py:26`。

它记录的是整次 forward 的几个边界事件：

- `frwd_s`
- `fstg_s`
- `mnbd_s`
- `mnbd_e`
- `lstg_e`
- `frwd_e`

定义见：`swiftllm/worker/model.py:31-38`

它的使用方式非常简单：

- `pf_record(name)` 只有在 `monitor_performance` 打开时才真正 `record()`（`swiftllm/worker/model.py:40-45`）

在 `_forward_batches(...)` 中，这些事件被插在整轮 forward 的关键边界上：

- forward 开始：`frwd_s`（`swiftllm/worker/model.py:307`）
- pre-layer 完成：`fstg_s`（`swiftllm/worker/model.py:312`）
- 主体层计算区间：`mnbd_s` / `mnbd_e`（顺序或流水内部记录）
- last-stage 结束：`lstg_e`（`swiftllm/worker/model.py:320`）
- post-layer 结束：`frwd_e`（`swiftllm/worker/model.py:323`）

#### 它不是业务状态类

很多人第一次看会疑惑：

> 为什么 worker/model.py 里要加这么个事件容器？

答案是：

**因为 NEO 不只要知道“每一层各部分多久”，还要知道“整次 forward 被粗分成哪几个大阶段，各阶段多久”。**

这正是 `ModelPerfResult` 需要的全局边界信息。

---

### 5.3 `ModelPerfResult`：把层级别与全局级别性能汇总起来

`ModelPerfResult` 定义于 `swiftllm/worker/model.py:48`。

它会在初始化时：

1. `torch.cuda.synchronize()`，确保事件都已完成（`swiftllm/worker/model.py:67`）
2. 从各层 `TransformerEvents` 收集：
   - `linr_times`
   - `pref_times`
   - `gdec_times`
   - `cdec_times`
   - `lnch_times`
   见 `swiftllm/worker/model.py:68-79`
3. 再利用 `ModelEvents` 计算整次 forward 的几个大阶段时间：
   - `prlr_time`
   - `fstg_time`
   - `mnbd_time`
   - `lstg_time`
   - `polr_time`
   见 `swiftllm/worker/model.py:81-85`
4. 再求各类平均值：
   - `avg_linr_time`
   - `avg_pref_time`
   - `avg_gdec_time`
   - `avg_cdec_time`
   - `avg_lnch_time`
   见 `swiftllm/worker/model.py:87-91`

#### 为什么它要区分 `use_pipline`？

因为 sequential 和双 batch pipeline 下，每层事件的解释方式不同。

如果是 pipeline，就要按两个 stage 的视角分别收集和对齐各层时间；如果是 sequential，就只要把同层两个 event 槽位的信息汇总即可。

这也说明：

**`ModelPerfResult` 不是为了 debug 打印，而是为了统一抽象不同执行模式下的性能观测结果。**

---

### 5.4 `ModelPerfResult` 的真实去向：不是打印给人看，而是喂给 profiler / predictor

这一点是理解这两个类最关键的地方。

在 `swiftllm/server/profiler.py` 中：

- `ModelProfiler` 直接 import 了 `ModelPerfResult`（`swiftllm/server/profiler.py:21`）
- `_run_test_case(...)` 会在人工构造的请求上执行多次 `executor.do_one_iteration(...)`（`swiftllm/server/profiler.py:79-134`）
- 中间会开启/关闭 perf monitor：
  - `turn_on_perf_monitor()`（`swiftllm/server/profiler.py:124-125`）
  - `turn_off_perf_monitor_and_flush_results()`（`swiftllm/server/profiler.py:133-134`）

然后 profiler 分别生成：

- `linr` 曲线：`_profile_linr(...)`，取 `avg_linr_time`（`swiftllm/server/profiler.py:136-187`）
- `pref` 曲线：`_profile_pref(...)`，取 `avg_pref_time`（`swiftllm/server/profiler.py:189-230`）
- `gdec` 曲线：`_profile_gdec(...)`，取 `avg_gdec_time`（`swiftllm/server/profiler.py:232-274`）
- `cdec` 曲面：`_profile_cdec(...)`，取 `avg_cdec_time`（`swiftllm/server/profiler.py:276-355`）
- `lnch` 开销：`_profile_lnch(...)`，取 `avg_lnch_time`（`swiftllm/server/profiler.py:357-400`）

接着 `init_profile_tables(...)` 把这些结果写回 `TablePerfPredictor`：

`swiftllm/server/profiler.py:41-55`

再然后 `AsyncEngine.initialize_async()` 初始化 scheduler 时把 predictor 传进去：

`swiftllm/server/engine.py:111-115`

因此完整链路是：

**TransformerEvents / ModelEvents → ModelPerfResult → ModelProfiler → TablePerfPredictor → Scheduler**

这就回答了用户的问题：

### `ModelEvents` / `ModelPerfResult` 为什么会出现在 `worker/model.py`？

因为 NEO 不仅要“执行推理”，还要“让调度器知道每种执行方式大概多贵”。

而要做到这点，就必须在 worker 端直接量测：

- 每层的线性、prefill、GPU decode、CPU decode、launch 开销
- 整次 forward 的阶段边界

所以这两个类虽然看起来像“分析工具”，但实际上是 NEO 调度系统的一部分，而不是外围附属类。

---

## 6. base-swiftLLM 与 NEO 的模块级对照

下面按模块总结“base 是什么 / NEO 改成什么 / 目的是什么”。

| 模块 | base-swiftLLM | NEO 改动 | 改动目的 |
|---|---|---|---|
| `structs.py` | 主要以普通请求/批次表达 forward | 新增 `BatchPerfData`、`SubBatch`，显式区分 `gprf/cprf/gdec/cdec` | 让调度器和 worker 都能理解异构执行路径 |
| `server/scheduler.py` | waiting/running/swapped 三队列，单 batch FCFS，GPU 不够就 swap | 新增 `gpu_decoding_q`、`cpu_decoding_q`、`_decide_mode_and_gen_batch()`、mode selection | 把 CPU 变成真实 decode 设备，并在 sequential/pipeline 之间自适应选择 |
| `server/block_manager.py` | block 生命周期更偏 worker 侧自治 | control plane 统一维护 CPU/GPU block 映射，每轮 `prepare(...)` 先编排 swap 与分配 | 把异构内存与 swap 管理前移，worker 只执行 |
| `server/engine.py` | 常规 engine 驱动 worker forward | 引入 profiler、predictor、scheduler、block_manager 的完整闭环 | 实现“先 profile，再调度”的系统逻辑 |
| `worker/model.py` | 单 batch forward，CPU 主要是 swap 空间 | 新增 `ModelEvents`、`ModelPerfResult`、`do_one_iteration()`、`_forward_pipeline()` | 支持双 sub-batch 流水执行，并为调度器提供性能观测 |
| `worker/layers/transformer_layer.py` | 主要是常规层 forward | 拆成 `preproj/attention/postproj`，支持 `paged_attention_cpu`、QKV 传输、双 stage 流水 | 落地 CPU decode + 非对称 pipeline |
| `worker/block_swapper.py` | 无 NEO 这种集中式 swapper 抽象 | 同时管理 GPU KV、CPU KV、CPU Q/K/V/O buffers、block tables | 为跨设备 attention 与 swap 提供统一承载层 |
| `perfpredictor.py` + `server/profiler.py` | 无完整在线性能建模链 | 新增 profile tables、插值预测、CPU/GPU 各阶段建模 | 支撑 load-aware scheduling 和 mode selection |

---

## 7. 还需要特别澄清的几个概念

### 7.1 `cprf` 不是“CPU 上执行 prefill”

这是最容易误读的点。

`cprf` 在本轮仍然属于 `num_prefs`，因此其 prompt attention 仍参与 prefill 路径；它更准确的含义是：

**这些请求本轮做 prefill，但其 KV 会被导向 CPU 驻留路径，方便后续 decode 阶段走 CPU。**

所以它是“CPU-resident prefill path”，而不是“CPU prefill compute”。

### 7.2 `ModelEvents` / `ModelPerfResult` 不是“推理功能类”，而是“调度支撑类”

如果只看名字，很容易以为它们是为了 debug 性能；
但从 `profiler.py -> perfpredictor.py -> scheduler.py` 的链路看，它们实际上支撑了 NEO 的核心调度逻辑。

### 7.3 `always_use_gpu` 是回退路径

在 `Scheduler.get_next_batch()` 中：
- 若 `always_use_gpu` 为真，就走 `_get_next_batch_old()`（`swiftllm/server/scheduler.py:393-402`）
- 否则走 NEO 新路径 `_get_next_batch_new()`

也就是说，这个选项本质上是退回更接近 base 的 GPU-only 调度路径。

### 7.4 `extra_layer_for_cprf` 不是小优化，而是影响 block split 的结构性开关

`DeviceBlockManager` 里：

```python
nsplits = 1 + engine_config.extra_layer_for_cprf
```

位置：`swiftllm/server/block_manager.py:34`

说明它会影响 block 管理的 split 数量；而在 layer 的 `_preproj()` / `_swap_out_blocks()` 中，它又会影响 cprf 的 KV 存放层位选择。

因此它不是“可有可无的小参数”，而是 NEO 对 cprf 路径进行中间态管理的重要设计开关。

---

## 8. 最终总结：NEO 的改动主线应该怎样理解？

如果把 NEO 的所有改动压缩成一条逻辑主线，我认为可以这样表述：

### 第一步：把请求从“只分运行状态”改成“显式区分设备执行路径”

通过 `SubBatch`、`BatchPerfData`、`cpu_decoding_q` 等结构，NEO 让系统能表达：
- 谁在 GPU 上 prefill
- 谁在 GPU 上 decode
- 谁在 CPU 上 decode
- 哪些 prefill 请求之后会转入 CPU 路径

### 第二步：把 CPU 从“swap 仓库”升级成“真实计算设备”

通过 `Swapper`、CPU block table、Q/K/V/O pinned buffers 和 `paged_attention_cpu(...)`，NEO 让 CPU 能真正承担 decode attention。

### 第三步：把 worker 从“单 batch 顺序执行器”改造成“异构双 sub-batch 流水执行器”

通过 `do_one_iteration()`、`_forward_pipeline()`、`forward_double()` 等逻辑，NEO 把两个 batch 的不同阶段与 GPU/CPU 的不同工作重叠起来。

### 第四步：为了让这些选择不是拍脑袋，新增性能建模链

通过 `TransformerEvents`、`ModelEvents`、`ModelPerfResult`、`ModelProfiler` 和 `TablePerfPredictor`，NEO 能先 profile 出不同阶段成本，再让 scheduler 做 load-aware scheduling 和 mode selection。

---

## 9. 用一句更学术化但也更准确的话总结 NEO 相对 base 的本质改动

**base-swiftLLM 是一个以 GPU 为唯一计算中心、CPU 主要承担换出存储的推理系统；而 NEO 将 CPU 提升为参与在线推理 decode attention 的协同计算设备，并围绕这一点重构了请求抽象、block 管理、worker 执行与性能建模，使系统能够在 sequential 与 asymmetric GPU-CPU pipeline 之间自适应选择，从而用 CPU 资源换取更大的有效 GPU batch 与更高吞吐。**
