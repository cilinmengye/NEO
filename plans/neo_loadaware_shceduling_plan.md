# NEO 论文 §3.2 Load-Aware Scheduling 复现笔记

## 1. 目标与范围

本文档只聚焦 NEO 论文中的 **Load-Aware Scheduling**，并把三层内容串起来：

1. 论文 §3.2 的核心思想；
2. 附录 **A PSEUDO CODE FOR LOAD-AWARE SCHEDULING** 的主流程；
3. NEO 仓库真实实现里调度器、block manager、worker 的执行路径。

最终目标不是复述论文，而是建立一个可落地的复现心智模型：

- 请求在系统里如何流动；
- 调度器每轮 iteration 到底在决定什么；
- 为什么要在 sequential 和 pipelined 之间切换；
- 为什么 CPU/GPU swap、prefill admission、CPU decode split 必须一起考虑；
- 真实代码相比论文伪代码多了哪些工程化 heuristic；
- 这种调度方式的局限在哪里。

---

## 2. 一句话理解 Load-Aware Scheduling

NEO 的 load-aware scheduling 不是“把请求排个队”这么简单，而是：

> 在每一轮 iteration 里，根据当前 GPU decode 负载、CPU decode 负载、KV block 容量、batch/token 预算，以及 profile 得到的耗时模型，贪心地决定：
> - 哪些请求继续留在 GPU；
> - 哪些请求要 swap out 到 CPU；
> - 哪些 CPU 请求值得 swap back 到 GPU；
> - 新到的 prefill 请求应该进 GPU 还是 CPU；
> - 本轮是跑单 batch 的 sequential mode，还是跑双 sub-batch 的 pipelined mode。

所以它本质上是一个 **profile-guided、per-iteration greedy scheduler**，而不是静态规则，也不是全局最优求解器。

---

## 3. 论文里的核心问题：为什么需要这种调度

如果只有 GPU decoding，而不做 CPU offloading，系统会很快碰到两个问题：

1. **KV cache 容量不够**
   - 长上下文或并发请求多时，GPU block 很快被占满。
2. **资源使用不均衡**
   - 有些请求适合留在 GPU；
   - 有些请求虽然还能继续服务，但若继续挤占 GPU，会让新请求 prefill 无法进入，整体吞吐反而变差。

NEO 的思路不是简单地“优先 decode”或“优先 prefill”，而是把系统看成一个异构执行流水：

- GPU 擅长做 linear + prefill attention + GPU paged attention；
- CPU 可以承担部分 decoding attention；
- 两边如果能 overlap，总吞吐会提高；
- 但 overlap 并不总是成立，所以需要 scheduler 每轮判断。

因此 Load-Aware Scheduling 的核心目标是：

- 在 **内存受限** 的前提下保住更多活跃请求；
- 在 **CPU/GPU 异构执行** 的前提下提高吞吐；
- 在 **prefill 与 decode 竞争资源** 时做更好的折中；
- 让 pipeline 只在“确实划算”时才启用。

---

## 4. 论文附录伪代码的主流程重建

虽然论文附录是抽象伪代码，但结合实现可以把它概括成下面 6 步。

### 4.1 维护三类请求集合

调度器持续维护三类核心状态：

- `waiting_q`：还没正式进入执行系统的新请求；
- `gpu_decoding_q`：当前在 GPU decode 路径上的请求；
- `cpu_decoding_q`：当前在 CPU decode 路径上的请求。

真实代码就在 `swiftllm/server/scheduler.py:106-118` 初始化这些结构。

### 4.2 先看当前 GPU decode 还能不能承受

每一轮 iteration 先把当前 `gpu_decoding_q` 当作“默认应该继续服务”的集合，然后检查：

- batch budget 是否超了；
- token budget 是否超了；
- GPU blocks 是否超了。

如果超了，就必须 preempt，一部分请求从 GPU 路径退出。

### 4.3 必要时做 swap out

当 GPU 压力过大时，调度器从运行队列尾部拿请求，把它们移到 CPU decode 路径。

这一步的逻辑是：

- 尽量保留先到的请求；
- 更晚加入、占资源更多的请求更容易被换出；
- 这样可以近似维持 FCFS。

### 4.4 有余量时做 swap in

如果 swap out 之后 GPU 还有容量，则尝试把 `cpu_decoding_q` 队首的一些请求搬回 GPU。

这里体现的是：

- 先出队到 CPU 的老请求，优先获得回到 GPU 的机会；
- 不是随便挑一个最短或最便宜的请求，而是尽量保持 arrival order。

### 4.5 尝试接纳新的 prefill 请求

对于 `waiting_q` 中的新请求，调度器需要判断：

- 本轮 batch/token budget 是否允许；
- GPU blocks 是否允许；
- intermediate blocks 是否允许；
- CPU blocks 是否允许；
- request id 是否充足。

能进系统的请求，进一步决定：

- 走 GPU prefill；
- 还是走 CPU-prefill / partial offload 路径。

### 4.6 比较 sequential 与 pipelined 两种模式

候选请求确定后，调度器不会默认总是 pipeline，而是要比较：

- 单 batch sequential 的吞吐；
- 双 sub-batch pipelined 的吞吐。

只有预测 `pipelined_rate > sequential_rate` 时，才会真正返回两个 batch。

这就是论文里“load-aware”的关键：

> 调度器并不是只关心能不能放得下，而是关心 **以当前负载和重叠关系来看，哪种执行模式吞吐更高**。

---

## 5. 真实代码的总控制流：调度不独立，它嵌在一次 iteration 里

很多人读论文时容易把 scheduling 当成孤立算法，但在真实系统里它只是主循环的一部分。

完整控制流在 `swiftllm/server/engine.py:189-223`：

1. `scheduler.get_next_batch()` 产出：
   - `batches`
   - `cur_swap_out`
   - `cur_swap_in`
2. `block_manager.prepare(...)` 把高层调度决策翻译成 block mapping 和 swap 参数；
3. `executor.do_one_iteration(...)` 在 worker 侧真正跑一次 forward；
4. `block_manager.update_and_free(...)` 更新输出 token，并释放 finished requests 的 block；
5. `scheduler.remove_finished_requests(...)` 从调度器内部队列移除完成请求并回收 request id。

也就是说，论文里的“调度一轮”在真实实现里并不是只返回一个 batch，而是同时决定：

- 谁本轮 forward；
- 谁本轮 swap；
- 谁继续在 GPU；
- 谁退到 CPU；
- 谁刚从 waiting 进入系统；
- worker 应该走 sequential 还是 pipeline 分支。

---

## 6. Scheduler 的状态与约束：真实实现比论文伪代码更工程化

### 6.1 三个核心队列

`scheduler.py:106-109`：

- `waiting_q: deque[Request]`
- `gpu_decoding_q: list[Request]`
- `cpu_decoding_q: deque[Request]`

它们按到达时间维持顺序，是实现“近似 FCFS”的基础。

### 6.2 RequestIdManager

`scheduler.py:19-53` 的 `RequestIdManager` 负责管理 block table 中的 `request_id` 命名空间。

这意味着调度并不只受显存约束，还受 **可用 request slot** 约束。

### 6.3 ScheduleBudget

`scheduler.py:54-88` 的 `ScheduleBudget` 同时跟踪：

- `remaining_batch_size`
- `remaining_tokens_in_batch`

这是非常关键的一点：

> NEO 不是只根据 block 数做调度，还同时约束单轮的 batch size 和 token 数，避免 runtime OOM 或 batch 超界。

### 6.4 block 数的三种含义

在 `_get_next_batch_new()` 中会同时看三类 block 约束：

- `num_gpu_blocks`：GPU 上真正可用于 decode/prefill 的 block；
- `num_itm_blocks`：中间缓存 / intermediate block 预算；
- `num_cpu_blocks`：CPU swap 空间能承载的 block。

所以真实实现不是“GPU 放不下就扔 CPU”，而是要同时满足 GPU、CPU、intermediate 三边容量。

---

## 7. `_get_next_batch_new()`：真实调度主流程逐步拆解

核心实现位于 `swiftllm/server/scheduler.py:237-328`。

下面按代码顺序解释。

### 7.1 Step 0：初始化本轮预算与候选集合

函数一开始创建：

- `budget`
- `pref_to_cpu`
- `pref_to_gpu`
- `swpout_reqs`
- `swpin_reqs`

这表明调度器在一轮里同时决定三类事情：

1. 新 prefill 去哪；
2. 哪些老请求 swap out；
3. 哪些 CPU 请求 swap in。

### 7.2 Step 1：先把现有 GPU decode 请求当作默认保留集合

`scheduler.py:259-262`：

- 统计 `gpu_decoding_q` 需要多少 GPU blocks；
- 先从 budget 中扣掉这些请求占的 batch/token 预算。

也就是说，调度器默认认为：

- 正在 GPU 上 decode 的请求优先继续运行；
- 除非预算或 block 不够，否则不轻易动它们。

### 7.3 Step 2：必要时 swap out

`scheduler.py:264-272`：

```python
while budget.overspent or gpu_block_needed > swap_out_threshold:
    victim = self.gpu_decoding_q.pop()
    self.cpu_decoding_q.appendleft(victim)
    swpout_reqs.append(victim)
    gpu_block_needed -= self._get_block_needed(victim)
    budget.add(1)
```

含义非常明确：

- 如果 GPU decode 集合太大，就从尾部弹出请求；
- 被弹出的请求进入 `cpu_decoding_q` 的头部；
- 当前轮会记录为 `swpout_reqs`，稍后由 `BlockManager` 物化 swap；
- 同时把预算还回来。

这里有两个重要性质：

1. **近似 FCFS**
   - 后来的请求更容易成为 victim；
   - 老请求更稳定地留在 GPU。
2. **swap out 是为了解决预算和 block 双重压力**
   - 不只是显存满了才换出；
   - batch/token 超预算也会触发换出。

### 7.4 Step 3：有余量时 swap in

`scheduler.py:273-284`：

- 只看 `cpu_decoding_q` 的队首；
- 若 GPU block 和 budget 允许，就把它搬回 GPU；
- 搬回成功后从 `cpu_decoding_q` 弹出，并 append 到 `gpu_decoding_q`。

这里的关键工程点是：

- swap in/out 不会同时发生，代码用 `assert not swpout_reqs or not swpin_reqs` 保证这一点；
- swap in 按队首做，继续体现 arrival-order fairness。

### 7.5 Step 4：估算新 prefill 最多能接纳多少

`scheduler.py:286-307` 遍历 `waiting_q`，检查：

- `itm_block_needed + cur_block_needed <= self.num_itm_blocks`
- `cpu_block_needed + cur_block_needed <= cpu_threshold`
- `request_id` 数量够不够
- `budget.check_and_substract(candidate.prompt_len)` 是否成功

也就是新请求进入系统要同时满足：

- intermediate cache 容量；
- CPU swap 容量；
- id 空间；
- 本轮 batch/token 预算。

### 7.6 Step 4 中的 GPU/CPU prefill 分流 heuristic

同一段里还有一个非常关键的工程 heuristic：

```python
if not pref_to_cpu and gpu_block_needed + cur_block_needed <= self.num_gpu_blocks:
    pref_to_gpu.append(candidate)
else:
    pref_to_cpu.append(candidate)
```

对应的策略是：

1. 优先把新 prefill 放到 GPU；
2. 如果 GPU 压力大，再把请求导向 CPU；
3. 一旦前面某个新请求已经被分到 CPU，后面的新请求也不要轻易插回 GPU。

第 3 点就是代码注释里写的 fairness heuristic：

> 如果更早到达的请求已经因为资源压力进了 CPU，后面更晚到达的请求不应因为“正好卡位”而回到 GPU 抢先。

这不是论文抽象伪代码里会强调的内容，但是真实系统里非常重要。

### 7.7 Step 5：调用 `_decide_mode_and_gen_batch(...)`

`scheduler.py:308-309` 把前面得到的：

- `pref_to_gpu`
- `pref_to_cpu`
- `budget`

交给模式决策函数。

这一步才真正把“可进入系统的候选请求”变成：

- 一个 batch 的 sequential 方案；
- 或两个 sub-batch 的 pipelined 方案。

### 7.8 Step 6：把试探 admission 变成真实提交

`scheduler.py:311-327` 先根据返回的 batch 数量反推本轮真实接纳多少 prefill，再：

- 从 `waiting_q` 中真正 `popleft()` 这些请求；
- 为它们分配 `request_id`；
- 把 GPU-prefill 请求追加到 `gpu_decoding_q`；
- 把 CPU-prefill 请求追加到 `cpu_decoding_q`。

这一步很重要，因为 Step 4 只是“试探上限”，Step 6 才是“正式提交”。

---

## 8. `_decide_mode_and_gen_batch()`：论文 claim 在代码里的核心落点

函数位于 `swiftllm/server/scheduler.py:142-235`。

它不是简单把请求拼成 batch，而是在做三件事：

1. 形成 sequential 候选；
2. 形成 pipelined 候选；
3. 用性能预测比较两者吞吐。

### 8.1 Step 1：先把 prefill 和 GPU decode 放进 batch 0

代码 `scheduler.py:161-173`：

- 所有 `gpu_prefill_reqs` 加入 `batches[0]` 和 `gpu_only_batch`；
- 所有 `cpu_prefill_reqs` 也先加入 `batches[0]` 和 `gpu_only_batch`；
- 所有 `gpu_decoding_q` 请求都加入 `batches[0]` 和 `gpu_only_batch`。

这里的含义是：

- `gpu_only_batch` 表示“如果本轮不做 pipeline，只跑一个 batch”的候选；
- `batches[0]` 是 pipeline 方案中的第一个 sub-batch 初稿。

### 8.2 Step 2：先裁剪 sequential 候选中的 prefill

`scheduler.py:177-183`：

```python
while gpu_only_batch.get_num_prefs():
    req, is_gpu = gpu_only_batch.pop_pref()
    if is_gpu or gpu_only_batch.perfdata.s < self.predictor.linr_S_threshold:
        gpu_only_batch.add_pref(req, is_gpu)
        break
```

逻辑是：

- 如果 batch 里 prefill 太多，会让 `s` 过大；
- 当 `s` 仍高于 `linr_S_threshold` 时，代码倾向继续砍掉末尾的 CPU-prefill；
- 直到剩余 prefill 数量让线性层代价回到合理区间。

这表明 NEO 不会为了“多塞几个 prefill”而让 linear 部分膨胀过头。

### 8.3 Step 3：把 CPU decode 请求拆到两个 sub-batch

`scheduler.py:184-204` 是整个函数最有代表性的地方。

它维护：

- `min_out_cpu_len`
- `next_batch_idx`

然后遍历 `cpu_decoding_q`：

1. 先做 budget 检查；
2. 尝试把当前 cdec 请求加到某个 batch；
3. 调用 `_get_remains(...)` 估计两个 batch 的 CPU capacity 余额；
4. 如果 `min(remains) < 0`，说明当前放法会让某个 batch 的 CPU decode 无法被 GPU 工作充分隐藏；
5. 这时先跳过当前请求，并更新 `min_out_cpu_len`，避免后面更长的请求重复无效尝试；
6. 若可以接受，则根据 `remains[1] > remains[0]` 决定下一次更倾向把 cdec 加到哪个 batch。

这段逻辑体现出两点：

- **pipeline 不是为了并发而并发，而是为了让 CPU 负载被 GPU 负载隐藏**；
- **CPU decode 请求会被按负载平衡拆到两个 sub-batch，而不是平均切半。**

### 8.4 `_get_remains()` 的真实含义

`scheduler.py:132-140`：

```python
return [
    batches[j^1].perfdata.linr_T +
    batches[j].perfdata.pref_T +
    batches[j].perfdata.gdec_T -
    batches[j].perfdata.cpu_time
    for j in range(2)
]
```

可解释为：

对于 batch `j`，当另一个 batch 在做 linear、当前 batch 在做 prefill/gdec 时，CPU 是否还有足够时间完成当前 batch 的 cdec。

若结果为正，代表：

- CPU decode 还有机会被隐藏；
- overlap 成立的可能性较大。

若结果为负，代表：

- 当前 CPU 负载太重；
- pipeline 结构下会暴露 CPU 瓶颈；
- 再把更多 cdec 塞进去意义不大。

这正是“load-aware”里最关键的 **overlap-aware** 部分。

### 8.5 `min_out_cpu_len` 是避免反复尝试无效候选的 heuristic

当某个 CPU decode 请求因为太长而导致 `min(remains) < 0` 时，代码会：

- 记下 `min_out_cpu_len = req.seq_len`；
- 后面遇到更长或一样长的 cdec 请求，直接跳过。

这不是论文伪代码层面的核心算法，而是很典型的工程优化：

- 减少重复尝试明显不适合进入 pipeline 的请求；
- 保持 scheduler 在在线场景里的决策开销较小。

### 8.6 Step 4：如果 CPU 太空闲，再减少 batch0 里的 prefill

`scheduler.py:217-223`：

```python
while batches[0].get_num_prefs():
    req, is_gpu = batches[0].pop_pref()
    if is_gpu or batches[0].perfdata.s < self.predictor.linr_S_threshold or min(self._get_remains(batches)) < 0:
        batches[0].add_pref(req, is_gpu)
        break
```

这一步和 Step 2 的方向不一样。

它的目标不是为了保护 sequential，而是为了保护 pipeline 的重叠关系：

- 如果 batch0 prefill 太多，GPU 侧可能过长或结构失衡；
- 代码会砍掉一些 prefill，让 pipeline 的 CPU/GPU overlap 更合适；
- 直到 `s` 降到阈值附近，或继续砍已不再必要。

### 8.7 Step 5：最终比较 sequential 与 pipelined 吞吐

`scheduler.py:224-234`：

```python
seqential_time = gpu_only_batch.perfdata.gpu_time * self.model_config.num_layers
pipelined_time = (batches[0].perfdata.gpu_time + batches[1].perfdata.gpu_time) * self.model_config.num_layers
seqential_rate = len(gpu_only_batch) / seqential_time
pipelined_rate = sum(len(batches[i]) for i in range(2)) / pipelined_time
if seqential_rate < pipelined_rate:
    return batches
else:
    return [gpu_only_batch]
```

这一步非常重要：

- 代码没有硬编码“只要有 CPU decode 就 pipeline”；
- 也没有硬编码“prefill 多就 sequential”；
- 而是直接比较预测吞吐率。

因此，论文里“load-aware scheduling dynamically chooses execution mode”的说法，在代码里最直接的对应就是这段逻辑。

---

## 9. 负载感知来自哪里：`BatchPerfData` + `PerfPredictor`

如果没有性能模型，上面的模式选择就会退化成拍脑袋 heuristics。

### 9.1 `BatchPerfData` 是调度器的局部性能状态

`swiftllm/structs.py:148-208` 定义了 `BatchPerfData`。

它维护：

- `x`：当前 sub-batch 请求数；
- `s`：iteration width；
- `n_g`：GPU decode token 总量；
- `x_c`：CPU decode 请求数；
- `n_c`：CPU decode token 总量。

并通过 predictor 暴露：

- `linr_T`
- `pref_T`
- `gdec_T`
- `cdec_T`
- `lnch_T`
- `gpu_time = linr_T + pref_T + gdec_T`
- `cpu_time = cdec_T + lnch_T`

这里就完成了论文抽象时间项到真实代码字段的映射。

### 9.2 `SubBatch` 里四类请求的真实语义

`swiftllm/structs.py:211-294` 的 `SubBatch` 把请求细分为：

- `gprf_reqs`：GPU prefill；
- `cprf_reqs`：CPU prefill / partial offload prefill；
- `gdec_reqs`：GPU decode；
- `cdec_reqs`：CPU decode。

调用 `set_model_forward_args()` 后，真实 forward 会用到：

- `num_cprfs`
- `num_gprfs`
- `num_gdecs`
- `num_cdecs`
- `num_prefs`
- `num_prgds`
- `sum_pref_toks`
- `sum_prgd_toks`
- `seq_block_size`
- `num_seq_blocks`

这说明 scheduler 选择的不是抽象“类别”，而是会直接影响 worker 里 attention kernel 的执行布局。

---

## 10. 为什么这些性能预测是可信的：`TablePerfPredictor` 与 `ModelProfiler`

### 10.1 Predictor 不是常数规则，而是 profile table + interpolation

`swiftllm/perfpredictor.py:70-196` 中的 `TablePerfPredictor` 维护多张表：

- `linr_S_list / linr_T_list`
- `pref_S_list / pref_T_list`
- `gdec_N_list / gdec_T_list`
- `cdec_S_list / cdec_N_lists / cdec_T_lists`
- `lnch_T`

然后通过：

- 一维线性插值；
- CPU decode 的双线性插值；

去预测没被精确 profile 过的点。

### 10.2 `linr_S_threshold` 是显式 heuristic

`perfpredictor.py:89`：

- `self.linr_S_threshold = 128`

它不是从论文公式直接推出来的，而是代码里额外引入的经验阈值，用来抑制过大的 linear 负载。

### 10.3 `lnch_T = 0.8` 也是工程近似

`perfpredictor.py:127-129`：

- `self.lnch_T = 0.8`

它表示 CPU decode 之外的一部分固定 launch / Python 开销。这个值在真实实现里是个经验常数。

### 10.4 Profiler 真正去跑模型生成表格

`swiftllm/server/profiler.py:41-55` 的 `init_profile_tables()` 会调用：

- `_profile_linr(...)`
- `_profile_pref(...)`
- `_profile_gdec(...)`
- `_profile_cdec(...)`

这些函数不是静态写表，而是构造人工 test case，让 executor 真正跑模型，再从 `ModelPerfResult` 中读取：

- `avg_linr_time`
- `avg_pref_time`
- `avg_gdec_time`
- `avg_cdec_time`

因此 NEO 的 scheduler 并非“写死规则系统”，而是 **offline profiling + online interpolation + greedy decision**。

---

## 11. 调度之后如何落地：`BlockManager.prepare(...)`

调度器只决定逻辑上的“谁进谁出”，真正把这些决策翻译成底层 block 操作的是 `swiftllm/server/block_manager.py:195-261`。

### 11.1 普通 swap 的准备

`block_manager.py:224-230`：

- 如果本轮有 `cur_swap_out` 或 `cur_swap_in`；
- 调用 `_initiate_swap(...)`；
- 生成 source block PIDs / destination block VIDs / destination block PIDs。

### 11.2 为 batch 分配 block

`block_manager.py:232-248`：

对每个 batch：

- 先 `batch.set_model_forward_args(...)`；
- 再给 GPU/CPU 两侧分配 block；
- 同时校验 batch size 和 iter width 不越界。

这说明 scheduler 的 budget 只是第一层防线，block manager 这里还有一次硬约束检查。

### 11.3 cprf 的额外 swap

`block_manager.py:250-259`：

CPU-prefill 请求会在 batch 分配后，再做一轮从 intermediate cache 到 CPU 的特殊 swap。

这也是为什么调度器必须额外考虑 `num_itm_blocks`，因为 CPU prefill 并不是“纯 CPU 上完成”，中间仍依赖 GPU 侧的临时缓存路径。

### 11.4 iteration 结束时更新和释放

`block_manager.py:264-275`：

- 把输出 token 写回请求对象；
- 找出 finished requests；
- 释放其 GPU/CPU block。

然后 `scheduler.remove_finished_requests(...)` 再负责把完成请求从 decoding 队列剔除，并归还 `request_id`。这一点在 `scheduler.py:404-413`。

---

## 12. worker 侧如何执行 scheduler 的模式选择

如果只看 scheduler，很容易误以为“两个 batch”只是逻辑拆分。其实 worker 侧真的有两条不同执行路径。

### 12.1 sequential vs pipeline 的分支入口

`swiftllm/worker/model.py:297-330` 的 `_forward_batches(...)`：

- `len(batches) == 1` 时走 `_forward_sequential(...)`；
- `len(batches) == 2` 时走 `_forward_pipeline(...)`。

这说明 scheduler 返回一个 batch 还是两个 batch，会直接改变 worker 的执行语义。

### 12.2 `_forward_pipeline(...)` 不是普通并发，而是 layer-wise 流水

`worker/model.py:278-294`：

- 先用最后一层对象执行 `forward_first_stage(...)`；
- 中间各层循环执行 `forward_double(...)`；
- 最后执行 `forward_last_stage(...)`。

这意味着 pipelined mode 的本质是：

- batch0 和 batch1 在不同 layer stage 上交错推进；
- 目标是让一批的 linear/postproj/preproj 与另一批的 attention/CPU decode 尽量重叠。

### 12.3 layer 内部确实实现了两批交错

`swiftllm/worker/layers/transformer_layer.py:397-448` 的 `_forward_pipeline_stage(...)` 与 `forward_double(...)` 展示了这种交错：

- 一批在做 `post-projection[i] -> pre-projection[i+1]`；
- 另一批在做 `attention[i]`；
- 两者交替推进。

### 12.4 CPU decode 如何被嵌入同一层 attention 流程

`transformer_layer.py:331-355` 的 `_attention(...)` 中：

- GPU prefill 走 flash attention / customized prefill；
- GPU decode 走 `paged_attention(...)`；
- CPU decode 走 `torch.ops.pacpu.paged_attention_cpu(...)`；
- 完成后再把 CPU 结果拷回 GPU output buffer。

因此 scheduler 里对 `cdec_T`、`lnch_T`、`_get_remains()` 的估计，并不是抽象想象，而是对应真实的 CPU attention 路径。

---

## 13. 论文抽象与真实实现的一致处

下面列出最重要的一致点。

### 13.1 都在做 per-iteration greedy scheduling

无论论文还是代码，本质都是：

- 每一轮根据当前状态做局部最优决策；
- 不回看全局未来；
- 不求解长期最优计划。

### 13.2 都同时考虑 prefill、GPU decode、CPU decode

代码中的 `gprf/cprf/gdec/cdec` 四类请求，正是论文抽象中不同服务阶段和不同设备路径的落地版。

### 13.3 都依赖 profile 信息，而不是固定公式

论文讲的是 load-aware / profile-guided；
代码里对应的是：

- `BatchPerfData`
- `TablePerfPredictor`
- `ModelProfiler`

### 13.4 都把 sequential 和 pipeline 当成可切换候选

不是所有轮次都 pipeline，也不是所有轮次都 sequential。真实代码确实保留了两种候选，并做吞吐比较。

### 13.5 都把 CPU/GPU overlap 视为核心收益来源

论文强调异构流水；
代码里的 `_get_remains()`、`cdec_T`、`forward_double(...)` 正是在估计和实现这种 overlap。

---

## 14. 论文抽象与真实实现的偏离处

真实代码并不是附录伪代码的逐行翻译，而是工程化实现。下面这些点尤其重要。

### 14.1 `swap_in_threshold = round(swap_out_threshold * 0.95)`

在 `scheduler.py:252-257`：

- `swap_out_threshold = self.num_gpu_blocks`
- `swap_in_threshold = round(swap_out_threshold * 0.95)`

这就是典型的 **hysteresis**：

- swap out 的门槛稍高；
- swap in 的门槛稍低；
- 防止边界条件下频繁来回抖动。

论文一般只会说“根据容量做 swap in/out”，不会展开这种工程防抖细节。

### 14.2 `linr_S_threshold`

在 `perfpredictor.py:89`：

- `linr_S_threshold = 128`

它用来限制 batch 的 linear 部分过大，是一个非常明显的工程 heuristic，而非论文里的理论变量。

### 14.3 `min_out_cpu_len`

`scheduler.py:185-203` 用它避免反复尝试明显不适合进入 pipeline 的长 CPU decode 请求。

这是在线系统常见的“剪枝式 heuristic”。

### 14.4 `pref_to_cpu` 的公平性约束

`scheduler.py:296-306` 中，只要有较早请求被导向 CPU，后面的新请求也倾向继续导向 CPU，而不是重新插到 GPU。

这使得系统更接近 FCFS，但也会带来保守性。

### 14.5 真实代码显式考虑 request-id 与 intermediate block 约束

论文里的伪代码通常强调 block 和 mode selection；
真实实现还要处理：

- `request_id` 命名空间；
- `num_itm_blocks`；
- cprf 特殊 swap。

这都是系统工程里绕不开的约束。

### 14.6 真实模式选择主要按 GPU 侧吞吐估计

`scheduler.py:224-234` 的吞吐比较主要用：

- `gpu_only_batch.perfdata.gpu_time`
- `batches[0].perfdata.gpu_time + batches[1].perfdata.gpu_time`

CPU 的影响并未直接写进最终 rate 公式，而是通过前面的 `_get_remains()` 和 cdec admission 过程间接体现。

这说明真实实现不是一个精确的全系统最优模型，而是：

- 先用 heuristic 保证 overlap 可行；
- 再用较简单的 throughput 公式做最终模式选择。

---

## 15. 为什么 NEO 要这样调度

这里把“流程是什么”和“为什么这样做”分开讲。

### 15.1 为什么要每轮做 greedy decision

因为在线 serving 的状态每轮都在变化：

- 新请求不断到达；
- 老请求不断增长序列长度；
- 哪些请求完成、哪些需要 swap 都是动态的；
- GPU/CPU 当前负载不是静态值。

如果做全局优化，不但代价高，而且预测会很快失真。

所以 NEO 选择每轮贪心：

- 只利用当前最可靠的状态；
- 结合 profile 表快速做局部最优决策；
- 让调度开销足够低，适合在线系统。

### 15.2 为什么要同时比较 sequential 与 pipelined

因为 pipeline 不是免费午餐。

如果强行 pipeline，可能会出现：

- CPU decode 太重，反而拖累整体；
- batch 被切碎后，GPU 侧 kernel 效率变差；
- prefill 太多时，线性层成为主瓶颈。

所以 NEO 不默认 pipeline，而是：

- 先形成两种候选；
- 再比较吞吐；
- 让 pipeline 只在真正划算时启用。

### 15.3 为什么要同时考虑 GPU compute、CPU decode、swap、batch budget

因为它们互相耦合：

- 你能不能接纳新 prefill，取决于 GPU/CPU/intermediate blocks；
- 你能不能把 CPU 请求搬回 GPU，取决于 block 和 budget；
- 你能不能让 pipeline 有收益，取决于 CPU decode 是否能被 GPU 工作隐藏；
- 你能不能避免 OOM，又取决于 batch size 和 token 数。

任何只看其中一个维度的调度策略都会不稳定。

### 15.4 为什么必须有 profiler，而不是手写解析模型

因为以下耗时都很难靠纸上公式精确建模：

- flash attention 的实际开销；
- paged attention 的开销；
- CPU paged attention 的开销；
- launch/Python overhead；
- batch 规模变化后的 kernel efficiency。

NEO 的方案是：

- 离线先 profile；
- 在线再插值；
- 在速度和精度之间做工程折中。

---

## 16. 这种调度方式的缺点与局限

这是复现时必须保持清醒的部分。

### 16.1 heuristic 对硬件和工作负载敏感

像下面这些量都带明显硬件依赖：

- `linr_S_threshold`
- `lnch_T`
- profile table 的形状
- swap in/out 的经验阈值

换 GPU、换 CPU、换模型尺寸、换 block size，这些 heuristic 很可能都需要重新校正。

### 16.2 profile table 可能陈旧

如果 profile 是在某种负载、某种驱动版本、某种 kernel 实现下生成的，后续环境变化后表格可能不再准确。

一旦预测失真：

- `_get_remains()` 会错估 overlap；
- mode selection 会选错；
- scheduler 会变得更保守或更激进。

### 16.3 它只保证局部贪心，不保证全局最优

NEO 是 per-iteration greedy scheduler，不会显式优化：

- 长期平均 latency；
- 某类请求的 tail latency；
- 多轮之后的 block fragmentation 或 future contention。

因此它能在在线场景里高效工作，但不能保证全局最优排程。

### 16.4 fairness 只是近似 FCFS

虽然代码很多地方都在维护 arrival order：

- swap out 从 GPU 队尾；
- swap in 从 CPU 队首；
- `waiting_q` 顺序 admission；
- `pref_to_cpu` 的公平性保护；

但它终究不是严格 FCFS。因为：

- 资源约束会改变谁先真正执行；
- 长序列和短序列对 block/budget 消耗不同；
- CPU/GPU 路径不同，实际完成时间自然不同。

### 16.5 CPU/GPU overlap 的判断是近似的

`_get_remains()` 很聪明，但它仍然只是近似模型。

真实系统中还有很多额外因素：

- CUDA stream 同步细节；
- PCIe / NUMA 影响；
- kernel launch 抖动；
- CPU attention 实际缓存行为；
- 不同长度请求混合时的非线性效应。

这些都可能让“预测可重叠”和“真实可重叠”之间出现偏差。

### 16.6 边界条件下仍可能出现 mode oscillation 或保守决策

虽然 `swap_in_threshold` 引入了 hysteresis，但在边界附近仍可能出现：

- 某几轮偏向 sequential；
- 某几轮偏向 pipeline；
- admission 数量反复变化。

另一方面，为了避免这种振荡，代码又引入了比较保守的 heuristic，于是也可能错过本来可行的更激进方案。

### 16.7 真实最终吞吐比较没有显式纳入全部 CPU 代价

最终 `seqential_rate` / `pipelined_rate` 主要基于 GPU 侧时间，CPU 的影响主要在更早阶段由 admission heuristic 间接体现。

这使得实现更简单，但也意味着：

- 它并不是一个严格的联合优化目标；
- 某些极端场景下，最终模式选择可能不够精确。

---

## 17. 一个可直接拿去复现的心智模型

如果你之后要重新实现或解释 NEO，可以把每轮 iteration 想成下面这张脑内流程图。

### 17.1 先看当前活跃 decode 请求

- GPU 上哪些请求已经在跑；
- CPU 上哪些请求还在等机会回来。

### 17.2 先保住 GPU decode 的主体，再做必要 preemption

- 若 GPU block 或 budget 超了，就从 GPU 队尾开始 swap out。

### 17.3 若有余量，按顺序把一些 CPU decode 拉回 GPU

- 只看 CPU 队首；
- 能回就回，不能回就停。

### 17.4 再决定本轮能新接多少 prefill

- 看 batch/token budget；
- 看 request id；
- 看 GPU / CPU / intermediate block。

### 17.5 对这些候选请求生成两个执行方案

- 方案 A：一个 batch，走 sequential；
- 方案 B：两个 sub-batch，走 pipeline。

### 17.6 用 profile 估计 overlap 是否成立、哪个方案吞吐更高

- `_get_remains()` 先判断 CPU decode 是否有机会被隐藏；
- 最后再比较 `sequential_rate` 与 `pipelined_rate`。

### 17.7 把高层决策交给 block manager 和 worker

- `BlockManager.prepare(...)` 负责 block table 和 swap 参数；
- `LlamaModel._forward_batches(...)` 负责实际顺序执行或双 batch 流水执行。

这就是 NEO load-aware scheduling 的完整闭环。

---

## 18. 论文伪代码到真实实现的映射表

| 论文抽象步骤 | 真实代码落点 | 说明 |
| --- | --- | --- |
| 维护 waiting / GPU / CPU 三类请求 | `swiftllm/server/scheduler.py:106-109` | 三队列就是调度状态机 |
| 检查当前可运行 decode 集合 | `scheduler.py:259-262` | 先扣预算，再看 block |
| 必要时 preempt / swap out | `scheduler.py:264-272` | 从 GPU 队尾弹出，进 CPU 队首 |
| 有余量则 swap in | `scheduler.py:273-284` | 从 CPU 队首搬回 GPU |
| 接纳新 prefill | `scheduler.py:286-307` | 同时检查 budget / block / request id |
| 形成 batch 并决定模式 | `scheduler.py:308-309`, `142-235` | sequential vs pipelined |
| 性能预测 | `swiftllm/structs.py:148-208`, `swiftllm/perfpredictor.py:70-196` | `BatchPerfData + TablePerfPredictor` |
| profile 表生成 | `swiftllm/server/profiler.py:41-55` | 离线真实测量 |
| 物化 swap 与 block mapping | `swiftllm/server/block_manager.py:195-261` | 高层决策变底层映射 |
| 顺序执行或双 batch 流水执行 | `swiftllm/worker/model.py:297-330` | worker 最终分支 |

---

## 19. 结论

NEO 的 Load-Aware Scheduling 真正有价值的地方，不是“把一些请求放 CPU”这么简单，而是把以下因素统一进了一轮在线决策里：

- GPU decode 持续服务；
- CPU offloading 扩容；
- 新 prefill 的 admission；
- batch/token/buffer/block 约束；
- CPU/GPU overlap 是否成立；
- sequential 与 pipelined 两种执行模式的吞吐比较。

从论文视角看，它强调的是 **profile-guided load-aware scheduling**；
从代码视角看，它落实为：

- `Scheduler._get_next_batch_new(...)` 决定队列演化、swap 和 admission；
- `Scheduler._decide_mode_and_gen_batch(...)` 决定 batch 结构与模式选择；
- `BatchPerfData + TablePerfPredictor` 提供负载感知；
- `BlockManager.prepare(...)` 与 `LlamaModel._forward_batches(...)` 把决策真正执行出来。

因此，若要复现 NEO，最重要的不是逐行抄附录伪代码，而是把下面这个原则吃透：

> **每轮都根据当前负载和 profile 结果，动态决定“谁留 GPU、谁去 CPU、谁能进系统、这一轮到底值不值得 pipeline”。**
