# NEO × Chunked Prefill 代码级解读

## 0. 先给结论

如果把论文那段话翻成一句更直白的话，就是：

> **NEO 现在已经会做 mixed prefill + decode，也已经会做 GPU/CPU 分流；但它当前是“按整条 prompt request”为粒度做 prefill admission。chunked prefill 的作用，是把这个 admission 粒度从“整条请求”细化成“一个个 prompt chunk”，让 scheduler 对 GPU/CPU 负载有更细粒度的控制。**

所以这里最关键的不是“NEO 现在已经怎么用了 chunked prefill”，而是：

1. **chunked prefill 本身是什么；**
2. **NEO 当前真实代码里 prefill / decode 是怎么进 batch 的；**
3. **如果把 chunked prefill 接到 NEO，最自然的切入点在哪；**
4. **为什么它可能提升吞吐；又为什么不一定总有效。**

---

## 1. 论文原话到底在说什么

你引用的那段话，本质上在对比两种“减少 decode-only 轮次”的思路：

- **Sarathi-Serve / chunked prefill**：
  把一个长 prompt 拆成多个 chunk，后续每轮只 prefill 一段 chunk，这样更容易在这些 prefill chunk 旁边“挂靠”一些 decode 请求，于是系统里会出现更多 **prefill+decode mixed batches**，更少 **decode-only batches**。

- **NEO**：
  不是把长 prompt 切 chunk，而是把一部分 **不够 batchable 的 decoding attention** 放去 CPU 做，于是 GPU 上能腾出更多空间给更 batchable 的工作（prefill / 其他 decode），也能减少 GPU 上那些“只有 decode、而且 decode attention 又很难 batch 得很漂亮”的时间占比。

所以论文说二者“精神类似”，指的是：

> **它们都在想办法减少 decode-only iteration 的占比，让 GPU 更多时间花在更好 batch 的工作上。**

但它们的实现方式完全不同：

- chunked prefill：还是都在 GPU 上做，只是把 prefill 切碎；
- NEO：把一部分 decode attention 真正挪到 CPU 上做。

---

## 2. 什么是 chunked prefill

## 2.1 常规 prefill 的问题

在在线 serving 里，一个请求通常经历两阶段：

1. **prefill**：把整段 prompt 喂进去，建立 KV cache；
2. **decode**：后面每轮只生成 1 个 token，再继续更新 KV cache。

如果按“整条 prompt 一次性 prefill”的方式做调度，那么一个很长的 prompt 会有两个典型问题：

- 它会一下子吃掉很多 iteration token budget；
- 它也会一下子占掉很多 GPU KV block；
- 结果是这一轮很可能只能主要服务这个长 prompt，自由挂靠 decode 请求的空间很少；
- 下一轮系统又可能回到 **decode-only**。

而 **decode-only batch** 往往不是理想状态，因为 decode attention 本身 batch 性更差、每次只处理 1 token，GPU 不一定容易吃满。

---

## 2.2 chunked prefill 的核心动作

chunked prefill 的核心动作很简单：

> **不要把一个长 prompt 一次性 prefill 完，而是拆成多个 chunk，分多轮做。**

例如一条 prompt 长度 2048，chunk size = 512，那么它不再是：

- 第 1 轮直接 prefill 2048

而变成：

- 第 1 轮 prefill 前 512
- 第 2 轮 prefill 第 513~1024
- 第 3 轮 prefill 第 1025~1536
- 第 4 轮 prefill 第 1537~2048

这样一来，每轮 prefill 占用的 token budget 小了，scheduler 更容易把一些 decode 请求一并塞进来，形成更多 mixed batch。

---

## 2.3 它为什么可能提升吞吐

直觉上，chunked prefill 的收益是两层：

### 第一层：减少 decode-only 轮次

因为长 prompt 不再独占一整个大 prefill iteration，而是被拆散到多个 chunk 里，于是 decode 请求更容易“搭车”。

也就是：

- 原来：
  - 一轮大 prefill
  - 接着几轮 decode-only
- chunked 之后：
  - 多轮较小 prefill，每轮都更有机会顺手带一些 decode

这就是论文说的：

> launch more prefill-decode mixed batches and thus less decoding-only batches

### 第二层：让 GPU 时间分配更平滑

因为大 prompt 不再一下子冲进来占满预算，系统更容易在 iteration 级别做细粒度调度。

---

## 2.4 它为什么会有代价

论文强调的两个缺点，本质上也很合理。

### 缺点 1：更吃 GPU memory bandwidth

后续 chunk 在做 attention 时，并不是“只看这一段 chunk 自己”。
它仍然需要看前面已经 prefill 过的历史上下文，因此会反复依赖已有 KV/cache 状态。

所以 chunked prefill 不是“白送吞吐”：

- chunk 切得越细；
- iteration 数越多；
- 历史 KV 被反复参与后续 chunk 计算的次数越多；
- GPU memory bandwidth 压力也越明显。

### 缺点 2：在显存很紧的 GPU 上，chunk 太小未必划算

如果 GPU 本来就很紧，你可能被迫把 chunk size 设得很小。问题是：

- chunk 太小，每轮 prefill 的计算变短；
- 那么可“piggyback”进去的 decode 数量也未必多；
- GPU 可能没有因此更饱和，反而只是调度更碎、带宽更重。

所以 chunked prefill 并不是“切得越小越好”。

---

## 3. NEO 当前代码里到底有没有原生 chunked prefill

先说结论：

> **NEO 当前仓库的在线 serving 路径，没有实现自己的 chunked prefill 调度。**

这个结论可以直接从代码看出来。

### 3.1 仓库里出现 chunked prefill 的地方，主要是 vLLM baseline

在 `NEO/evaluation/server.py:26-47`，vLLM 启动参数里明确传了：

- `--enable-chunked-prefill`
- `--max-num-batched-tokens = chunk_size`

也就是这里的 `vllm256` / `vllm512`，本质上是：

- 用 vLLM serve；
- 开 chunked prefill；
- chunk size 由 `--max-num-batched-tokens` 控制。

而同一个文件里，NEO / base / fsdc 走的是另一条路径：`NEO/evaluation/server.py:55-87`

- `sys.executable -m swiftllm.server.api_server`

这条路径并没有任何 `--enable-chunked-prefill` 对应逻辑。

所以：

- **仓库里有 chunked prefill 的实验对照；**
- **但那是给 vLLM baseline 打开的，不是 NEO 自己已经实现了 chunked prefill。**

---

## 4. NEO 当前真实的 prefill / decode 路径是什么

要理解“怎么把 chunked prefill 用到 NEO 上”，先要看清：**NEO 现在 admission 的单位是什么。**

## 4.1 Request 当前没有“prefill 到第几个 chunk”的状态

`NEO/swiftllm/structs.py:27-62` 里的 `Request` 只有这些核心状态：

- `prompt_token_ids`
- `prompt_len`
- `output_len`
- `max_output_len`
- `request_id`
- `output_token_ids`

并没有类似：

- `prefilled_prompt_len`
- `next_chunk_start`
- `next_chunk_len`
- `is_prefill_finished`

这种“chunk 级 prefill 进度”的字段。

这件事非常关键，因为它说明当前 NEO 的 request 生命周期只有两种明显状态：

1. **还没开始 decode：说明这轮是第一次 prefill**
2. **已经有 output_len > 0：说明已经进入 decode 阶段**

但它没有办法表达：

> “这个请求还没开始 decode，但 prompt 其实已经 prefill 了前 1024 个 token，这轮只该再 prefill 下一段 512。”

也就是说，从数据结构上看，NEO 当前并没有内建 chunked prefill 的 request state machine。

---

## 4.2 现在的 prefill 输入粒度是“整条 prompt”

看 `NEO/swiftllm/structs.py:83-87`：

```python
return sum([req.prompt_token_ids if req.output_len == 0 else req.output_token_ids[-1:] for req in reqs], [])
```

这里的含义很直接：

- 如果 `req.output_len == 0`，说明它还是 prefill 请求；
- 那么模型输入直接就是 **整段 `req.prompt_token_ids`**；
- 如果已经开始 decode，则每轮只送最后一个生成 token。

所以当前 NEO 的语义是：

> **只要一个请求还处在 prefill 状态，一旦它被 admission 进 batch，送进模型的就是完整 prompt，而不是某个 chunk。**

---

## 4.3 当前 scheduler 的 prefill admission 也是 request-level 的

最关键代码在 `NEO/swiftllm/server/scheduler.py:286-317`。

### Step 4：先估算这一轮最多能接多少个 prefill request

`Scheduler._get_next_batch_new()` 在 Step 4 遍历 `waiting_q`：

- 用 `candidate.prompt_len` 扣 token budget：`scheduler.py:294`
- 用整条 request 的 block 需求 `cur_block_needed` 做 GPU/CPU/ITM 容量判断：`scheduler.py:290-305`
- 决定这条 request 先去 `pref_to_gpu` 还是 `pref_to_cpu`

关键点在这里：

```python
not budget.check_and_substract(candidate.prompt_len)
```

说明当前调度预算扣减单位是：

> **完整 prompt 长度**

而不是 chunk 长度。

### Step 6：真正发车时，也是整条 request 出队

在 `scheduler.py:311-317`：

- `real_num_prefs = sum(b.get_num_prefs() for b in batches)`
- 然后连续 `popleft()` 对应数量的 waiting request
- 并给它们分配 `request_id`

这里仍然没有“同一条请求只消费一个 chunk、剩余 chunk 还留在 waiting 状态”的逻辑。

因此当前 NEO 的 prefill admission 粒度可以明确总结成：

> **一个 waiting request 一旦被选中，就是以“整条 prompt”作为一个 prefill 单元进入本轮。**

---

## 4.4 NEO 已经具备 mixed prefill + decode 执行能力

虽然 NEO 没有 chunked prefill，但它已经不是“prefill 和 decode 完全分开”的执行器了。

从 `NEO/swiftllm/structs.py:211-272` 可以看到 `SubBatch` 已经明确区分：

- `cprf_reqs`：CPU-prefill requests
- `gprf_reqs`：GPU-prefill requests
- `gdec_reqs`：GPU-decode requests
- `cdec_reqs`：CPU-decode requests

而 `set_model_forward_args()` 在 `structs.py:265-282` 又把这些请求聚合成同一轮 forward 所需的布局：

- `num_cprfs`
- `num_gprfs`
- `num_gdecs`
- `num_cdecs`
- `sum_pref_toks`
- `sum_prgd_toks`

这说明：

> **NEO 已经能在同一轮 forward 里同时承载 prefill + GPU decode + CPU decode。**

这点在 worker 侧也能直接看到。

### worker 里的 attention 路径本来就是混合的

在 `NEO/swiftllm/worker/layers/transformer_layer.py:274-355`：

- `batch.num_prefs > 0` 时跑 prefill attention：`transformer_layer.py:274-307`
- `batch.num_gdecs > 0` 时跑 GPU paged attention：`transformer_layer.py:311-329`
- `batch.num_cdecs > 0` 时跑 `torch.ops.pacpu.paged_attention_cpu(...)`：`transformer_layer.py:331-355`

也就是说，NEO 的执行路径已经支持：

- 一部分 token 走 prefill；
- 一部分 decode 留在 GPU；
- 另一部分 decode offload 到 CPU。

因此如果以后接 chunked prefill，**主要缺的不是底层 forward 能力，而是上层 request/scheduler 的 admission 粒度。**

---

## 5. NEO 当前是怎么做 load-aware scheduling 的

这个问题很重要，因为论文里说要“修改 step 5”，本质上是在现有调度框架上细化，而不是另起炉灶。

## 5.1 现有调度先做哪些事

`Scheduler._get_next_batch_new()` 大致是：

1. **先把能留在 GPU 的 gdec 留住**：`scheduler.py:259-262`
2. **如果 GPU 太挤，就把一些 gdec swap out 到 CPU**：`scheduler.py:264-272`
3. **如果 GPU 还有空间，再把一些 cdec swap back 到 GPU**：`scheduler.py:273-284`
4. **再看 waiting_q 里能接多少新 prefill requests**：`scheduler.py:286-306`
5. **调用 `_decide_mode_and_gen_batch()` 把 CPU decode 分配到 1 个或 2 个 sub-batch**：`scheduler.py:308-309`
6. **真正让前面的 prefilling requests 出队并拿到 request_id**：`scheduler.py:311-317`

然后 `AsyncEngine` 在 `NEO/swiftllm/server/engine.py:193-212` 中：

- 从 scheduler 取出 `batches, cur_swap_out, cur_swap_in`
- 调用 `block_manager.prepare(...)`
- 再调用 `executor.do_one_iteration(...)`
- 最后 `update_and_free(...)`

这条主路径说明：

> chunked prefill 想接到 NEO 上，必须沿着 **scheduler → block_manager → model forward** 这条已有控制流接进去。

---

## 5.2 论文说“修改 step 5”在代码语境里该怎么理解

论文原文说：

> instead of removing the whole prefilling requests, NEO could remove chunks of prefilling requests

如果映射到当前代码，最接近的落点其实是 `_decide_mode_and_gen_batch()` 里的这段：

- `NEO/swiftllm/server/scheduler.py:217-222`

```python
while batches[0].get_num_prefs():
    req, is_gpu = batches[0].pop_pref()
    if is_gpu or batches[0].perfdata.s < self.predictor.linr_S_threshold or min(self._get_remains(batches)) < 0:
        batches[0].add_pref(req, is_gpu)
        break
```

这段逻辑的意思是：

- 先把 prefill 都尽量塞进第一个 batch；
- 如果发现这样会让 CPU 太闲 / 平衡不好；
- 就从 batch 里往外移掉一些 prefill requests。

但当前“移掉”的单位是：

> **整个 prefill request**

论文说的 chunked prefill 版 NEO，本质上就是把这里的粒度细化为：

> **不是把整条 prefill request 从 batch 里拿掉，而是把这条 request 本轮只保留一部分 chunk，其余 chunk 留到下轮。**

所以论文虽然写“modify step 5”，但落到当前代码结构，更准确地说是两层改动：

1. `waiting_q` admission 单位从 request 改成 chunk；
2. `_decide_mode_and_gen_batch()` 的“减 prefill”逻辑也从 request-level 改成 chunk-level。

---

## 6. 如果把 chunked prefill 用到 NEO，上层应该怎么改

下面讲最重要的部分：**怎样把它落到 NEO 现有代码结构上。**

## 6.1 第一层：Request 需要有 prefill-progress 状态

当前 `Request` 只能表达：

- 整个 prompt 还没 prefill；
- 或者已经进入 decode。

如果要支持 chunked prefill，最先要补的不是 worker，而是 `Request` 状态。

至少要能表达：

- 当前 prompt 已经 prefill 到哪个 token 位置；
- 下一轮要 prefill 的 chunk 从哪里开始；
- 这一轮 chunk 长度是多少；
- 什么时候才算 full prompt prefill 完成；
- full prompt 完成之前，**还不能真正进入普通 decode**。

可以把它抽象成类似这样的语义（这里只是概念，不是说当前代码已有）：

- `prefilled_prompt_len`
- `remaining_prefill_len`
- `next_chunk_len`
- `prefill_complete`

没有这层状态，scheduler 就无法表达：

> “这条 request 已经 prefill 了前 1024，本轮只该再做 512，不该再把整段 prompt 送进去。”

---

## 6.2 第二层：scheduler 的 admission 单位从 request 改成 chunk

当前关键问题在 `scheduler.py:286-306`：

- budget 扣的是 `candidate.prompt_len`
- block 估算用的也是整条 request 当前 `seq_len`

而 chunked prefill 版 NEO 应该把这里改成：

- 对于一个还没 prefill 完的 request，先生成一个“本轮 chunk candidate”；
- 用 `chunk_len` 扣 iteration token budget；
- 决定这轮是否让这段 chunk 进入 batch；
- 本轮只推进 request 的一部分 prefill 进度，而不是一次性把整条 prompt 从 waiting 队列里彻底消费掉。

也就是说，当前：

- `waiting_q` 里元素 = 一个完整 request

而改造后更自然的语义会变成：

- `waiting_q` 里还是 request
- 但 scheduler 每轮从 request 上派生出一个 **chunk-sized admission decision**

这里要特别注意：

- **逻辑请求还是一个 request**；
- **调度单位变成这个 request 的下一段 prefill chunk**。

---

## 6.3 第三层：`Request.get_input_tokens()` 也要变成 chunk-aware

当前 `NEO/swiftllm/structs.py:83-87` 的语义是：

- 只要 `output_len == 0`，就把 `prompt_token_ids` 全送进去。

这和 chunked prefill 直接冲突。

因为 chunked prefill 需要的是：

- 第一次 prefill：送 `prompt[0:chunk0_end]`
- 第二次 prefill：送 `prompt[chunk0_end:chunk1_end]`
- ...

也就是说，这里不能再简单用“是否 `output_len == 0`”判断输入，而要依据：

- 当前 request 是否 still in prefill mode；
- 如果是 prefill mode，本轮 chunk 的 token slice 是哪一段。

所以 **Request 输入构造逻辑** 也要跟着 chunk state 一起改。

---

## 6.4 第四层：`BatchPerfData / PerfPredictor` 必须 chunk-aware

这一层很容易被忽略，但其实很关键。

当前 `NEO/swiftllm/structs.py:165-168`：

```python
self.s += prompt_len
self.pref_T += self.predictor.get_pref_T(prompt_len)
```

以及 `SubBatch.add_pref()` 在 `NEO/swiftllm/structs.py:226-231` 里直接传的是：

- `req.prompt_len`

也就是说当前 perf model 默认认为：

> 一条 prefill 请求的成本主要由“这整条 prompt 的长度”决定。

同时 `NEO/swiftllm/perfpredictor.py:166-170` 里的 `get_pref_T(S)` 也是以 **prefill iteration width = S** 为输入做插值。

问题在于 chunked prefill 下：

- 本轮新增输入只有 `chunk_len`；
- 但它的 attention 代价又不仅仅只和“新 chunk 长度”有关；
- 因为后续 chunk 还会依赖前面已经建立的上下文/KV；
- 所以单纯用 `get_pref_T(chunk_len)` 未必还能准确反映真实代价。

因此若真把 chunked prefill 接进 NEO，**至少有两种层次的改法**：

### 简单版

先把它近似当成：

- prefill 成本主要看本轮 chunk 的 token 数

也就是先把：

- `add_pref(req.prompt_len)`

改成：

- `add_pref(chunk_len)`

这样 scheduler 至少能先在“预算”和“粗粒度 cost model”上工作起来。

### 更准确版

再进一步把 predictor 改成：

- 输入不只是本轮 `chunk_len`
- 还要考虑已有 context 长度 / 已 prefilled 长度
- 甚至需要重新 profile chunked-prefill 下的 pref attention 曲线

这也是为什么论文说 chunked prefill 接入 NEO 不只是 scheduler 改几行，而是会牵涉到 cost model。

---

## 6.5 第五层：BlockManager 的 offload 时机要后移

这是 NEO 语境下最容易被忽视、但论文原话其实已经明确点出来的部分。

论文说的建议是：

> NEO can compute all chunked prefilling on the GPU, and doesn’t offload the KV-cache until the full prompt of a request is prefilled.

这句话放到当前代码里，含义非常明确。

### 当前 NEO 的 cprf 会在 prepare 阶段安排 swap out

看 `NEO/swiftllm/server/block_manager.py:250-255`：

```python
for batch in batches:
    sp, dv, dp = self._initiate_swap(
        batch.all_reqs[:batch.num_cprfs], is_swap_out=True,
        use_itm=self.engine_config.extra_layer_for_cprf, omit_last=False
    )
```

也就是说当前只要一个 request 被当成 `cprf`，prepare 阶段就会为它准备 GPU→CPU 的 cprf swap。

worker 侧对应的真正 swap 发生在：

- `NEO/swiftllm/worker/layers/transformer_layer.py:389-390`
- `_swap_out_blocks(batch)` 内部实际执行 `swapper.swap_blocks(...)`：`transformer_layer.py:181-196`

### chunked prefill 版 NEO 不应该每个 chunk 完就 swap out

因为 chunked prefill 的建议是：

- **所有 chunked prefilling 先在 GPU 上完成；**
- **等 full prompt prefill 完成之后，再决定要不要把 KV offload 到 CPU。**

所以如果接入 chunked prefill，block manager / scheduler 对这些“还没 full prompt prefill 完成”的请求，语义应该更接近：

- 它们仍然是“GPU-side ongoing prefill request”
- 不是已经完成 prefill、可以直接按 cprf 语义换出去的对象

换句话说：

> **chunked prefill 接进 NEO 后，不能把“chunk 完成”误当成“整条 prompt prefill 完成”。**

否则你会在 chunk 之间反复把它 swap 到 CPU，再下轮又拉回 GPU，反而把本来想优化的东西搞得更糟。

---

## 7. 为什么 chunked prefill 对 NEO 可能带来额外收益

这一部分要分成两层看。

## 7.1 第一层：和 Sarathi 一样，减少 decode-only 轮次

这个很好理解。

如果长 prompt 不再整条 admission，而是只 admission 一个 chunk，那么：

- 一轮里留给 decode 的 token / batch / block 空间通常更多；
- 更容易形成 mixed prefill+decode batch；
- decode-only 轮次减少；
- GPU 上那些“不够 batchable 的 decode-only 时间”占比下降。

这就是论文说它与 NEO 第二个收益“精神类似”的地方。

---

## 7.2 第二层：对 NEO 更重要的是，chunked prefill 提供了 chunk-level 的 CPU-GPU balancing

这才是 NEO 视角下更关键的收益。

当前 NEO 的 admission 粒度偏粗：

- 一个长 prompt 要么这轮整条进来；
- 要么这轮整条不进来；
- 如果要减少 prefill 压力，也是把**整条 prefill request**从 batch 里拿掉：`scheduler.py:217-222`

这种粗粒度控制在长上下文场景下会比较难受。

因为一个超长 prompt 可能意味着：

- 一次进来就消耗大量 token budget；
- 同时还需要占掉较多 block；
- 于是 scheduler 的选择很僵硬。

而如果改成 chunk 粒度，scheduler 可以做更细的平衡：

- 本轮这个长 prompt 只先进 256 / 512 token；
- 给 decode 多留点空间；
- 如果 CPU decode 当前已经很重，就少放一点 prefill chunk；
- 如果 CPU 这边比较空，就可以多放一点 prefill chunk；
- 不需要在“整条长 prompt 进 / 不进”之间二选一。

所以对 NEO 来说，chunked prefill 最重要的价值不是“重复 Sarathi”，而是：

> **把当前 request-level 的 load balancing，进一步细化成 chunk-level 的 load balancing。**

这也正是论文里所谓：

> finer-grained control for CPU-GPU balancing

---

## 8. 具体案例：一个超长 prompt 和一堆 decode 请求同时存在时，调度会怎样不同

下面给一个具体案例。

### 场景设定

当前系统里有：

- 1 个新到达的大请求 `R0`
  - `prompt_len = 2048`
  - 还没开始 decode
- GPU 上已经有一些正在 decode 的请求
  - `G1, G2`
- CPU 上还有很多 decode 请求
  - `C1 ... C32`
- 当前 iteration 的 token budget 比较紧
- GPU block 也比较紧

---

### 情况 A：当前 NEO 的做法（整条 prompt admission）

在当前 `scheduler.py:286-306` 这套逻辑下，`R0` 被看成一个整体：

- 它要么以 `prompt_len = 2048` 去扣预算；
- 要么因为预算不够，整条进不来；
- 要么勉强进来，但会明显挤压这轮 decode 的空间。

于是容易出现两种情况：

#### A1. 整条进不来

那这轮 scheduler 只能先服务 decode 请求，`R0` 继续卡在 waiting_q。

问题是：

- `R0` 一直很重；
- 后面很多轮它都可能因为“整条太大”而很难被接纳；
- admission 粒度太粗。

#### A2. 整条进来

那这轮可能变成：

- 一个很大的 prefill + 少量 decode
- 甚至 decode 空间被挤得很少

下一轮系统又容易回到 decode-only 主导。

---

### 情况 B：chunked prefill 版 NEO 的做法

假设把 `R0` 的 chunk size 设成 512，那么 scheduler 本轮看到的就不再是：

- “要不要接一条 2048 的 prefill request”

而是：

- “要不要先接 `R0` 的下一个 512-token chunk”

于是这轮可能变成：

- `R0.chunk0 = 512`
- 再加上若干 `G-dec` 请求
- 甚至再保留一部分 `C-dec` 请求在 two-batch pipeline 中与 GPU 工作重叠

下一轮再决定：

- 是继续接 `R0.chunk1`
- 还是先让 CPU / GPU decode 更平衡后再接

这个时候 scheduler 拥有了当前 NEO 没有的自由度：

- **不是在“整条 R0 进 / 不进”之间选；**
- **而是在“R0 这轮先推进多少 prefill”之间选。**

这就是 chunk-level admission 给 NEO 带来的最实际的变化。

---

## 9. 为什么它并不保证一定更强

虽然上面说了很多收益，但一定要注意：**论文并没有说 chunked prefill 接到 NEO 上就一定无脑更强。**

## 9.1 带宽问题仍然在

即使 NEO 把部分 decode attention 放去 CPU，chunked prefill 本身在 GPU 上的额外 memory bandwidth 压力仍然存在。

NEO 解决的是：

- 部分 unbatchable decode attention 挪到 CPU

但它并没有消除 chunked prefill 自身的代价：

- chunk 越碎，GPU 上后续 chunk 对历史 KV 的依赖和反复读取问题仍然在。

---

## 9.2 小 chunk 不一定能形成足够好的 piggyback

如果 GPU 很紧，被迫把 chunk 设得很小，那么：

- 本轮 prefill 的 GPU 计算时间本身不长；
- 可让 decode“搭车”的空间有限；
- 甚至还不够支撑 NEO 当前 two-batch pipeline 的平衡收益。

于是最后可能出现：

- chunk 切细了；
- 调度更复杂了；
- predictor 误差更大了；
- 但 GPU 不一定更饱和。

---

## 9.3 predictor 不改，scheduler 的判断可能会失真

当前 mode selection 是依赖 perf model 的。

例如：

- `_decide_mode_and_gen_batch()` 会根据 `perfdata` 比较 sequential / pipelined：`scheduler.py:224-234`
- `perfdata` 又依赖 `BatchPerfData` 的 `pref_T / gdec_T / cdec_T`
- 而 `pref_T` 目前来自 `predictor.get_pref_T(S)`：`perfpredictor.py:166-170`

如果引入 chunked prefill，却还继续把 pref 代价简单按“整条 prompt”或“裸 chunk 长度”近似，那么 mode selection 可能就会越来越不准。

所以这不是单纯 scheduler 改几处即可的功能。

---

## 10. 最后总结：怎么理解“把 chunked prefill 技术运用到 NEO 上”

可以把最终结论浓缩成下面 5 句话。

### 1. NEO 当前没有原生 chunked prefill

证据最直接的是：

- vLLM baseline 在 `evaluation/server.py:26-47` 显式开了 `--enable-chunked-prefill`
- NEO 自己的 `swiftllm.server.api_server` 路径没有对应实现：`evaluation/server.py:55-87`

### 2. NEO 当前已经有 mixed prefill + decode 执行能力

证据是：

- `SubBatch` 已区分 `cprf/gprf/gdec/cdec`：`structs.py:211-245`
- worker attention 路径已能同时处理 pref / gdec / cdec：`transformer_layer.py:274-355`

所以它缺的不是底层执行，而是上层调度粒度。

### 3. 真正要改的是 request / scheduler / predictor / block-manager 这一层

最自然的切入点是：

- `Request` 增加 prefill-progress 状态：`structs.py:27-62`
- `Request.get_input_tokens()` 改为 chunk-aware：`structs.py:83-87`
- `Scheduler._get_next_batch_new()` Step 4 改为 chunk admission：`scheduler.py:286-317`
- `_decide_mode_and_gen_batch()` 的“减 prefill”改成减 chunk 而不是减整条 request：`scheduler.py:217-222`
- `BatchPerfData / PerfPredictor` 改成 chunk-aware：`structs.py:165-168`, `perfpredictor.py:166-170`
- full prompt prefill 完成前不要按 cprf 逻辑过早 offload：`block_manager.py:250-255`

### 4. 它为什么可能提升

因为它让 NEO 获得了：

- 更少的 decode-only iteration；
- 更细粒度的 prefill/decode 时间分配；
- 更细粒度的 CPU-GPU balancing。

### 5. 它为什么也可能失效

因为 chunked prefill 自身仍然有：

- GPU memory bandwidth 开销；
- 小 chunk 下 piggyback 空间不足的问题；
- 还会要求 predictor / scheduler 的建模同步升级。

---

## 11. 一句话版理解

如果只用一句话总结“如何把 chunked prefill 运用到 NEO 上”：

> **不是把 NEO 现有的 CPU offloading 替换掉，而是在 NEO 现有 mixed-batch + load-aware scheduling 框架里，把 prefill admission 从“整条 prompt”细化成“一个个 prompt chunk”，并且在 full prompt prefill 完成前继续把这条请求视作 GPU 上的 ongoing prefill，从而让 scheduler 能以 chunk 粒度做更细的 CPU-GPU 平衡。**

这就是论文那段话在 NEO 代码语境里的最准确理解。
