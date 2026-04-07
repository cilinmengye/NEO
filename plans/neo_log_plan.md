# NEO 运行时日志代码级解读

这份说明只聚焦 **NEO 在线服务路径** 下两类 runtime 日志到底在描述什么调度动作：

1. `swiftllm.server.scheduler`：
   - `Gdecs: ..., Cdecs: ..., Pr2gs: ..., Pr2cs: ..., Waiting: ...`
2. `swiftllm.server.engine`：
   - `Forwarding batches with sizes [...]`
   - `swap out: X`
   - `swap in: Y`

你给的样例：

```text
INFO:swiftllm.server.scheduler:Gdecs: 1, Cdecs: 45, Pr2gs: 0, Pr2cs: 1, Waiting: 0
INFO:swiftllm.server.engine:Forwarding batches with sizes [(1, 0, 1, 0)], swap out: 0, swap in: 0
INFO:swiftllm.server.scheduler:Gdecs: 1, Cdecs: 46, Pr2gs: 1, Pr2cs: 0, Waiting: 0
INFO:swiftllm.server.engine:Forwarding batches with sizes [(0, 1, 1, 0)], swap out: 0, swap in: 0
INFO:swiftllm.server.scheduler:Gdecs: 2, Cdecs: 46, Pr2gs: 0, Pr2cs: 1, Waiting: 0
INFO:swiftllm.server.engine:Forwarding batches with sizes [(1, 0, 2, 0)], swap out: 0, swap in: 0
INFO:swiftllm.server.scheduler:Gdecs: 2, Cdecs: 47, Pr2gs: 0, Pr2cs: 1, Waiting: 0
INFO:swiftllm.server.engine:Forwarding batches with sizes [(1, 0, 2, 0)], swap out: 0, swap in: 0
```

如果先用一句话概括：

> `scheduler` 行是“**本轮准备怎么调度**”的摘要，`engine` 行是“**本轮实际要拿什么 batch 去 forward**”的摘要。

两者在同一次 iteration 中由 `AsyncEngine._main_event_loop()` 串起来，调用链是：

```text
scheduler.get_next_batch()
  -> block_manager.prepare(...)
  -> executor.do_one_iteration(...)
  -> block_manager.update_and_free(...)
  -> scheduler.remove_finished_requests(...)
```

对应源码见 `NEO/swiftllm/server/engine.py:189-212`。

---

## 1. 先看日志到底是在哪打印的

### 1.1 `scheduler` 行：调度器刚决定“本轮要新发起哪些 prefill”

打印点在 `NEO/swiftllm/server/scheduler.py:319-323`：

```python
logger.info(
    "Gdecs: %d, Cdecs: %d, Pr2gs: %d, Pr2cs: %d, Waiting: %d",
    len(self.gpu_decoding_q), len(self.cpu_decoding_q), len(pref_to_gpu), len(pref_to_cpu), len(self.waiting_q)
)
```

所以这条日志不是在 worker 里打印的，也不是模型 forward 之后打印的，而是 **scheduler 刚刚决定好本轮要把哪些 waiting request 拉出来 prefill** 时打印的。

### 1.2 `engine` 行：主循环马上要把本轮 batch 真正送进 forward

打印点在 `NEO/swiftllm/server/engine.py:205-207`：

```python
logger.info(
    f"Forwarding batches with sizes {[(b.num_cprfs, b.num_gprfs, b.num_gdecs, b.num_cdecs) for b in batches]}, "
    f"swap out: {len(cur_swap_out)}, swap in: {len(cur_swap_in)}"
)
```

所以这条日志已经比 `scheduler` 更晚一步：

1. scheduler 已经返回了 `batches, cur_swap_out, cur_swap_in`
2. block manager 已经开始 `prepare(...)`
3. engine 此刻打印“本轮要跑什么 batch、顺带要做多少 swap”

因此，同一轮里通常应该这样理解：

- `scheduler` 行回答：**本轮调度器决定了什么**
- `engine` 行回答：**这些决定落实成了什么 batch 结构**

---

## 2. 看懂这些日志前，只需要先记住 3 个队列 + 1 个 batch 结构

## 2.1 Scheduler 内部最关键的 3 条队列

`Scheduler.__init__()` 在 `NEO/swiftllm/server/scheduler.py:95-118` 初始化了三条核心状态：

```python
self.waiting_q: deque[Request] = deque()
self.gpu_decoding_q: list[Request] = []
self.cpu_decoding_q: deque[Request] = deque()
```

它们分别表示：

- `waiting_q`
  - 已经 tokenized，但还没有开始 prefill 的请求
- `gpu_decoding_q`
  - 当前驻留 GPU，并会在接下来 iteration 里按 GPU decode 方式继续前进的请求
- `cpu_decoding_q`
  - 当前留在 CPU 侧 decode / 等待未来可能 swap in 的请求

于是 scheduler 日志里的 5 个字段里，有 3 个就是这三条队列的长度：

- `Waiting = len(self.waiting_q)`
- `Gdecs = len(self.gpu_decoding_q)`
- `Cdecs = len(self.cpu_decoding_q)`

这三个数字描述的是 **scheduler 的持久状态**，不是单轮 forward 的 batch 内部组成。

---

## 2.2 `Request` 的几个字段决定这些计数为什么会变化

`Request` 定义在 `NEO/swiftllm/structs.py:27-106`。这里最值得记住的是：

- `prompt_len`
- `output_len`
- `seq_len = prompt_len + output_len`，见 `NEO/swiftllm/structs.py:48-50`
- `request_id`
- `is_finished()`，见 `NEO/swiftllm/structs.py:64-65`

也就是说，一个请求在调度层面的大致生命线是：

```text
原始请求
  -> tokenization 完成
  -> waiting_q
  -> 本轮被挑出来做 prefill（pref_to_gpu / pref_to_cpu）
  -> 之后进入 gpu_decoding_q 或 cpu_decoding_q
  -> 每轮 decode 推进 output_len
  -> finished
  -> remove_finished_requests()
```

`remove_finished_requests()` 在 `NEO/swiftllm/server/scheduler.py:404-413`：

```python
self.gpu_decoding_q = list(filter(not_finished_func, self.gpu_decoding_q))
self.cpu_decoding_q = deque(filter(not_finished_func, self.cpu_decoding_q))
self.request_id_manager.free_ids([req.request_id for req in reqs])
```

所以你看到 `Gdecs` 或 `Cdecs` 变少，不一定是 swap；也可能是某些请求已经完成，被从解码队列里移除了。

---

## 2.3 `SubBatch` 的四类 request 槽位，决定了 engine tuple 的四元组含义

`SubBatch` 定义在 `NEO/swiftllm/structs.py:211-293`。

最关键的是 `set_model_forward_args()` 里这几行，见 `NEO/swiftllm/structs.py:265-272`：

```python
self.num_cprfs = len(self.cprf_reqs)
self.num_gprfs = len(self.gprf_reqs)
self.num_gdecs = len(self.gdec_reqs)
self.num_cdecs = len(self.cdec_reqs)
self.num_prefs = self.num_cprfs + self.num_gprfs
self.num_prgds = self.num_prefs + self.num_gdecs

self.all_reqs = self.cprf_reqs + self.gprf_reqs + self.gdec_reqs + self.cdec_reqs
```

因此 engine 日志里的：

```text
Forwarding batches with sizes [(1, 0, 1, 0)]
```

四元组顺序就是：

```text
(num_cprfs, num_gprfs, num_gdecs, num_cdecs)
= (cprf, gprf, gdec, cdec)
```

四类分别表示：

- `cprf`：本轮要做 CPU-prefill 的请求数
- `gprf`：本轮要做 GPU-prefill 的请求数
- `gdec`：本轮在 GPU 上 decode 的请求数
- `cdec`：本轮在 CPU 上 decode 的请求数

注意：这 4 个数字是 **request 个数**，不是 token 数，也不是 block 数。

---

## 3. 按一次真实 iteration 的调用链来读日志

最清楚的办法不是按概念拆，而是顺着 `AsyncEngine._main_event_loop()` 的真实顺序看。

代码在 `NEO/swiftllm/server/engine.py:189-212`：

```python
while True:
    batches, cur_swap_out, cur_swap_in = self.scheduler.get_next_batch()
    if not (len(batches) or len(cur_swap_in) or len(cur_swap_out)):
        await asyncio.sleep(0.001)
        continue

    forward_args = self.block_manager.prepare(batches, cur_swap_out, cur_swap_in)

    if any(b.num_prefs for b in batches):
        logger.info(...)

    output_token_ids = await self._run_on_model_executor_async(
        self.executor.do_one_iteration, batches, *forward_args
    )

    finished_reqs = self.block_manager.update_and_free(batches, output_token_ids)
    self.scheduler.remove_finished_requests(finished_reqs)
```

这段代码可以直接翻译成：

### 第 1 步：scheduler 决定本轮做什么

```python
batches, cur_swap_out, cur_swap_in = self.scheduler.get_next_batch()
```

这里会：

- 看看当前 GPU decode 队列有多少请求
- 看看 CPU decode 队列有多少请求
- 必要时先决定要不要 swap out / swap in
- 再决定本轮要不要从 waiting 里拉一些请求来做 prefill
- 最后把这些请求组织成 1 个或 2 个 `SubBatch`

### 第 2 步：block manager 把 request-level 决策翻译成 block-level 参数

```python
forward_args = self.block_manager.prepare(batches, cur_swap_out, cur_swap_in)
```

这里才开始做真正的 block 映射、swap 参数整理、batch block 分配。

### 第 3 步：engine 打印 forward 摘要

如果本轮 batch 里有 prefill，请打印：

- 本轮 batch 里每个 `SubBatch` 的 `(cprf, gprf, gdec, cdec)`
- 本轮 conventional swap out / swap in 的 request 个数

### 第 4 步：executor 跑一轮 model iteration

```python
self.executor.do_one_iteration(...)
```

### 第 5 步：更新输出 token，释放 finished request 的 block

```python
finished_reqs = self.block_manager.update_and_free(...)
```

### 第 6 步：把 finished request 从 scheduler 队列里移掉

```python
self.scheduler.remove_finished_requests(finished_reqs)
```

所以，**同一对相邻日志行**，其实就是同一轮 iteration 的“前半段调度摘要”。

---

## 4. `scheduler` 这 5 个字段到底各自对应什么

打印代码再次放一遍，见 `NEO/swiftllm/server/scheduler.py:319-323`：

```python
logger.info(
    "Gdecs: %d, Cdecs: %d, Pr2gs: %d, Pr2cs: %d, Waiting: %d",
    len(self.gpu_decoding_q), len(self.cpu_decoding_q), len(pref_to_gpu), len(pref_to_cpu), len(self.waiting_q)
)
```

于是 5 个字段可以逐个精确对应：

## 4.1 `Gdecs`

```text
Gdecs = len(self.gpu_decoding_q)
```

即：**当前已经在 GPU decode 队列里的请求数**。

这些请求会在本轮 batch 里以 `gdec` 的形式被加入，见 `NEO/swiftllm/server/scheduler.py:170-172`：

```python
for req in self.gpu_decoding_q:
    batches[0].add_gdec(req)
```

## 4.2 `Cdecs`

```text
Cdecs = len(self.cpu_decoding_q)
```

即：**当前已经在 CPU decode 队列里的请求数**。

这些请求之后可能被拆进 batch 的 `cdec` 槽位，见 `NEO/swiftllm/server/scheduler.py:184-204`。

## 4.3 `Pr2gs`

```text
Pr2gs = len(pref_to_gpu)
```

即：**本轮刚从 `waiting_q` 挑出来、准备进入 GPU-prefill 的请求数**。

它们来自 `_get_next_batch_new()` Step 4，见 `NEO/swiftllm/server/scheduler.py:286-307`。

## 4.4 `Pr2cs`

```text
Pr2cs = len(pref_to_cpu)
```

即：**本轮刚从 `waiting_q` 挑出来、准备进入 CPU-prefill 的请求数**。

它们同样来自 Step 4，只是因为 GPU block 预算或公平性策略，被分流到了 CPU prefill 侧。

## 4.5 `Waiting`

```text
Waiting = len(self.waiting_q)
```

即：**此刻 waiting 队列里还剩多少尚未被启动 prefill 的请求**。

---

## 5. 这条 `scheduler` 日志有两个特别容易误读的时序点

## 5.1 它不是每轮都打印

注意 `scheduler.py:319` 的条件：

```python
if pref_to_gpu or pref_to_cpu:
```

也就是说，只有当本轮真的新拉起了 prefill 请求时，才会打印：

```text
Gdecs, Cdecs, Pr2gs, Pr2cs, Waiting
```

如果某轮只是 decode / swap，没有新 prefill，这条日志不会出现。

## 5.2 它打印时，`pref_to_gpu/pref_to_cpu` 还没真正 extend 进队列

打印之后，源码才做，见 `NEO/swiftllm/server/scheduler.py:325-326`：

```python
self.gpu_decoding_q.extend(pref_to_gpu)
self.cpu_decoding_q.extend(pref_to_cpu)
```

这意味着：

- 日志里的 `Gdecs/Cdecs` 是 **扩充前** 的旧队列规模
- 日志里的 `Pr2gs/Pr2cs` 是 **准备扩充进去** 的新 prefill 请求数

所以你看到：

```text
Gdecs: 1, Cdecs: 45, Pr2gs: 0, Pr2cs: 1
```

不要读成“现在系统里已经有 46 个 CPU 侧请求”。更精确的读法是：

- 当前已有 1 个 GPU decode request
- 当前已有 45 个 CPU decode request
- 本轮还准备再从 waiting 里拉 1 个请求去做 CPU prefill
- 日志打印之后，`cpu_decoding_q` 才会扩充到 46

---

## 6. `engine` 日志里的 batch tuple 到底怎么读

打印点在 `NEO/swiftllm/server/engine.py:205-207`：

```python
logger.info(
    f"Forwarding batches with sizes {[(b.num_cprfs, b.num_gprfs, b.num_gdecs, b.num_cdecs) for b in batches]}, "
    f"swap out: {len(cur_swap_out)}, swap in: {len(cur_swap_in)}"
)
```

所以：

```text
[(1, 0, 1, 0)]
```

应该翻译成：

- 本轮只有 **1 个 `SubBatch`**
- 这个 `SubBatch` 内部包含：
  - `1` 个 `cprf`
  - `0` 个 `gprf`
  - `1` 个 `gdec`
  - `0` 个 `cdec`

即：

```text
1 个 CPU prefill + 1 个 GPU decode
```

同理：

```text
[(0, 1, 1, 0)]
```

表示：

```text
1 个 GPU prefill + 1 个 GPU decode
```

再比如：

```text
[(1, 0, 2, 0)]
```

表示：

```text
1 个 CPU prefill + 2 个 GPU decode
```

---

## 7. 为什么 `sizes [...]` 里有时只有一个 tuple

这不表示 NEO 不支持双 sub-batch。

`BlockManager.prepare()` 明确允许 `len(batches) in (1, 2)`，见 `NEO/swiftllm/server/block_manager.py:216-217`：

```python
assert len(batches) in (1, 2)
```

而 `Scheduler._decide_mode_and_gen_batch()` 在 `NEO/swiftllm/server/scheduler.py:142-234` 里会根据情况返回：

- `[]`
- `[gpu_only_batch]`
- `[batch0, batch1]`

所以 engine 日志里的：

- `[(1, 0, 1, 0)]`
  - 表示这轮只有 1 个 `SubBatch`
- `[(a, b, c, d), (e, f, g, h)]`
  - 才表示这轮真正形成了 2 个 `SubBatch`

只有一个 tuple 时，正确理解是：

> 本轮 scheduler 最终选择返回单 batch 执行，而不是双 sub-batch pipeline。

---

## 8. `swap out / swap in` 到底按什么统计

这是最容易误读的点之一。

先说结论：

> engine 日志里的 `swap out: X, swap in: Y` 统计的是 **request 数量**，不是 block 数量。

原因非常直接：`engine.py:195` 拿到的是：

```python
batches, cur_swap_out, cur_swap_in = self.scheduler.get_next_batch()
```

而 `scheduler.get_next_batch()` 返回的第二、第三个值类型就是：

```python
list[Request], list[Request]
```

见 `NEO/swiftllm/server/scheduler.py:393-402`。

所以 engine 打印：

```python
len(cur_swap_out), len(cur_swap_in)
```

自然是在数 **request 个数**。

---

## 9. scheduler 是怎样决定 swap out / swap in 的

在新路径 `_get_next_batch_new()` 里，关键逻辑在 `NEO/swiftllm/server/scheduler.py:264-284`。

## 9.1 Step 2：必要时把 GPU decode request 挪去 CPU

```python
while budget.overspent or gpu_block_needed > swap_out_threshold:
    victim = self.gpu_decoding_q.pop()
    self.cpu_decoding_q.appendleft(victim)
    swpout_reqs.append(victim)
```

这就是典型的：

```text
gpu_decoding_q -> cpu_decoding_q
```

因此：

- `swap out: X`
  - 表示本轮有 `X` 个 request 被从 GPU decode 队列移去 CPU decode 队列

## 9.2 Step 3：如果资源允许，再把 CPU 侧 request 拉回 GPU

```python
candidate = self.cpu_decoding_q[0]
...
swpin_reqs.append(candidate)
self.cpu_decoding_q.popleft()
self.gpu_decoding_q.append(candidate)
```

这就是：

```text
cpu_decoding_q -> gpu_decoding_q
```

因此：

- `swap in: Y`
  - 表示本轮有 `Y` 个 request 被从 CPU decode 队列拉回 GPU decode 队列

---

## 10. 但真正 block 级 swap 的准备发生在 `BlockManager.prepare()`

这一步必须和上一节区分开。

`scheduler` 只是在 request 级别决定：

- 哪些 request 要 swap out
- 哪些 request 要 swap in

而 `BlockManager.prepare()` 才把这些 request 变成真实 block 参数，见 `NEO/swiftllm/server/block_manager.py:195-261`。

尤其是 Step 1，见 `NEO/swiftllm/server/block_manager.py:224-230`：

```python
is_swap_out = bool(cur_swap_out)
sp, dv, dp = self._initiate_swap(cur_swap_out or cur_swap_in, is_swap_out)
mappings[is_swap_out][0].extend(dv)
mappings[is_swap_out][1].extend(dp)
swappings[0].extend(sp)
swappings[1].extend(dp)
```

而 `_initiate_swap()` 在 `NEO/swiftllm/server/block_manager.py:172-192` 做的事情是：

```python
src_blk_pids = src_block_manager.free(reqs, int(use_itm))
dst_blk_vids, dst_blk_pids = dst_block_manager.alloc(reqs, omit_last=omit_last)
return src_blk_pids, dst_blk_vids, dst_blk_pids
```

所以：

- engine 日志里的 `swap out: 3`
  - 只能读成“3 个 request 被安排 swap out”
- **不能**直接读成“只交换了 3 个 KV block”

因为每个 request 对应多少 block，要由它自己的 `seq_len` 和 `block_size` 决定，scheduler 里 `_get_block_needed()` 见 `NEO/swiftllm/server/scheduler.py:120-124`：

```python
return cdiv(request.seq_len, self.engine_config.block_size)
```

同一个 request 可能占多个 block。

---

## 11. 还有一个更隐蔽的点：`swap out: 0` 也不一定等于“本轮完全没 swap 动作”

因为 `BlockManager.prepare()` 里除了 conventional swap 之外，还有一段专门给 `cprf` 做的 swap setup，见 `NEO/swiftllm/server/block_manager.py:250-259`：

```python
for batch in batches:
    sp, dv, dp = self._initiate_swap(
        batch.all_reqs[:batch.num_cprfs], is_swap_out=True,
        use_itm=self.engine_config.extra_layer_for_cprf, omit_last=False
    )
    batch.src_blk_ids = sp
    batch.dst_blk_ids = dp
    mappings[1][0].extend(dv)
    mappings[1][1].extend(dp)
```

这意味着：

- engine 日志里的 `swap out / swap in`
  - 只反映 scheduler 返回的 `cur_swap_out / cur_swap_in`
  - 也就是 **conventional request-level swap**
- 但 CPU prefill (`cprf`) 相关的 block-level swap setup
  - 可能仍然在 `prepare()` 里发生

所以：

```text
swap out: 0, swap in: 0
```

更准确的意思是：

> 本轮没有 scheduler 层面显式安排的 conventional swap in/out request；但不自动排除 `cprf` 相关 block 处理。

---

## 12. 用你的样例日志逐组翻译

下面按“相邻两行是一轮”来读。

---

### 12.1 第一组

```text
INFO:swiftllm.server.scheduler:Gdecs: 1, Cdecs: 45, Pr2gs: 0, Pr2cs: 1, Waiting: 0
INFO:swiftllm.server.engine:Forwarding batches with sizes [(1, 0, 1, 0)], swap out: 0, swap in: 0
```

#### scheduler 行怎么读

- 当前已经有 `1` 个 request 在 `gpu_decoding_q`
- 当前已经有 `45` 个 request 在 `cpu_decoding_q`
- 本轮新从 waiting 里选出 `0` 个 GPU-prefill request
- 本轮新从 waiting 里选出 `1` 个 CPU-prefill request
- `waiting_q` 此刻剩 `0`

注意这里的 `Cdecs: 45` 是打印时刻的旧值；打印后 `cpu_decoding_q.extend(pref_to_cpu)` 才会发生，所以 CPU 侧队列规模随后会变成 46。

#### engine 行怎么读

```text
[(1, 0, 1, 0)]
```

表示本轮只有一个 `SubBatch`，其组成是：

- `1` 个 `cprf`
- `0` 个 `gprf`
- `1` 个 `gdec`
- `0` 个 `cdec`

即：

```text
1 个 CPU-prefill + 1 个 GPU-decode
```

同时：

```text
swap out: 0, swap in: 0
```

表示 scheduler 本轮没有额外安排 conventional swap。

#### 这一轮实际发生了什么

最自然的翻译就是：

> 系统当时已经有 1 个请求在 GPU 侧持续 decode，45 个请求在 CPU 侧排着 decode；本轮又从 waiting 里拉起了 1 个新的 CPU-prefill 请求，并把它和现有的 1 个 GPU decode 请求一起组成单个 `SubBatch` 去 forward。

---

### 12.2 第二组

```text
INFO:swiftllm.server.scheduler:Gdecs: 1, Cdecs: 46, Pr2gs: 1, Pr2cs: 0, Waiting: 0
INFO:swiftllm.server.engine:Forwarding batches with sizes [(0, 1, 1, 0)], swap out: 0, swap in: 0
```

#### scheduler 行怎么读

和上一轮相比，这里已经变成：

- `Gdecs: 1`
- `Cdecs: 46`

这正好说明上一轮新发起的那个 `Pr2cs: 1` 在日志打印后已经被并入 CPU 侧持久状态了。

而这一轮又新拉起：

- `Pr2gs: 1`
- `Pr2cs: 0`

即：

> 本轮不是新增 CPU prefill，而是新增 1 个 GPU prefill。

#### engine 行怎么读

```text
[(0, 1, 1, 0)]
```

表示：

```text
1 个 GPU-prefill + 1 个 GPU-decode
```

#### 这一轮实际发生了什么

> 当前系统状态变成 1 个 GPU decode、46 个 CPU decode；这时 scheduler 又从 waiting 里拉起了 1 个新的 GPU-prefill 请求，并把它和那 1 个现有 GPU decode 请求一起组成单个 batch 去跑。

这也正好说明：

- `Pr2gs: 1` 对应 engine tuple 里的 `num_gprfs = 1`
- `Pr2cs: 0` 对应 tuple 里的 `num_cprfs = 0`

---

### 12.3 第三组

```text
INFO:swiftllm.server.scheduler:Gdecs: 2, Cdecs: 46, Pr2gs: 0, Pr2cs: 1, Waiting: 0
INFO:swiftllm.server.engine:Forwarding batches with sizes [(1, 0, 2, 0)], swap out: 0, swap in: 0
```

#### scheduler 行怎么读

这里 `Gdecs` 从 1 变成了 2，说明前一轮新增的那个 GPU prefill 请求在随后状态推进中已经进入了 GPU decode 队列。

当前状态是：

- `2` 个 GPU decode
- `46` 个 CPU decode
- 本轮再新拉起 `1` 个 CPU prefill

#### engine 行怎么读

```text
[(1, 0, 2, 0)]
```

表示：

```text
1 个 CPU-prefill + 2 个 GPU-decode
```

#### 这一轮实际发生了什么

> 此时 GPU 侧已经累积到 2 个持续 decode 请求，CPU 侧保持 46 个 decode 请求；本轮又新增 1 个 CPU-prefill 请求，并把它与这 2 个 GPU decode 请求组成一个单 batch forward。

---

### 12.4 第四组

```text
INFO:swiftllm.server.scheduler:Gdecs: 2, Cdecs: 47, Pr2gs: 0, Pr2cs: 1, Waiting: 0
INFO:swiftllm.server.engine:Forwarding batches with sizes [(1, 0, 2, 0)], swap out: 0, swap in: 0
```

#### scheduler 行怎么读

上一轮的 `Pr2cs: 1` 已经并入持久队列，所以这次 `Cdecs` 从 46 变成了 47。

当前状态：

- `2` 个 GPU decode
- `47` 个 CPU decode
- 本轮又新增 `1` 个 CPU prefill

#### engine 行怎么读

仍然是：

```text
[(1, 0, 2, 0)]
```

即：

```text
1 个 CPU-prefill + 2 个 GPU-decode
```

#### 这一轮实际发生了什么

> 当前系统状态稳定在 2 个 GPU decode，同时 CPU 侧 decode 队列继续增长；本轮再次从 waiting 拉起 1 个 CPU-prefill 请求，与现有的 2 个 GPU decode 请求共同 forward。

---

## 13. 把这几组日志串起来看，会看到怎样的状态演化

如果只抽取 scheduler 的持久状态，可以近似记成：

```text
初始观测：G=1, C=45
第 1 轮新增：Pr2cs=1  -> 之后 CPU 侧变成 46
第 2 轮新增：Pr2gs=1  -> 之后 GPU 侧变成 2
第 3 轮新增：Pr2cs=1  -> 之后 CPU 侧变成 47
第 4 轮新增：Pr2cs=1  -> 再继续往后会变成 48（若中间没有完成/交换抵消）
```

而 engine 视角则在告诉你：

```text
第 1 轮跑的是 (1 cprf, 0 gprf, 1 gdec, 0 cdec)
第 2 轮跑的是 (0 cprf, 1 gprf, 1 gdec, 0 cdec)
第 3 轮跑的是 (1 cprf, 0 gprf, 2 gdec, 0 cdec)
第 4 轮跑的是 (1 cprf, 0 gprf, 2 gdec, 0 cdec)
```

也就是：

- GPU decode 常驻部分在逐渐累积：`1 -> 1 -> 2 -> 2`
- CPU decode 常驻部分也在逐渐累积：`45 -> 46 -> 46 -> 47`
- 每一轮又还在继续拉新的 prefill 进来
- 这几轮都没有发生 conventional swap in/out

---

## 14. 为什么这些日志明显是新 server 路径，而不是旧 GPU-only 路径

`Scheduler.get_next_batch()` 在 `NEO/swiftllm/server/scheduler.py:393-402`：

```python
if self.engine_config.always_use_gpu:
    return self._get_next_batch_old()
return self._get_next_batch_new()
```

而旧路径 `_get_next_batch_old()` 打印的格式在 `NEO/swiftllm/server/scheduler.py:367-368`：

```python
logger.info(
    f"Waiting: {len(self.waiting_q)}, Prefs: {len(cur_batch.gprf_reqs)}, Gdecs: {len(self.gpu_decoding_q)}, Cdecs: {len(self.cpu_decoding_q)}"
)
```

旧格式是：

```text
Waiting: ..., Prefs: ..., Gdecs: ..., Cdecs: ...
```

而你给的样例是：

```text
Gdecs: ..., Cdecs: ..., Pr2gs: ..., Pr2cs: ..., Waiting: ...
```

这显然是 `_get_next_batch_new()` 的日志格式，不是旧的 `always_use_gpu` 路径。

---

## 15. 最容易误读的点，集中纠正

### 误解 1：`Gdecs/Cdecs` 就是当前 forward batch 的组成

不对。

更准确地说：

- `Gdecs/Cdecs` 是 **scheduler 持久队列** 的规模
- 当前 forward batch 的内部组成要看 engine 打印的 tuple

---

### 误解 2：`Pr2gs/Pr2cs` 是“已经跑完 prefill 的数量”

不对。

它们表示的是：

- 本轮刚从 `waiting_q` 被挑出来
- 准备发起 prefill
- 并在日志打印后才 extend 进 decode 队列

---

### 误解 3：`[(1, 0, 1, 0)]` 里的 4 个数是 token 数

不对。

它们是：

```text
(cprf, gprf, gdec, cdec)
```

也就是四类 request 个数。

---

### 误解 4：`swap out: 4` 表示只交换了 4 个 KV blocks

不对。

它只表示：

- 有 4 个 request 被安排 conventional swap out

至于真实 block 数，要看 `BlockManager.prepare()` 里 `_initiate_swap()` 对这些 request 分别分配/释放了多少 blocks。

---

### 误解 5：`swap out: 0` 就代表本轮完全没有 swap 相关动作

也不完全对。

更准确地说：

- 没有 scheduler 层面显式安排的 conventional swap request
- 但 CPU prefill (`cprf`) 相关 block setup 仍可能在 `BlockManager.prepare()` 内发生

---

### 误解 6：看不到 engine 那条日志就代表本轮没跑 forward

也不对。

`engine.py:205` 的条件是：

```python
if any(b.num_prefs for b in batches):
```

因此这条摘要日志只在本轮 batch 里有 prefill 时才打印。

纯 decode 轮次完全可能：

- forward 实际发生了
- 但这条 `Forwarding batches with sizes ...` 没打印

---

## 16. 把整件事压缩成一句话

如果要用一句话记住这两类日志：

> `scheduler` 行告诉你“当前三条核心队列有多大，以及本轮又准备新拉起多少 GPU/CPU prefill 请求”；`engine` 行告诉你“这些决定最终被组织成了哪些 `SubBatch`，每个 batch 里有多少 `(cprf, gprf, gdec, cdec)`，以及本轮 scheduler 还显式安排了多少 request-level swap in/out”。

---

## 17. 建议你之后顺着源码继续看的顺序

如果你想自己拿着日志逐跳验证，最推荐的阅读顺序是：

1. `NEO/swiftllm/server/engine.py:189-212`
   - 先看 `_main_event_loop()`，建立整轮 iteration 的主线
2. `NEO/swiftllm/server/scheduler.py:237-328`
   - 看 `_get_next_batch_new()`，理解 `Gdecs/Cdecs/Pr2gs/Pr2cs/Waiting`
3. `NEO/swiftllm/structs.py:211-293`
   - 看 `SubBatch`，理解 `(cprf, gprf, gdec, cdec)`
4. `NEO/swiftllm/server/block_manager.py:172-261`
   - 看 `prepare()` 和 `_initiate_swap()`，理解为什么 `swap out / swap in` 是 request 计数而不是 block 计数
5. `NEO/swiftllm/server/scheduler.py:404-413`
   - 看 `remove_finished_requests()`，理解为什么队列规模会在后续 iteration 里变化

按这个顺序最不容易混淆“持久队列状态”和“单轮 batch 组成”。
