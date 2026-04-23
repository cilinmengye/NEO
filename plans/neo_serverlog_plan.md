# NEO server log 异常现象分析

## 1. 现象描述

在 `ours-server.log` 中观察到如下日志片段：

```text
INFO:swiftllm.server.scheduler:Gdecs: 29, Cdecs: 282, Pr2gs: 1, Pr2cs: 0, Waiting: 0
INFO:swiftllm.server.engine:Forwarding batches with sizes [(0, 1, 29, 0)], swap out: 0, swap in: 0
INFO:     127.0.0.1:53758 - "POST /v1/completions HTTP/1.1" 200 OK
INFO:     127.0.0.1:42614 - "POST /v1/completions HTTP/1.1" 200 OK
INFO:     127.0.0.1:52936 - "POST /v1/completions HTTP/1.1" 200 OK
INFO:swiftllm.server.scheduler:Gdecs: 29, Cdecs: 279, Pr2gs: 0, Pr2cs: 1, Waiting: 0
INFO:swiftllm.server.engine:Forwarding batches with sizes [(1, 0, 29, 0)], swap out: 0, swap in: 0
```

直观上会觉得奇怪：

- 第一轮 engine 日志里 `num_cdecs = 0`，说明本轮 batch 没有 CPU decode；
- 同时 `swap out: 0, swap in: 0`，说明也没有发生显式换入换出；
- 但下一次 scheduler 日志里，`Cdecs` 却从 `282` 变成了 `279`。

表面上看，似乎“既没跑 cdec，也没 swap，为什么 CPU decode 队列少了 3 个请求”。

## 2. 结论先行

这并不是 scheduler 和 engine 日志互相矛盾，也不意味着代码存在明显错误。

根因是：

1. `scheduler` 日志里的 `Cdecs` 表示的是 **`cpu_decoding_q` 的长度快照**；
2. `engine` 日志里的 `num_cdecs=0` 表示的是 **该次被 launch 的 batch 中没有 cdec**；
3. `cpu_decoding_q` 的长度变化，不只会由 swap 引起，还会由 **forward 结束后 finished request 被统一清理** 引起；
4. 这两类日志都不是“每轮 iteration 全量打印”，因此中间可能存在 **没有显示出来的静默 decode 轮次**。

因此，`Cdecs: 282 -> 279` 最合理的解释是：**有 3 个 request 在 forward 后完成，然后被 `remove_finished_requests(...)` 从 `cpu_decoding_q` 中过滤掉了。**

---

## 3. scheduler 日志到底在打印什么

`scheduler` 中相关代码位于：

- `swiftllm/server/scheduler.py:440-449`

关键逻辑是：

```python
if pref_to_gpu or pref_to_cpu:
    logger.info(
        "Gdecs: %d, Cdecs: %d, Pr2gs: %d, Pr2cs: %d, Waiting: %d",
        len(self.gpu_decoding_q), len(self.cpu_decoding_q), len(pref_to_gpu), len(pref_to_cpu), len(self.waiting_q)
    )

self.gpu_decoding_q.extend(pref_to_gpu)
self.cpu_decoding_q.extend(pref_to_cpu)
```

这说明：

- `Gdecs` = `len(self.gpu_decoding_q)`
- `Cdecs` = `len(self.cpu_decoding_q)`
- `Pr2gs` = 本轮新接纳到 GPU 的 prefill 数
- `Pr2cs` = 本轮新接纳到 CPU 路径的 prefill 数
- `Waiting` = 当前 waiting queue 长度

更重要的是：

**这条日志打印发生在 `extend(pref_to_gpu/pref_to_cpu)` 之前。**

所以这不是“本轮调度完成后的最终队列状态”，而是：

> **pre-append 队列快照 + 本轮 admission 计数**

也就是说：

```text
Gdecs: 29, Cdecs: 282, Pr2gs: 1, Pr2cs: 0
```

真正含义是：

- 当时 scheduler 看到已有 29 个 GPU decode request；
- 已有 282 个 CPU decode request；
- 本轮又决定新接纳 1 个 GPU prefill；
- 这 1 个 prefill 在日志打印时还没有 `extend` 进 decode queue。

## 4. engine 日志里的 tuple 到底表示什么

`engine` 中相关代码位于：

- `swiftllm/server/engine.py:264-266`

```python
if any(b.num_prefs for b in batches):
    logger.info(f"Forwarding batches with sizes {[(b.num_cprfs, b.num_gprfs, b.num_gdecs, b.num_cdecs) for b in batches]}, "
                f"swap out: {len(cur_swap_out)}, swap in: {len(cur_swap_in)}")
```

而 tuple 字段的定义来自：

- `swiftllm/structs.py:374-381`

```python
self.num_cprfs = len(self.cprf_reqs)
self.num_gprfs = len(self.gprf_reqs)
self.num_gdecs = len(self.gdec_reqs)
self.num_cdecs = len(self.cdec_reqs)
```

因此：

```text
[(0, 1, 29, 0)]
```

表示：

- `num_cprfs = 0`
- `num_gprfs = 1`
- `num_gdecs = 29`
- `num_cdecs = 0`

也就是：

> **1 个 GPU prefill + 29 个 GPU decode + 0 个 CPU decode**

而：

```text
[(1, 0, 29, 0)]
```

表示：

> **1 个 CPU prefill + 29 个 GPU decode + 0 个 CPU decode**

这里最容易犯的错是把：

- “本轮 batch 里没有 cdec”
- 和 “`cpu_decoding_q` 在下一轮前不会变”

当成同一件事。

实际上它们不是同一个量。

## 5. 真正会让队列变短的，不只有 swap

主循环在：

- `swiftllm/server/engine.py:252-271`

核心顺序是：

1. `scheduler.get_next_batch()`
2. `block_manager.prepare(...)`
3. `executor.do_one_iteration(...)`
4. `block_manager.update_and_free(...)`
5. `scheduler.remove_finished_requests(...)`

其中，`block_manager.update_and_free(...)` 位于：

- `swiftllm/server/block_manager.py:264-275`

```python
all_reqs = sum([b.all_reqs for b in batches], [])
finished_reqs = Request.update_output(all_reqs, output_token_ids)
self._free_blocks_of_requests(finished_reqs)
return finished_reqs
```

而 `Request.update_output(...)` 位于：

- `swiftllm/structs.py:93-108`

```python
for req, tok in zip(reqs, output_toks):
    req.output_len += 1
    req.output_token_ids.append(tok)
    req.output_q.put_nowait(StepOutput(tok, req))
    if req.is_finished():
        req.finished_event.set()
        finished_reqs.append(req)
```

也就是说：

- 本轮参与 forward 的 request 会在这里更新输出；
- 如果更新后达到了结束条件，就会被放入 `finished_reqs`；
- 然后返回给 engine。

接着 engine 会调用：

- `swiftllm/server/scheduler.py:533-542`

```python
def remove_finished_requests(self, reqs: list[Request]):
    def not_finished_func(req: Request) -> bool:
        return not req.is_finished()
    self.gpu_decoding_q = list(filter(not_finished_func, self.gpu_decoding_q))
    self.cpu_decoding_q = deque(filter(not_finished_func, self.cpu_decoding_q))
```

这一点非常关键：

`remove_finished_requests(...)` 不是“只从当前 batch 对应的局部列表里删几个 request”，而是：

> **把整个 `gpu_decoding_q` 和整个 `cpu_decoding_q` 都按 `req.is_finished()` 重新过滤一遍。**

因此，`cpu_decoding_q` 的长度减少，完全可能是因为：

- 某些 request 在 forward 之后已经 finished；
- 然后它们在下一次 scheduler 观察队列前，被统一清理掉了。

这一步并不要求：

- 本轮有 `cdec batch`；
- 或者本轮有 `swap in/out`。

## 6. 为什么从日志上看起来像“什么都没发生”

因为这两类日志本身是**稀疏日志**，不是每一轮 iteration 都会打印。

### 6.1 scheduler 这条日志不是每轮都打

`scheduler` 的打印条件是：

- `swiftllm/server/scheduler.py:440`

```python
if pref_to_gpu or pref_to_cpu:
```

也就是说：

只有当本轮 admission 了新的 prefill，这条日志才会出现。

### 6.2 engine 这条日志也不是每轮都打

`engine` 的打印条件是：

- `swiftllm/server/engine.py:264`

```python
if any(b.num_prefs for b in batches):
```

也就是说：

只有当本轮 `batches` 中存在 prefill，这条 `Forwarding batches with sizes ...` 才会出现。

因此，下面这些轮次都可能是**静默的**：

- 纯 decode 轮次；
- 没有 prefill 的轮次；
- 某些只推进已有 decode 请求的轮次。

这正是容易误判的地方：

> 用户看到的两对日志之间，不代表系统只执行了这两次可见的调度/执行。

完全可能中间夹着一轮或多轮没有打印的 iteration。

## 7. 如何重建这段日志的真实语义

结合代码语义，这段日志应理解为：

### 第一组可见日志

```text
Gdecs: 29, Cdecs: 282, Pr2gs: 1, Pr2cs: 0, Waiting: 0
Forwarding batches with sizes [(0, 1, 29, 0)], swap out: 0, swap in: 0
```

表示：

- scheduler 快照时，已有 29 个 GPU decode request；
- scheduler 快照时，已有 282 个 CPU decode request；
- 本轮新接纳了 1 个 GPU prefill；
- 本轮 launch 的 batch 是：1 个 GPU prefill + 29 个 GPU decode；
- 本轮确实没有 cdec；
- 本轮也没有显式 swap。

### 中间三条 completion

```text
POST /v1/completions HTTP/1.1 200 OK
POST /v1/completions HTTP/1.1 200 OK
POST /v1/completions HTTP/1.1 200 OK
```

这三条日志与 `Request.update_output(...) -> req.finished_event.set()` 的完成路径是对得上的。

也就是说，这段时间里确实有 3 个 request 完成了。

### 第二组可见日志

```text
Gdecs: 29, Cdecs: 279, Pr2gs: 0, Pr2cs: 1, Waiting: 0
Forwarding batches with sizes [(1, 0, 29, 0)], swap out: 0, swap in: 0
```

表示：

- 下一次可见的 scheduler 快照里，`cpu_decoding_q` 已经变成了 279；
- 本轮新接纳了 1 个 CPU prefill；
- 本轮 launch 的 batch 仍没有 cdec；
- 但这不妨碍在它之前某些 finished request 已经被从 `cpu_decoding_q` 过滤掉。

## 8. 为什么 `282 -> 279` 与 3 条 completion 高度一致

最强的旁证就是数量正好对上：

- 你看到 `Cdecs` 从 `282` 下降到 `279`，减少了 `3`；
- 中间恰好出现了 `3` 条 completion 日志。

因此最自然、最符合源码控制流的解释是：

> **这 3 个 completion 对应 3 个已经结束的 request，而这些 request 在下一次 scheduler 看队列之前，被 `remove_finished_requests(...)` 从 `cpu_decoding_q` 中清掉了。**

这里不需要额外假设：

- 发生了某个没记录的 swap；
- 或者本轮其实偷偷跑了 cdec。

只靠现有控制流就足够解释这个现象。

## 9. 最终结论

这段日志的正确理解应该是：

1. `scheduler` 日志中的 `Cdecs` 是 `cpu_decoding_q` 的队列长度快照，不是本轮 batch 中的 `num_cdecs`；
2. `engine` 日志中的 `[(0, 1, 29, 0)]` 和 `[(1, 0, 29, 0)]` 只表示这两轮被 launch 的 batch 内容都没有 cdec；
3. `cpu_decoding_q` 的长度变化，不只由 swap 决定，也会由 forward 结束后的 `remove_finished_requests(...)` 清理 finished request 决定；
4. 由于日志是稀疏打印的，两次可见日志之间可以存在静默的 decode iteration；
5. 因此，“没有 cdec batch、没有 swap，却看到 `Cdecs` 从 282 变成 279”是**可以被代码正常解释的现象**，并不构成 scheduler 与 engine 的逻辑矛盾。

一句话总结：

> **你看到的是“本轮 batch 内容”和“下一次 scheduler 队列快照”之间的差异，而不是同一个量的前后变化；`Cdecs` 的下降来自 finished request cleanup，而不需要依赖 cdec batch 或 swap。**
