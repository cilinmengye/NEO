# `rate = -1` 吞吐测试导致客户端/服务端崩溃的调研结论

## 1. 结论先行

你遇到的现象，本质上不是“vLLM 吞吐太低”，也不是“少了一个 `sleep` 就一定会崩”，而是：

1. **当前 throughput 模式会在极短时间内创建海量并发任务，并立即发起海量 TCP 建连。**
2. **每个请求又单独新建一个 `aiohttp.ClientSession`，进一步放大了连接风暴。**
3. **客户端先在本机 socket / ephemeral port / connector 这一层耗尽资源，于是报 `Cannot assign requested address`。**
4. **一旦 `asyncio.gather(*tasks)` 中有一个请求抛异常，`run_test()` 就会异常退出；外层 `try/finally` 立刻停服，于是 vLLM 把还没处理完的请求统一记成 `Aborted request`。**

所以：

- `Cannot assign requested address` 是**客户端本地资源耗尽**的信号；
- `Aborted request` 大量出现，更多是**测试流程被中途打断后的连锁反应**，不一定说明 vLLM 本身内部逻辑有 bug；
- 真正需要修的，不是“补一个很小的 sleep”这么简单，而是要把 throughput test 从“无限制瞬时洪峰”改成“**受控高并发饱和**”。

---

## 2. 代码证据链

### 2.1 throughput 模式会瞬时创建全部请求任务

`NEO/evaluation/benchmark.py:30-58` 的逻辑是：

- `rate > 0` 时才会根据指数分布 gap 做发送间隔控制；
- `rate <= 0` 时，`gaps` 直接变成全 0；
- 循环里持续 `asyncio.create_task(...)`；
- 然后一次性 `await asyncio.gather(*tasks)`。

关键位置：

- `NEO/evaluation/benchmark.py:35`：`rate: float = -1 # -1 means throughput test`
- `NEO/evaluation/benchmark.py:52`：`gaps = ... if rate > 0 else [0] * len(prompts)`
- `NEO/evaluation/benchmark.py:54`：`asyncio.create_task(request_completions_task(...))`
- `NEO/evaluation/benchmark.py:56-57`：只有 `rate > 0` 才 `await asyncio.sleep(...)`
- `NEO/evaluation/benchmark.py:58`：`times = await asyncio.gather(*tasks)`

这意味着在 `rate = -1` 时，**不是“尽可能快地持续压测”**，而是**先把所有请求几乎同时放出去**。

---

### 2.2 每个请求都新建一个 `aiohttp.ClientSession`

`NEO/evaluation/api_client.py:10-30` 里：

- 每调用一次 `request_completions(...)`
- 就执行一次 `async with aiohttp.ClientSession(...) as session:`
- 然后立刻 `session.post(...)`

关键位置：

- `NEO/evaluation/api_client.py:16`：`async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:`
- `NEO/evaluation/api_client.py:25`：`async with session.post(url=api_url, json=payload) as response:`

这说明当前实现不是“一个测试复用一个 session / connector”，而是：

- **一个请求 = 一个 Session = 一套独立连接管理开销**。

当 `benchmark.py` 在 throughput 模式下一口气创建海量任务时，这种写法会把连接创建压力进一步放大。

---

### 2.3 `fig10b` 的请求规模非常大

`NEO/evaluation/reproduce-fig10b.py:41-59` 中：

- `num_threshold = 200000`
- 分别读取 code trace 和 conv trace
- 然后把两者拼起来

关键位置：

- `NEO/evaluation/reproduce-fig10b.py:49`：`num_threshold = 200000`
- `NEO/evaluation/reproduce-fig10b.py:50-51`：分别读取两份 trace
- `NEO/evaluation/reproduce-fig10b.py:53-56`：两份数据拼接成一个 `prompts` / `output_lens`
- `NEO/evaluation/reproduce-fig10b.py:68`、`NEO/evaluation/reproduce-fig10b.py:70`：对 `ours` 和 `vllm` 都调用 `run_test(..., rate=-1)`

也就是说，吞吐测试不是几百个、几千个请求，而是**接近 40 万个请求**会被并入同一轮测试。

这和“瞬时全部建连”的组合非常危险。

---

### 2.4 服务端的并发承载能力是有限的

`NEO/evaluation/server.py:26-49` 和 `NEO/evaluation/configs/config-4090-8b.json:1-14` 显示：

- vLLM 启动时显式限制了 `max_num_seqs`
- 以及 `max_num_batched_tokens`

关键位置：

- `NEO/evaluation/server.py:28`：`max_num_seqs = min(int(chunk_size_str), config["max_num_seqs"])`
- `NEO/evaluation/server.py:34`：`--max-num-seqs`
- `NEO/evaluation/server.py:35`：`--max-num-batched-tokens`
- `NEO/evaluation/configs/config-4090-8b.json:7`：`"max_num_seqs": 1024`
- `NEO/evaluation/configs/config-4090-8b.json:8`：`"max_num_batched_tokens": 26400`

这说明服务端本来就是按照“**有限并发 + 排队调度**”来工作的。

而客户端现在做的是“**40 万级任务瞬时抢建连接**”，两者不匹配。

---

## 3. `Cannot assign requested address` 到底是什么意思？

你的 traceback 最关键的是这一段：

- `aiohttp/connector.py` 在 `_wrap_create_connection` 阶段失败
- 底层是 `loop.sock_connect(sock, address)`
- 最终抛出 `OSError: [Errno 99] Cannot assign requested address`
- 再被包装成 `aiohttp.client_exceptions.ClientConnectorError`

这类错误说明失败发生在：

- **客户端试图创建/分配本地 socket 并连接到 `localhost:8000` 的阶段**。

它通常不是以下问题：

- 不是 HTTP payload 格式错了；
- 不是服务端返回了 4xx/5xx；
- 不是模型推理超时；
- 不是 tokenizer 逻辑错误。

它更像是以下资源被打满：

1. **ephemeral port 耗尽**
   本机向同一目标地址/端口发起大量并发 TCP 连接时，需要为每条连接分配本地临时端口。端口数量是有限的，而且关闭后的连接还会经历 `TIME_WAIT`，不能立即复用。

2. **瞬时 connect 风暴过大**
   即使理论端口还没完全耗尽，短时间内海量并发 `connect()` 也可能让本地网络栈、accept backlog、文件描述符或调度资源先扛不住。

3. **每请求独立 session/connector 带来的额外资源膨胀**
   当前不是复用连接池，而是为每个请求重复创建连接管理对象，导致资源消耗更快。

结合当前代码，我认为你这里最可能的直接机制是：

> **throughput 模式一次性触发几十万协程，每个协程都新建 `ClientSession` 并尝试连接 `localhost:8000`，导致客户端本地 TCP 连接资源分配失败，于是报 `Cannot assign requested address`。**

这和服务端是不是 vLLM、是不是 localhost，其实关系不大；只要客户端无上限建连，任何 HTTP server 都可能被这种压测方式拖死。

---

## 4. 为什么 vLLM 会打印大量 `Aborted request`？

这部分从日志和控制流都能解释通。

### 4.1 日志先显示服务端进入 shutdown，再出现大批 aborted

`NEO/evaluation/vllm-server.log:9964-9966` 开始出现：

- `Waiting for background tasks to complete.`
- `Waiting for application shutdown.`
- 随后大量 `Aborted request ...`

关键片段：

- `NEO/evaluation/vllm-server.log:9964`：`Waiting for background tasks to complete. (CTRL+C to force quit)`
- `NEO/evaluation/vllm-server.log:9965`：`Waiting for application shutdown.`
- `NEO/evaluation/vllm-server.log:9966-10087`：大量 `Aborted request ...`

这说明 `Aborted request` 不是凭空出现的，而是发生在**服务正在退出**的时候。

---

### 4.2 外层 `try/finally` 会在 `run_test()` 抛异常后立即停服

`NEO/evaluation/reproduce-fig10b.py:62-73`：

- `start_server(...)`
- `await run_test(...)`
- `finally: stop_server()`

关键位置：

- `NEO/evaluation/reproduce-fig10b.py:63`：`start_server(server_name, config)`
- `NEO/evaluation/reproduce-fig10b.py:68` / `70`：`await run_test(..., rate=-1)`
- `NEO/evaluation/reproduce-fig10b.py:71-72`：`finally: stop_server()`

而 `run_test()` 里又是：

- `NEO/evaluation/benchmark.py:58`：`await asyncio.gather(*tasks)`

`asyncio.gather()` 默认行为是：**只要有一个任务抛异常，外层 await 就会收到异常并提前返回**。于是调用栈会马上离开 `run_test()`，进入外层 `finally`，然后 `stop_server()`。

因此链路是：

1. 某些请求先因为客户端资源耗尽而连接失败；
2. `gather(*tasks)` 收到异常；
3. `run_test()` 异常退出；
4. `reproduce-fig10b.py` 的 `finally` 立即停掉 vLLM；
5. vLLM 对仍在队列/执行中的请求统一做 abort；
6. 日志里就刷出大量 `Aborted request ...`。

所以这些 aborted request 很大概率是**后果**，不是最初的根因。

---

## 5. 为什么“吞吐测试”不应该等于“无限制瞬时洪峰”？

这是这次问题里最关键的理解点。

### 5.1 你现在测到的，其实混入了大量“客户端自伤”成本

当前 throughput 模式测到的，不只是：

- 服务端排队能力
- GPU 解码能力
- 调度器吞吐能力

还混入了：

- Python 一次性创建几十万任务的开销
- `aiohttp` 创建几十万个 `ClientSession` 的开销
- 本地 TCP 海量并发建连开销
- 内核端口与 socket 资源耗尽

这意味着当前测试方式已经偏离了“测服务端吞吐”的目标。

---

### 5.2 真正合理的吞吐测试应当是“持续压满”，不是“瞬时炸满”

吞吐测试的合理目标通常是：

- 让服务端一直处于高负载；
- 始终有足够多的请求等待/执行；
- 但客户端自身不应该先崩溃。

也就是说，更合理的方式是：

- **不控制到达率（rate=-1 仍然成立）**，但
- **控制最大在途请求数（outstanding / inflight requests）**。

这叫“受控饱和”而不是“无限 fan-out”。

举例说：

- 如果你设置 `max_inflight = 256 / 512 / 1024`，
- 那么客户端会一直把服务端维持在高压状态；
- 某个请求结束，就立刻补下一个；
- 这样仍然能测可持续吞吐；
- 但不会因为客户端一次性去建 40 万个 TCP 连接而自爆。

---

## 6. 这次问题的根因拆解

我把根因分成三层：

### 6.1 第一层根因：无上限的任务 fan-out

`NEO/evaluation/benchmark.py:53-58` 在 throughput 模式下没有任何 in-flight 上限，直接创建所有任务。

这是最核心的问题。

---

### 6.2 第二层根因：每请求新建 `ClientSession`

`NEO/evaluation/api_client.py:16` 的 session 生命周期太短。

这会导致：

- 连接池无法复用；
- 连接复用几乎失效；
- connector 无法承担“限流缓冲器”的作用；
- 本地 socket/端口更快耗尽。

这是放大器。

---

### 6.3 第三层根因：异常传播方式导致整轮测试被首个失败打断

`NEO/evaluation/benchmark.py:58` 的 `asyncio.gather(*tasks)` 一旦遇到一个失败，整轮就直接抛出；而 `reproduce-fig10b.py:71-72` 会立即停服。

这使得：

- 你很难区分到底失败了多少请求；
- 很难产出完整结果文件；
- 服务端日志会被大量 aborted request 污染。

这是“雪上加霜”的问题。

---

## 7. 我建议的解决方向

下面是按优先级排序的建议。

### 7.1 第一优先级：给 throughput 模式增加 `max_inflight`

建议把 `rate = -1` 保留为“**不做 arrival-rate sleep 节流**”，但不要等价为“无限并发建连”。

应该改成：

- 最多允许 `max_inflight` 个请求同时在途；
- 有请求完成就补发下一个；
- 始终让服务端保持饱和；
- 但不制造客户端连接风暴。

推荐形式：

- `run_test(..., rate=-1, max_inflight=...)`
- 或 throughput 模式下内部启用 bounded concurrency 调度。

这一步是最关键的。

---

### 7.2 第二优先级：一个测试轮次复用一个共享 `ClientSession`

建议把 `aiohttp.ClientSession` 从“每请求创建”改成“每轮测试创建一次并复用”。

同时建议显式设置：

- `aiohttp.TCPConnector(limit=..., limit_per_host=...)`

并让 connector 的 limit 与 `max_inflight` 对齐。

这样有几个好处：

- 避免每请求创建一套 session/connector；
- 允许连接复用；
- 让客户端并发形态更稳定；
- 把测试重点重新放回服务端吞吐，而不是客户端建连极限。

---

### 7.3 第三优先级：不要因为首个请求失败就整轮崩掉

建议把 throughput 模式的结果收集改成：

- 记录成功请求；
- 记录失败请求及错误类型；
- 不要因为第一个 `ClientConnectorError` 就直接退出整轮。

否则现在的行为会导致：

- 测试一失败就停服；
- 服务端日志刷大量 `Aborted request`；
- 结果文件不完整；
- 无法判断到底是“偶发失败”还是“整体压测配置错误”。

---

### 7.4 第四优先级：把 `fig10b` 的数据量和并发上限参数化

`NEO/evaluation/reproduce-fig10b.py:49` 现在把 `num_threshold` 固定死为 `200000`。

建议把以下两个参数暴露出来：

1. `num_threshold`
2. `max_inflight`

这样你可以采用更稳妥的验证顺序：

1. 先用较小数据量（例如 1k / 5k）验证机制是否正确；
2. 再逐步提高 `max_inflight`；
3. 最后再扩大数据量，找到既能打满服务端又不让客户端自爆的区间。

---

## 8. 哪些方案不建议作为主方案？

### 8.1 只加一个很小的 `sleep`

如果只是简单在 throughput 模式里补一个固定小延迟，比如几毫秒：

- 短期可能缓解崩溃；
- 但它实际上把测试变成了“限速到达率测试”；
- 不能从根本上解决“请求在途数量无上限”的问题。

所以它可以作为辅助平滑手段，但**不应是主方案**。

---

### 8.2 直接增大系统内核参数

比如去调：

- `ulimit -n`
- ephemeral port range
- TCP reuse / TIME_WAIT 相关参数

这些在极端压测时当然可能有帮助，但这里**不是首选**，因为：

- 当前压测方式本身就不合理；
- 即使调大系统参数，也只是把崩溃点往后推；
- 对测试可重复性和可移植性都不好。

先修测试逻辑，再考虑系统参数，顺序不能反。

---

### 8.3 把 `max_num_seqs` 一味调大

服务端调度上限不是没有意义，但这也不是这次问题的第一修复点。

因为当前首先崩的是：

- 客户端建连层
- 而不是服务端排队调度层

所以先修客户端注入方式，比先改 vLLM 启动参数更重要。

---

## 9. 推荐的验证顺序

如果后续要改代码，我建议这样验证：

### 第一步：先验证文档里的判断

先确认是否满足以下现象：

- `rate=-1` 时问题出现；
- 小规模请求时不容易出现；
- 请求量和并发度升高时更容易出现；
- 先出现客户端连接错误，再出现服务端 aborted request。

这和当前代码/日志是吻合的。

---

### 第二步：先做小规模 throughput test

例如：

- 降低 `num_threshold`
- 同时给 throughput 模式设置较小 `max_inflight`

检查：

- 是否还出现 `Cannot assign requested address`
- 是否还会在刚开始时就触发服务端 shutdown
- 是否结果文件能正常产出

---

### 第三步：逐步提高 `max_inflight`

例如逐步尝试：

- 64
- 128
- 256
- 512
- 1024

观察：

- 客户端是否稳定；
- 服务端是否持续繁忙；
- throughput 是否趋于稳定；
- 是否出现新的瓶颈（例如服务端 5xx、显存、CPU tokenize、文件描述符等）。

---

### 第四步：最后再扩大数据量

等确认机制稳定后，再把 `num_threshold` 提高到更接近原始规模。

这样得到的吞吐结果才更可信。

---

## 10. 最终判断

### 10.1 这是什么原因？

根因是：

> `rate = -1` 时，`benchmark.py` 把所有请求瞬时 fan-out；而 `api_client.py` 又为每个请求新建 `aiohttp.ClientSession`；在 `fig10b` 的几十万请求规模下，客户端先耗尽本地 TCP 连接资源，因此报 `Cannot assign requested address`。

### 10.2 为什么会有大量 `Aborted request`？

因为：

> 某些客户端请求先连接失败，导致 `asyncio.gather(*tasks)` 抛异常；外层 `try/finally` 立刻停掉 vLLM；服务端在 shutdown 时将尚未完成的请求统一 abort，所以日志里出现大量 `Aborted request`。

### 10.3 应该如何解决？

推荐的主方案是：

1. **throughput 模式增加 `max_inflight`，改成受控高并发饱和测试；**
2. **单轮测试复用共享 `aiohttp.ClientSession` + `TCPConnector`；**
3. **改进异常收集方式，不要首个请求失败就整轮中断并停服；**
4. **把 `fig10b` 的数据规模与并发参数化，先小规模验证再逐步放大。**

这四点里，前两点最关键。

---

## 11. 最简短版本

如果只用一句话概括：

> 你现在的 throughput test 不是“把服务端打满”，而是“把客户端先打爆了”；`Cannot assign requested address` 是客户端本机建连资源耗尽，`Aborted request` 则是测试异常后服务端被提前停掉的连锁反应。
