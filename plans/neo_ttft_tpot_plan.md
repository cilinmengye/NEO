# Context

当前 NEO 的评测链路只能稳定产出“整条 request 的完成延迟”，也就是从请求发送到完整响应结束的总时长。现有实现里：

- `evaluation/api_client.py` 走的是 non-streaming 请求，等完整 body 返回后一次性读取；
- `evaluation/benchmark.py` 只记录 `start/end/input_len/output_len/ok/error`；
- `evaluation/reproduce-fig6b.py` 复用 `run_test(...)` 的结果文件；
- `evaluation/illustrator.py` 再用 `(end - start) / output_len` 画当前的平均 per-token latency 曲线。

因此现在能比较的，本质上是“请求级 completion latency 的平均化变体”，而不是严格意义上的：

- **TTFT**: Time To First Token
- **TPOT**: Time Per Output Token / inter-token generation latency

要实现对 vLLM 与 NEO 的 TTFT / TPOT 对比，核心不是重写整个评测框架，而是：

1. 在 **evaluation 层** 把“完整响应计时”扩展成“流式 token 到达计时”；
2. 保持现有结果文件、绘图脚本、复现实验脚本的组织方式不变；
3. 让旧的 Avg token latency 路径继续可用，同时新增 TTFT / TPOT 路径。

从现有代码看，NEO 服务端已经具备可复用的 streaming 基础：

- `swiftllm/server/api_server.py` 已支持 `stream=True`；
- `swiftllm/server/engine.py` 的 `add_request_and_stream(...)` 会逐 token yield；
- `swiftllm/structs.py` 的 `Request.update_output(...)` 每生成一个 token 就会向 `output_q` 推一个 `StepOutput`。

因此，这次实现的推荐方向是：**优先用客户端流式观测来测量用户视角的 TTFT / TPOT，而不是先改 NEO 运行时内部埋点。**

# Recommended approach

1. **把 TTFT / TPOT 作为现有 latency benchmark 的增强，而不是新增一套平行框架。**
   - 保留 `evaluation/benchmark.py -> results/*.json -> evaluation/illustrator.py -> reproduce-fig6b.py` 这条主链路。
   - 不新建一套完全独立的 benchmark 脚本。
   - 继续沿用当前 `-lat-{rate}.json` 的结果文件风格，只是在每条请求记录里补充流式时间字段。

2. **先在 `evaluation/api_client.py` 增加统一的“流式 completion trace”抽象。**
   - 当前 `request_completions(...)` 只做 non-streaming：`session.post(...)` 后 `await response.text()` 一次性读完。
   - 推荐改成支持两种消费模式：
     - **legacy non-streaming**：保持现有行为，兼容旧 benchmark；
     - **streaming trace mode**：按 token 边界增量读取响应，并记录每个 token 到达时刻。
   - 这个模块应负责“不同后端的协议归一化”，不要把 NEO/vLLM 的响应细节泄露到 `benchmark.py`。
   - 推荐在客户端层输出统一结构，例如：
     - `output_token_ids` 或 `tokens_received`
     - `token_offsets`：相对于请求 `start` 的每个 token 到达偏移
     - `first_token_offset`
     - `streamed=True/False`
   - 对 NEO：
     - 利用现有 `/v1/completions` + `stream=True`
     - 按行消费 `text/plain`，每行一个 token id
   - 对 vLLM：
     - 在同一个文件里加一个适配路径
     - 把它的流式响应归一化成相同的 `token_offsets`
   - 这样后续 benchmark/plotter 都只依赖统一 trace，不依赖具体后端协议。

3. **在 `evaluation/benchmark.py` 扩展请求结果 schema，但保留旧字段完全兼容。**
   - 当前每条请求记录只有：
     - `input_len`
     - `output_len`
     - `start`
     - `end`
     - `ok`
     - `error`
   - 推荐新增可选字段，而不是重构为新的顶层 JSON 结构。建议新增：
     - `streamed`: bool
     - `tokens_received`: int
     - `token_offsets`: list[float]
     - `first_token_offset`: float | null
   - 保留 `start/end`，这样旧的 Avg token latency 计算仍然可用。
   - `run_test(...)` 推荐新增一个显式参数，例如：
     - `collect_stream_metrics=False`
     - 必要时再加 `backend` / `server_name`
   - 这样：
     - 旧的调用点不变；
     - 只有需要 TTFT/TPOT 的 rate benchmark 才开启流式记录。

4. **把 TTFT 和 TPOT 的定义固定为“客户端观察到的 token 到达时间”。**
   - 基于 `token_offsets = [t1, t2, ..., tn]`：
     - `TTFT = t1`
     - `TPOT = mean(t2 - t1, t3 - t2, ..., tn - t(n-1))`
   - 这一定义有两个优点：
     - 与用户真实感知一致；
     - 不需要先改 `swiftllm` 运行时代码。
   - 需要明确边界处理：
     - 若 `tokens_received == 0`：该请求不计入 TTFT/TPOT；
     - 若 `tokens_received == 1`：该请求可计入 TTFT，但不计入 TPOT；
     - 若 `tokens_received != output_len`：默认视为异常或不一致记录，应在 benchmark 中显式报错或剔除，并在日志中说明。

5. **让 `evaluation/illustrator.py` 变成“按指标选择 reducer”的绘图模块。**
   - 当前 `draw_one_rl_diagram(...)` 固定用 `get_lat_avg(...)`，而 `get_lat_avg(...)` 固定计算 `(end - start) / output_len`。
   - 推荐改造成：
     - 保留当前默认指标 `avg_per_token_latency`；
     - 新增 `ttft` 和 `tpot` 两种 reducer；
     - `draw_one_rl_diagram(...)` 新增 `metric` 参数。
   - 推荐保留旧语义为默认值：
     - 不显式传 `metric` 时，仍画现有的 Average per token latency。
   - 新增辅助函数：
     - `get_ttft_avg(file)`
     - `get_tpot_avg(file)`
     - 或者统一为一个 `get_metric_avg(file, metric)`
   - 旧结果文件若不包含 `token_offsets` / `first_token_offset`，则：
     - 对 `avg_per_token_latency` 仍可继续使用；
     - 对 `ttft` / `tpot` 应明确报错，提示需要重新以 streaming mode 跑 benchmark。

6. **在 `evaluation/reproduce-fig6b.py` 中复用同一批 rate sweep，同时输出三类图。**
   - 当前 `reproduce-fig6b.py` 的职责是：
     - 启动 server
     - 调 `run_test(...)`
     - 再调用 `draw_one_rl_diagram(...)`
   - 推荐保持这个结构不变，只做两处升级：
     1. rate benchmark 调用 `run_test(..., collect_stream_metrics=True, ...)`
     2. 图生成阶段分别画：
        - 现有 Avg token latency 图
        - TTFT 图
        - TPOT 图
   - 推荐不要覆盖当前 `fig6b.pdf` 的语义，而是：
     - 保留 `fig6b.pdf` 继续表示现有 Average per token latency；
     - 新增例如：
       - `fig6b-ttft.pdf`
       - `fig6b-tpot.pdf`
   - 这样不会影响当前已有结果的解释方式，也更容易和论文原图及新指标并存。

7. **NEO 服务端第一阶段不改运行时代码，只复用现有 streaming 接口。**
   - 当前服务端已具备实现用户视角 TTFT / TPOT 所需的基本能力：
     - `swiftllm/server/api_server.py:16-45`
     - `swiftllm/server/engine.py:167-181`
     - `swiftllm/structs.py:93-108`
   - 因此推荐第一阶段**不修改 `swiftllm/` 下运行时代码**。
   - 只有在实现时发现以下问题时，才再考虑服务端变更：
     - 需要更明确的流结束标记；
     - 需要统一 NEO 与 vLLM 的 stream 格式；
     - 需要服务端时间戳来做更细粒度的“阶段拆解 TTFT”。
   - 但这不属于当前“比较 TTFT / TPOT”这个目标的最小实现范围。

8. **处理 benchmark 缓存兼容性，避免旧 JSON 被误用。**
   - 当前 `run_test(...)` 若发现结果文件已存在，会直接复用。
   - 这会导致一个问题：
     - 老的 `-lat-*.json` 不含 `token_offsets`；
     - 新的 TTFT / TPOT 图如果直接读取，会得到错误结论或报不清晰的错。
   - 推荐在 `collect_stream_metrics=True` 时，增加缓存兼容检查：
     - 若缓存文件缺少流式字段，则视为“不兼容缓存”；
     - 自动重跑，或清晰报错提示必须重新采样。
   - 推荐实现成显式检查逻辑，而不是用近似值兜底。

9. **把验证流程设计成“客户端 trace 正确性 + 聚合指标正确性 + 复现实验正确性”三层。**
   - 第一层：客户端 streaming trace
     - 小请求下确认 token 数、token_offsets 长度、单调性、first_token_offset 正确。
   - 第二层：指标计算
     - 人工核对某条请求的 `TTFT` 与 `TPOT` 是否和 `token_offsets` 一致。
   - 第三层：端到端复现实验
     - 用小规模 rate sweep 同时跑 `ours` 与 `vllm`
     - 确认生成三张图，并且旧 Avg latency 图不被破坏。

# Critical files to modify

推荐执行阶段重点修改以下文件：

- `/home/yxlin/github/swift/NEO/evaluation/api_client.py`
  - 增加 streaming 请求与流式 trace 归一化逻辑；
  - 在同一层处理 NEO / vLLM 的流协议差异。

- `/home/yxlin/github/swift/NEO/evaluation/benchmark.py`
  - 扩展单请求结果结构；
  - 增加 `collect_stream_metrics` 控制参数；
  - 增加缓存兼容检查与 streaming 字段写出逻辑。

- `/home/yxlin/github/swift/NEO/evaluation/illustrator.py`
  - 新增 TTFT / TPOT 聚合逻辑；
  - 让 `draw_one_rl_diagram(...)` 支持 `metric` 选择，同时保留当前默认行为。

- `/home/yxlin/github/swift/NEO/evaluation/reproduce-fig6b.py`
  - 在当前 rate benchmark 中开启 streaming metric 采样；
  - 在同一套结果文件上新增 TTFT / TPOT 绘图输出。

在第一阶段，**不建议主动修改**以下文件，除非实现时发现协议契约不足：

- `/home/yxlin/github/swift/NEO/swiftllm/server/api_server.py`
- `/home/yxlin/github/swift/NEO/swiftllm/server/engine.py`
- `/home/yxlin/github/swift/NEO/swiftllm/structs.py`

# Existing functions and utilities to reuse

- `/home/yxlin/github/swift/NEO/evaluation/api_client.py`
  - `request_completions(...)`
  - 当前是统一的 completion 请求入口，适合作为扩展 streaming client 的主入口。

- `/home/yxlin/github/swift/NEO/evaluation/benchmark.py`
  - `request_completions_task(...)`
  - `_run_rate_test(...)`
  - `run_test(...)`
  - 当前已负责单请求计时、rate sweep、结果缓存与 JSON 输出，适合直接扩展而不是另起新 benchmark。

- `/home/yxlin/github/swift/NEO/evaluation/reproduce-fig6b.py`
  - `one_round(...)`
  - 当前已经是 Fig 6b 的 server 启动 + benchmark + plotting 总入口，适合直接复用。

- `/home/yxlin/github/swift/NEO/evaluation/illustrator.py`
  - `_get_successful_records(...)`
  - `get_lat_avg(...)`
  - `draw_one_rl_diagram(...)`
  - 当前已承担结果聚合与绘图职责，应在此处扩展 TTFT / TPOT reducer，而不是新建另一个 plotting 模块。

- `/home/yxlin/github/swift/NEO/swiftllm/server/api_server.py`
  - `generate(...)`
  - 已支持 `stream=True`，可直接作为 NEO 客户端 TTFT / TPOT 观测的服务端基础。

- `/home/yxlin/github/swift/NEO/swiftllm/server/engine.py`
  - `add_request_and_stream(...)`
  - 已经逐 token yield `StepOutput`，说明 NEO 的流式接口在实现上天然适合客户端计时。

- `/home/yxlin/github/swift/NEO/swiftllm/structs.py`
  - `Request.update_output(...)`
  - 每个 token 生成后都会 `put_nowait(StepOutput(...))`，这是 NEO 当前 streaming 粒度的根源。

# Verification

1. **NEO streaming 基础验证**
   - 用一个小请求对 `/v1/completions` 发送 `stream=True`；
   - 确认客户端确实能逐行收到 token id；
   - 确认记录到的 `token_offsets` 数量与请求的 `output_len` 一致。

2. **vLLM streaming 适配验证**
   - 确认当前评测环境中的 vLLM 端点支持 streaming；
   - 确认客户端能从其响应中稳定提取 token 到达边界；
   - 若只能提取 first chunk 而不能稳定识别每个 token，则本轮实现只能先落地 TTFT，TPOT 需要额外适配。

3. **结果 schema 兼容性验证**
   - 新跑出的 `-lat-*.json` 应同时包含旧字段和新增 streaming 字段；
   - 旧 `illustrator.py` 默认 Avg token latency 路径在新文件上仍可正常工作；
   - 新的 `ttft` / `tpot` 路径在旧文件上应报出清晰的缺字段错误，而不是静默使用近似值。

4. **指标计算正确性验证**
   - 抽样检查至少一条请求记录：
     - `TTFT == token_offsets[0]`
     - `TPOT == mean(diff(token_offsets))`（从第二个 token 开始的间隔均值）
   - 对 `tokens_received <= 1` 的特殊记录确认边界处理符合预期。

5. **端到端复现实验验证**
   - 用小规模 rate list 分别对 `ours` 与 `vllm` 跑一次 `reproduce-fig6b.py`；
   - 确认能产出：
     - 旧 Avg token latency 图；
     - 新 TTFT 图；
     - 新 TPOT 图；
   - 确认三张图使用相同的 rate sweep 与系统图例。

6. **回归验证**
   - 确认其他不关心 streaming 指标的评测脚本仍可运行；
   - 特别是 throughput 路径不应被强制依赖 streaming 字段。

7. **实现阶段需优先验证的关键假设**
   - vLLM 在当前环境下是否提供可用于 token 级测量的 streaming 输出；
   - vLLM streaming 输出是否能稳定对齐到“每个 token 一个事件”，而不是任意文本 chunk；
   - 当前 benchmark 设置下，请求的实际输出 token 数是否稳定等于 `output_len` / `max_tokens`；
   - `aiohttp` 的增量读取应按逻辑记录边界（line / SSE event）计时，而不是按底层 transport chunk 计时。

# vLLM streaming token mismatch diagnosis

## Context

当前报错的直接表现是：`evaluation/benchmark.py` 在读取 vLLM 的流式结果后，发现 `tokens_received != output_len`，例如 `expected 167, got 163`，于是抛出 `RuntimeError`。

这不是 Docker 挂载路径 `/home/yxlin/github/swift/NEO -> /workspace/NEO` 本身导致的问题。挂载只会让报错里的文件路径显示成容器内路径；真正的问题在于 **当前 `evaluation/api_client.py` 对 vLLM streaming 响应的解析假设过强**。

现状里：
- NEO 的 `/v1/completions` + `stream=True` 是 `text/plain`，服务端每生成 1 个 token 就输出 1 行 token id；
- vLLM 是 SSE 风格流，事件里的 `choices[*].text` 更像“文本 chunk”，**不保证一条 event 就是一个 token**；
- 当前实现把没有 `token_ids` 的非空 `text` 直接按 `1 token` 计数；
- 因此对 vLLM 来说，`tokens_received` 实际上统计的是“带内容的 chunk/event 数”，而不是真实输出 token 数；
- 随后 `benchmark.py` 又对所有后端统一强制校验 `tokens_received == output_len`，于是触发当前报错。

所以这次修复的核心目标不是改 Docker，也不是推翻整条评测链路，而是：
1. 把 **NEO 的 token-level streaming** 和 **vLLM 的 chunk-level/SSE streaming** 区分开；
2. 只在“确实能证明是 token 级观测”的场景下计算精确 TPOT；
3. 保留现有 `evaluation/` 目录下的整体结构不变，用最小改动修正指标语义和校验逻辑。

## Recommended approach

1. **先把根因固定为“协议粒度不一致”，不要把 Docker 挂载当作修复方向。**
   - `expected 167, got 163` 的本质不是路径错了，而是 vLLM streaming 里一个 SSE event 可能携带多个 token，对应当前 `api_client.py` 的计数被低估。
   - 因此后续实现应围绕“stream event 是否等于 token”来修，而不是围绕容器路径做兼容。

2. **在 `evaluation/api_client.py` 中显式区分 streaming 协议的粒度与可靠性。**
   - 保留当前 NEO 的 plain-text per-line 解析方式，因为它确实对应 `1 line = 1 token`。
   - 对 vLLM 的 SSE 路径，不要再把 `choice.text` 的非空内容机械地当作 `1 token`。
   - 推荐把客户端流式返回结构补充为“观测语义”而不只是“数量”：
     - `streamed`: 是否走了流式接口
     - `stream_observation`: 例如 `token` / `chunk`
     - `events_received`: 接收到多少个有效流事件
     - `tokens_received`: 仅在能确认 token 级边界时写入
     - `token_offsets`: 仅当能确认 token 级边界时才写入真正的 token offsets
     - `first_token_offset`: 首个带输出内容事件的时间；对 vLLM 可继续作为 TTFT 候选值
   - 若 vLLM 当前响应里没有可靠 `token_ids` 或其他 token-level 边界信息，则应把它视为 **chunk-level observation**，而不是伪装成 token-level。

3. **在 `evaluation/benchmark.py` 中把“严格 token 数校验”改成按观测粒度决定。**
   - 现在的全局校验 `tokens_received == output_len` 只适用于 NEO 这种 token-level streaming。
   - 推荐改成：
     - 当 `stream_observation == "token"` 时，保留严格校验；
     - 当 `stream_observation == "chunk"` 时，不再要求 `events_received == output_len` 或 `tokens_received == output_len`；
     - 仍然保留 `start/end` 等旧字段，保证原有 avg latency 路径不受影响。
   - 对 vLLM，如果只有 chunk-level 观测，则 benchmark 不应因为“chunk 数不等于 token 数”直接失败。

4. **重新定义这次实现中 vLLM 与 NEO 的指标支持范围。**
   - **NEO**：
     - TTFT：可精确支持
     - TPOT：可精确支持
   - **vLLM（基于当前 `/v1/completions` streaming 形态）**：
     - TTFT：通常仍可用“首个带输出内容的流事件时间”近似/定义
     - TPOT：只有在流里能拿到稳定 token-level 边界时才应支持
   - 如果当前 vLLM 只能提供 chunk-level stream，则不应该继续把 chunk 间隔伪装成 token 间隔来画 TPOT。

5. **在 `evaluation/illustrator.py` 中按“指标是否有可靠语义”决定是否允许绘图。**
   - `avg_per_token_latency` 继续沿用现有 `(end - start) / output_len` 聚合逻辑。
   - `ttft`：
     - 允许消费 NEO token-level 数据；
     - 也允许消费 vLLM 的首个有效输出事件时间，但需要依赖 benchmark 写出的统一字段。
   - `tpot`：
     - 只应在 token-level streaming 数据上计算；
     - 若结果文件标明是 chunk-level 观测，则应报出清晰错误，而不是静默计算一个伪 TPOT。
   - 换句话说，绘图层要消费 benchmark 提供的“语义标签”，而不是仅凭字段存在就默认可算 TPOT。

6. **在 `evaluation/reproduce-fig6b.py` 中对 vLLM 的图生成做最小安全收缩。**
   - 推荐保持现有 Avg latency 图继续生成。
   - 对 TTFT：若 vLLM benchmark 已写出可接受的 `first_token_offset` 语义，则可以继续画 NEO vs vLLM。
   - 对 TPOT：
     - 如果验证后确认 vLLM 当前 stream 不是 token-level，就不要再把 vLLM 纳入 TPOT 对比图；
     - 可选方案是：只画 NEO TPOT，或直接跳过 vLLM 的 TPOT 图并给出清晰提示。
   - 关键点是：**宁可少画一张图，也不要输出语义错误的 TPOT 结论。**

7. **优先做“最小修复”，不要在第一步就改 SwiftLLM/服务端。**
   - 当前问题是评测客户端把 vLLM 的 SSE chunk 误解为 token，不是 NEO 服务端 streaming 的问题。
   - 因此第一阶段仍应只修改 `evaluation/` 下的客户端、benchmark、plotter、复现实验脚本。
   - 只有当你明确需要 vLLM 的精确 TPOT，且当前 API 无法暴露 token-level 边界时，再考虑：
     - 切换 vLLM 的另一种返回格式；
     - 或在服务端/API 层寻找可稳定暴露 token ids 的接口；
     - 但这属于第二阶段，不是当前报错的最小修复范围。

8. **把修复后的语义写清楚，避免未来再次误用缓存结果。**
   - benchmark 结果文件需要能区分：
     - 这是 token-level metrics 还是 chunk-level metrics；
     - 哪些指标是“精确支持”的，哪些只是“首事件时间”。
   - 这样 `illustrator.py` 才能根据结果文件做正确决策，而不是再靠隐含假设运行。

## Critical files to modify

- `/home/yxlin/github/swift/NEO/evaluation/api_client.py`
  - 修正 vLLM streaming 解析语义；
  - 区分 token-level 与 chunk-level 观测；
  - 输出更明确的 stream metadata。

- `/home/yxlin/github/swift/NEO/evaluation/benchmark.py`
  - 去掉对所有后端统一的 `tokens_received == output_len` 强校验；
  - 改为按观测粒度/协议能力决定是否校验；
  - 保证缓存结果里带有足够的语义字段供绘图层判断。

- `/home/yxlin/github/swift/NEO/evaluation/illustrator.py`
  - 让 `ttft` / `tpot` 的计算依赖结果文件中的观测语义；
  - 对 chunk-level 数据禁止计算伪 TPOT。

- `/home/yxlin/github/swift/NEO/evaluation/reproduce-fig6b.py`
  - 调整图生成策略；
  - 在 vLLM 不具备 token-level 数据时，不再强行输出 vLLM TPOT 对比图。

第一阶段**不建议修改**：
- `/home/yxlin/github/swift/NEO/swiftllm/server/api_server.py`
- `/home/yxlin/github/swift/NEO/swiftllm/server/engine.py`
- `/home/yxlin/github/swift/NEO/swiftllm/structs.py`

## Existing functions and utilities to reuse

- `/home/yxlin/github/swift/NEO/evaluation/api_client.py`
  - `_build_completion_payload(...)`
  - `request_completions(...)`
  - `request_completions_stream(...)`
  - 这些已经是统一请求入口，适合继续扩展协议识别与 streaming metadata。

- `/home/yxlin/github/swift/NEO/evaluation/benchmark.py`
  - `request_completions_task(...)`
  - `_results_match_streaming_mode(...)`
  - `run_test(...)`
  - 继续复用这条 benchmark 主链，只修正其对 streaming 结果的解释与校验。

- `/home/yxlin/github/swift/NEO/evaluation/illustrator.py`
  - `_require_streaming_metrics(...)`
  - `get_metric_avg(...)`
  - `draw_one_rl_diagram(...)`
  - 应在这里增强“指标是否可计算”的判断，而不是新建绘图框架。

- `/home/yxlin/github/swift/NEO/evaluation/reproduce-fig6b.py`
  - `one_round(...)`
  - `main()`
  - 继续复用当前复现实验入口，只调整哪些指标允许对哪些系统绘图。

- `/home/yxlin/github/swift/NEO/swiftllm/server/api_server.py`
  - `generate(...)`
  - 已确认 NEO streaming 是每 token 一行，这是 NEO 可精确测 TTFT/TPOT 的依据。

## Verification

1. **协议粒度验证**
   - 分别对 NEO 与 vLLM 发一个小的 `stream=True` completion 请求；
   - 确认 NEO 的每次流输出严格对应 1 token；
   - 确认 vLLM 的每个 SSE event 是否可能包含多 token 文本 chunk。

2. **客户端结果语义验证**
   - 检查修复后的单条结果记录：
     - NEO 应标记为 token-level；
     - vLLM 若没有 token ids，应标记为 chunk-level；
     - `first_token_offset` 应在两者中都只对应“首次有输出内容”的时间。

3. **benchmark 校验逻辑验证**
   - NEO 路径应继续通过严格 token 数校验；
   - vLLM 路径不应再因为 `chunk 数 != output_len` 直接抛错；
   - 但若某条本应 token-level 的记录缺字段，仍要明确失败。

4. **指标语义验证**
   - 对 NEO：
     - 验证 `TTFT == token_offsets[0]`
     - 验证 `TPOT == mean(diff(token_offsets))`
   - 对 vLLM：
     - 验证 TTFT 取的是首个有效输出事件时间；
     - 若无 token-level 边界，确认 TPOT 路径被明确拒绝。

5. **端到端复现实验验证**
   - 重新运行 `evaluation/reproduce-fig6b.py` 的小规模 rate sweep；
   - 确认：
     - Avg latency 图仍可生成；
     - TTFT 图按新语义生成；
     - TPOT 图不会再基于 vLLM chunk-level 数据输出错误结论。

6. **缓存兼容性验证**
   - 确认新旧结果文件不会被混淆使用；
   - 若旧缓存缺少新的观测语义字段，应清晰提示重新采样，而不是静默沿用旧假设。
