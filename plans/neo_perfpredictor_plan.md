# NEO 中 `perfpredictor.py` 的运行时作用：代码级调研

这份说明只聚焦一个问题：**`swiftllm/perfpredictor.py` 在 NEO 运行时到底起什么作用，以及它在 `reproduce-fig6c.py` 里是否真的会生效。**

结论先写在最前面：

- `perfpredictor.py` **不是** worker 每层 forward 里的执行模块，它不直接参与 attention、swap、CPU kernel 或双 batch pipeline 的实际计算。
- 它的主要职责是：**给 scheduler 提供 batch 级性能估计**，让 scheduler 在构造 `SubBatch` 时判断：
  1. CPU decode 能否被另一边 GPU 工作“盖住”
  2. prefill 要不要裁掉一部分
  3. 本轮应该走 **sequential** 还是 **pipelined**
- 真正执行计算的是 executor / worker；`PerfPredictor` 只提供时间估计值。
- 在运行 `NEO/evaluation/reproduce-fig6c.py` 时：
  - `ours` 路径会经过 `swiftllm.server.api_server -> AsyncEngine.initialize_async() -> ModelProfiler.init_profile_tables() -> Scheduler(..., self.profiler.pp)`，**因此会用到 perfpredictor**。
  - `vllm` 路径直接启动 vLLM CLI，**不会用到 perfpredictor**。

---

## 1. 先看清定位：`perfpredictor.py` 是 scheduler 的估时器，不是 worker 的执行模块

`swiftllm/perfpredictor.py:7-44` 先定义了一个纯接口 `PerfPredictor`，提供 5 个查询函数：

- `get_linr_T(S)`
- `get_pref_T(S)`
- `get_gdec_T(N)`
- `get_cdec_T(S, N)`
- `get_lnch_T()`

从命名就能看出来，它提供的不是“算子”，而是**时间估计值**。

真正的两个实现是：

### 1.1 `ZeroPerfPredictor`

`swiftllm/perfpredictor.py:46-68`

它所有接口都返回 `0.0`。这个类的意义不是参与正式调度，而是：

- 在某些“只想先构造 batch / 跑真实执行，但不想依赖预测值”的场景里，给出一个最小占位实现。
- 后面 `ModelProfiler` 构造人工测试 batch 时，就用到了这一点。

### 1.2 `TablePerfPredictor`

`swiftllm/perfpredictor.py:70-196`

这才是运行时真正使用的 predictor。它有两部分职责：

1. **定义采样点**
   - `linr_S_list`：线性部分按 iteration width `S` 建表，见 `swiftllm/perfpredictor.py:80-90`
   - `pref_S_list`：prefill 按 token 数 `S` 建表，见 `swiftllm/perfpredictor.py:91-98`
   - `gdec_N_list`：GPU decode attention 按总 token 数 `N` 建表，见 `swiftllm/perfpredictor.py:99-105`
   - `cdec_S_list` / `cdec_N_lists`：CPU decode 按 `(S_c, N_c)` 二维建表，见 `swiftllm/perfpredictor.py:107-125`

2. **提供运行时查询**
   - `get_linr_T/get_pref_T/get_gdec_T` 通过 `_interp_1d()` 做一维线性插值，见 `swiftllm/perfpredictor.py:148-176`
   - `get_cdec_T` 对 `S` 和 `N` 做二维插值，见 `swiftllm/perfpredictor.py:178-193`

也就是说，**运行时不会重新 profile，也不会在线学习；运行时只是在查表 + 插值。**

### 1.3 `lnch_T` 当前是常量，不是当前启动时重新测出来的

`swiftllm/perfpredictor.py:127-129`：

```python
self.lnch_T = 0.8
# self.lnch_T = self._profile_lnch(lnch_S_list)
```

这点很重要：

- 当前代码里，launch 开销不是动态 profile 后再塞进 predictor 的。
- `TablePerfPredictor.get_lnch_T()` 最终返回的是这个固定常量，见 `swiftllm/perfpredictor.py:195-196`。

所以如果问“perfpredictor 里的 perf 数据是不是全部来自真实 profile”，答案是：

- `linr/pref/gdec/cdec`：主要是
- `lnch_T`：当前实现里不是，它是硬编码 `0.8`

---

## 2. predictor 并不是直接喂给 scheduler 一个总结果，而是先进入 `BatchPerfData`

理解 runtime 的关键，不是只看 `perfpredictor.py`，而是看 **谁在消费它**。

真正的第一消费层在 `swiftllm/structs.py:148-207` 的 `BatchPerfData`。

### 2.1 `BatchPerfData` 保存的是“当前 batch 的估计工作量”

初始化：`swiftllm/structs.py:153-163`

```python
self.x = 0
self.s = 0
self.n_g = 0
self.x_c = 0
self.n_c = 0

self.predictor = predictor
self.pref_T = 0
self.gdec_T = 0
self.lnch_T = predictor.get_lnch_T()
```

这里几个量的含义可以直接按 scheduler 的视角理解：

- `x`：batch 内请求数
- `s`：本轮总 iteration width
- `n_g`：GPU decode 相关的总 token 量
- `x_c`：CPU decode request 数
- `n_c`：CPU decode 相关总 token 量

### 2.2 predictor 的输出是怎么累积进去的

#### prefill

`swiftllm/structs.py:165-168`

```python
def add_pref(self, prompt_len):
    self.x += 1
    self.s += prompt_len
    self.pref_T += self.predictor.get_pref_T(prompt_len)
```

也就是说：

- 每加一个 prefill request，`pref_T` 立刻增加该 prompt 长度对应的估计 prefill 时间。

#### GPU decode

`swiftllm/structs.py:175-179`

```python
def add_gdec(self, seq_len):
    self.x += 1
    self.s += 1
    self.n_g += seq_len
    self.gdec_T = self.predictor.get_gdec_T(self.n_g)
```

这里不是累加每个 request 的独立时间，而是：

- 先累计总 `n_g`
- 再根据新的总量重新查询 `get_gdec_T(self.n_g)`

说明调度器关心的是**这一整个 sub-batch 的 GPU decode 工作量**，不是单条请求的独立 cost。

#### CPU decode

`swiftllm/structs.py:181-185`

```python
def add_cdec(self, seq_len):
    self.x += 1
    self.s += 1
    self.x_c += 1
    self.n_c += seq_len
```

这里 `add_cdec()` 本身不立即写 `cdec_T`，因为 CPU decode 代价是二维函数：

- `S_c = x_c`
- `N_c = n_c`

真正读取时才查 predictor。

### 2.3 `gpu_time` / `cpu_time` 才是 scheduler 真正比较的量

`swiftllm/structs.py:193-207`

```python
@property
def linr_T(self) -> float:
    return self.predictor.get_linr_T(self.s)

@property
def cdec_T(self) -> float:
    return self.predictor.get_cdec_T(self.x_c, self.n_c)

@property
def gpu_time(self) -> float:
    return self.linr_T + self.pref_T + self.gdec_T

@property
def cpu_time(self) -> float:
    return self.cdec_T + self.lnch_T
```

所以对 scheduler 来说，predictor 最终产出的核心不是五张表，而是两个 batch 级指标：

- `gpu_time`
- `cpu_time`

后面所有模式选择，几乎都围绕这两个量展开。

---

## 3. `SubBatch` 是 predictor 进入 scheduler 的实际载体

`swiftllm/structs.py:211-245` 定义了 `SubBatch`。

最关键的一句是 `swiftllm/structs.py:216-221`：

```python
def __init__(self, predictor: PerfPredictor=ZeroPerfPredictor()):
    ...
    self.perfdata = BatchPerfData(predictor)
```

这句说明：

- `SubBatch` 一创建，就会绑定一个 predictor
- 后续 `add_pref/add_gdec/add_cdec` 的同时，`perfdata` 也会同步更新

也就是说，scheduler 在“试着往 batch 里塞请求”的时候，不需要额外做一次离线估算；它边塞边能读到新的估计 `gpu_time/cpu_time`。

### 3.1 `set_model_forward_args()` 进一步证明 predictor 的使命主要在调度阶段

`swiftllm/structs.py:254-263`

```python
self.batch_size = self.perfdata.x
self.iter_width = self.perfdata.s
del self.perfdata
```

这里非常关键：

- 一旦真正进入 model forward 准备阶段，`perfdata` 就被删掉了。
- `SubBatch` 只保留 worker 真正执行需要的字段，比如 `iter_width`、请求分类、token 边界等。

这说明：

> predictor 主要服务于 **batch 形成和调度决策**；进入 worker 真正做一轮 forward 时，它的主要使命已经完成了。

换句话说，**worker 跑的是 `SubBatch` 结果，不是 `PerfPredictor` 本体。**

---

## 4. `perfpredictor` 是如何接入在线运行时的

这一部分需要顺着真实启动链往下追。

### 4.1 服务入口先构造 `AsyncEngine`

`swiftllm/server/api_server.py:47-58`

```python
parser = argparse.ArgumentParser()
parser.add_argument("--host", type=str, default="localhost")
parser.add_argument("--port", type=int, default=8000)
swiftllm.EngineConfig.add_cli_args(parser)
...
engine = swiftllm.AsyncEngine(swiftllm.EngineConfig(**args))
```

也就是说，在线服务模式里：

- CLI 参数先进入 `EngineConfig`
- 再进入 `AsyncEngine`

这里的 `EngineConfig` 包含了与 predictor 直接相关的两个外部路径字段，见 `swiftllm/engine_config.py:30-32`：

- `library_path`
- `profile_result_path`

其中：

- `library_path` 是 pacpu 自定义库路径
- `profile_result_path` 是 perf table JSON 缓存路径

### 4.2 `Engine.initialize()` 先初始化 executor / profiler / block manager

`swiftllm/server/engine.py:49-67`

```python
self.executor = self.executor_class(self.engine_config, self.model_config)
self.profiler = ModelProfiler(self.executor)
self.profiler.profile_num_blocks()
self.block_manager = BlockManager(self.engine_config, self.model_config)
self.executor.init_kvcache_and_swap()
```

这里需要区分两件事：

1. `profile_num_blocks()`
   - 只是推断 `num_gpu_blocks` / `num_cpu_blocks`
   - 它不等于 predictor 的时间表初始化

2. `ModelProfiler(self.executor)`
   - 这一步只是把 profiler 对象建出来
   - 此时 `self.profiler.pp` 还没有真正填好运行时要用的表

### 4.3 真正把 `TablePerfPredictor` 接进运行时的是 `initialize_async()`

`swiftllm/server/engine.py:103-121`

```python
async def initialize_async(self):
    self.event_loop = asyncio.get_event_loop()
    super().initialize()

    logger.info("Initializing performance table...")
    self.profiler.init_profile_tables(self.block_manager)

    logger.info("Initializing scheduler...")
    self.scheduler = Scheduler(self.engine_config, self.model_config, self.profiler.pp)
```

这几行就是 perfpredictor 的主接入点：

1. `self.profiler.init_profile_tables(self.block_manager)`
   - 构造 `TablePerfPredictor`
   - 读取 / 生成 profile tables
   - 最终写进 `self.profiler.pp`

2. `Scheduler(..., self.profiler.pp)`
   - 把这个 predictor 传给 scheduler

到这里，运行时关系已经很清楚了：

```text
api_server.py
  -> AsyncEngine(EngineConfig)
  -> AsyncEngine.initialize_async()
  -> ModelProfiler.init_profile_tables(...)
  -> self.profiler.pp = TablePerfPredictor(...)
  -> Scheduler(..., predictor=self.profiler.pp)
```

因此，`perfpredictor.py` 并不是一个“被 worker import 的内核辅助模块”，而是一个**在 engine 初始化阶段接进 scheduler 的估时模块**。

---

## 5. perf 数据到底从哪里来

这一节最容易被误解。答案不是“运行过程中自动学出来”，而是：

- **主要在启动阶段由 `ModelProfiler` 真实跑出来**
- 然后缓存到 `profile_result_path`
- 运行时只查表 / 插值

### 5.1 `TablePerfPredictor` 自己并不产出真实数据

`swiftllm/perfpredictor.py:76-129` 的 `__init__()` 只是：

- 生成采样点列表
- 建 lower-bound index
- 初始化表字段为 `None`

比如：

- `self.linr_T_list = None`
- `self.pref_T_list = None`
- `self.gdec_T_list = None`
- `self.cdec_T_lists = [None]`

这说明 `TablePerfPredictor` 本身只是**表结构 + 查询器**，不是数据生产者。

### 5.2 真正填表的是 `ModelProfiler.init_profile_tables()`

`swiftllm/server/profiler.py:41-54`

```python
self.bm = block_manager
self.pp = TablePerfPredictor(engine_config)

self.pp.linr_S_list, self.pp.linr_T_list = self._profile_linr(self.pp.linr_S_list)
self.pp.pref_S_list, self.pp.pref_T_list = self._profile_pref(self.pp.pref_S_list)
self.pp.gdec_N_list, self.pp.gdec_T_list = self._profile_gdec(self.pp.gdec_N_list)
self.pp.cdec_S_list, self.pp.cdec_N_lists, self.pp.cdec_T_lists = self._profile_cdec(...)
```

这四步分别把：

- `linr`
- `pref`
- `gdec`
- `cdec`

的表填满。

### 5.3 profiler 是怎么拿到这些数的：它会构造人工 test case，然后跑真实 executor

核心逻辑在 `swiftllm/server/profiler.py:79-134` 的 `_run_test_case()`。

它会：

1. 构造假的 request
2. 构造假的 `SubBatch`
3. 调 `executor.do_one_iteration(...)`
4. 打开 / 关闭 perf monitor
5. 从真实执行结果里取平均值

最关键的构造代码是 `swiftllm/server/profiler.py:96-114`：

```python
batches = []
...
for i in range(nbatches):
    batch = SubBatch()
    ...
    batch.add_pref(...)
    batch.add_gdec(...)
    batch.add_cdec(...)
```

这里的 `SubBatch()` 没传 predictor，因此走默认参数：

- `SubBatch(predictor=ZeroPerfPredictor())`

这点很关键。它说明：

> profiler 在建表时测的是**真实执行成本**，不是“让 predictor 再递归地预测自己”。

也就是说，profile table 的数据来源是：

- 构造人工 batch
- 走真实 executor
- 由 `ModelPerfResult.mean(...)` 提取各阶段均值

### 5.4 这些结果具体取自哪里

例如：

- `swiftllm/server/profiler.py:164-170`：`avg_linr_time`
- `swiftllm/server/profiler.py:207-213`：`avg_pref_time`
- `swiftllm/server/profiler.py:251-257`：`avg_gdec_time`
- `swiftllm/server/profiler.py:307-313`：`avg_cdec_time`

这些都来自：

```python
ModelPerfResult.mean(res, ...)
```

而 `res` 又来自：

- `self.executor.do_one_iteration(...)`
- `self.executor.turn_off_perf_monitor_and_flush_results()`

`Executor` 再进一步连到真实模型，见 `swiftllm/server/executor.py:61-90` 与 `swiftllm/server/executor.py:93-125`：

- `SingleProcExecutor.do_one_iteration()` 直接调 `LlamaModel.do_one_iteration()`
- `RayExecutor.do_one_iteration()` 通过 remote model 调同一条执行链

所以这些表不是 mock 数据，而是**真实 model iteration 的 profile 结果**。

### 5.5 profile 结果会缓存到 `profile_result_path`

`swiftllm/server/profiler.py` 中各 profile 函数会读写这些文件：

- `linr.json`：`swiftllm/server/profiler.py:143`
- `pref.json`：`swiftllm/server/profiler.py:196`
- `gdec.json`：`swiftllm/server/profiler.py:239`
- `cdec.json`：`swiftllm/server/profiler.py:284`
- `lnch.json`：`swiftllm/server/profiler.py:364`

典型行为是：

1. 若 JSON 已存在且覆盖范围足够，则直接读取返回
2. 否则补跑 profile
3. 写回 JSON
4. 顺带画出 png

例如 `linr` 的缓存逻辑见 `swiftllm/server/profiler.py:145-149`。

因此更准确地说：

- **首次启动**：可能会真实 profiling
- **后续启动**：可能直接复用 `profile_result_path` 下已有 JSON
- **运行中调度**：只查表，不重新测

### 5.6 `lnch.json` 虽然有 profile 逻辑，但当前 predictor 没真正接上它

`swiftllm/server/profiler.py:357-400` 里有 `_profile_lnch()`，也会读写 `lnch.json`。

但当前 `ModelProfiler.init_profile_tables()` 并没有把它的结果写回 `self.pp.lnch_T`；同时 `TablePerfPredictor.__init__()` 里 `lnch_T` 直接固定成 `0.8`，见 `swiftllm/perfpredictor.py:127-129`。

所以当前版本更准确的说法是：

- 代码库里**存在** `lnch` 的 profile 逻辑
- 但运行时 scheduler 真正用到的 `lnch_T` 仍是固定常量 `0.8`

---

## 6. predictor 在 scheduler 里到底影响了哪些决策

核心文件是 `swiftllm/server/scheduler.py`。

### 6.1 scheduler 初始化时拿到 predictor

`swiftllm/server/scheduler.py:95-104`

```python
def __init__(..., predictor: PerfPredictor):
    ...
    self.predictor = predictor
```

这意味着后面它构造出的 `SubBatch` 都可以显式绑定这个 predictor。

### 6.2 真正进入 batch 构造的是 `SubBatch(self.predictor)`

`swiftllm/server/scheduler.py:157-159`

```python
batches = [SubBatch(self.predictor) for _ in range(2)]
gpu_only_batch = SubBatch(self.predictor)
```

这一步非常关键。因为从这里开始：

- 每次 `add_pref/add_gdec/add_cdec`
- 都会同步更新 `perfdata`
- scheduler 于是能边试边读估计代价

### 6.3 `_get_remains()`：用 predictor 判断 CPU decode 能否被另一边 GPU 工作隐藏

`swiftllm/server/scheduler.py:132-140`

```python
def _get_remains(self, batches: list[SubBatch]) -> float:
    assert len(batches) == 2
    return [
        batches[j^1].perfdata.linr_T +
        batches[j].perfdata.pref_T +
        batches[j].perfdata.gdec_T -
        batches[j].perfdata.cpu_time
        for j in range(2)
    ]
```

这个函数不是在做精确 wall-clock 模拟，而是在估计：

- 对于 batch `j` 的 CPU decode 而言
- 另一边 batch 的 `linr`，加上自己这边的 `pref` 和 `gdec`
- 是否足以覆盖 `cpu_time`

如果这个余量变成负值，就说明：

- CPU decode 太重
- 已经不能再被流水隐藏住
- 继续塞这个 request 会让 CPU 变成瓶颈

这就是 predictor 最直接的运行时作用之一。

### 6.4 决策一：prefill 裁剪

先看 `gpu_only_batch` 的裁剪，`swiftllm/server/scheduler.py:177-182`

```python
while gpu_only_batch.get_num_prefs():
    req, is_gpu = gpu_only_batch.pop_pref()
    if is_gpu or gpu_only_batch.perfdata.s < self.predictor.linr_S_threshold:
        gpu_only_batch.add_pref(req, is_gpu)
        break
```

这里直接使用了 `TablePerfPredictor` 中的 heuristic：

- `linr_S_threshold = 128`，见 `swiftllm/perfpredictor.py:88-90`

它的作用是：

- 不让纯 GPU batch 的线性部分 `linr_T` 因为 `s` 太大而过重
- 也就是说，predictor 不只是给时间数值，还暴露了一个启发式阈值供 scheduler 控制 batch 宽度

后面在 pipeline batch 的第一子批上也有类似裁剪，见 `swiftllm/server/scheduler.py:217-222`。

### 6.5 决策二：CPU decode 请求怎样分配到两个 sub-batch

`swiftllm/server/scheduler.py:184-203`

```python
for req in self.cpu_decoding_q:
    ...
    batches[next_batch_idx].add_cdec(req)
    remains = self._get_remains(batches)
    ...
    if min(remains) < 0 and self.num_gpu_blocks > 0:
        ...
        batches[next_batch_idx].pop_cdec()
        continue
    next_batch_idx = remains[1] > remains[0]
```

这里的逻辑可以直接翻译成：

1. 试着把当前 CPU decode request 塞进某个 batch
2. 立刻重算两边的剩余量 `remains`
3. 如果最小余量已经小于 0，说明某边 CPU 太重，会破坏隐藏关系，于是回退
4. 否则把下一个请求优先塞向“剩余空间更大”的那一边

这就是 predictor 对 **CPU decode 分配** 的直接影响。

### 6.6 决策三：最终选 sequential 还是 pipelined

`swiftllm/server/scheduler.py:224-234`

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

这几行几乎可以当成 perfpredictor 在 scheduler 中的“总开关”。

它不是问：

- 某个 batch 能不能跑

而是在问：

- 用 predictor 估出来看，本轮是：
  - 单 batch sequential 吞吐更高？
  - 还是双 sub-batch pipeline 吞吐更高？

然后按估算吞吐率来选模式。

因此更准确的总结是：

> perfpredictor 对运行时的影响，不在于“直接让模型算得更快”，而在于**让 scheduler 选出更合适的 batch 形状与执行模式**。

### 6.7 `always_use_gpu` 会直接绕开这条新调度逻辑

`swiftllm/server/scheduler.py:237-328` 是 `_get_next_batch_new()`，而 `get_next_batch()` 的分流在 `swiftllm/server/scheduler.py:...` 尾部：

```python
def get_next_batch(self) -> tuple[list[SubBatch], list[Request], list[Request]]:
    if self.engine_config.always_use_gpu:
        return self._get_next_batch_old()
    return self._get_next_batch_new()
```

这点在后面分析 `reproduce-fig6c.py` 时很关键：

- 只要 `always_use_gpu=True`
- 就不会进入依赖 predictor 的这条新调度链

---

## 7. 运行 `reproduce-fig6c.py` 时，perfpredictor 到底会不会生效

现在回到用户最关心的问题。

### 7.1 `reproduce-fig6c.py` 自己并不直接 import perfpredictor

`NEO/evaluation/reproduce-fig6c.py:34-52`

```python
async def one_round(server_name: str):
    start_server(server_name, config)
    try:
        if server_name == "ours":
            for rate in ours_rates:
                await run_test(*prepare_real_test("osc", config, server_name), rate=rate)
        if server_name == "vllm":
            for rate in vllm_rates:
                await run_test(*prepare_real_test("osc", config, server_name), rate=rate)
    finally:
        stop_server()
```

所以不能只看这个脚本本身。真正要看的是：

- `start_server("ours", config)` 启动了什么
- `start_server("vllm", config)` 启动了什么

### 7.2 `ours` 路径：会用到 perfpredictor

关键代码在 `NEO/evaluation/server.py:55-87`。

当 `name == "ours"` 时，命令行是：

```python
cmd = numacmd + [
    sys.executable, "-m", "swiftllm.server.api_server",
    "--port", "8000",
    "--model-path", config["model_path"],
    ...
    "--library-path", f"{neo_dir}/pacpu/build/{config['library']}",
    "--profile-result-path", f"{neo_dir}/profile_results/",
] + cmd
```

这里直接证明了两件事：

1. `ours` 启动的是 `swiftllm.server.api_server`
2. 它显式传入了 `--profile-result-path`

顺着这条链往下走：

#### 第一步：进入 `api_server.py`

`swiftllm/server/api_server.py:58`

```python
engine = swiftllm.AsyncEngine(swiftllm.EngineConfig(**args))
```

#### 第二步：服务启动时调用 `initialize_async()`

`swiftllm/server/api_server.py:69-73`

```python
async def main_coroutine():
    await engine.initialize_async()
    uvicorn_task = asyncio.create_task(uvicorn_server.serve())
    engine_task = asyncio.create_task(engine.start_all_event_loops())
```

#### 第三步：初始化 performance table

`swiftllm/server/engine.py:111-115`

```python
self.profiler.init_profile_tables(self.block_manager)
self.scheduler = Scheduler(self.engine_config, self.model_config, self.profiler.pp)
```

#### 第四步：scheduler 构造 `SubBatch(self.predictor)` 并在调度中使用它

见前面 `swiftllm/server/scheduler.py:157-234`。

所以结论必须写得非常明确：

> 运行 `reproduce-fig6c.py` 的 `ours` 分支时，`perfpredictor.py` **会生效**。
>
> 它不是在 benchmark 脚本本身里被直接 import 的，而是经由 `start_server("ours") -> api_server -> AsyncEngine.initialize_async() -> ModelProfiler.init_profile_tables() -> Scheduler(..., self.profiler.pp)` 这条服务初始化链，被接入到运行时调度中。

### 7.3 `vllm` 路径：不会用到 perfpredictor

`NEO/evaluation/server.py:26-53` 中，当 `name[:4] == "vllm"` 时，启动命令是：

```python
server_proc = subprocess.Popen(
    numacmd + [
        VLLM_BIN, "serve", config["model_path"], "--port", "8000",
        ...
    ]
)
```

这条链：

- 不会进入 `swiftllm.server.api_server`
- 不会构造 `AsyncEngine`
- 不会构造 `ModelProfiler`
- 不会初始化 `TablePerfPredictor`
- 也不会构造 `Scheduler(..., predictor=...)`

因此结论也必须明确：

> `reproduce-fig6c.py` 的 `vllm` 路径 **不会** 用到 NEO 的 `perfpredictor.py`。

### 7.4 `base` 路径也要顺手说明一下

虽然 `reproduce-fig6c.py` 当前只跑 `vllm` 和 `ours`，但 `NEO/evaluation/server.py:55-70` 还定义了 `base` 和 `fsdc` 路径。

其中 `base` 会加：

```python
cmd=["--always-use-gpu"]
```

再结合 `swiftllm/server/scheduler.py` 的分流可知：

- `base` 仍然会初始化 `profiler` 和 `scheduler`
- 但真正调度时会走 `_get_next_batch_old()`
- 因而不会走依赖 predictor 的新批次形成逻辑

这也能反过来说明为什么 `ours` 路径更关键：

- `ours` 没带 `--always-use-gpu`
- 所以会走 `_get_next_batch_new()`
- predictor 才会在新调度逻辑里真正发挥作用

---

## 8. 把整条调用链压缩成一张图

如果只想记住 runtime 主线，可以记成下面这条：

```text
evaluation/reproduce-fig6c.py
  -> evaluation/server.py:start_server("ours", config)
  -> python -m swiftllm.server.api_server ... --profile-result-path ...
  -> api_server.py: AsyncEngine(EngineConfig(...))
  -> await engine.initialize_async()
  -> Engine.initialize()
       -> ModelProfiler(self.executor)
       -> profile_num_blocks()
  -> profiler.init_profile_tables(self.block_manager)
       -> TablePerfPredictor(...)
       -> profile/load linr/pref/gdec/cdec tables
  -> Scheduler(..., predictor=self.profiler.pp)
  -> SubBatch(self.predictor)
  -> BatchPerfData 使用 predictor 维护 gpu_time/cpu_time
  -> scheduler 用这些估计值做：
       1. CPU decode 分配
       2. prefill 裁剪
       3. sequential / pipelined 模式选择
```

如果是 `vllm`，这条链在第一步就分叉掉了：

```text
evaluation/server.py:start_server("vllm", config)
  -> vllm serve ...
  -> 不进入 swiftllm.server.api_server
  -> 不会使用 NEO perfpredictor
```

---

## 9. 最容易误解的点，集中纠正

### 误解 1：`perfpredictor.py` 直接参与 worker forward

不对。

更准确的说法是：

- 它主要服务 scheduler 的 batch 形成与模式选择
- 真正进入 worker forward 时，`SubBatch.set_model_forward_args()` 已经把 `perfdata` 删掉了，见 `swiftllm/structs.py:254-263`

### 误解 2：运行时在不停“学习”新的性能模型

不对。

更准确的说法是：

- 服务初始化时由 `ModelProfiler` 真实跑测试 batch，或从 JSON 缓存读取
- 运行时只是查表和插值

### 误解 3：`reproduce-fig6c.py` 没有 import predictor，所以它没被用到

不对。

更准确的说法是：

- `ours` 路径通过启动在线 server，间接进入 `AsyncEngine.initialize_async()`
- predictor 在这条初始化链里被接入 scheduler

### 误解 4：所有 predictor 数据都来自真实 profiling

不完全对。

更准确的说法是：

- `linr/pref/gdec/cdec` 基本是 `ModelProfiler` 跑真实 executor 得到的
- 但当前 `lnch_T` 在 `TablePerfPredictor` 中是固定常量 `0.8`

### 误解 5：scheduler 用 predictor 在做精确执行时间仿真

不对。

更准确的说法是：

- 它是在做**启发式决策**
- 目标是判断 batch 形状和模式选择
- 不是做 cycle-accurate 的严格时序模拟

### 误解 6：profiler 构造测试 batch 时，也依赖 predictor 自己来算自己

不对。

更准确的说法是：

- profiler 里人工构造的 `SubBatch()` 默认用的是 `ZeroPerfPredictor()`
- 因此测出来的是实际执行成本，而不是递归预测结果

---

## 10. 最后的结论

如果只用一句话概括 `swiftllm/perfpredictor.py` 在 NEO 中的作用，可以这样说：

> 它是一个**面向 scheduler 的表驱动性能估计器**：在服务初始化时由 `ModelProfiler` 生成或加载 `linr/pref/gdec/cdec` 性能表，运行时通过 `SubBatch -> BatchPerfData` 把这些估计转成 batch 级 `gpu_time/cpu_time`，再用于 CPU decode 分配、prefill 裁剪，以及 sequential / pipelined 模式选择；在 `reproduce-fig6c.py` 中，只有 `ours` 路径会通过 `api_server -> AsyncEngine.initialize_async()` 间接启用它，而 `vllm` 路径不会。
