# NEO 中 CPU decode attention 的等待机制与 pacpu 自定义算子加载链路

这份说明只聚焦两个问题：

1. `swiftllm/worker/layers/transformer_layer.py` 里到底是怎样**等待 CPU 完成 decode attention** 的。
2. `torch.ops.pacpu.paged_attention_cpu(...)` 这个符号到底是怎样从 `pacpu/` 目录里的源码，变成 Python 侧可调用的自定义 PyTorch op 的。

重点不是 scheduler 怎样决定哪些请求下放 CPU，也不是双 sub-batch pipeline 的总体交错；这里只看 **worker 侧的同步边界、数据路径、构建—加载—注册—调用链**。

本文主要对应以下代码：

- `NEO/swiftllm/worker/layers/transformer_layer.py:139-150`：`_comm_wait_compute()` / `_compute_wait_comm()`
- `NEO/swiftllm/worker/layers/transformer_layer.py:158-179`：`_transfer_qkv()`
- `NEO/swiftllm/worker/layers/transformer_layer.py:258-355`：`_attention()` 中 CPU decode 分支
- `NEO/swiftllm/worker/block_swapper.py:21-115`：CPU pinned buffer / CPU KV swap / block table
- `NEO/swiftllm/structs.py:254-293`：`SubBatch.set_model_forward_args()`
- `NEO/pacpu/pacpu.cpp:72-131`：`paged_attention_cpu(...)` 与 `TORCH_LIBRARY(pacpu, m)`
- `NEO/pacpu/core.h:130-260`：CPU attention 的实际实现
- `NEO/pacpu/build.sh:1-7`：构建入口
- `NEO/pacpu/CMakeLists.txt:40-56`：shared library 目标生成
- `NEO/swiftllm/engine_config.py:30-32, 68-166`：`library_path` 配置入口
- `NEO/swiftllm/server/api_server.py:47-58`：CLI 参数进入 `EngineConfig`
- `NEO/swiftllm/server/engine.py:49-64`：`Engine.initialize()`
- `NEO/swiftllm/server/executor.py:61-125`：executor 把 `engine_config` 传给 worker
- `NEO/swiftllm/worker/model.py:150-153`：`torch.ops.load_library(...)`
- `NEO/examples/example.py:85-104`：`library_path` 的直接构造示例

---

## 1. 先直接回答核心问题：谁在等谁？

最短答案是：

> **CUDA stream 并没有直接调度 CPU attention。** NEO 真正依赖的是“**host 线程先阻塞等待输入就绪，再同步调用 CPU C++ op，最后用 CUDA stream 只管理回传到 GPU 的 copy 和后续 GPU 依赖**”。

把 `NEO/swiftllm/worker/layers/transformer_layer.py:158-179, 258-355` 压缩成一条主线，就是：

1. `_transfer_qkv()` 在 `cpu_communication_stream` 上发起 **GPU→CPU** 的 QKV 拷贝。
2. `qkvtr_e.record()` 记录一个 CUDA event，表示这批 copy 在 communication stream 上何时完成。
3. `_attention()` 里的 `self.events[cur_stage].qkvtr_e.synchronize()` 在 **host 线程** 上阻塞，直到这批 QKV 已经真正在 CPU pinned buffer 里可读。
4. 然后 `torch.ops.pacpu.paged_attention_cpu(...)` 作为一次**同步的 C++/CPU 函数调用**执行 CPU attention；函数返回时，`o_cpu` 已经写好。
5. 算子返回后，再在 `cpu_communication_stream` 上发起 **CPU→GPU** 的 `o.copy_(oc, non_blocking=True)`。
6. 函数尾部 `self._compute_wait_comm()` 让默认 CUDA stream 等待 `cpu_communication_stream`，这样后续 GPU `_postproj()` 只能在回拷完成后继续。

所以这里有两类完全不同的“等待”：

### 1.1 host-side blocking

负责两件事：

- 等 QKV 已经从 GPU 拷到 CPU
- 真正执行 CPU attention 本体

对应代码是：

```python
self.events[cur_stage].qkvtr_e.synchronize()
torch.ops.pacpu.paged_attention_cpu(...)
```

这两步都不是 CUDA stream 在“控制 CPU”，而是 **Python/host 调用线程自己在同步地等和跑**。

### 1.2 CUDA stream dependency

负责一件事：

- 保证 CPU attention 的输出**回到 GPU 后**，后续 GPU kernel 才能继续

对应代码是：

```python
with torch.cuda.stream(self.cpu_communication_stream):
    o[-batch.num_cdecs:, :].copy_(oc, non_blocking=True)
...
self._compute_wait_comm()
```

也就是说：

- **CPU attention 本体**靠 host 调用栈同步
- **CPU 结果何时可被后续 GPU `_postproj()` 使用**，靠 CUDA stream 依赖

这两件事不能混为一谈。

---

## 2. 先建立最小状态模型：哪些数据真的走了 CPU decode 路径

## 2.1 CPU decode 真实使用的 buffer

`NEO/swiftllm/worker/block_swapper.py:53-74` 已经把 CPU decode 相关存储写得很清楚：

```python
self.k_swap = torch.zeros(..., device="cpu", pin_memory=True)
self.v_swap = torch.zeros(..., device="cpu", pin_memory=True)

self.q_cpu = torch.zeros(..., device="cpu", pin_memory=True)
self.k_cpu = torch.zeros(..., device="cpu", pin_memory=True)
self.v_cpu = torch.zeros(..., device="cpu", pin_memory=True)
self.o_cpu = torch.zeros(..., dtype=torch.float32, device="cpu", pin_memory=True)

self.cpu_block_table = torch.zeros(..., dtype=torch.int32, device="cpu")
```

因此 worker 侧 CPU decode 真正涉及的对象是：

- `self.swapper.q_cpu / k_cpu / v_cpu`
  - CPU decode attention 的输入 QKV buffer
  - 都在 **pinned host memory** 上
- `self.swapper.o_cpu`
  - CPU decode attention 的输出 buffer
  - 也在 **pinned host memory** 上
- `self.swapper.k_swap / v_swap`
  - CPU KV cache 的 swap 区
  - 也是 **CPU 侧存储**
- `self.swapper.cpu_block_table`
  - CPU KV block table
- `batch.attn_out_buf`
  - 当前 sub-batch 的 GPU 侧 attention 输出 buffer
  - CPU decode 的结果最终会回写到它的尾部区域

这说明 CPU decode 不是调用另一个独立 runtime，也不是把数据复制到某个 Python list 里再算；它直接消费 `Swapper` 提供的 **pinned host buffers + CPU KV swap cache + CPU block table**。

## 2.2 哪些 token 会走 CPU decode

`NEO/swiftllm/structs.py:265-281` 先把一个 `SubBatch` 拆成几段：

```python
self.num_cprfs = len(self.cprf_reqs)
self.num_gprfs = len(self.gprf_reqs)
self.num_gdecs = len(self.gdec_reqs)
self.num_cdecs = len(self.cdec_reqs)
self.num_prefs = self.num_cprfs + self.num_gprfs
self.num_prgds = self.num_prefs + self.num_gdecs

self.sum_pref_toks = sum(self.seq_lens_list[:self.num_prefs])
self.sum_prgd_toks = self.sum_pref_toks + self.num_gdecs
```

因此一个 sub-batch 的 token 布局可以近似记成：

```text
|<------ prefill tokens ------>|<-- gpu decode rows -->|<-- cpu decode rows -->|
0                         sum_pref_toks           sum_prgd_toks             iter_width
```

在 `NEO/swiftllm/worker/layers/transformer_layer.py:170-178, 331-349` 中：

- `_transfer_qkv()` 取的是：
  - `q[-batch.num_cdecs:]`
  - `k[-batch.num_cdecs:]`
  - `v[-batch.num_cdecs:]`
- `paged_attention_cpu(...)` 的序列元数据取的是：
  - `batch.seq_ids_list[batch.num_prgds:]`
  - `batch.seq_lens_list[batch.num_prgds:]`

所以很明确：

> 走 CPU decode 的，就是当前 `SubBatch` **尾部那一段 CPU decode rows**，不是“第三个 batch”，也不是单独切出去的一套 forward。

---

## 3. `_attention()` 里的 CPU decode 路径，实际上分成 4 个严格阶段

下面只沿着 `NEO/swiftllm/worker/layers/transformer_layer.py:158-179, 258-355` 的代码顺序讲。

## 3.1 阶段 A：先把 CPU decode 所需 QKV 从 GPU 送到 pinned host buffer

入口在 `_transfer_qkv()`：`NEO/swiftllm/worker/layers/transformer_layer.py:158-179`

```python
def _transfer_qkv(self, q, k, v, batch, cur_stage=0):
    self._comm_wait_compute()
    if batch.num_cdecs > 0:
        with torch.cuda.stream(self.cpu_communication_stream):
            qc = self.swapper.q_cpu[:batch.num_cdecs]
            kc = self.swapper.k_cpu[:batch.num_cdecs]
            vc = self.swapper.v_cpu[:batch.num_cdecs]
            qc.copy_(q[-batch.num_cdecs:], non_blocking=True)
            kc.copy_(k[-batch.num_cdecs:], non_blocking=True)
            vc.copy_(v[-batch.num_cdecs:], non_blocking=True)
            self.events[cur_stage].qkvtr_e.record()
```

先看最容易误读的一句：

```python
self._comm_wait_compute()
```

而 `NEO/swiftllm/worker/layers/transformer_layer.py:139-143` 的定义是：

```python
def _comm_wait_compute(self):
    self.cpu_communication_stream.wait_stream(torch.cuda.default_stream())
```

它的含义不是“让 CPU 等 GPU”，而是：

- communication stream 上即将发起的 GPU→CPU copy
- 必须先等默认计算流把 QKV 张量算完

也就是：

> **先确保 QKV 在 GPU 上已经生产完成，再允许 communication stream 去搬运它。**

然后在 `cpu_communication_stream` 上发起三次 `copy_`：

- `q[-num_cdecs:] -> q_cpu`
- `k[-num_cdecs:] -> k_cpu`
- `v[-num_cdecs:] -> v_cpu`

因为目标是 pinned host memory，所以这是标准的 **device→host async copy** 用法。

最后：

```python
self.events[cur_stage].qkvtr_e.record()
```

这里的 `qkvtr_e` 只表示：

> **communication stream 上这批 QKV copy 何时完成**。

它**不表示** CPU attention 已完成；这点必须分清。

---

## 3.2 阶段 B：host 显式等待 QKV copy 完成

继续看 `_attention()` 中 CPU decode 分支：`NEO/swiftllm/worker/layers/transformer_layer.py:331-349`

```python
if batch.num_cdecs > 0:
    oc = self.swapper.o_cpu[:batch.num_cdecs]
    events.pf_time("lnch_m")
    self.events[cur_stage].qkvtr_e.synchronize()
    events.pf_time("cdec_s")
    torch.ops.pacpu.paged_attention_cpu(...)
```

最关键的一句是：

```python
self.events[cur_stage].qkvtr_e.synchronize()
```

这是一个 **host-side blocking call**。

它的真实含义是：

- 当前 Python / host 线程停在这里
- 直到前面 `qkvtr_e.record()` 对应的 CUDA event 已经完成
- 也就是：`q_cpu / k_cpu / v_cpu` 这批 pinned host buffer 已经可安全读取

因此，NEO 在这里并不是“用 CUDA stream 通知 CPU 算子什么时候启动”。

更准确的说法是：

> **主机线程自己先阻塞，等数据真的已经落到 host buffer，再去同步调用 CPU 算子。**

这是全文最核心的同步边界之一。

### 纠正常见误解

误解：`qkvtr_e.synchronize()` 在等 CPU attention 做完。

不对。它只是在等：

- `_transfer_qkv()` 发起的 GPU→CPU QKV copy 完成。

CPU attention 这时甚至还没开始执行。

---

## 3.3 阶段 C：CPU attention 真正发生在 `torch.ops.pacpu.paged_attention_cpu(...)`

接着还是同一段代码：

```python
torch.ops.pacpu.paged_attention_cpu(
    cur_layer_id,
    self.model_config.softmax_scale,
    batch.seq_ids_list[batch.num_prgds:],
    batch.seq_lens_list[batch.num_prgds:],

    self.swapper.q_cpu[:batch.num_cdecs],
    self.swapper.k_cpu[:batch.num_cdecs],
    self.swapper.v_cpu[:batch.num_cdecs],
    self.swapper.k_swap,
    self.swapper.v_swap,
    self.swapper.cpu_block_table,
    oc
)
```

这一调用的本体在 `NEO/pacpu/pacpu.cpp:72-124`：

```cpp
void paged_attention_cpu(
  int64_t cur_layer,
  double softmax_scale,
  const std::vector<int64_t> &seq_ids,
  const std::vector<int64_t> &seq_lengths,
  at::Tensor q,
  at::Tensor k,
  at::Tensor v,
  at::Tensor k_cache,
  at::Tensor v_cache,
  at::Tensor block_table,
  at::Tensor o
) {
  ...
  auto qbatch_p = (data_t*) q.data_ptr<at_data_t>();
  auto kbatch_p = (data_t*) k.data_ptr<at_data_t>();
  auto vbatch_p = (data_t*) v.data_ptr<at_data_t>();
  auto obatch_p = o.data_ptr<otpt_t>();
  auto kcache_p = (data_t*) k_cache.data_ptr<at_data_t>();
  auto vcache_p = (data_t*) v_cache.data_ptr<at_data_t>();
  auto block_table_p = block_table.data_ptr<int32_t>();

  ispc_attention_tasks(
    cur_layer, num_blocks, batch_size, block_table_width, softmax_scale,
    seq_ids, seq_lengths,
    qbatch_p, kbatch_p, vbatch_p, obatch_p, kcache_p, vcache_p, block_table_p
  );
}
```

这里已经把语义写死了：

- 输入是 `at::Tensor`，但函数体里马上把它们变成裸指针
- 然后调用 `ispc_attention_tasks(...)`
- 返回前，没有任何异步 dispatch 的迹象

所以这一步应该理解为：

> `torch.ops.pacpu.paged_attention_cpu(...)` 是一次**同步的 C++ operator 调用**。当 Python 看到它“返回”时，这次 CPU attention 已经完成，结果已经写在 `o_cpu` 对应的 `oc` 里。

### 3.3.1 CPU attention 的实现确实在 CPU 线程里

`NEO/pacpu/core.h:222-260` 的 `ispc_attention_tasks(...)` 明确是 CPU 侧任务切分逻辑：

```cpp
int ws = omp_get_max_threads();
...
std::vector<std::tuple<int, int, int, int> > tasks;
...
```

而 `NEO/pacpu/core.h` 更前面的实现里也能看到 OpenMP / ISPC / CPU 内存访问，例如：

- `brute::store_kv(...)` 把本轮 K/V 写入 CPU KV cache：`core.h:11-31`
- `brute::qk_product(...)` / `softmax(...)` / `av_product(...)`：`core.h:33-126`
- `#pragma omp parallel` 的并行区（在后半段实现中）

这说明 pacpu 的计算语义是：

- 在 host/CPU 上执行
- 由 C++ / OpenMP / ISPC 完成并行化
- **不是**某个隐藏的 CUDA kernel
- **也不在** CUDA stream 上运行

### 3.3.2 这一步还顺手完成了 CPU KV cache 的写入

`NEO/pacpu/core.h:152-165` 的 `brute_attention(...)` 里，单个 sequence 的顺序是：

```cpp
brute::store_kv(cur_layer, num_blocks, seq_len, kip, vip, kcache_p, vcache_p, btp);
brute::qk_product(...);
brute::softmax(...);
brute::av_product(...);
```

也就是 CPU decode attention 不只是“读 CPU KV cache”，还会把本轮 decode 的 K/V 先写入 `k_cache / v_cache`，然后再参与 attention。

在 worker 侧对应到这次调用的实参，就是：

- `self.swapper.k_swap`
- `self.swapper.v_swap`
- `self.swapper.cpu_block_table`

所以这套路径确实是围绕 CPU swap 区展开的。

---

## 3.4 阶段 D：把 CPU 输出回拷到 GPU，并让后续 GPU 计算等待它

CPU op 返回后，`_attention()` 继续执行：

```python
events.pf_time("cdec_e")
with torch.cuda.stream(self.cpu_communication_stream):
    o[-batch.num_cdecs:, :].copy_(oc, non_blocking=True)
...
self._compute_wait_comm() # Wait for CPU decoding to finish
```

其中 `oc = self.swapper.o_cpu[:batch.num_cdecs]`，而 `o` 是：

```python
o = batch.attn_out_buf.view(batch.iter_width, -1, self.model_config.head_dim)
```

所以这一步的物理意义是：

- `o_cpu` 中 CPU 算好的输出
- 被异步拷回 GPU 上 `batch.attn_out_buf` 的尾部区域
- 也就是 `o[-batch.num_cdecs:, :]`

因为 `oc` 在 pinned host memory 中，所以这一步可以作为 **host→device async copy** 发给 `cpu_communication_stream`。

然后看 `_compute_wait_comm()` 的定义：`NEO/swiftllm/worker/layers/transformer_layer.py:146-150`

```python
def _compute_wait_comm(self):
    torch.cuda.default_stream().wait_stream(self.cpu_communication_stream)
```

它的真实含义是：

> 默认 CUDA stream 上后续所有计算，都必须等 communication stream 上尚未完成的工作结束后才能继续。

在当前上下文里，这些工作主要包括：

- 刚才发起的 `o_cpu -> attn_out_buf` 回拷
- 同一 communication stream 上还没做完的 swap/copy

于是接下来的 `_postproj()` 才能安全读取完整的 `batch.attn_out_buf`。

### 3.4.1 这里为什么注释写的是“Wait for CPU decoding to finish”

源码注释是：

```python
self._compute_wait_comm() # Wait for CPU decoding to finish
```

这句话如果字面理解，很容易误读成：

- “default stream 在直接等待 CPU 内核执行完”

其实不准确。

更精确地说：

- 到执行这行代码时，`torch.ops.pacpu.paged_attention_cpu(...)` 已经**同步返回**
- 所以 CPU attention 本体其实已经结束
- `_compute_wait_comm()` 等的并不是 CPU op 本身
- 而是 **CPU attention 的输出回拷到 GPU** 以及 communication stream 上挂着的其它 copy/swap

所以这句注释更接近下面这个意思：

> 这里形成了 CPU decode 整条路径的 GPU-side join point，保证后续 GPU 看到的是完整结果。

---

## 4. 把整个等待链压成一个时序图

如果只看一个 `batch.num_cdecs > 0` 的分支，可以把 `NEO/swiftllm/worker/layers/transformer_layer.py:158-179, 331-355` 压成下面这张图：

```text
default CUDA stream:
    计算 q/k/v
         |
         |  _comm_wait_compute() 让 communication stream 依赖这里
         v

cpu_communication_stream:
    q/k/v 尾部 GPU->CPU copy 到 q_cpu/k_cpu/v_cpu
    record(qkvtr_e)

host thread:
    qkvtr_e.synchronize()      # 等 copy 真正完成，CPU 输入就绪
    torch.ops.pacpu.paged_attention_cpu(...)  # 同步 CPU C++ op，写 o_cpu

cpu_communication_stream:
    o_cpu -> attn_out_buf 尾部 CPU->GPU copy

default CUDA stream:
    _compute_wait_comm()       # 等 communication stream
    _postproj()                # 此时读到完整 attn_out_buf
```

因此：

- **真正阻塞 host 的点**：
  - `qkvtr_e.synchronize()`
  - `torch.ops.pacpu.paged_attention_cpu(...)`
- **真正约束 GPU 后续计算的点**：
  - `_compute_wait_comm()`

这就是 NEO 等待 CPU decode attention 的完整答案。

---

## 5. 再看 pacpu：它是怎样从源码变成 `torch.ops.pacpu.paged_attention_cpu`

这一部分按“构建 → 配置 → 传递 → 加载 → 注册 → 调用”讲最清楚。

## 5.1 构建阶段：`.so` 是怎样编出来的

入口是 `NEO/pacpu/build.sh:1-7`：

```sh
Torch_DIR=$(python -c 'import torch;print(torch.utils.cmake_prefix_path)')/Torch
CUDA_HOST_COMPILER_PATH=$(which g++-11)
CXX_COMPILER_PATH=$(which g++-13)

mkdir -p build
cmake -B build -S . -DTorch_DIR=$Torch_DIR -DModel=$1 -DTP=$2 -DCMAKE_CUDA_HOST_COMPILER=${CUDA_HOST_COMPILER_PATH} -DCMAKE_CXX_COMPILER=${CXX_COMPILER_PATH}
cmake --build build
```

这里做了两件关键事：

1. 用 `-DModel=$1 -DTP=$2` 把模型名和 tensor parallel 度传给 CMake。
2. 用 `Torch_DIR=...` 让 CMake 能找到 libtorch / Torch CMake config。

接着看 `NEO/pacpu/CMakeLists.txt:40-56`：

```cmake
function(gen_torch_lib model_name tp_degree)
  set(TARGET_NAME "${TARGET_NAME_PREFIX}-${model_name}-tp${tp_degree}")

  add_library(${TARGET_NAME} SHARED)
  target_sources(${TARGET_NAME} PRIVATE ${TARGET_SOURCES})
  target_add_common_options(${TARGET_NAME} ${model_name} ${tp_degree})
  target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:C,CXX>:${TORCH_CXX_FLAGS}>)
  target_link_libraries(${TARGET_NAME} PRIVATE "${TORCH_LIBRARIES}")
endfunction()
...
gen_torch_lib(${Model} ${TP})
```

所以最终产物是一个 **Torch shared library**，目标名规则为：

```text
pacpu-{model_name}-tp{tp_degree}
```

按 Linux 默认命名，文件名就是：

```text
libpacpu-{model_name}-tp{tp_degree}.so
```

这和 `NEO/examples/example.py:99` 完全一致：

```python
library_path=f"{repo_dir}/pacpu/build/libpacpu-{args.model_name}-tp{args.tp_degree}.so"
```

也和链接产物 `NEO/pacpu/build/CMakeFiles/pacpu-llama2_7b-tp2.dir/link.txt:1` 一致：

```text
-o libpacpu-llama2_7b-tp2.so
```

### 5.1.1 这个库不是普通 `import pacpu`

这里产出的不是一个通过 `import pacpu` 导入的 Python 扩展模块；它的定位是：

- 一个可被 `torch.ops.load_library(...)` 动态装入的 Torch operator library。

这是后面链路的关键前提。

---

## 5.2 配置阶段：`.so` 路径怎样进入 `EngineConfig`

`NEO/swiftllm/engine_config.py:30-32` 把它作为配置字段：

```python
library_path: str
profile_result_path: str
```

而 `NEO/swiftllm/engine_config.py:135-139` 又提供了 CLI 参数：

```python
parser.add_argument(
    "--library-path",
    type=str,
    help="Path to the external library",
)
```

所以从系统设计上，pacpu `.so` 的物理路径并不是 worker 自己扫描出来的，而是**外部传入配置**。

### 5.2.1 API server 模式下怎么传

`NEO/swiftllm/server/api_server.py:47-58`：

```python
parser = argparse.ArgumentParser()
swiftllm.EngineConfig.add_cli_args(parser)
...
engine = swiftllm.AsyncEngine(swiftllm.EngineConfig(**args))
```

也就是：

- CLI 收到 `--library-path`
- 最终构造成 `EngineConfig.library_path`

### 5.2.2 offline example 里怎么传

`NEO/examples/example.py:85-104` 直接手工构造：

```python
engine_config = swiftllm.EngineConfig(
    ...
    library_path=f"{repo_dir}/pacpu/build/libpacpu-{args.model_name}-tp{args.tp_degree}.so",
    ...
)
```

所以无论是 server 入口还是 example 入口，`.so` 路径都在 **engine 初始化之前** 已经准备好了。

---

## 5.3 传递阶段：配置怎样一路到达 worker model

先看 `NEO/swiftllm/server/engine.py:49-55`：

```python
def initialize(self):
    ...
    self.executor = self.executor_class(self.engine_config, self.model_config)
```

然后看 `NEO/swiftllm/server/executor.py`：

### 单进程模式

`executor.py:65-75`

```python
class SingleProcExecutor(Executor):
    def __init__(self, engine_config, model_config):
        ...
        self.model = LlamaModel(engine_config, model_config, rank=0)
```

### Ray 模式

`executor.py:98-110`

```python
class RayExecutor(Executor):
    def __init__(self, engine_config, model_config):
        ...
        self.models = [RemoteLlamaModel.remote(engine_config, model_config, rank=i) for i in range(num_workers)]
```

因此不管是：

- `SingleProcExecutor`
- 还是 `RayExecutor`

`engine_config` 都会被传到 worker 侧 `LlamaModel` / `RemoteLlamaModel`，其中自然也包括 `engine_config.library_path`。

---

## 5.4 加载阶段：`torch.ops.load_library(...)` 什么时候执行

真正的加载点在 `NEO/swiftllm/worker/model.py:150-153`：

```python
if engine_config.library_path:
    torch.ops.load_library(engine_config.library_path)
self.cpu_communication_stream = torch.cuda.Stream()
```

这说明：

1. worker model 初始化时，如果给了 `library_path`，先动态加载 `.so`
2. 然后才继续初始化后续对象

`torch.ops.load_library(...)` 的作用可以直接理解为：

- 把这个 shared library 装入当前 Python 进程
- 执行库内部静态注册逻辑
- 把其中通过 PyTorch dispatcher 注册的 operator 放到 `torch.ops` 命名空间下

所以对 NEO 而言：

> pacpu 不是通过 Python import 被加载，而是通过 `torch.ops.load_library(...)` 被动态装入。

---

## 5.5 注册阶段：为什么 Python 里会出现 `torch.ops.pacpu.paged_attention_cpu`

这一步在 `NEO/pacpu/pacpu.cpp:126-131`：

```cpp
TORCH_LIBRARY(pacpu, m) {
#ifdef USE_ATEN_OPER
  m.def("paged_attention_cpu_torch", &paged_attention_cpu_torch);
#endif
  m.def("paged_attention_cpu", &paged_attention_cpu);
}
```

这里的语义非常直接：

- 用 `TORCH_LIBRARY(pacpu, m)` 注册一个名为 `pacpu` 的 Torch library namespace
- 在这个 namespace 下注册：
  - `paged_attention_cpu`

因此当 `.so` 被 `torch.ops.load_library(...)` 成功装入后，Python 侧就会出现：

```python
torch.ops.pacpu.paged_attention_cpu
```

所以它不是：

- 普通 Python 函数
- `ctypes` 导出的手工符号
- pybind11 自己绑定的模块函数

而是：

> **通过 PyTorch dispatcher 注册出来的 C++ operator entrypoint。**

---

## 5.6 调用阶段：worker 是怎样实际调用这个 op 的

最后收口到 worker 侧：`NEO/swiftllm/worker/layers/transformer_layer.py:336-349`

```python
torch.ops.pacpu.paged_attention_cpu(
    cur_layer_id,
    self.model_config.softmax_scale,
    batch.seq_ids_list[batch.num_prgds:],
    batch.seq_lens_list[batch.num_prgds:],
    self.swapper.q_cpu[:batch.num_cdecs],
    self.swapper.k_cpu[:batch.num_cdecs],
    self.swapper.v_cpu[:batch.num_cdecs],
    self.swapper.k_swap,
    self.swapper.v_swap,
    self.swapper.cpu_block_table,
    oc
)
```

所以这个调用链可以完整写成：

```text
pacpu/build.sh
  -> pacpu/CMakeLists.txt
  -> libpacpu-{model}-tp{tp}.so
  -> EngineConfig.library_path
  -> Engine / Executor
  -> LlamaModel.__init__()
  -> torch.ops.load_library(library_path)
  -> TORCH_LIBRARY(pacpu, m)
  -> torch.ops.pacpu.paged_attention_cpu(...)
```

这就是从 `pacpu/` 源码到 Python 调用点的完整闭环。

---

## 6. 为什么这套机制能保证 `_postproj()` 读到正确结果

把前面的同步边界和数据路径合起来，答案其实很明确：

## 6.1 CPU op 启动前，输入已经稳定

因为有：

```python
self.events[cur_stage].qkvtr_e.synchronize()
```

所以当 `paged_attention_cpu(...)` 开始时：

- `q_cpu / k_cpu / v_cpu` 已经不再是“正在 DMA 中”的未完成数据
- 它们已经是可供 CPU 同步读取的 pinned host buffer

## 6.2 CPU op 返回时，输出已经写进 `o_cpu`

因为 `torch.ops.pacpu.paged_attention_cpu(...)` 是同步 C++ 调用，所以返回后：

- `oc` / `o_cpu` 中的结果已经生成完成

## 6.3 `_postproj()` 开始前，GPU 侧 `attn_out_buf` 已经拿到 CPU 输出

因为有：

```python
with torch.cuda.stream(self.cpu_communication_stream):
    o[-batch.num_cdecs:, :].copy_(oc, non_blocking=True)
...
self._compute_wait_comm()
```

所以默认 stream 继续跑 `_postproj()` 之前：

- `batch.attn_out_buf` 尾部区域的 CPU decode 输出已经回写完成

因此 `_postproj()` 看到的是完整 attention 输出：

- prefill 输出
- GPU decode 输出
- CPU decode 输出

三部分都已经在同一个 `attn_out_buf` 里会合。

---

## 7. 最容易误读的点集中纠正

## 误解 1：CUDA stream 直接控制了 CPU attention 的启动和结束

不对。

更准确地说：

- CUDA stream 只控制 GPU 相关 copy 和后续 GPU 依赖
- CPU attention 本体由 host 同步调用的 C++ op 执行

## 误解 2：`qkvtr_e.synchronize()` 等的是 CPU attention 完成

不对。

它等的是：

- `_transfer_qkv()` 发起的 GPU→CPU QKV copy 完成

CPU attention 还没开始。

## 误解 3：`_compute_wait_comm()` 直接等待 CPU kernel 执行

不对。

执行到这行时，`paged_attention_cpu(...)` 已经同步返回。

`_compute_wait_comm()` 等的是：

- `cpu_communication_stream` 上尚未完成的 CPU→GPU 回拷
- 以及同一 stream 上的其它 copy/swap

## 误解 4：`torch.ops.pacpu.paged_attention_cpu` 是 Python 里定义的自定义函数

不对。

它是 `.so` 里通过 `TORCH_LIBRARY(pacpu, m)` 注册出来的 C++ op。

## 误解 5：pacpu 是通过普通 `import pacpu` 加载的

不对。

NEO 通过：

```python
torch.ops.load_library(engine_config.library_path)
```

动态加载 shared library。

## 误解 6：CPU decode 用的是一套独立于 worker 的数据结构

不对。

它直接使用：

- `Swapper.q_cpu / k_cpu / v_cpu / o_cpu`
- `Swapper.k_swap / v_swap`
- `Swapper.cpu_block_table`

## 误解 7：只要有 `_compute_wait_comm()`，就不需要 `qkvtr_e.synchronize()`

不对。

二者解决的是不同问题：

- `qkvtr_e.synchronize()`：保证 **CPU op 启动前输入已就绪**
- `_compute_wait_comm()`：保证 **后续 GPU op 启动前输出已回到 GPU**

少任何一个，边界都不完整。

---

## 8. 把整条链路压成一句话

如果只用一句话概括 NEO 的 worker 侧 CPU decode wait 机制，可以这样说：

> 它先在 `cpu_communication_stream` 上把 CPU decode 所需尾部 QKV 从 GPU 异步拷到 pinned host buffer，再由 host 线程用 `qkvtr_e.synchronize()` 显式等输入就绪、同步调用 `torch.ops.pacpu.paged_attention_cpu(...)` 在 CPU 上完成 attention，随后把 `o_cpu` 异步拷回 GPU，并用 `_compute_wait_comm()` 让默认 CUDA stream 在 `_postproj()` 前等待这次回拷完成。

如果只用一句话概括 pacpu 的加载链路，可以这样说：

> `pacpu/build.sh` 与 `CMakeLists.txt` 先生成 `libpacpu-{model}-tp{tp}.so`，再由 `EngineConfig.library_path` 把路径传到 worker，在 `LlamaModel.__init__()` 中通过 `torch.ops.load_library(...)` 动态装库，触发 `TORCH_LIBRARY(pacpu, m)` 注册，最终让 Python 侧可以调用 `torch.ops.pacpu.paged_attention_cpu(...)`。

---

## 9. 建议顺着源码自行验证的阅读顺序

如果你想自己顺一遍源码，最不容易迷路的顺序是：

1. `NEO/swiftllm/worker/layers/transformer_layer.py:158-179`
   - `_transfer_qkv()`：先看 event 到底记录了什么
2. `NEO/swiftllm/worker/layers/transformer_layer.py:331-355`
   - `_attention()`：看 host 阻塞、CPU op、回拷、join point
3. `NEO/swiftllm/worker/layers/transformer_layer.py:139-150`
   - `_comm_wait_compute()` / `_compute_wait_comm()`：看 stream 依赖方向
4. `NEO/swiftllm/worker/block_swapper.py:53-74`
   - 看 CPU decode 用到哪些 pinned host buffer
5. `NEO/swiftllm/structs.py:265-281`
   - 看 `num_cdecs / num_prgds / sum_pref_toks / sum_prgd_toks`
6. `NEO/pacpu/pacpu.cpp:72-131`
   - 看 `paged_attention_cpu(...)` 与 `TORCH_LIBRARY(pacpu, m)`
7. `NEO/pacpu/core.h:130-260`
   - 看 CPU op 实际怎样跑在 CPU 上
8. `NEO/pacpu/build.sh:1-7` + `NEO/pacpu/CMakeLists.txt:40-56`
   - 看 `.so` 是怎样构建出来的
9. `NEO/swiftllm/worker/model.py:150-153`
   - 看 `.so` 在 worker 初始化时怎样被动态加载

按这个顺序读，最容易把“host 同步”和“CUDA stream 依赖”拆开看清楚。
