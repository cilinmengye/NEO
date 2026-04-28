# NEO GPU/CPU KV Cache 与 Paged Attention 开发者说明

本文从开发者视角解释 NEO 如何组织 GPU/CPU 上的 KV Cache，如何维护逻辑 block 到物理 block 的映射，以及 GPU / CPU paged attention 如何通过 block table 读写物理 KV Cache。

重点代码入口：

- `swiftllm/server/engine.py`：初始化 executor、profiling 可用 block 数、创建 `BlockManager`、初始化 worker KV cache。
- `swiftllm/server/scheduler.py`：决定请求进入 prefill、GPU decode、CPU decode，以及是否 swap in/out。
- `swiftllm/server/block_manager.py`：server 侧权威 block 分配、释放、swap 映射。
- `swiftllm/worker/block_swapper.py`：worker 侧真实 GPU/CPU KV cache、block table、CPU staging buffer。
- `swiftllm/worker/model.py`：每个 iteration 更新 block table、执行 swap、准备 batch tensor、调用 transformer layers。
- `swiftllm/worker/layers/transformer_layer.py`：prefill KV 写入、GPU paged attention、CPU paged attention、CPRF swap out。
- `swiftllm/worker/kernels/paged_attn.py`：当前活跃的 GPU decode paged attention Triton 实现。
- `csrc/src/small_kernels.cu`：当前活跃的 `swiftllm_c.store_kvcache` prefill KV 写入实现。
- `csrc/src/block_swapping.cpp`：`swiftllm_c.swap_blocks` 物理 KV block copy 实现。
- `pacpu/pacpu.cpp`、`pacpu/core.h`、`pacpu/pacpu.ispc`：CPU paged attention 实现。

---

## 1. 总览：control-plane 与 data-plane

NEO 的 KV Cache 管理可以分成两层：

### 1.1 Server/control-plane：决定“逻辑 block 映射到哪个物理 block”

server 侧的核心对象是：

- `BlockManager`
- `DeviceBlockManager`

它们不持有真实 K/V tensor，而是维护 allocation metadata：

- 某个 request / sequence 当前有多少个逻辑 KV block；
- 每个逻辑 KV block 被放在哪个物理 block；
- GPU / CPU 物理 block 哪些空闲；
- 当前 iteration 需要把哪些 block table entry 更新到 worker；
- 当前 iteration 是否需要做 GPU <-> CPU 的物理 block copy。

换句话说，server 侧是 KV cache 的“控制面”。它回答的问题是：

> seq_id=3 的第 2 个逻辑 block，应该映射到 GPU 物理 block 11，还是 CPU 物理 block 25？

### 1.2 Worker/data-plane：真正持有和使用物理 KV cache

worker 侧的核心对象是 `Swapper`。

它真正分配并持有：

- GPU KV cache tensor：`k_cache` / `v_cache`；
- CPU KV cache tensor：`k_swap` / `v_swap`；
- GPU block table tensor：`gpu_block_table`；
- CPU block table tensor：`cpu_block_table`；
- CPU decode 用 pinned staging buffers：`q_cpu` / `k_cpu` / `v_cpu` / `o_cpu`。

worker 侧是 KV cache 的“数据面”。它回答的问题是：

> GPU paged attention kernel 要读 seq_id=3 的第 2 个逻辑 block 时，应该从 `k_cache[layer, 11, kv_head, :, :]` 读取。

### 1.3 一句话链路

一次 iteration 中：

1. `Scheduler` 决定哪些 request 做 prefill / GPU decode / CPU decode / swap。
2. `BlockManager.prepare()` 在 server 侧分配逻辑 block 到物理 block，产出：
   - `mappings`：需要写入 worker block table 的 `(vid, pid)`；
   - `swappings`：需要复制的物理 block id；
   - `is_swap_out`：本轮主 swap 方向。
3. `LlamaModel.do_one_iteration()` 在 worker 侧：
   - 调 `Swapper.set_block_tables(mappings)` 更新 worker block table；
   - 调 `Swapper.swap_blocks(...)` 做物理 KV block copy；
   - 进入 `_forward_batches()` 执行 prefill / decode attention。
4. transformer layer 内部：
   - prefill 通过 `swiftllm_c.store_kvcache()` 写 GPU KV cache；
   - GPU decode 通过 Triton `paged_attention()` 读写 GPU KV cache；
   - CPU decode 通过 `torch.ops.pacpu.paged_attention_cpu()` 读写 CPU KV cache。

---

## 2. 关键概念：seq_id、逻辑 block、物理 block、vid、pid

### 2.1 `request_id` / `seq_id`

在 NEO 中，一个 request 会被分配一个 `request_id`。在 KV cache 管理语义里，它也就是 `seq_id`：

- `Request.request_id`：server 侧 request 在 block table 中的行号；
- worker 侧 attention kernel 看到的 `seq_ids` 也是这些 request id；
- block table 的第 `seq_id` 行保存该 sequence 的逻辑 block 到物理 block 映射。

因此：

```text
seq_id == request.request_id == block_table 的行号
```

### 2.2 逻辑 block 与物理 block

对每个 sequence，KV cache 按 token position 被切成固定大小的逻辑 block。

如果：

```text
block_size = 16
```

那么：

- token 0..15 属于逻辑 block 0；
- token 16..31 属于逻辑 block 1；
- token 32..47 属于逻辑 block 2。

逻辑 block 是每个 sequence 自己的 block 编号；物理 block 是 GPU/CPU KV cache tensor 中真实的 block index。

例如：

```text
seq_id=3 的 logical_block_pos=2 -> gpu_block_table[3, 2] = 11
```

含义是：

```text
seq_id=3 的第 2 个逻辑 block 存在 GPU 物理 block 11
```

### 2.3 映射公式

对任意 token：

```text
logical_block_pos = token_pos // block_size
block_offset      = token_pos % block_size
vid               = seq_id * max_blocks_per_seq + logical_block_pos
physical_block    = block_table[seq_id, logical_block_pos]
```

其中：

- `vid` 是 flattened virtual block id，用于一次性更新 `block_table.view(-1)[vid]`；
- `physical_block` 是真实 KV cache tensor 的物理 block 维度下标。

物理 cache 访问下标形如：

```text
[layer, physical_block, kv_head, block_offset, :]
```

### 2.4 贯穿数值例子

后文反复使用这个小例子：

```text
block_size = 4
max_blocks_per_seq = 8
seq_id = 3
seq_len = 10
num_layers = 2
num_gpu_blocks = 16
num_cpu_blocks = 32
local_num_kv_heads = 2
head_dim = 128
```

对于 `token_pos=9`：

```text
logical_block_pos = 9 // 4 = 2
block_offset      = 9 % 4  = 1
vid               = 3 * 8 + 2 = 26
```

如果：

```text
gpu_block_table[3, 2] = 11
```

那么 token 9 的 K/V 会位于 GPU KV cache：

```text
k_cache[layer, 11, kv_head, 1, :]
v_cache[layer, 11, kv_head, 1, :]
```

如果：

```text
cpu_block_table[3, 2] = 25
```

那么 token 9 的 K/V 会位于 CPU KV cache：

```text
k_swap[layer, 25, kv_head, 1, :]
v_swap[layer, 25, kv_head, 1, :]
```

---

## 3. 核心数据结构与 tensor shape

### 3.1 Server 侧 `DeviceBlockManager`

`DeviceBlockManager` 负责一个 device namespace 内的 block 管理。NEO 会分别为 GPU 和 CPU 建立一个 `DeviceBlockManager`。

#### `seq_num_blks`

```text
shape  = [max_seqs_in_block_table]
dtype  = torch.int32
device = CPU
```

维度含义：

- 第 `seq_id` 个元素表示该 sequence 当前已经分配了多少个逻辑 block。

例子：

```text
seq_num_blks[3] = 3
```

表示 seq_id=3 当前有 logical block 0、1、2。

#### `block_table`

```text
shape  = [max_seqs_in_block_table, max_blocks_per_seq]
dtype  = torch.int32
device = CPU
```

维度含义：

- 第 0 维：`seq_id` / request id；
- 第 1 维：该 sequence 内的 logical block position；
- 值：物理 block id。

server 侧 `block_table` 是权威 metadata；worker 侧的 `gpu_block_table` / `cpu_block_table` 是执行时使用的镜像。

#### `is_block_free`

```text
shape  = [num_blocks]
dtype  = torch.bool
device = CPU
```

NEO 中它是按 split 保存的 list：

```text
is_block_free[split_id][physical_block_id]
```

含义：该 split 中某个物理 block 是否空闲。

#### `num_free_blocks`

```text
Python list[int]
长度 = nsplits
```

表示每个 split 当前还有多少空闲物理 block。

### 3.2 Worker 侧 GPU KV cache：`k_cache` / `v_cache`

在 `Swapper.__init__()` 中分配。

```text
shape = [num_layers + extra_layer_for_cprf,
         num_gpu_blocks,
         local_num_kv_heads,
         block_size,
         head_dim]
dtype  = torch.float16
device = CUDA
```

维度含义：

1. `num_layers + extra_layer_for_cprf`
   - transformer layer 维度；
   - 如果 `extra_layer_for_cprf=True`，额外多一层 intermediate layer，用于 CPU-destined prefill 的中转。
2. `num_gpu_blocks`
   - GPU 物理 block id。
3. `local_num_kv_heads`
   - tensor parallel 后当前 rank 本地的 KV head 数；
   - `local_num_kv_heads = model_config.num_kv_heads // model_config.world_size`。
4. `block_size`
   - 一个物理 block 内的 token slot。
5. `head_dim`
   - 每个 head 的 hidden dimension。

访问格式：

```text
k_cache[layer, gpu_physical_block, kv_head, block_offset, head_dim_index]
v_cache[layer, gpu_physical_block, kv_head, block_offset, head_dim_index]
```

以贯穿例子为例，如果 `extra_layer_for_cprf=True`：

```text
shape = [2 + 1, 16, 2, 4, 128] = [3, 16, 2, 4, 128]
```

其中：

- layer 0、1 是真实 transformer layer；
- layer 2 是 CPRF intermediate layer。

### 3.3 Worker 侧 CPU KV cache：`k_swap` / `v_swap`

同样在 `Swapper.__init__()` 中分配。

```text
shape = [num_layers,
         num_cpu_blocks,
         local_num_kv_heads,
         block_size,
         head_dim]
dtype  = torch.float16
device = CPU pinned memory
```

维度含义：

1. `num_layers`
   - transformer layer 维度；CPU KV cache 没有额外 CPRF intermediate layer。
2. `num_cpu_blocks`
   - CPU 物理 block id。
3. `local_num_kv_heads`
   - 当前 TP rank 本地 KV head 数。
4. `block_size`
   - 物理 block 内 token slot。
5. `head_dim`
   - 每个 head 的 hidden dimension。

访问格式：

```text
k_swap[layer, cpu_physical_block, kv_head, block_offset, head_dim_index]
v_swap[layer, cpu_physical_block, kv_head, block_offset, head_dim_index]
```

以贯穿例子为例：

```text
shape = [2, 32, 2, 4, 128]
```

### 3.4 Worker 侧 block table

#### `gpu_block_table`

```text
shape  = [max_seqs_in_block_table, max_blocks_per_seq]
dtype  = torch.int32
device = CUDA
```

用途：GPU prefill store 和 GPU paged attention 使用它，把 sequence 的 logical block 映射到 GPU physical block。

#### `cpu_block_table`

```text
shape  = [max_seqs_in_block_table, max_blocks_per_seq]
dtype  = torch.int32
device = CPU
```

用途：CPU paged attention 使用它，把 sequence 的 logical block 映射到 CPU physical block。

### 3.5 CPU decode staging buffers

CPU decode 不是直接从 GPU tensor 里读 Q/K/V。`_transfer_qkv()` 会把 CPU decode requests 对应的 Q/K/V 从 GPU 异步 copy 到 pinned CPU buffer，然后 CPU op 使用这些 buffer。

#### `q_cpu`

```text
shape  = [max_batch_size, local_num_q_heads, head_dim]
dtype  = torch.float16
device = CPU pinned memory
```

维度含义：

- 第 0 维：当前 CPU decode batch 中第几个 request；
- 第 1 维：当前 TP rank 本地 query head；
- 第 2 维：head dimension。

#### `k_cpu` / `v_cpu`

```text
shape  = [max_batch_size, local_num_kv_heads, head_dim]
dtype  = torch.float16
device = CPU pinned memory
```

维度含义：

- 第 0 维：当前 CPU decode batch 中第几个 request；
- 第 1 维：当前 TP rank 本地 KV head；
- 第 2 维：head dimension。

#### `o_cpu`

```text
shape  = [max_batch_size, local_num_q_heads, head_dim]
dtype  = torch.float32
device = CPU pinned memory
```

用途：CPU paged attention 的输出 buffer。CPU op 写入 `o_cpu` 后，worker 再把它异步 copy 回 GPU 的 `o` / attention output buffer。

---

## 4. 初始化链路：KV cache 什么时候创建

KV cache 的实际大小依赖 profiling 阶段测出的可用 block 数。

典型链路：

1. `Engine` 初始化 executor 和 model worker。
2. profiling 阶段估计可用 GPU block 数与 CPU block 数：
   - `num_gpu_blocks`
   - `num_cpu_blocks`
3. `Engine` 创建 server 侧 `BlockManager`。
4. `Executor` 调 worker 的 `LlamaModel.init_kvcache_and_swap(engine_config)`。
5. `LlamaModel.init_kvcache_and_swap()`：
   - 保存 `num_gpu_blocks` / `num_cpu_blocks`；
   - 创建 `Swapper(engine_config, model_config)`；
   - 给每个 transformer layer 调 `layer.set_swapper(self.swapper)`。
6. `Swapper.__init__()` 真正分配：
   - `k_cache` / `v_cache`；
   - `k_swap` / `v_swap`；
   - `gpu_block_table` / `cpu_block_table`；
   - `q_cpu` / `k_cpu` / `v_cpu` / `o_cpu`。

注意：`BlockManager` 与 `Swapper` 是一对控制面/数据面结构。前者知道“应该怎么映射”，后者持有“真实 tensor 与执行时 block table”。

---

## 5. 每个 iteration 如何修改 KV cache 相关数据

### 5.1 Scheduler 先决定请求类别

`SubBatch` 中 request 的顺序固定为：

```text
all_reqs = cprf_reqs + gprf_reqs + gdec_reqs + cdec_reqs
```

含义：

- `cprf_reqs`：CPU-destined prefill requests；prefill 计算仍在 GPU 上做，但这批 KV 最终要去 CPU。
- `gprf_reqs`：GPU-destined prefill requests。
- `gdec_reqs`：GPU decode requests。
- `cdec_reqs`：CPU decode requests。

`SubBatch.set_model_forward_args()` 会派生出：

```text
num_cprfs = len(cprf_reqs)
num_gprfs = len(gprf_reqs)
num_gdecs = len(gdec_reqs)
num_cdecs = len(cdec_reqs)
num_prefs = num_cprfs + num_gprfs
num_prgds = num_prefs + num_gdecs
```

其中 `num_prgds` 表示 prefill + GPU decode 的数量范围，主要用于 GPU-side attention/cache 路径。

### 5.2 BlockManager.prepare() 生成 mappings 与 swappings

`BlockManager.prepare()` 是 server 侧每轮 KV cache metadata 更新的关键边界。

它返回：

```text
mappings = ((gpu_vids, gpu_pids), (cpu_vids, cpu_pids))
swappings = (src_block_ids, dst_block_ids)
is_swap_out = bool
```

#### `mappings`

`mappings` 表示需要写进 worker block table 的增量映射。

例如：

```text
gpu_vids = [26]
gpu_pids = [11]
```

worker 会执行：

```text
gpu_block_table.view(-1)[26] = 11
```

由于：

```text
vid = 3 * 8 + 2 = 26
```

这等价于：

```text
gpu_block_table[3, 2] = 11
```

#### `swappings`

`swappings` 表示要复制哪些物理 block 的内容。

例如 swap out：

```text
src_block_ids = [11]
dst_block_ids = [25]
is_swap_out = True
```

含义：

```text
从 GPU 物理 block 11 copy 到 CPU 物理 block 25
```

注意这里 copy 的是物理 KV block 内容，而不是 block table entry。

### 5.3 Worker 更新 block table

`LlamaModel.do_one_iteration()` 进入 forward 前先做：

```python
self.swapper.set_block_tables(mappings)
```

`Swapper.set_block_tables()` 的语义是：

```text
if gpu_vids:
    gpu_block_table.view(-1)[gpu_vids] = gpu_pids
if cpu_vids:
    cpu_block_table.view(-1)[cpu_vids] = cpu_pids
```

这是 worker 执行时 block table 的增量更新。

### 5.4 Worker 执行物理 block copy

如果本轮有 swap in / swap out，`LlamaModel.do_one_iteration()` 会在 `cpu_communication_stream` 上调用：

```python
self.swapper.swap_blocks(*swappings, is_swap_out, layer_id, layer_id)
```

并对每个 layer 做一次。

底层 `swiftllm_c.swap_blocks` 做的是整块 K 和整块 V 的物理 copy：

- `is_swap_out=True`：GPU -> CPU；
- `is_swap_out=False`：CPU -> GPU。

### 5.5 Worker 准备 attention runtime tensor

`LlamaModel._prepare_inputs()` 为每个 batch 构造：

#### `batch.prgd_seq_ids`

```text
shape  = [num_prgds]
dtype  = torch.int32
device = CUDA
```

内容：`all_reqs[:num_prgds]` 的 request id。

它供 GPU prefill store / GPU paged attention 访问 `gpu_block_table` 行。

#### `batch.prgd_seq_lens`

```text
shape  = [num_prgds]
dtype  = torch.int32
device = CUDA
```

内容：`all_reqs[:num_prgds]` 当前 sequence length。

对于 prefill，它通常是 prompt length；对于 GPU decode，它是包含当前 decode token 后的 sequence length。

#### `batch.pref_st_locs_we`

```text
shape  = [num_prefs + 1]
dtype  = torch.int32
device = CUDA
```

内容：prefill requests 在拼接后的 token tensor 中的 start locations with end。

例如两个 prefill request 长度分别是 5 和 3：

```text
pref_st_locs_we = [0, 5, 8]
```

这使 `store_kvcache()` 能知道第 i 个 prefill request 的 K/V 在拼接 `k` / `v` tensor 里的起始 offset。

---

## 6. Prefill KV 写入路径

### 6.1 入口：`LlamaTransformerLayer._preproj()`

每层 transformer 先由 `_preproj()` 从 embeddings 计算 Q/K/V。对 prefill request，K/V 需要写入 KV cache。

核心路径是：

```text
_preproj()
  -> self._compute_wait_comm()
  -> swiftllm_c.store_kvcache(...)
```

`_compute_wait_comm()` 的作用是让默认 CUDA stream 等待 `cpu_communication_stream`。这是为了避免前面还没完成的 swap / copy 与当前 cache 写入在同一片物理 cache 上发生冲突。

### 6.2 `store_kvcache()` 相关 tensor shape

#### `k` / `v`

在 `_preproj()` 之后，prefill + decode 的 K/V 被拼接在同一个 tensor 中。

从 `store_kvcache()` 的角度看，prefill 部分的语义是：

```text
shape  = [num_prefill_tokens, local_num_kv_heads, head_dim]
dtype  = torch.float16
device = CUDA
```

实际传入的 `k` / `v` 可能还包含 decode token，但 `store_kvcache()` 根据 `num_prefs`、`seq_start_locs`、`seq_lens` 和 `max_pref_toks` 只处理 prefill 部分。

#### `k_cache` / `v_cache`

```text
shape  = [num_layers + extra_layer_for_cprf,
          num_gpu_blocks,
          local_num_kv_heads,
          block_size,
          head_dim]
dtype  = torch.float16
device = CUDA
```

#### `gpu_block_table`

```text
shape  = [max_seqs_in_block_table, max_blocks_per_seq]
dtype  = torch.int32
device = CUDA
```

#### `seq_ids`

```text
shape  = [num_prefs]
dtype  = torch.int32
device = CUDA
```

对应：

```python
batch.prgd_seq_ids[:batch.num_prefs]
```

#### `seq_start_locs` / `pref_st_locs_we`

```text
shape  = [num_prefs + 1]
dtype  = torch.int32
device = CUDA
```

用于定位每个 prefill request 在拼接 K/V tensor 中的起始 token offset。

#### `seq_lens`

```text
shape  = [num_prefs]
dtype  = torch.int32
device = CUDA
```

对应每个 prefill sequence 的长度。

### 6.3 `store_kvcache()` 的关键参数语义

`_preproj()` 调用：

```python
store_kvcache(
    k,
    v,
    self.swapper.k_cache,
    self.swapper.v_cache,
    self.swapper.gpu_block_table,
    batch.prgd_seq_ids[:batch.num_prefs],
    batch.pref_st_locs_we,
    batch.prgd_seq_lens[:batch.num_prefs],
    itm_layer,
    gpu_layer,
    batch.num_cprfs,
    batch.max_pref_toks
)
```

逐项解释：

| 参数 | 含义 |
| --- | --- |
| `k`, `v` | 当前 layer pre-projection 得到的 K/V，prefill token 的源数据在这里。 |
| `k_cache`, `v_cache` | GPU 物理 KV cache，目标写入位置。 |
| `gpu_block_table` | seq/logical block 到 GPU physical block 的映射表。 |
| `seq_ids` | 每个 prefill request 的 block table row。 |
| `pref_st_locs_we` | 每个 prefill request 在拼接 K/V tensor 中的起止位置。 |
| `seq_lens` | 每个 prefill request 的长度。 |
| `itm_layer` | CPRF 使用的 intermediate GPU cache layer。 |
| `gpu_layer` | 正常 GPU prefill 使用的真实 layer id。 |
| `num_cprfs` | 前 `num_cprfs` 个 prefill request 是 CPU-destined prefill。 |
| `max_pref_toks` | kernel launch 使用的最大 prefill 长度上界。 |

### 6.4 CPRF：不是 CPU 上做 prefill

`cprf_reqs` 的名字容易误解。

CPRF 的含义是：

```text
CPU-destined prefill
```

不是：

```text
CPU-computed prefill
```

也就是说：

- prefill 的 Q/K/V projection 和 attention 仍然在 GPU 上做；
- prefill 产生的 KV 先写入 GPU cache；
- 如果 request 后续要放到 CPU decode 队列，则这些 KV 会再 swap out 到 CPU KV cache。

当 `extra_layer_for_cprf=True` 时，CPRF 的 prefill KV 可以先写入 `k_cache/v_cache` 的额外 intermediate layer：

```text
itm_layer = num_layers
```

而普通 GPU prefill 写入：

```text
gpu_layer = 当前真实 layer id
```

### 6.5 Prefill 写入数值例子

仍然使用：

```text
block_size = 4
max_blocks_per_seq = 8
seq_id = 3
seq_len = 10
local_num_kv_heads = 2
head_dim = 128
```

server 已经分配并同步到 worker：

```text
gpu_block_table[3, 0] = 5
gpu_block_table[3, 1] = 8
gpu_block_table[3, 2] = 11
```

因为 seq_len=10，需要 3 个逻辑 block：

```text
logical block 0: token 0,1,2,3 -> physical GPU block 5
logical block 1: token 4,5,6,7 -> physical GPU block 8
logical block 2: token 8,9     -> physical GPU block 11
```

对 token 9：

```text
logical_block_pos = 9 // 4 = 2
block_offset      = 9 % 4 = 1
physical_block    = gpu_block_table[3, 2] = 11
```

`store_kvcache()` 会把该 token 的 K/V 写入：

```text
k_cache[cur_layer, 11, kv_head, 1, :]
v_cache[cur_layer, 11, kv_head, 1, :]
```

如果这是 CPRF 且 `extra_layer_for_cprf=True`，写入 layer 可能不是 `cur_layer`，而是 intermediate layer：

```text
k_cache[num_layers, 11, kv_head, 1, :]
v_cache[num_layers, 11, kv_head, 1, :]
```

随后 `_swap_out_blocks()` 再把这些 physical GPU blocks copy 到 CPU physical blocks。

---

## 7. GPU paged attention 如何使用 GPU KV cache

### 7.1 入口

GPU decode 使用当前活跃的 Triton 实现：

```text
swiftllm/worker/kernels/paged_attn.py::paged_attention()
```

在 `LlamaTransformerLayer._attention()` 中调用：

```python
paged_attention(
    q[batch.sum_pref_toks:batch.sum_prgd_toks],
    k[batch.sum_pref_toks:batch.sum_prgd_toks],
    v[batch.sum_pref_toks:batch.sum_prgd_toks],
    o[batch.sum_pref_toks:batch.sum_prgd_toks],
    self.swapper.k_cache,
    self.swapper.v_cache,
    self.model_config.softmax_scale,
    self.swapper.gpu_block_table,
    batch.prgd_seq_ids[batch.num_prefs:],
    batch.prgd_seq_lens[batch.num_prefs:],
    cur_layer_id,
    batch.seq_block_size,
    batch.num_seq_blocks,
)
```

这里的切片范围：

```text
[sum_pref_toks : sum_prgd_toks]
```

正好是 GPU decode requests 的单 token Q/K/V。

### 7.2 GPU decode 输入输出 tensor shape

#### `q`

```text
shape  = [num_gdecs, local_num_q_heads, head_dim]
dtype  = torch.float16
device = CUDA
```

#### `k` / `v`

```text
shape  = [num_gdecs, local_num_kv_heads, head_dim]
dtype  = torch.float16
device = CUDA
```

这是当前 decode token 的 K/V。GPU paged attention 会先把它写入 `k_cache/v_cache`，然后再参与 attention。

#### `o`

```text
shape  = [num_gdecs, local_num_q_heads, head_dim]
dtype  = torch.float16
device = CUDA
```

输出 attention result。

#### `k_cache` / `v_cache`

```text
shape  = [num_layers + extra_layer_for_cprf,
          num_gpu_blocks,
          local_num_kv_heads,
          block_size,
          head_dim]
dtype  = torch.float16
device = CUDA
```

#### `seq_ids`

```text
shape  = [num_gdecs]
dtype  = torch.int32
device = CUDA
```

#### `seq_lens`

```text
shape  = [num_gdecs]
dtype  = torch.int32
device = CUDA
```

对于 decode，`seq_len` 包含当前刚生成/输入的 decode token，因此 kernel 用 `seq_len - 1` 找当前 token 的 cache 写入位置。

### 7.3 Triton phase1 如何通过 block table 写当前 token K/V

在 `_fwd_paged_attention_phase1` 中，每个 program 处理：

```text
一个 decoding sequence + 一个 q head + 一个 seq block
```

关键变量：

```text
my_batch_id     = 当前 GPU decode batch 中第几个 request
my_q_head_id    = 当前 query head
my_seq_block_id = 当前 sequence block chunk
my_kv_head_id   = my_q_head_id // (num_q_heads // num_kv_heads)
my_seq_id       = seq_ids[my_batch_id]
my_seq_len      = seq_lens[my_batch_id]
```

当前 decode token 的位置是：

```text
token_pos = my_seq_len - 1
```

对应：

```text
my_block_pos    = (my_seq_len - 1) // block_size
my_block_offset = (my_seq_len - 1) % block_size
my_block_index  = gpu_block_table[my_seq_id, my_block_pos]
```

然后写入：

```text
k_cache[cur_layer, my_block_index, my_kv_head_id, my_block_offset, :]
v_cache[cur_layer, my_block_index, my_kv_head_id, my_block_offset, :]
```

### 7.4 GPU decode 写当前 token 的数值例子

假设 GPU decode 的某个 request：

```text
seq_id = 3
seq_len = 10
block_size = 4
gpu_block_table[3, 2] = 11
cur_layer = 1
```

当前 decode token 是 token 9：

```text
token_pos = seq_len - 1 = 9
my_block_pos = 9 // 4 = 2
my_block_offset = 9 % 4 = 1
my_block_index = gpu_block_table[3, 2] = 11
```

如果 `my_kv_head_id=0`，则当前 token K/V 写入：

```text
k_cache[1, 11, 0, 1, :]
v_cache[1, 11, 0, 1, :]
```

### 7.5 GPU paged attention 如何读取历史 K/V

GPU paged attention 把长 sequence 再切成 `seq_block_size` 粒度的 chunks。

在每个 chunk 内，它遍历若干个 KV cache block：

```text
start_block_idx = my_seq_block_id * (seq_block_size // block_size)
block_idx       = start_block_idx + block_i
physical_block  = gpu_block_table[my_seq_id, block_idx]
```

然后读取整块：

```text
k_cache[cur_layer, physical_block, my_kv_head_id, :, :]
v_cache[cur_layer, physical_block, my_kv_head_id, :, :]
```

对最后一个 sequence block，kernel 会用 mask 避免读超过 `seq_len` 的 token slot。

### 7.6 GPU paged attention 读取数值例子

仍然假设：

```text
seq_id = 3
seq_len = 10
block_size = 4
gpu_block_table[3, 0] = 5
gpu_block_table[3, 1] = 8
gpu_block_table[3, 2] = 11
cur_layer = 1
```

这个 sequence 的 token 分布是：

```text
token 0..3 -> k_cache[1, 5,  kv_head, 0..3, :]
token 4..7 -> k_cache[1, 8,  kv_head, 0..3, :]
token 8..9 -> k_cache[1, 11, kv_head, 0..1, :]
```

GPU attention 计算 token 9 的 attention 时，会按 block table 依次读：

```text
logical block 0 -> physical block 5
logical block 1 -> physical block 8
logical block 2 -> physical block 11
```

也就是说，attention 并不要求 sequence 的 KV 在物理 cache 中连续。只要 block table 正确，logical sequence 就能由分散的 physical blocks 拼起来。

---

## 8. CPU paged attention 如何使用 CPU KV cache

### 8.1 CPU decode 的整体路径

CPU decode 的路径是：

```text
_preproj() 产生 GPU 上的 q/k/v
  -> _transfer_qkv() 把 CPU decode 对应的 q/k/v copy 到 pinned CPU buffer
  -> _attention() 调 torch.ops.pacpu.paged_attention_cpu(...)
  -> CPU op 写当前 token K/V 到 k_swap/v_swap
  -> CPU op 通过 cpu_block_table 读取历史 K/V 并计算 attention
  -> 输出写入 o_cpu
  -> worker 把 o_cpu copy 回 GPU attention output buffer
```

### 8.2 `_transfer_qkv()` 的 staging tensor shape

`_transfer_qkv()` 取 CPU decode 部分：

```python
q[-batch.num_cdecs:]
k[-batch.num_cdecs:]
v[-batch.num_cdecs:]
```

copy 到：

```python
self.swapper.q_cpu[:batch.num_cdecs]
self.swapper.k_cpu[:batch.num_cdecs]
self.swapper.v_cpu[:batch.num_cdecs]
```

对应 shape：

```text
q_cpu[:num_cdecs]
shape  = [num_cdecs, local_num_q_heads, head_dim]
dtype  = torch.float16
device = CPU pinned memory

k_cpu[:num_cdecs]
v_cpu[:num_cdecs]
shape  = [num_cdecs, local_num_kv_heads, head_dim]
dtype  = torch.float16
device = CPU pinned memory
```

### 8.3 `paged_attention_cpu` API 与 tensor shape

调用位置在 `_attention()`：

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

参数语义：

| 参数 | shape / 类型 | 含义 |
| --- | --- | --- |
| `cur_layer_id` | Python int | 当前 transformer layer。 |
| `softmax_scale` | Python float | attention score scaling。 |
| `seq_ids_list` | `list[int]`, 长度 `num_cdecs` | CPU decode requests 的 block table row。 |
| `seq_lens_list` | `list[int]`, 长度 `num_cdecs` | CPU decode requests 的当前 sequence length。 |
| `q_cpu` | `[num_cdecs, local_num_q_heads, head_dim]`, fp16, CPU pinned | 当前 decode token Q。 |
| `k_cpu` | `[num_cdecs, local_num_kv_heads, head_dim]`, fp16, CPU pinned | 当前 decode token K。 |
| `v_cpu` | `[num_cdecs, local_num_kv_heads, head_dim]`, fp16, CPU pinned | 当前 decode token V。 |
| `k_swap` | `[num_layers, num_cpu_blocks, local_num_kv_heads, block_size, head_dim]`, fp16, CPU pinned | CPU physical K cache。 |
| `v_swap` | `[num_layers, num_cpu_blocks, local_num_kv_heads, block_size, head_dim]`, fp16, CPU pinned | CPU physical V cache。 |
| `cpu_block_table` | `[max_seqs_in_block_table, max_blocks_per_seq]`, int32, CPU | logical block 到 CPU physical block 映射。 |
| `o_cpu` | `[num_cdecs, local_num_q_heads, head_dim]`, fp32, CPU pinned | CPU attention 输出。 |

### 8.4 CPU op 如何写当前 token K/V

在 `pacpu/core.h` 中，CPU decode 也会先把当前 token 的 K/V 写入 CPU KV cache。

关键公式：

```text
block_pos = (seq_len - 1) / BLOCK_SIZE
block_id  = cpu_block_table[seq_id, block_pos]
block_off = (seq_len - 1) % BLOCK_SIZE
```

物理写入位置：

```text
k_swap[cur_layer, block_id, kv_head, block_off, :]
v_swap[cur_layer, block_id, kv_head, block_off, :]
```

### 8.5 CPU decode 写当前 token 的数值例子

假设：

```text
seq_id = 3
seq_len = 10
block_size = 4
cpu_block_table[3, 2] = 25
cur_layer = 1
```

当前 decode token 是 token 9：

```text
token_pos = 9
block_pos = 9 // 4 = 2
block_off = 9 % 4 = 1
block_id  = cpu_block_table[3, 2] = 25
```

如果 `kv_head=0`，则 CPU op 写入：

```text
k_swap[1, 25, 0, 1, :]
v_swap[1, 25, 0, 1, :]
```

### 8.6 CPU paged attention 如何读取历史 K/V

CPU op 为每个 CPU decode request 取出它的 block table row：

```text
btp = cpu_block_table + seq_id * block_table_width
```

然后对历史 token 所在 block 做：

```text
physical_block = btp[token_pos // block_size]
block_offset   = token_pos % block_size
```

读取：

```text
k_swap[cur_layer, physical_block, kv_head, block_offset, :]
v_swap[cur_layer, physical_block, kv_head, block_offset, :]
```

ISPC 路径也是同一个语义，只是把一段 sequence 切成 segment 并并行计算 QK、softmax、AV，最后合并 partial result。

### 8.7 CPU paged attention 读取数值例子

假设：

```text
seq_id = 3
seq_len = 10
block_size = 4
cpu_block_table[3, 0] = 21
cpu_block_table[3, 1] = 24
cpu_block_table[3, 2] = 25
cur_layer = 1
```

那么 CPU attention 看到的 logical sequence 是：

```text
token 0..3 -> k_swap[1, 21, kv_head, 0..3, :]
token 4..7 -> k_swap[1, 24, kv_head, 0..3, :]
token 8..9 -> k_swap[1, 25, kv_head, 0..1, :]
```

它和 GPU paged attention 的核心思想完全一样：

```text
logical token position -> logical block -> block table -> physical block -> physical KV cache tensor
```

区别是：

- GPU 路径使用 `gpu_block_table` + `k_cache/v_cache` + Triton kernel；
- CPU 路径使用 `cpu_block_table` + `k_swap/v_swap` + C++/ISPC op。

---

## 9. Swap 与 block table 的一致性

### 9.1 block table 只记录映射，不移动数据

例如：

```text
cpu_block_table[3, 2] = 25
```

只表示：

```text
seq_id=3 的 logical block 2 现在应该在 CPU physical block 25
```

它不会自动把数据从 GPU copy 到 CPU。

### 9.2 swap copy 只移动物理 block 内容，不表达逻辑归属

例如：

```text
swap_blocks(src_block_ids=[11], dst_block_ids=[25], is_swap_out=True)
```

只表示：

```text
把 GPU physical block 11 的 K/V 内容 copy 到 CPU physical block 25
```

它本身不知道这个 block 属于哪个 seq_id / logical_block_pos。

### 9.3 一致性依赖 `BlockManager.prepare()` 同时生成两类信息

要让系统正确，必须同时满足：

1. block table entry 指向新的 physical block；
2. physical block 中真的有对应的 K/V 内容。

因此 swap out 的完整语义应该是：

```text
原来：gpu_block_table[3, 2] = 11，并且 k_cache/v_cache 的 GPU block 11 有数据
现在：cpu_block_table[3, 2] = 25，并且 k_swap/v_swap 的 CPU block 25 有同一份数据
```

这需要两步配合：

```text
mappings:  cpu_block_table[3, 2] = 25
swappings: copy GPU physical block 11 -> CPU physical block 25
```

### 9.4 Swap 数值例子

假设 seq_id=3 的 logical block 2 当前在 GPU：

```text
gpu_block_table[3, 2] = 11
```

它包含 token 8、9：

```text
k_cache[layer, 11, kv_head, 0, :]  # token 8
k_cache[layer, 11, kv_head, 1, :]  # token 9
```

如果 scheduler 决定把该 request swap out 到 CPU，server 侧可能分配：

```text
cpu_block_table[3, 2] = 25
```

worker 执行：

```text
swap_blocks(src=11, dst=25, is_swap_out=True)
```

copy 后：

```text
k_swap[layer, 25, kv_head, 0, :]  # token 8
k_swap[layer, 25, kv_head, 1, :]  # token 9
```

之后 CPU paged attention 只看 `cpu_block_table[3, 2] = 25`，就能正确读到 token 8、9 的历史 KV。

---

## 10. 从 request 生命周期看 KV cache 修改点

### 10.1 Prefill request 首次进入系统

1. scheduler 从 waiting queue 取 request。
2. request 被分配 `request_id`。
3. `BlockManager.prepare()` 给它的 prompt tokens 分配 GPU logical blocks。
4. worker `set_block_tables()` 更新 `gpu_block_table`。
5. `_preproj()` 调 `store_kvcache()` 把 prompt K/V 写入 `k_cache/v_cache`。
6. 如果是 GPRF，请求后续进入 GPU decode 队列。
7. 如果是 CPRF，请求的 KV 会再 swap out 到 CPU，后续进入 CPU decode 队列。

### 10.2 GPU decode request 每轮生成一个 token

1. `BlockManager.prepare()` 按新的 `seq_len` 检查是否需要新 logical block。
2. 如果当前 token 落入新 block，则分配新的 GPU physical block，并更新 `gpu_block_table`。
3. GPU paged attention kernel：
   - 先把当前 token K/V 写入 GPU cache；
   - 再通过 `gpu_block_table` 读取历史 K/V；
   - 输出 attention result。

### 10.3 CPU decode request 每轮生成一个 token

1. `BlockManager.prepare()` 检查是否需要新的 CPU logical block。
2. 如果需要，分配新的 CPU physical block，并更新 `cpu_block_table`。
3. `_transfer_qkv()` 把当前 token Q/K/V copy 到 CPU pinned buffers。
4. `paged_attention_cpu()`：
   - 把当前 token K/V 写入 CPU cache；
   - 通过 `cpu_block_table` 读取历史 K/V；
   - 输出 attention result 到 `o_cpu`。
5. worker 把 `o_cpu` copy 回 GPU attention output。

### 10.4 Request 结束

request 完成后，server 侧 `BlockManager` 释放对应 blocks：

- GPU blocks 回到 GPU free list；
- CPU blocks 回到 CPU free list；
- `seq_num_blks[seq_id]` 清零。

真实 tensor 内容通常不需要清零，因为后续是否有效完全由 block table 与 allocation metadata 决定。

---

## 11. 常见误解与开发注意事项

### 11.1 CPRF 不是 CPU prefill

CPRF 表示 CPU-destined prefill，不表示 prefill 在 CPU 上计算。

prefill 仍然在 GPU 上执行，区别只是 KV 目的地最终是 CPU decode 路径。

### 11.2 活跃 GPU decode path 是 Triton `paged_attn.py`

当前 `_attention()` 中实际调用的是：

```text
swiftllm/worker/kernels/paged_attn.py::paged_attention()
```

旧的 `csrc/src/attention.cu` 不应当作为当前主路径理解。

### 11.3 活跃 prefill store 是 `swiftllm_c.store_kvcache`

`transformer_layer.py` 中导入的是 C++/CUDA extension：

```text
from swiftllm_c import store_kvcache
```

旧的 Triton `swiftllm/worker/kernels/kvcache_mgmt.py` 不是当前活跃路径。

### 11.4 GPU physical block id 与 CPU physical block id 是不同 namespace

`gpu_block_table[seq_id, block_pos] = 11` 和 `cpu_block_table[seq_id, block_pos] = 11` 不表示同一个物理内存位置。

它们分别表示：

```text
GPU k_cache/v_cache 的 physical block 11
CPU k_swap/v_swap 的 physical block 11
```

不能混用。

### 11.5 `seq_id/request_id` 必须稳定

attention kernel 只知道 `seq_ids` 和 block table。如果 request id 改变但 block table 没有同步迁移，kernel 会读错行。

因此 request 生命周期内 `request_id` 必须稳定对应 block table row。

### 11.6 block table 更新和 physical copy 缺一不可

swap 相关 bug 常见于只做了其中一半：

- 只更新 block table，没有 copy physical KV 内容；
- 只 copy physical KV 内容，没有更新 block table；
- copy 的 physical block id 与 block table 指向的 physical block id 不一致。

正确性依赖 `BlockManager.prepare()` 同时产生正确的 `mappings` 与 `swappings`。

### 11.7 `block_offset` 是物理 block 内 offset，不是全局 token id

例如 token 9 在 `block_size=4` 时：

```text
logical_block_pos = 2
block_offset = 1
```

访问物理 cache 时用的是：

```text
[physical_block, 1, :]
```

不是：

```text
[physical_block, 9, :]
```

---

## 12. 最小 mental model

理解 NEO 的 KV cache，可以抓住三句话：

1. Server 侧 `BlockManager` 维护逻辑 block 到物理 block 的权威映射，但不持有真实 K/V tensor。
2. Worker 侧 `Swapper` 持有真实 GPU/CPU KV cache tensor 与执行时 block table。
3. GPU/CPU paged attention 都遵循同一个公式：

```text
seq_id + token_pos
  -> logical_block_pos = token_pos // block_size
  -> block_offset      = token_pos % block_size
  -> physical_block    = block_table[seq_id, logical_block_pos]
  -> KV tensor[layer, physical_block, kv_head, block_offset, :]
```

GPU 路径把这个公式应用到：

```text
gpu_block_table + k_cache/v_cache
```

CPU 路径把这个公式应用到：

```text
cpu_block_table + k_swap/v_swap
```

只要 block table 与 physical block 内容保持一致，logical sequence 的 KV 可以分散存放在任意 physical blocks 中，paged attention 仍然能正确重建完整上下文。
