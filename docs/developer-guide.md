# NEO 开发者指南

本文档用于集中记录 NEO 项目中关键代码路径的开发者视角解释。后续如果继续分析其他函数或模块，应按章节追加到本文档中，而不是创建分散的说明文件。

## `_preproj` 与 prefill KV cache 写入

本节解释 `swiftllm/worker/layers/transformer_layer.py` 中的 `_preproj()`，重点回答三个问题：

1. `_preproj()` 在 NEO 整体执行链路中做什么；
2. 为什么 `store_kvcache()` 前要调用 `_compute_wait_comm()`；
3. `store_kvcache()` 如何把 prefill 的 K/V 写入 paged KV cache，以及每个参数的含义。

### 先说结论

`_preproj()` 是每个 transformer layer 在进入 attention 前的“前投影”阶段，主要做：

- tensor parallel all-reduce；
- residual + RMSNorm；
- Q/K/V linear projection；
- RoPE；
- 对 prefill request，将新生成的 K/V 写入 paged KV cache。

`store_kvcache()` 前的 `_compute_wait_comm()` 是一个 CUDA stream 同步屏障：它让默认 compute stream 等待共享的 `cpu_communication_stream`。这样可以确保之前异步发起的 swap、GPU/CPU copy 等通信任务完成后，再写入 KV cache，避免写到仍在被 swap 的区域，或在 block/cache 状态尚未可见时继续计算。

当前 `_preproj()` 调用的是 C++/CUDA extension 中的 `swiftllm_c.store_kvcache`，不是 `swiftllm/worker/kernels/kvcache_mgmt.py` 中的 Triton 版本。后者在当前代码中已被注释掉，且 cache layout 与当前 `Swapper` / C++ 路径不一致，不应作为运行时行为的权威参考。

### 1. NEO 中 KV cache 写入的整体上下文

一次模型迭代大致经过以下链路：

```text
Engine.step()
  -> BlockManager.prepare(...)
       - 为本轮 request 分配 GPU/CPU blocks
       - 准备 logical block id -> physical block id 的映射
       - 准备 swap in/out 参数
  -> Executor.do_one_iteration(...)
  -> LlamaModel.do_one_iteration(...)
       - Swapper.set_block_tables(mappings)
       - 如有需要，在 cpu_communication_stream 上发起 swap_blocks(...)
       - _forward_batches(batches)
  -> TransformerLayer.forward / forward_double / forward_first_stage
       - _preproj(...)
       - _attention(...)
       - _swap_out_blocks(...)
       - _postproj(...)
```

关键文件：

- `swiftllm/server/engine.py`：server 侧 iteration 入口；
- `swiftllm/server/block_manager.py`：control plane 的 block 分配、释放、swap 参数准备；
- `swiftllm/worker/model.py`：worker 侧 model forward、CUDA stream 创建、block table 安装、swap 发起；
- `swiftllm/worker/block_swapper.py`：GPU KV cache、CPU swap space、block tables、swap 操作的持有者；
- `swiftllm/worker/layers/transformer_layer.py`：每层 transformer 的 compute、attention、KV cache 写入和异步通信编排。

`BlockManager` 不直接参与模型计算。它在 control plane 决定“某个 sequence 的第几个逻辑 block 对应哪个物理 block”。这些映射通过 `mappings` 传给 worker，worker 再调用 `Swapper.set_block_tables()` 写入真正供 CUDA kernel 使用的 `gpu_block_table` / `cpu_block_table`。

### 2. `SubBatch` 的 runtime 布局

`SubBatch` 是 NEO 调度和 worker forward 之间的混合执行单元。它不是简单的“一个 GPU batch”，而是可以同时包含四类 request：

| 类别 | 字段 | 含义 |
|---|---|---|
| CPU prefill | `cprf_reqs` | prefill 计算仍在 GPU 上做，但新生成的 KV 本轮后会被 swap 到 CPU |
| GPU prefill | `gprf_reqs` | prefill 计算在 GPU 上做，KV 保留在 GPU KV cache |
| GPU decode | `gdec_reqs` | decode attention 使用 GPU paged attention 和 GPU KV cache |
| CPU decode | `cdec_reqs` | decode attention 使用 CPU kernel 和 CPU swap space |

进入 worker forward 前，`SubBatch.set_model_forward_args()` 会固定 request 顺序：

```python
all_reqs = cprf_reqs + gprf_reqs + gdec_reqs + cdec_reqs
num_prefs = num_cprfs + num_gprfs
num_prgds = num_prefs + num_gdecs
```

这几个边界很重要：

| 字段 | 含义 | 被谁使用 |
|---|---|---|
| `num_cprfs` | CPU prefill request 数量 | `store_kvcache()` 用它判断前多少条 prefill 写入 intermediate layer |
| `num_prefs` | prefill request 总数，即 CPRF + GPRF | `_preproj()` 只对前 `num_prefs` 条 request 存 KV |
| `num_prgds` | prefill + GPU decode request 数量 | GPU 侧 `prgd_seq_ids/prgd_seq_lens` 的范围 |
| `sum_pref_toks` | 所有 prefill tokens 数量 | prefill attention 和 K/V flatten 边界 |
| `sum_prgd_toks` | prefill tokens + GPU decode tokens 数量 | GPU decode attention 的切片边界 |
| `max_pref_toks` | 本 sub-batch 中最长 prefill 长度 | `store_kvcache()` kernel launch grid |

`store_kvcache()` 依赖一个关键约定：prefill 序列中前 `num_cprfs` 条一定是 CPRF，后面才是 GPRF。CUDA kernel 用 `batch_pos < num_cprfs` 来决定当前序列写入 `itm_layer` 还是 `gpu_layer`。

### 3. `_preproj()` API 说明

位置：`swiftllm/worker/layers/transformer_layer.py:276-335`

签名：

```python
def _preproj(
    self,
    embeddings: torch.Tensor,
    batch: SubBatch,
    layer_off: int = 0
) -> tuple[torch.Tensor]:
```

#### 3.1 参数

| 参数 | 类型 / 形状 | 含义 |
|---|---|---|
| `embeddings` | `torch.Tensor`，逻辑形状 `[batch.iter_width, hidden_size]` | 当前 layer 输入 hidden states。函数会在其上做 fused residual + RMSNorm，然后投影出 Q/K/V。 |
| `batch` | `SubBatch` | 当前 sub-batch 的 runtime 元数据，提供 request 顺序、token 边界、RoPE position、KV cache 写入所需的 seq ids / lens / block 起点等。 |
| `layer_off` | `int`，默认 `0` | layer 偏移。`0` 表示当前层；pipeline 路径中 `1` 表示提前计算“下一层要用的 QKV”。 |

返回值：

| 返回值 | 形状 | 含义 |
|---|---|---|
| `q` | `[batch.iter_width, local_num_q_heads, head_dim]` | 当前 sub-batch 的 query。 |
| `k` | `[batch.iter_width, local_num_kv_heads, head_dim]` | 当前 sub-batch 的 key。prefill 部分会被写入 KV cache。 |
| `v` | `[batch.iter_width, local_num_kv_heads, head_dim]` | 当前 sub-batch 的 value。prefill 部分会被写入 KV cache。 |

其中 `local_num_q_heads = num_q_heads / world_size`，`local_num_kv_heads = num_kv_heads / world_size`。

#### 3.2 执行步骤

`_preproj()` 的执行顺序如下：

```text
选择权重
  -> tensor parallel all-reduce embeddings
  -> fused_add_rmsnorm_inplace(...)
  -> q_proj / k_proj / v_proj
  -> view 成 [iter_width, heads, head_dim]
  -> rotary_embedding_inplace(q, k, ...)
  -> 如果存在 prefill request：
       _compute_wait_comm()
       store_kvcache(...)
  -> return q, k, v
```

代码中：

```python
weight = self.weight if not layer_off else self.next_layer_weight
```

这行在 pipeline 路径里非常关键。NEO 的 double sub-batch pipeline 会在某些 stage 中提前为下一层计算 Q/K/V。因此当 `layer_off=1` 时：

- Q/K/V 使用 `next_layer_weight`；
- KV cache 写入层号为：

```python
gpu_layer = (self.layer_id + layer_off) % self.model_config.num_layers
```

也就是说，`layer_off=1` 不是“当前层多算一次”，而是“当前 pipeline stage 预先生成下一层要消费的 QKV”。

#### 3.3 为什么只对 prefill 存 KV

`_preproj()` 中有这个判断：

```python
if batch.num_prefs > 0 and self.swapper is not None:
    ...
    store_kvcache(...)
```

这表示 `_preproj()` 只显式存 prefill request 的 K/V。原因是 prefill 会一次生成多个历史 tokens 的 K/V，需要批量写入 paged KV cache。

decode request 的当前 token K/V 不在 `_preproj()` 里通过 `store_kvcache()` 显式写入。GPU decode 路径使用 `paged_attention()`，CPU decode 路径会把 Q/K/V 搬到 CPU buffer 并调用 CPU attention kernel；它们的 KV 使用和更新由对应 attention 路径处理。

### 4. `_compute_wait_comm()`：为什么 `store_kvcache()` 前要等待 communication stream

位置：`swiftllm/worker/layers/transformer_layer.py:209-216`

```python
def _compute_wait_comm(self):
    torch.cuda.default_stream().wait_stream(self.cpu_communication_stream)
```

它的语义是：让默认 CUDA compute stream 等待 `cpu_communication_stream` 上已经排队的任务完成。

注意它只等待“调用此函数之前已经 enqueue 到 communication stream 的任务”，不会等待未来才 enqueue 的任务。

#### 4.1 NEO 中的两个 stream 方向

`LlamaModel` 创建一个共享 stream：

```python
self.cpu_communication_stream = torch.cuda.Stream()
```

然后传给每个 `LlamaTransformerLayer`。每层用两个 helper 表达同步方向：

| helper | 代码语义 | 使用场景 |
|---|---|---|
| `_comm_wait_compute()` | `cpu_communication_stream.wait_stream(default_stream)` | “先算后拷”：例如 Q/K/V 已在 compute stream 上算好，communication stream 再异步 copy 到 CPU。 |
| `_compute_wait_comm()` | `default_stream.wait_stream(cpu_communication_stream)` | “先拷/先 swap 后算”：例如后续 compute 要读写的 cache/buffer 必须等 communication stream 完成。 |

#### 4.2 `_preproj()` 前的 communication 从哪里来

`_preproj()` 前可能已经有多类任务排在 `cpu_communication_stream` 上。

##### 来源 1：iteration 开始时的 full-layer swap

在 `LlamaModel.do_one_iteration()` 中，如果本轮有 conventional swap in/out：

```python
with torch.cuda.stream(self.cpu_communication_stream):
    for layer_id in range(self.model_config.num_layers):
        self.swapper.swap_blocks(*swappings, is_swap_out, layer_id, layer_id)
```

这些 swap 发生在 `_forward_batches()` 之前。它们可能还在异步执行。进入第一层 attention 或 `_preproj()` 写 KV cache 前，需要确保相关 physical block 的内容和状态已经完成迁移。

##### 来源 2：CPRF 的 `_swap_out_blocks()`

`_swap_out_blocks()` 会把 CPU prefill 产生的新 KV blocks 从 GPU swap 到 CPU：

```python
with torch.cuda.stream(self.cpu_communication_stream):
    self.swapper.swap_blocks(
        batch.src_blk_ids,
        batch.dst_blk_ids,
        is_swap_out=True,
        gpu_layer=...,
        cpu_layer=self.layer_id
    )
```

在 pipeline steady stage 中，执行顺序是：

```text
_transfer_qkv(...)
_swap_out_blocks(batches[cur_stage])
_postproj(...)
_preproj(..., layer_off=1)
```

因此 `_preproj(..., layer_off=1)` 前面可能刚刚异步发起了 CPRF swap-out。如果马上 `store_kvcache()`，就可能写到仍在被 swap 的 cache 区域，所以要先 `_compute_wait_comm()`。

##### 来源 3：CPU decode 的 Q/K/V GPU->CPU transfer

CPU decode 前，`_transfer_qkv()` 会：

1. 调用 `_comm_wait_compute()`，确保 Q/K/V 已经在 compute stream 上算完；
2. 在 `cpu_communication_stream` 上把尾部 CPU decode requests 的 Q/K/V copy 到 pinned CPU buffer；
3. 记录 `qkvtr_e`，供 CPU attention 启动前同步。

这类 copy 也和其他 communication 共享同一个 stream，因此会影响 `_compute_wait_comm()` 的等待边界。

##### 来源 4：CPU decode output 的 CPU->GPU 回拷

CPU decode attention 是同步 CPU op，完成后会把输出异步 copy 回 GPU：

```python
with torch.cuda.stream(self.cpu_communication_stream):
    o[-batch.num_cdecs:, :].copy_(oc, non_blocking=True)
```

随后 `_attention()` 会调用 `_compute_wait_comm()`，确保 post-projection 使用 `attn_out_buf` 前，CPU decode 输出已经回到 GPU。

#### 4.3 `store_kvcache()` 前等待的核心目的

对 `_preproj()` 来说，`_compute_wait_comm()` 的目的不是等待 Q/K/V projection 本身。Q/K/V projection 已经在默认 compute stream 上顺序执行，后面的 `store_kvcache()` 同样在默认 stream 上 enqueue，自然有 compute-stream 内部顺序保证。

它真正等待的是另一条 stream 上的异步数据搬运：

```text
cpu_communication_stream:
    previous swap / qkv copy / output copy  ---------------------> done

compute/default stream:
    RMSNorm + QKV + RoPE  -> wait(cpu_communication_stream) -> store_kvcache
```

如果没有这个等待，默认 stream 可能在 communication stream 尚未完成时开始写 KV cache，产生典型的跨 stream data hazard。

### 5. `store_kvcache()` API 说明

当前 `_preproj()` 实际调用的是：

```python
from swiftllm_c import store_kvcache
```

C++ signature 位于 `csrc/src/small_kernels.h`：

```cpp
void store_kvcache(
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor block_table,
    torch::Tensor seq_ids,
    torch::Tensor seq_start_locs,
    torch::Tensor seq_lens,
    const int64_t itm_layer,
    const int64_t gpu_layer,
    const int64_t num_cprfs,
    const int64_t max_pref_len
);
```

CUDA kernel 位于 `csrc/src/small_kernels.cu:204-313`。

#### 5.1 功能

`store_kvcache()` 将 `_preproj()` 生成的 prefill K/V 从连续 tensor 布局写入 paged KV cache。

输入 `k/v` 的逻辑布局是：

```text
[num_prefill_tokens + decode_tokens, local_num_kv_heads, head_dim]
```

但传给 `store_kvcache()` 的 metadata 只覆盖前面的 prefill 部分。因此 kernel 实际只写：

```text
k[:batch.sum_pref_toks]
v[:batch.sum_pref_toks]
```

写入目标 `k_cache/v_cache` 的布局是：

```text
[layer, physical_block_id, local_num_kv_heads, block_size, head_dim]
```

每条 sequence 的第 `block_pos` 个逻辑 block 会通过 `block_table` 映射到一个 physical block id。

#### 5.2 `_preproj()` 调用参数对应关系

`_preproj()` 中的调用是：

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

参数说明：

| 参数 | 来源 | 类型 / 形状 | 含义 |
|---|---|---|---|
| `k` | `_preproj()` 的 K projection 输出 | `float16` CUDA tensor，逻辑形状 `[batch.iter_width, local_num_kv_heads, head_dim]` | 当前 sub-batch 所有 token 的 K。kernel 根据 `seq_start_locs` / `seq_lens` 只读取 prefill 部分。 |
| `v` | `_preproj()` 的 V projection 输出 | 同 `k` | 当前 sub-batch 所有 token 的 V。 |
| `k_cache` | `self.swapper.k_cache` | `float16` CUDA tensor，`[num_layers + extra_layer_for_cprf, num_gpu_blocks, local_num_kv_heads, block_size, head_dim]` | GPU paged KV cache 的 K 存储。 |
| `v_cache` | `self.swapper.v_cache` | 同 `k_cache` | GPU paged KV cache 的 V 存储。 |
| `block_table` | `self.swapper.gpu_block_table` | `int32` CUDA tensor，`[max_seqs_in_block_table, max_blocks_per_seq]` | 将 `(seq_id, logical_block_index)` 映射到 GPU physical block id。 |
| `seq_ids` | `batch.prgd_seq_ids[:batch.num_prefs]` | `int32` CUDA tensor，`[num_prefs]` | 本次要写 KV 的 prefill sequences 的 request id / sequence id。kernel 用它索引 `block_table`。 |
| `seq_start_locs` | `batch.pref_st_locs_we` | `int32` CUDA tensor，长度至少 `num_prefs`，实际构造为 `[0] + prefix_sum(prefill_lens)` | 每条 prefill sequence 在 flatten K/V 中的起始 token offset。 |
| `seq_lens` | `batch.prgd_seq_lens[:batch.num_prefs]` | `int32` CUDA tensor，`[num_prefs]` | 每条 prefill sequence 的长度。 |
| `itm_layer` | `_preproj()` 局部变量 | `int64` | CPRF sequence 的写入 layer。`extra_layer_for_cprf=True` 时为 `num_layers`，否则等于 `gpu_layer`。 |
| `gpu_layer` | `_preproj()` 局部变量 | `int64` | 普通 GPU prefill sequence 的写入 layer，即 `(self.layer_id + layer_off) % num_layers`。 |
| `num_cprfs` | `batch.num_cprfs` | `int64` | prefill 序列中前多少条属于 CPRF。kernel 用 `batch_pos < num_cprfs` 判断写入 `itm_layer` 还是 `gpu_layer`。 |
| `max_pref_len` | `batch.max_pref_toks` | `int64` | 本 sub-batch 最长 prefill 长度。只用于决定 kernel launch grid 的第二维。 |

#### 5.3 Kernel 的索引逻辑

kernel launch 形状：

```cpp
const dim3 grid(num_seqs, (max_pref_len - 1) / block_size + 1);
const int64_t block_dim_x = head_dim;
```

也就是说，一个 CUDA block 负责一个：

```text
(sequence batch_pos, logical block_pos)
```

kernel 内部核心逻辑：

```cpp
const int64_t batch_pos = blockIdx.x;
const int64_t block_pos = blockIdx.y;
const int64_t seq_len = seq_lens[batch_pos];

if (block_size * block_pos >= seq_len) {
    return;
}

const int64_t cur_layer = batch_pos < num_cprfs ? itm_layer : gpu_layer;
const int32_t seq_id = seq_ids[batch_pos];
const int32_t seq_start_loc = seq_start_locs[batch_pos];
const int32_t block_idx = block_table[seq_id * block_table_width + block_pos];
```

含义是：

1. `batch_pos` 选择当前 prefill sequence；
2. `block_pos` 选择该 sequence 的第几个逻辑 KV block；
3. 如果这个逻辑 block 超出 sequence 长度，直接返回；
4. 如果这是 CPRF，即 `batch_pos < num_cprfs`，写入 `itm_layer`；否则写入 `gpu_layer`；
5. 用 `seq_id` 和 `block_pos` 到 `block_table` 中查 physical block id；
6. 把该 logical block 内的 K/V tokens 写入：

```text
k_cache[cur_layer, block_idx, kv_head, token_offset_in_block, head_dim]
v_cache[cur_layer, block_idx, kv_head, token_offset_in_block, head_dim]
```

#### 5.4 `seq_start_locs` 为什么来自 `pref_st_locs_we`

worker 在 `_prepare_inputs()` 中构造：

```python
batch.pref_st_locs_we = torch.tensor(
    [0] + list(itertools.accumulate(batch.seq_lens_list[:batch.num_prefs])),
    dtype=torch.int32,
    device='cuda'
)
```

例如有 3 条 prefill，长度分别为 `[5, 7, 2]`，则：

```text
pref_st_locs_we = [0, 5, 12, 14]
```

`store_kvcache()` 使用前 `num_prefs` 个起点：

```text
sequence 0 starts at k/v offset 0
sequence 1 starts at k/v offset 5
sequence 2 starts at k/v offset 12
```

最后一个元素 `14` 对 `store_kvcache()` 本身不是必须的，但它同时被 flash attention 的 varlen API 当作 cumulative sequence lengths 使用，因此命名里有 `we`，可以理解为 “with end”。

### 6. Swapper、block table 与 `extra_layer_for_cprf`

`Swapper` 位于 `swiftllm/worker/block_swapper.py`。它持有运行时 KV 相关的主要数据结构。

#### 6.1 GPU KV cache

```python
kvcache_shape = (
    model_config.num_layers + engine_config.extra_layer_for_cprf,
    engine_config.num_gpu_blocks,
    num_kv_heads,
    engine_config.block_size,
    model_config.head_dim
)
self.k_cache = torch.zeros(kvcache_shape, dtype=torch.float16, device="cuda")
self.v_cache = torch.zeros(kvcache_shape, dtype=torch.float16, device="cuda")
```

第一维通常是 `num_layers`。如果 `extra_layer_for_cprf=True`，则多出一个 layer，索引为 `num_layers`，专门作为 CPRF 的 intermediate GPU cache layer。

#### 6.2 CPU swap space

```python
kvswap_shape = (
    model_config.num_layers,
    engine_config.num_cpu_blocks,
    num_kv_heads,
    engine_config.block_size,
    model_config.head_dim
)
self.k_swap = torch.zeros(kvswap_shape, dtype=torch.float16, device="cpu", pin_memory=True)
self.v_swap = torch.zeros(kvswap_shape, dtype=torch.float16, device="cpu", pin_memory=True)
```

CPU swap space 没有 extra layer。CPRF 的目标是最终把每个真实 transformer layer 的 KV 放到 CPU 对应 layer 上。

#### 6.3 block table

```python
self.gpu_block_table = torch.zeros(
    (max_seqs_in_block_table, max_blocks_per_seq),
    dtype=torch.int32,
    device="cuda"
)
```

block table 的语义是：

```text
block_table[seq_id, logical_block_index] = physical_block_id
```

在 control plane 中，`DeviceBlockManager.alloc()` 生成 flattened virtual id：

```python
vid = seq_id * block_table_width + logical_block_index
```

worker 通过：

```python
self.gpu_block_table.view(-1)[gpu_vids] = gpu_pids
```

把这些 mapping 写入 GPU block table。

`store_kvcache()` 再用同样的扁平化公式读取：

```cpp
block_idx = block_table[seq_id * block_table_width + block_pos];
```

#### 6.4 `extra_layer_for_cprf` 的意义

CPRF 的语义是：prefill 的计算仍在 GPU 上完成，但生成的 KV 要放到 CPU，供后续 CPU decode 使用。

如果没有 intermediate layer，CPRF 新 KV 可能暂时写在真实 GPU layer 的 cache 区域，再 swap 到 CPU。启用 `extra_layer_for_cprf` 后，GPU KV cache 多出一层：

```text
normal layers: 0 ... num_layers - 1
CPRF intermediate layer: num_layers
```

此时 `_preproj()` 中：

```python
itm_layer = self.model_config.num_layers if self.engine_config.extra_layer_for_cprf else gpu_layer
```

因此：

- CPRF prefill K/V 写入 `itm_layer = num_layers`；
- GPRF prefill K/V 写入真实 `gpu_layer`；
- 后续 `_swap_out_blocks()` 再把 CPRF 的 intermediate layer 数据 swap 到 CPU 的真实 `self.layer_id`。

这也是 `store_kvcache()` 需要同时接收 `itm_layer` 和 `gpu_layer` 的原因。

### 7. 常见坑点与维护注意事项

#### 7.1 不要把 `kvcache_mgmt.py` 当成当前运行路径

`swiftllm/worker/layers/transformer_layer.py` 中当前导入的是：

```python
from swiftllm_c import store_kvcache
```

而下面这行是注释掉的：

```python
# from swiftllm.worker.kernels.kvcache_mgmt import store_kvcache
```

`swiftllm/worker/kernels/kvcache_mgmt.py` 中的 Triton 版本使用的 cache 注释布局是：

```text
[num_blocks, num_layers, num_kv_heads, block_size, head_dim]
```

但当前 `Swapper` 和 C++ kernel 使用的是：

```text
[num_layers, num_blocks, num_kv_heads, block_size, head_dim]
```

因此分析当前行为时，应以 `csrc/src/small_kernels.cu` 为准。

#### 7.2 `num_cprfs` 依赖 request 顺序

CUDA kernel 判断 CPRF 的方式是：

```cpp
cur_layer = batch_pos < num_cprfs ? itm_layer : gpu_layer;
```

它没有检查 request 类型，只相信 batch 中 prefill sequence 的排序。因此 `SubBatch.all_reqs = cprf + gprf + gdec + cdec` 这个顺序不能随意改。

#### 7.3 `max_pref_len` 只决定 launch 范围

`max_pref_len` 不参与每条 sequence 的真实长度判断。真实长度来自：

```cpp
seq_len = seq_lens[batch_pos]
```

`max_pref_len` 必须大于等于本次所有 prefill sequence 的最大长度，否则 grid 的 block 数不够，会漏写后面的 logical blocks。

#### 7.4 C++ kernel 的隐含约束

`csrc/src/small_kernels.cu` 中的 `store_kvcache()` 有一些强约束：

- `head_dim` 必须是 `128`；
- 支持的 local `num_kv_heads` 只有 `1, 2, 4, 8, 16, 32, 40`；
- `k/v/k_cache/v_cache` 按 `half*` 解释，实际应为 `float16`；
- `block_table/seq_ids/seq_start_locs/seq_lens` 按 `int32_t*` 解释，实际应为 `int32`；
- C++ wrapper 没有完整的 dtype / contiguous assert，调用侧需要保证 tensor layout 与 dtype 正确。

#### 7.5 `_compute_wait_comm()` 不是全局同步

`_compute_wait_comm()` 不是 `torch.cuda.synchronize()`。它只建立 default stream 对 `cpu_communication_stream` 的依赖：

```text
default stream waits for work already queued on cpu_communication_stream
```

这比全局同步更轻量，也保留了 NEO 通过 compute / communication overlap 提升性能的设计目标。

### 8. 一句话总结

`_preproj()` 不是单纯的 QKV projection helper。它处在 NEO 的 compute / communication overlap、paged KV cache、CPU/GPU 混合 decode、CPRF swap-out 机制的交界处。`_compute_wait_comm()` 的存在，是为了在尽可能保留异步 overlap 的同时，给 `store_kvcache()` 建立必要的跨 stream 数据依赖；`store_kvcache()` 则负责把 flatten 的 prefill K/V 按 block table 写入正确的 GPU KV cache layer 和 physical block。
