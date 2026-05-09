# NEO GPU -> CPU KV Cache Transfer 开发者说明

本文专门解释 NEO 中 **GPU 将 KV Cache 传递给 CPU** 的路径。这里的“传递”不是抽象的数据结构更新，而是 worker 侧真实地把 `k_cache/v_cache` 中的物理 KV block 通过 `cudaMemcpyAsync(..., cudaMemcpyDeviceToHost, ...)` 拷贝到 CPU pinned memory 中的 `k_swap/v_swap`。

相关代码主要分布在：

- `swiftllm/server/scheduler.py`：决定哪些 request 需要从 GPU 转到 CPU，或哪些 prefill 的 KV 最终要落到 CPU。
- `swiftllm/server/block_manager.py`：把 request 级别的 swap 决策转成物理 block id 与 block table mapping。
- `swiftllm/worker/block_swapper.py`：持有真实 GPU/CPU KV cache tensor，并封装 `swap_blocks()`。
- `swiftllm/worker/model.py`：在 iteration 开头发起常规 swap-out。
- `swiftllm/worker/layers/transformer_layer.py`：在每层内部发起 CPRF 的 KV swap-out，并处理 stream 等待。
- `csrc/src/block_swapping.cpp`：真正执行 GPU->CPU `cudaMemcpyAsync`。
- `pacpu/pacpu.cpp`、`pacpu/core.h`：CPU paged attention 如何在 CPU 侧消费已经收到的 KV cache。

---

## 1. 先回答四个核心问题

### 1.1 GPU 什么时候会将 KV Cache 传递给 CPU？

NEO 中有两类 GPU->CPU KV Cache transfer。

第一类是 **常规 swap-out**：已有 request 正在 GPU decode，但调度器判断 GPU decode 常驻集太大，需要把其中一部分 request 换到 CPU decode 队列。代码在 `swiftllm/server/scheduler.py` 的 `Scheduler._get_next_batch_new()`。当 `budget.overspent` 或 `gpu_block_needed > swap_out_threshold` 时，调度器从 `gpu_decoding_q` 尾部取出 victim：

```python
while budget.overspent or gpu_block_needed > swap_out_threshold:
    victim = self.gpu_decoding_q.pop()
    self.cpu_decoding_q.appendleft(victim)
    swpout_reqs.append(victim)
```

这里的 `swpout_reqs` 会作为 `cur_swap_out` 返回给 engine。之后 `BlockManager.prepare()` 会把这些 request 转成“从哪些 GPU physical block 拷贝到哪些 CPU physical block”。worker 在本轮 forward 前发起实际 GPU->CPU copy。

第二类是 **CPRF / `pref_to_cpu`**：某些新 prefill request 的 prefill 计算仍在 GPU 上做，但它生成出来的 KV cache 不常驻 GPU，而是写到 CPU KV swap space。`scheduler.py` 中有注释明确说明：`pref_to_cpu` 不是在 CPU 中执行 prefill，而是 prefill 计算仍经过 GPU，生成出的 KV 会被 swap 到 CPU。

这类请求在 `SubBatch` 中表现为 `cprf_reqs`。它的 GPU->CPU copy 不在 iteration 开头一次性完成，而是在每个 transformer layer 中，等当前层的 prefill KV 已经写入 GPU cache 后，由 `_swap_out_blocks()` 发起。

### 1.2 GPU 从哪里给出 KV Cache？CPU 接收到哪里？

GPU 侧真实 KV cache 是 `Swapper.k_cache` / `Swapper.v_cache`，创建于 `swiftllm/worker/block_swapper.py::Swapper.__init__()`：

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

其逻辑 shape 是：

```text
[num_layers + extra_layer_for_cprf,
 num_gpu_blocks,
 local_num_kv_heads,
 block_size,
 head_dim]
```

- 第 0 维：layer id。若 `extra_layer_for_cprf=True`，最后额外多出一个 intermediate layer slot，索引通常是 `num_layers`。
- 第 1 维：GPU physical block id。
- 第 2 维：本 tensor-parallel rank 上的 KV head id。
- 第 3 维：block 内 token offset。
- 第 4 维：head dimension。
- dtype/device：`torch.float16`，CUDA。

CPU 侧接收位置是 `Swapper.k_swap` / `Swapper.v_swap`：

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

其逻辑 shape 是：

```text
[num_layers,
 num_cpu_blocks,
 local_num_kv_heads,
 block_size,
 head_dim]
```

- 第 0 维：CPU KV 所属 transformer layer。
- 第 1 维：CPU physical block id。
- 第 2 维：本 rank 的 KV head id。
- 第 3 维：block 内 token offset。
- 第 4 维：head dimension。
- dtype/device：`torch.float16`，CPU pinned memory。

因此，单个 physical KV block 的传输语义可以理解为：

```text
GPU source:
  k_cache[gpu_layer, src_gpu_physical_block, :, :, :]
  v_cache[gpu_layer, src_gpu_physical_block, :, :, :]

CPU destination:
  k_swap[cpu_layer, dst_cpu_physical_block, :, :, :]
  v_swap[cpu_layer, dst_cpu_physical_block, :, :, :]
```

### 1.3 GPU 侧哪些操作需要等待 GPU->CPU transfer 完成？为什么？

不是所有 GPU 操作都会等待 GPU->CPU KV transfer。NEO 使用一个专门的 `cpu_communication_stream` 发起异步 copy，使部分 GPU compute 可以与通信重叠。真正需要等待的是那些可能依赖 KV cache 状态一致性、或可能复用/写入同一批 GPU physical blocks 的边界。

关键等待点包括：

1. `LlamaModel._forward_sequential()` 在进入第一层前执行：

   ```python
   torch.cuda.current_stream().wait_stream(self.cpu_communication_stream)
   ```

   这确保本轮 iteration 开头已经提交到 communication stream 的常规 swap 完成后，再进入 transformer body。

2. pipeline 路径的 `LlamaTransformerLayer.forward_first_stage()` 在第一段 attention 前调用：

   ```python
   self._compute_wait_comm()
   ```

   其注释说明这里必须确保前面的 async swap 已经完成。

3. `LlamaTransformerLayer._preproj()` 在 `store_kvcache()` 前调用：

   ```python
   self._compute_wait_comm()
   store_kvcache(...)
   ```

   这是最关键的 GPU 写入等待点。原因是 `BlockManager.prepare()` 可能已经在控制面释放了某些 GPU physical blocks，并把它们重新分配给新的 prefill。若前一个 GPU->CPU D2H copy 还没完成，新的 `store_kvcache()` 就写入同一个 GPU physical block，会导致 CPU 端收到被覆盖或混杂的 KV。

4. `_attention()` 末尾也调用 `_compute_wait_comm()`，但这里主要是等待 CPU decode output 从 CPU 回拷到 GPU，保证后续 post-projection 能消费正确的 attention output。它和 KV block swap 的等待方向相同，但语义不完全一样。

### 1.4 CPU 侧哪些操作需要等待 CPU receive 完成？为什么？

CPU paged attention 在进入 `torch.ops.pacpu.paged_attention_cpu(...)` 前需要等待。等待点在 `swiftllm/worker/layers/transformer_layer.py::_attention()`：

```python
self.events[cur_stage].qkvtr_e.synchronize()
torch.ops.pacpu.paged_attention_cpu(...)
```

`qkvtr_e` 是在 `_transfer_qkv()` 中记录在 `cpu_communication_stream` 上的 event。它本来表示 CPU decode 当前 token 的 `q_cpu/k_cpu/v_cpu` staging copy 已完成；同时，因为 CUDA stream 内操作是 FIFO 顺序，若在同一个 `cpu_communication_stream` 上更早提交过 GPU->CPU KV swap-out，那么同步 `qkvtr_e` 也会等待这些更早的 D2H copy 完成。

PACPU 自身不做 CUDA 同步。`pacpu/pacpu.cpp::paged_attention_cpu()` 直接取 CPU tensor raw pointers：

```cpp
auto qbatch_p = (data_t*) q.data_ptr<at_data_t>();
auto kbatch_p = (data_t*) k.data_ptr<at_data_t>();
auto vbatch_p = (data_t*) v.data_ptr<at_data_t>();
auto kcache_p = (data_t*) k_cache.data_ptr<at_data_t>();
auto vcache_p = (data_t*) v_cache.data_ptr<at_data_t>();
auto block_table_p = block_table.data_ptr<int32_t>();
```

所以 CPU 侧真正的等待边界在 Python layer。如果不等，PACPU 可能读到尚未完成 D2H copy 的 `q_cpu/k_cpu/v_cpu`，或者读到尚未完整写入 CPU pinned memory 的 `k_swap/v_swap`。

---

## 2. 控制面与数据面的分工

NEO 的 GPU->CPU KV transfer 可以分成 control-plane 与 data-plane 两层。

### 2.1 Server/control-plane：决定“哪些 block 要搬”

server 侧不直接搬 tensor bytes。它负责：

- 哪些 request 要从 GPU decode 变成 CPU decode；
- 哪些 prefill request 的 KV 最终要放到 CPU；
- 每个 request 的逻辑 block 对应哪个物理 block；
- swap 时 source physical block id 和 destination physical block id 是什么。

核心文件是 `swiftllm/server/scheduler.py` 和 `swiftllm/server/block_manager.py`。

`Scheduler` 产出 request 级别的决策：

```text
batches, cur_swap_out, cur_swap_in = scheduler.get_next_batch()
```

其中 `cur_swap_out` 只是 request 列表，不是 block id。随后 `BlockManager.prepare()` 把 request 列表转换为 worker 可执行的参数：

```text
mappings, swappings, is_swap_out = block_manager.prepare(...)
```

- `mappings`：新的 block table 映射，形如 `((gpu_vids, gpu_pids), (cpu_vids, cpu_pids))`。
- `swappings`：常规 swap 的物理 block copy 列表，形如 `(src_block_pids, dst_block_pids)`。
- `is_swap_out`：当前常规 swap 是不是 GPU->CPU。

### 2.2 Worker/data-plane：真正“搬 KV bytes”

worker 侧持有真实 tensor：

- GPU KV：`Swapper.k_cache / v_cache`；
- CPU KV：`Swapper.k_swap / v_swap`；
- GPU/CPU block table：`gpu_block_table / cpu_block_table`；
- CPU decode staging buffers：`q_cpu/k_cpu/v_cpu/o_cpu`。

`LlamaModel.do_one_iteration()` 先更新 worker 的 block table：

```python
self.swapper.set_block_tables(mappings)
```

然后如果 `swappings[0]` 非空，就在 `cpu_communication_stream` 上调用 `Swapper.swap_blocks()`。真正的 D2H copy 在 `csrc/src/block_swapping.cpp` 中执行。

---

## 3. Block table 与 physical block：传输前后什么变了？

GPU->CPU transfer 涉及两个层面的变化：

1. block table 映射变化；
2. physical KV block 内容拷贝。

这两者不要混淆。

对于一个 request，逻辑 block 的位置由 token 位置决定：

```text
logical_block_pos = token_pos // block_size
block_offset      = token_pos % block_size
vid               = seq_id * max_blocks_per_seq + logical_block_pos
```

在 GPU 上时，worker 通过：

```text
gpu_block_table[seq_id, logical_block_pos] = gpu_physical_block
```

找到 GPU physical block，然后访问：

```text
k_cache[layer_id, gpu_physical_block, kv_head, block_offset, :]
v_cache[layer_id, gpu_physical_block, kv_head, block_offset, :]
```

swap-out 到 CPU 后，CPU block table 被更新为：

```text
cpu_block_table[seq_id, logical_block_pos] = cpu_physical_block
```

CPU paged attention 之后访问的是：

```text
k_swap[layer_id, cpu_physical_block, kv_head, block_offset, :]
v_swap[layer_id, cpu_physical_block, kv_head, block_offset, :]
```

注意：block table 只保存映射；真实 K/V 数据内容必须由 `swap_blocks()` 搬到对应 physical block。若只更新 `cpu_block_table` 而没有完成 D2H copy，CPU attention 会按新映射读到错误或未完成的数据。

---

## 4. 场景一：常规 GPU decode swap-out

### 4.1 Scheduler 选择要换出的 request

代码位置：`swiftllm/server/scheduler.py::Scheduler._get_next_batch_new()`。

调度器先统计当前 `gpu_decoding_q` 中 request 需要的 GPU blocks：

```python
gpu_block_needed = sum(self._get_block_needed(req) for req in self.gpu_decoding_q)
```

然后若 batch budget 超限，或 GPU block 需求超过阈值，就做 swap-out：

```python
while budget.overspent or gpu_block_needed > swap_out_threshold:
    victim = self.gpu_decoding_q.pop()
    self.cpu_decoding_q.appendleft(victim)
    swpout_reqs.append(victim)
    gpu_block_needed -= self._get_block_needed(victim)
    budget.add(1)
```

变量含义：

- `gpu_decoding_q`：当前常驻 GPU 继续 decode 的 request 队列。
- `cpu_decoding_q`：KV cache 在 CPU 侧、由 CPU paged attention 继续 decode 的 request 队列。
- `victim`：本轮被从 GPU 常驻集移出去的 request。
- `swpout_reqs`：本轮需要执行 GPU->CPU KV transfer 的 request 列表。
- `swap_out_threshold`：当前允许 GPU 常驻 KV blocks 的上界。

旧路径 `_get_next_batch_old()` 中也有类似逻辑：当 `len(gpu_decoding_q)` 或 `num_decoding_gpu_blocks` 超过上限时，把 victim 加入 `newly_swapped_out`，最终返回给 engine。

### 4.2 BlockManager 生成 source/destination physical block ids

代码位置：`swiftllm/server/block_manager.py`。

`BlockManager.prepare()` 中常规 swap 的入口是：

```python
is_swap_out = bool(cur_swap_out)
sp, dv, dp = self._initiate_swap(cur_swap_out or cur_swap_in, is_swap_out)
mappings[is_swap_out][0].extend(dv)
mappings[is_swap_out][1].extend(dp)
swappings[0].extend(sp)
swappings[1].extend(dp)
```

当 `cur_swap_out` 非空时，`is_swap_out=True`，所以：

- `mappings[1]` 被更新，即 CPU block table 的映射被更新；
- `swappings=(sp, dp)` 表示 source 是 GPU physical block ids，destination 是 CPU physical block ids。

`_initiate_swap()` 的核心逻辑是：

```python
src_block_manager = self.gpu_block_manager if is_swap_out else self.cpu_block_manager
dst_block_manager = self.cpu_block_manager if is_swap_out else self.gpu_block_manager
src_blk_pids = src_block_manager.free(reqs, int(use_itm))
dst_blk_vids, dst_blk_pids = dst_block_manager.alloc(reqs, omit_last=omit_last)
return src_blk_pids, dst_blk_vids, dst_blk_pids
```

对 GPU->CPU swap-out：

- `src_block_manager` 是 GPU block manager；
- `dst_block_manager` 是 CPU block manager；
- `src_blk_pids` 是从 GPU block table 中取出的旧 GPU physical block ids；
- `dst_blk_vids` 是 CPU block table 中要更新的 virtual ids；
- `dst_blk_pids` 是新分配的 CPU physical block ids。

`DeviceBlockManager.alloc()` 中 virtual id 的计算方式是：

```python
new_blk_vids = [seq_ids[i] * self.block_table_width + j + seq_num_blks_list[i]
                for i, n in enumerate(new_num_blks_list)
                for j in range(n)]
```

也就是：

```text
vid = seq_id * max_blocks_per_seq + logical_block_pos
```

### 4.3 Worker 在 iteration 开头发起每层 copy

代码位置：`swiftllm/worker/model.py::LlamaModel.do_one_iteration()`。

worker 收到 `mappings` 后，先更新自己的 runtime block tables：

```python
if self.swapper is not None:
    self.swapper.set_block_tables(mappings)
```

`Swapper.set_block_tables()` 负责把 server 侧算出的 `vid -> pid` 写入真实 worker tensor：

```python
(gpu_vids, gpu_pids), (cpu_vids, cpu_pids) = mappings
if gpu_vids:
    self.gpu_block_table.view(-1)[gpu_vids] = torch.tensor(gpu_pids, dtype=torch.int32, device="cuda")
if cpu_vids:
    self.cpu_block_table.view(-1)[cpu_vids] = torch.tensor(cpu_pids, dtype=torch.int32, device="cpu")
```

随后，如果有常规 swap，worker 在 `cpu_communication_stream` 上对每一层发起 copy：

```python
if swappings[0]:
    with torch.cuda.stream(self.cpu_communication_stream):
        for layer_id in range(self.model_config.num_layers):
            self.swapper.swap_blocks(*swappings, is_swap_out, layer_id, layer_id)
```

这里每一层都使用同一组 physical block ids，但 `gpu_layer` / `cpu_layer` 分别是当前 `layer_id`。这表示：一个 request 的所有历史 KV blocks 都要对每个 transformer layer 搬一次。

---

## 5. 场景二：CPRF / pref_to_cpu swap-out

### 5.1 CPRF 的含义

代码位置：`swiftllm/server/scheduler.py::Scheduler._get_next_batch_new()`。

注释已经说明：

```python
# pref_to_cpu 并不是说在 CPU 中执行 prefill request,
# 而是 prefill 计算本身仍经过 GPU，但其生成出的 KV
# 会在本轮后被 swap 到 CPU
```

所以 CPRF 的准确含义是 **CPU-destined prefill**：

- prompt 的 prefill attention 在 GPU 上执行；
- Q/K/V projection、RoPE、prefill attention 都在 GPU 路径中完成；
- prefill 产生的 KV cache 先写到 GPU cache 或 intermediate GPU cache；
- 随后每一层把对应 KV block 从 GPU copy 到 CPU。

### 5.2 Scheduler 如何产生 CPRF request

`Scheduler._get_next_batch_new()` 扫描 `waiting_q` 时，会把可接纳的新 request 分为 `pref_to_gpu` 与 `pref_to_cpu`：

```python
if not pref_to_cpu and gpu_block_needed + cur_block_needed <= self.num_gpu_blocks:
    gpu_block_needed += cur_block_needed
    pref_to_gpu.append(candidate)
else:
    cpu_block_needed += cur_block_needed
    itm_block_needed += cur_block_needed
    pref_to_cpu.append(candidate)
```

含义是：

- 如果前面还没有 CPU-destined prefill，并且 GPU block 空间足够，就优先放到 `pref_to_gpu`；
- 否则放到 `pref_to_cpu`，并增加 CPU block 需求与 intermediate block 需求。

之后 `_decide_mode_and_gen_batch()` 会把这些 request 放进 `SubBatch`。CPRF request 在 `SubBatch` 中对应 `cprf_reqs`，并且在 `set_model_forward_args()` 后排在 `batch.all_reqs` 的最前面：

```text
all_reqs = cprf_reqs + gprf_reqs + gdec_reqs + cdec_reqs
```

这个顺序很重要，因为 `batch.all_reqs[:batch.num_cprfs]` 就是本 batch 的 CPRF requests。

### 5.3 BlockManager 为 CPRF 准备 source/destination blocks

`BlockManager._alloc_blocks_for_batch()` 会先给 batch 中需要 GPU 执行的部分分配 GPU blocks：

```python
self.gpu_block_manager.alloc(
    batch.all_reqs[:batch.num_prgds],
    split_point=batch.num_cprfs * self.extra_layer_for_cprf,
    omit_last=False
)
```

其中 `batch.num_prgds = num_prefs + num_gdecs`，包含 CPRF、GPRF、GPU decode。若 `extra_layer_for_cprf=True`，CPRF 的 GPU blocks 会被分配到 intermediate split，用于临时存放即将换出到 CPU 的 prefill KV。

随后 `BlockManager.prepare()` 专门处理 CPRF swap：

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

变量含义：

- `sp`：source GPU physical block ids。对 CPRF 来说，这是刚刚为 prefill 临时分配的 GPU/intermediate blocks。
- `dv`：destination CPU virtual block ids，即要写入 `cpu_block_table.view(-1)` 的位置。
- `dp`：destination CPU physical block ids，即 CPU `k_swap/v_swap` 中真实接收数据的 physical blocks。
- `batch.src_blk_ids` / `batch.dst_blk_ids`：后续 worker layer 内 `_swap_out_blocks()` 使用的 physical block id 列表。
- `omit_last=False`：CPRF 是完整 prompt prefill，CPU 侧需要为所有 prompt tokens 的 KV 分配 blocks，不能省略最后一个 token。

### 5.4 Layer 内先写 GPU KV，再换出到 CPU

代码位置：`swiftllm/worker/layers/transformer_layer.py`。

`_preproj()` 中，如果当前 batch 有 prefill request，就调用 `store_kvcache()` 把 K/V 写到 GPU cache：

```python
if batch.num_prefs > 0 and self.swapper is not None:
    gpu_layer = (self.layer_id + layer_off) % self.model_config.num_layers
    itm_layer = self.model_config.num_layers if self.engine_config.extra_layer_for_cprf else gpu_layer
    self._compute_wait_comm()
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

对 CPRF 来说，`store_kvcache()` 会把前 `batch.num_cprfs` 个 prefill request 的 KV 写到 `itm_layer`。如果 `extra_layer_for_cprf=True`，`itm_layer=num_layers`，也就是额外的 intermediate layer slot。

之后 `_swap_out_blocks()` 发起 GPU->CPU KV block copy：

```python
if batch.num_cprfs > 0:
    with torch.cuda.stream(self.cpu_communication_stream):
        self.swapper.swap_blocks(
            batch.src_blk_ids,
            batch.dst_blk_ids,
            is_swap_out=True,
            gpu_layer=self.model_config.num_layers if self.engine_config.extra_layer_for_cprf else self.layer_id,
            cpu_layer=self.layer_id
        )
```

这里的 layer 选择很关键：

- `gpu_layer=num_layers`：当 CPRF 使用 extra layer 时，从 intermediate slot 读出当前层的 CPRF KV；
- `gpu_layer=layer_id`：当不使用 extra layer 时，从当前真实 layer 的 GPU cache 读出；
- `cpu_layer=layer_id`：CPU 侧永远按真实 transformer layer 存入 `k_swap/v_swap`。

---

## 6. 底层 copy：`swiftllm_c.swap_blocks` 如何移动 bytes

Python wrapper 位于 `swiftllm/worker/block_swapper.py::Swapper.swap_blocks()`：

```python
def swap_blocks(
    self,
    src_block_ids: list[int],
    dst_block_ids: list[int],
    is_swap_out: bool,
    gpu_layer: int,
    cpu_layer: int
):
    assert len(src_block_ids) == len(dst_block_ids)
    if not src_block_ids:
        return
    swiftllm_c.swap_blocks(
        src_block_ids,
        dst_block_ids,
        is_swap_out,
        gpu_layer,
        cpu_layer,
        self.k_cache, self.v_cache,
        self.k_swap, self.v_swap
    )
```

它只传 physical block ids，不传 `seq_id`，也不传 logical block position。逻辑到物理的映射已经由 `BlockManager.prepare()` 和 `set_block_tables()` 处理完。

C++ 实现位于 `csrc/src/block_swapping.cpp::swap_blocks()`。

首先它获取当前 CUDA stream：

```cpp
cudaStream_t stream = at::cuda::getCurrentCUDAStream();
```

文件注释说明：C++ 函数不显式接收 stream，而是使用当前 stream，所以 Python 调用前必须通过：

```python
with torch.cuda.stream(self.cpu_communication_stream):
    self.swapper.swap_blocks(...)
```

把当前 stream 切到 communication stream。

然后 C++ 计算 layer 与 block 的 byte offset：

```cpp
size_t gpu_layer_size_in_bytes = getTensorSizeInBytes(k_cache) / k_cache.size(0);
size_t cpu_layer_size_in_bytes = getTensorSizeInBytes(k_swap) / k_swap.size(0);
size_t block_layer_size_in_bytes = gpu_layer_size_in_bytes / k_cache.size(1);

char* k_cache_ptr = (char*)k_cache.data_ptr() + gpu_layer * gpu_layer_size_in_bytes;
char* v_cache_ptr = (char*)v_cache.data_ptr() + gpu_layer * gpu_layer_size_in_bytes;
char* k_swap_ptr = (char*)k_swap.data_ptr() + cpu_layer * cpu_layer_size_in_bytes;
char* v_swap_ptr = (char*)v_swap.data_ptr() + cpu_layer * cpu_layer_size_in_bytes;
```

对 `is_swap_out=True`，它执行 GPU->CPU copy：

```cpp
cudaMemcpyAsync(
    k_swap_ptr + start_target_block_id * block_layer_size_in_bytes,
    k_cache_ptr + start_source_block_id * block_layer_size_in_bytes,
    cur_segment_size_in_bytes,
    cudaMemcpyDeviceToHost,
    stream
);

cudaMemcpyAsync(
    v_swap_ptr + start_target_block_id * block_layer_size_in_bytes,
    v_cache_ptr + start_source_block_id * block_layer_size_in_bytes,
    cur_segment_size_in_bytes,
    cudaMemcpyDeviceToHost,
    stream
);
```

这说明一段 K 和一段 V 分别被拷贝。源地址是 GPU `k_cache/v_cache` 的某个 layer 中从 `start_source_block_id` 开始的连续 physical blocks；目标地址是 CPU `k_swap/v_swap` 的某个 layer 中从 `start_target_block_id` 开始的连续 physical blocks。

C++ 还会合并连续 block：

```cpp
while (end_index < num_blocks_to_swap &&
       source_block_ids[end_index] == source_block_ids[end_index-1]+1 &&
       target_block_ids[end_index] == target_block_ids[end_index-1]+1) {
    end_index++;
}
```

如果 source block ids 与 target block ids 都连续，就把多个 blocks 合并成一次更大的 `cudaMemcpyAsync`，减少 copy 调用次数。

---

## 7. 具体数值例子：一个 logical block 如何从 GPU physical block 变成 CPU physical block

假设：

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

对于 token position `token_pos=9`：

```text
logical_block_pos = 9 // 4 = 2
block_offset      = 9 % 4  = 1
vid               = seq_id * max_blocks_per_seq + logical_block_pos
                  = 3 * 8 + 2 = 26
```

假设 swap-out 前：

```text
gpu_block_table[3, 2] = 5
```

那么 token 9 所在 logical block 在 GPU 上的 physical block 是 5。第 `layer_id=1` 层、第 `kv_head=0` 个 KV head 的 token 9 KV 位于：

```text
k_cache[1, 5, 0, 1, :]
v_cache[1, 5, 0, 1, :]
```

现在调度器决定把 `seq_id=3` 对应 request swap out 到 CPU。`BlockManager._initiate_swap()` 可能为 logical block 2 分配 CPU physical block 20，并生成 CPU mapping：

```text
cpu_block_table[3, 2] = 20
```

同时 `swappings` 中包含：

```text
src_block_ids = [..., 5, ...]
dst_block_ids = [..., 20, ...]
is_swap_out = True
```

worker 对每一层调用：

```python
self.swapper.swap_blocks(src_block_ids, dst_block_ids, True, layer_id, layer_id)
```

当 `layer_id=1` 时，C++ copy 的语义是：

```text
k_cache[1, 5, :, :, :]  ->  k_swap[1, 20, :, :, :]
v_cache[1, 5, :, :, :]  ->  v_swap[1, 20, :, :, :]
```

copy 完成后，CPU paged attention 若要读 token 9 的 K/V，会通过：

```text
logical_block_pos = 2
cpu_physical_block = cpu_block_table[3, 2] = 20
block_offset = 1
```

访问：

```text
k_swap[1, 20, kv_head, 1, :]
v_swap[1, 20, kv_head, 1, :]
```

这就是“逻辑 KV Cache 通过 block table 映射到物理 CPU KV Cache”的过程。

---

## 8. GPU 侧等待：哪些地方等，哪些地方不等

### 8.1 两个 stream helper 的含义

`swiftllm/worker/layers/transformer_layer.py` 中有两个 helper：

```python
def _comm_wait_compute(self):
    self.cpu_communication_stream.wait_stream(torch.cuda.default_stream())

def _compute_wait_comm(self):
    torch.cuda.default_stream().wait_stream(self.cpu_communication_stream)
```

它们方向相反：

- `_comm_wait_compute()`：communication stream 等 default stream。用于“先算后拷”。例如 Q/K/V projection 在 default stream 上算完后，communication stream 才能把 Q/K/V 拷到 CPU staging buffer。
- `_compute_wait_comm()`：default stream 等 communication stream。用于“先拷后算/写”。例如后续 GPU kernel 要读写与 communication stream copy 有关的数据时，需要先等 copy 完成。

### 8.2 常规 swap-out 后，forward 开始前会等

常规 swap-out 在 `LlamaModel.do_one_iteration()` 中被提交到 `cpu_communication_stream`。随后 `_forward_batches()` 进入 transformer body。

在 sequential 路径，`_forward_sequential()` 一开始就等待：

```python
torch.cuda.current_stream().wait_stream(self.cpu_communication_stream)
```

原因是常规 swap 不只是 GPU->CPU，也可能在其他轮次包含 CPU->GPU swap-in；统一等待可以保证 forward 看到的 block table 与 physical cache 内容一致。对 GPU->CPU swap-out 来说，这个等待还可以避免刚被释放/换出的 GPU blocks 在 copy 尚未完成时被后续 GPU attention/store 重用。

### 8.3 Pipeline first stage 前会等

在 pipeline 路径中，`forward_first_stage()` 先做 batch0 的 preproj，然后在第一段 attention 前调用：

```python
self._compute_wait_comm()
```

代码注释写明：

```python
# Here we must make sure all swaps are done before the first attention
```

这保证前面异步发起的 swaps 在第一段 attention 开始前完成，避免 attention 读取的 KV cache 与 block table 映射不一致。

### 8.4 `store_kvcache()` 前会等，这是避免覆盖 still-copying GPU block 的关键

`_preproj()` 中，prefill KV 写入 GPU cache 前有：

```python
self._compute_wait_comm()
store_kvcache(...)
```

这里的等待尤其重要。因为 server 侧的 `BlockManager._initiate_swap()` 对 swap-out request 调用了 GPU block manager 的 `free()`，从控制面看这些 GPU physical blocks 已经可以重新分配。如果 worker 上的 D2H copy 还没完成，而新的 prefill `store_kvcache()` 又写入了同一个 physical block，CPU 侧最终收到的可能不是被换出 request 的旧 KV，而是新 request 的 KV 或混合数据。

所以 `_preproj()` 的等待本质上是在保护 **GPU physical block reuse** 的正确性。

### 8.5 CPRF swap-out 之后，不是所有 GPU work 都立即等

CPRF 的 `_swap_out_blocks()` 在 communication stream 上异步发起：

```python
with torch.cuda.stream(self.cpu_communication_stream):
    self.swapper.swap_blocks(...)
```

在 sequential forward 中，它位于 `_attention()` 之后、`_postproj()` 之前：

```python
self._attention(q, k, v, batch)
self._swap_out_blocks(batch)
embeddings = self._postproj(batch)
```

`_postproj()` / FFN 通常不读写被搬走的 KV cache block，因此它不必为了 CPRF D2H copy 立即同步阻塞。NEO 允许这些计算与 communication stream 上的 D2H copy 重叠。

但后续如果又要写 KV cache，`_preproj()` 会通过 `_compute_wait_comm()` 等待，避免写到仍在 copy 的区域。

---

## 9. CPU 侧等待：PACPU 前必须等，PACPU 内部不等

### 9.1 CPU decode 的输入包括两类 CPU 数据

CPU decode attention 使用：

1. 当前 token 的 Q/K/V staging buffers：
   - `q_cpu`
   - `k_cpu`
   - `v_cpu`
2. 历史 KV cache：
   - `k_swap`
   - `v_swap`
3. CPU block table：
   - `cpu_block_table`
4. 输出 buffer：
   - `o_cpu`

这些参数在 `_attention()` 中传给 PACPU：

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

### 9.2 `_transfer_qkv()` 记录 CPU 输入 ready event

CPU decode 当前 token 的 Q/K/V 是在 `_transfer_qkv()` 中从 GPU copy 到 CPU pinned buffers：

```python
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

这里先 `_comm_wait_compute()`，保证 communication stream 等 default stream 上的 Q/K/V projection 完成。随后 Q/K/V D2H copy 和 event record 都排入同一个 `cpu_communication_stream`。

### 9.3 `_attention()` 在 PACPU 前同步 event

CPU op 启动前：

```python
self.events[cur_stage].qkvtr_e.synchronize()
events.pf_time("lnch_m")
events.pf_time("cdec_s")
torch.ops.pacpu.paged_attention_cpu(...)
```

`qkvtr_e.synchronize()` 是 host-side 等待。它保证 CPU 线程进入 PACPU 前，`q_cpu/k_cpu/v_cpu` 已经 ready。

更重要的是，`qkvtr_e` 在 `cpu_communication_stream` 上记录。CUDA stream 是 FIFO 的：同一个 stream 上早于 event 的 D2H copy 必须先完成，event 才会完成。因此，如果此前同一 stream 上排了 KV block swap-out，`qkvtr_e.synchronize()` 也会间接等待这些更早的 KV receive 完成。

### 9.4 PACPU 本身直接读 CPU pointer，不做 CUDA 同步

`pacpu/pacpu.cpp::paged_attention_cpu()` 读取 tensor shape 后，直接获取 raw pointers：

```cpp
auto qbatch_p = (data_t*) q.data_ptr<at_data_t>();
auto kbatch_p = (data_t*) k.data_ptr<at_data_t>();
auto vbatch_p = (data_t*) v.data_ptr<at_data_t>();
auto obatch_p = o.data_ptr<otpt_t>();
auto kcache_p = (data_t*) k_cache.data_ptr<at_data_t>();
auto vcache_p = (data_t*) v_cache.data_ptr<at_data_t>();
auto block_table_p = block_table.data_ptr<int32_t>();
```

然后调用 CPU/ISPC attention 实现。这里没有 `cudaStreamSynchronize()`、`cudaEventSynchronize()` 或 `cudaDeviceSynchronize()`。

`pacpu/core.h::store_kv()` 会把当前 decode token 的 K/V 写入 CPU KV cache：

```cpp
int block_pos = (seq_len - 1) / BLOCK_SIZE;
int block_id = block_table[block_pos];
int block_off = (seq_len - 1) % BLOCK_SIZE;
int64_t cache_off = (1ll * cur_layer * num_blocks + block_id) * BLOCK_NELEM + block_off * HEAD_DIM;
```

`qk_product()` 和 `av_product()` 再通过 `block_table[j / BLOCK_SIZE]` 读取历史 K/V blocks：

```cpp
auto kp = k_cache + (1ll * cur_layer * num_blocks + block_table[j / BLOCK_SIZE]) * BLOCK_NELEM;
auto vjp = v_cache + (1ll * cur_layer * num_blocks + block_table[j / BLOCK_SIZE]) * BLOCK_NELEM;
```

因此 PACPU 假设调用者已经保证 CPU memory 中的数据完整可读。这个保证来自 Python 层的 event synchronize 和 stream FIFO 顺序。

---

## 10. KV block swap 与 Q/K/V staging copy 不要混淆

NEO 中至少有两条 GPU->CPU 数据路径：

### 10.1 持久 KV block swap-out

路径是：

```text
Swapper.swap_blocks()
  -> swiftllm_c.swap_blocks()
  -> csrc/src/block_swapping.cpp
  -> cudaMemcpyAsync(DeviceToHost)
```

它移动的是历史 KV cache 的 physical blocks：

```text
k_cache/v_cache[gpu_layer, src_gpu_block, :, :, :]
  -> k_swap/v_swap[cpu_layer, dst_cpu_block, :, :, :]
```

这是本文讨论的核心。

### 10.2 CPU decode 当前 token Q/K/V staging copy

路径是：

```text
LlamaTransformerLayer._transfer_qkv()
  -> q_cpu/k_cpu/v_cpu.copy_(q/k/v, non_blocking=True)
```

它移动的是当前 layer 当前 token 的 Q/K/V，不是持久历史 KV blocks：

```text
q/k/v[-num_cdecs:] on GPU
  -> q_cpu/k_cpu/v_cpu on CPU pinned memory
```

PACPU 需要两者都 ready：

- 历史 KV 在 `k_swap/v_swap`；
- 当前 token Q/K/V 在 `q_cpu/k_cpu/v_cpu`。

`qkvtr_e.synchronize()` 名义上是等待 Q/K/V staging copy，但由于同 stream FIFO，也会等待它之前排队的 KV block copy。

---

## 11. 常见误解与开发注意事项

1. **CPRF 不是 CPU 上做 prefill。**

   CPRF 的 prefill compute 仍在 GPU 上，CPU 只是最终 KV cache 的存放位置。

2. **`swap_blocks` 的 block id 是 physical block id。**

   它不是 `seq_id`，也不是 logical block position。`seq_id/logical_block -> physical_block` 的转换由 `BlockManager` 和 block table 完成。

3. **GPU physical block id 与 CPU physical block id 不能混用。**

   GPU block ids 索引 `k_cache/v_cache` 的第 1 维；CPU block ids 索引 `k_swap/v_swap` 的第 1 维。它们来自不同 `DeviceBlockManager`。

4. **block table 更新不等于数据已经 copy 完。**

   `set_block_tables()` 只是把 `vid -> pid` 写入 worker block table。真实 KV bytes 需要 `cudaMemcpyAsync` 完成后才在 CPU `k_swap/v_swap` 中可读。

5. **PACPU 不主动等待 CUDA copy。**

   `paged_attention_cpu()` 直接读 CPU raw pointer。调用它前必须通过 Python 层 event/stream 同步保证 CPU data ready。

6. **不是所有 GPU work 都等待 CPRF swap-out。**

   NEO 会尽量让 postproj/MLP 等不依赖 KV cache 的 compute 与 D2H copy 重叠。真正需要等待的是 attention/cache 写入/physical block 复用边界。

7. **`extra_layer_for_cprf` 会改变 CPRF 的 GPU source layer。**

   若开启，CPRF prefill KV 先写到 `k_cache/v_cache[num_layers, ...]` 这个额外 layer slot，再被 copy 到 CPU 的真实 `cpu_layer=layer_id`。这能避免 CPRF intermediate KV 与正常 GPU-resident KV 混在同一 layer 空间里。

---

## 12. 最小 mental model

可以把 NEO 的 GPU->CPU KV transfer 理解成下面这条链：

```text
Scheduler:
  request 是否需要去 CPU？
    - 常规 decode swap-out
    - CPRF / pref_to_cpu

BlockManager:
  request 的 logical blocks 对应哪些 physical blocks？
  source GPU physical block ids 是哪些？
  destination CPU physical block ids 是哪些？
  CPU block table 应该更新哪些 vid -> pid？

Worker Swapper:
  set_block_tables(mappings)
  swap_blocks(src_gpu_pids, dst_cpu_pids, is_swap_out=True, gpu_layer, cpu_layer)

C++ extension:
  cudaMemcpyAsync(
    k_cache/v_cache[gpu_layer, src_gpu_pid, :, :, :]
      ->
    k_swap/v_swap[cpu_layer, dst_cpu_pid, :, :, :],
    cudaMemcpyDeviceToHost,
    cpu_communication_stream
  )

Synchronization:
  GPU cache 写入/attention 边界通过 stream wait 保证不与未完成 copy 冲突；
  CPU paged attention 前通过 qkvtr_e.synchronize() 保证 CPU buffers 可读；
  PACPU 自身只读 CPU pointer，不做 CUDA 同步。
```

只要记住：**block table 决定读哪里，`swap_blocks()` 决定数据是否真的搬过去，stream/event 决定什么时候可以安全读写**，NEO 的 GPU->CPU KV Cache transfer 就比较容易理解。
