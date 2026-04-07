# SwiftLLM 中 swap to CPU 路径调研

## 1. 先说结论

先直接回答你的两个核心判断。

### 1.1 当 `num_cpu_blocks` 过小，甚至 `num_cpu_blocks == 0` 时，scheduler 仍然可能生成 `swap to cpu`

是的，**你的理解是对的**。

当前 `scheduler` 在决定是否执行 `swap-out` 时，只看：

- 当前 `running_q` 的请求数是否超过 `max_batch_size`
- 当前 decoding 请求总共占用的 GPU blocks 是否超过 `num_gpu_blocks`

它**不看 CPU 侧是否还有空余 blocks 可接收这些被换出的序列**。

也就是说，哪怕：

- `num_cpu_blocks` 很小
- `num_cpu_blocks == 0`

只要 GPU 侧超限，scheduler 依然会把某些请求放进 `newly_swapped_out`，随后 engine 继续调用 worker 去执行真正的 `swap out`。

---

### 1.2 `num_cpu_blocks` 不够时，不会在 scheduler 阶段报错，而是执行到 worker 时才报错

是的，**这部分理解也对**。

而且严格说，实际情况比“只是晚一点报错”更严重一些：

- scheduler 不做 CPU 容量校验；
- 真正的 CPU block 校验发生在 worker 的 `BlockManager._allocate_blocks()`；
- 如果 CPU blocks 不够，会在这里抛 `RuntimeError`；
- 但在抛错之前，`_swap()` 已经先把源端 GPU blocks 做了 gather 并逻辑释放；
- 所以这不是一个“无副作用失败”，而是一个**先改状态、再失败**的路径；
- 换句话说，当前 `swap-out` **不是原子操作**。

因此更准确的结论是：

> 当前实现里，`num_cpu_blocks` 不足不会在调度层被阻止，而是会让系统继续走到 worker 执行阶段，在 CPU block 分配时抛异常；并且该异常发生前，GPU block manager 的状态可能已经被修改。

---

## 2. 本次调研覆盖的代码范围

本次主要阅读了下面这些文件：

- `swiftllm/server/scheduler.py`
- `swiftllm/server/engine.py`
- `swiftllm/server/llm_engine.py`
- `swiftllm/worker/model_runner.py`
- `swiftllm/worker/model_instance.py`
- `swiftllm/worker/block_manager.py`
- `csrc/src/block_swapping.cpp`
- `swiftllm/engine_config.py`

我的关注点主要是 5 个：

1. scheduler 在什么条件下决定 `swap out to cpu`
2. `num_cpu_blocks` 是否在调度阶段被纳入判断
3. 真正执行 `swap out` 的调用链是什么
4. CPU blocks 不足时，异常到底在哪里触发
5. 失败时系统状态是否已经部分被修改

---

## 3. swap to CPU 的整体调用链

先把整条链路串起来。

### 3.1 scheduler 决定是否 swap-out

入口在：

- `swiftllm/server/engine.py:121-133`

`Engine._main_event_loop()` 每轮都会调用：

- `self.scheduler.get_next_batch()`：`swiftllm/server/engine.py:122`

返回值是：

- `cur_batch`
- `cur_swap_in`
- `cur_swap_out`

如果 `cur_swap_out` 非空，就会调用：

- `self.model.swap_out_seqs(...)`：`swiftllm/server/engine.py:128-133`

也就是说：

- scheduler 只负责“决定哪些请求要被换出”
- engine 负责把这个决定交给 model/worker 去真正执行

---

### 3.2 server 层继续往下转发

后续调用链依次是：

1. `LLMEngine.swap_out_seqs()`
   - `swiftllm/server/llm_engine.py:111-112`
2. `ModelRunner.swap_out_seqs()`
   - `swiftllm/worker/model_runner.py:132-133`
3. `ModelInstance.swap_out_seqs()`
   - `swiftllm/worker/model_instance.py:365-372`
4. `ModelInstance._swap(..., False)`
   - `swiftllm/worker/model_instance.py:334-352`

因此，真正的执行核心在：

- `swiftllm/worker/model_instance.py:_swap`

---

### 3.3 最底层 C++ 只是搬运数据

在 `_swap()` 里最后会调用：

- `swiftllm_c.swap_blocks(...)`：`swiftllm/worker/model_instance.py:345-352`

对应实现是：

- `csrc/src/block_swapping.cpp:22-85`

这个 C++ 函数本质上只做一件事：

- 根据 source block ids 和 target block ids
- 用 `cudaMemcpyAsync` 把 K/V cache 在 GPU 和 CPU 之间拷贝

所以它的职责是：

- **搬运数据**

而不是：

- 决定是否该 swap
- 检查 CPU 容量是否足够
- 做事务性保护

这一点后面会很关键。

---

## 4. scheduler 到底在什么条件下决定 swap to CPU

关键逻辑在：

- `swiftllm/server/scheduler.py:68-129`

核心代码是：

- `self.num_decoding_gpu_blocks = sum(self._get_block_needed(req) for req in self.running_q)`：`swiftllm/server/scheduler.py:103`
- `while len(self.running_q) > self.engine_config.max_batch_size or self.num_decoding_gpu_blocks > self.num_gpu_blocks:`：`swiftllm/server/scheduler.py:105-106`
- `victim = self.running_q.pop()`：`swiftllm/server/scheduler.py:108`
- `newly_swapped_out.append(victim)`：`swiftllm/server/scheduler.py:110`

这段逻辑表明：

只要满足下面任一条件，scheduler 就会把 `running_q` 尾部请求换出：

1. `len(self.running_q) > self.engine_config.max_batch_size`
2. `self.num_decoding_gpu_blocks > self.num_gpu_blocks`

其中每个请求需要多少 block，由：

- `_get_block_needed()`：`swiftllm/server/scheduler.py:56-60`

计算：

- `cdiv(request.prompt_len + request.get_cur_output_len(), block_size)`

也就是说，随着输出 token 增长，即便没有新请求进来，已有请求也可能因为累计长度变长而导致总 block 需求超过 GPU 容量，进而触发 swap-out。

---

## 5. scheduler 是否考虑了 CPU blocks 的剩余容量

这里是这次调研最关键的一点之一。

在 `Scheduler.__init__` 里，确实有这样一行：

- `self.num_free_cpu_blocks = engine_config.num_cpu_blocks`：`swiftllm/server/scheduler.py:52`

乍看之下会让人以为 scheduler 可能会用这个值控制 swap-out。

但实际往后看，会发现：

- 这个变量只是初始化了；
- 后面没有参与任何判断；
- 也没有在 swap-in / swap-out 后更新；
- `get_next_batch()` 的所有决策条件都只围绕 `running_q`、`max_batch_size`、`num_decoding_gpu_blocks`、`num_gpu_blocks`。

因此可以明确说：

> 当前 scheduler 完全没有把 CPU blocks 的可用容量纳入 swap-out 决策。

换句话说，当前 scheduler 判断的是：

- “GPU 侧需不需要腾地方”

而不是：

- “这次 swap-out 端到端是否真的可执行”

这两者不是一回事。

因此：

- scheduler 可以生成“从 GPU 视角看合理，但从 CPU 容量视角看不可执行”的 swap-out。

---

## 6. `num_cpu_blocks == 0` 时为什么启动阶段不会报错

这一点也很重要，因为它解释了为什么问题会被拖到运行时才暴露。

### 6.1 配置层没有限制 `num_cpu_blocks > 0`

在：

- `swiftllm/engine_config.py:61-66`

CLI 参数只是这样定义的：

- `--num-cpu-blocks`
- `type=int`
- `default=2048`

没有任何 `> 0` 的约束。

对应的 `set_engine_config()`：

- `swiftllm/engine_config.py:100-136`

也是直接把 `num_cpu_blocks` 接收进去，没有额外校验。

---

### 6.2 engine 初始化时只检查了 GPU blocks

在：

- `swiftllm/server/engine.py:52-54`

只做了：

- `assert model_config.num_gpu_blocks > 0`

并没有：

- `assert num_cpu_blocks > 0`

所以 `num_cpu_blocks = 0` 时，引擎初始化本身不会报错。

---

### 6.3 worker 初始化时会正常创建 0 长度 CPU swap 空间

在：

- `swiftllm/worker/model_instance.py:147-156`

CPU swap tensor 的形状是：

- 第一维 = `self.engine_config.num_cpu_blocks`

所以当 `num_cpu_blocks == 0` 时，会创建：

- `k_swap.shape[0] == 0`
- `v_swap.shape[0] == 0`

同时 CPU block manager 也会按 0 个 blocks 初始化：

- `swiftllm/worker/model_instance.py:166-172`

而 `BlockManager.__init__()` 中：

- `self.num_free_blocks = num_blocks`：`swiftllm/worker/block_manager.py:18-22`
- `self.is_block_free = torch.ones((num_blocks,), ...)`：`swiftllm/worker/block_manager.py:37-41`

于是：

- CPU block manager 的总块数是 0
- 空闲块数也是 0
- `is_block_free` 是一个长度为 0 的张量

这一切从 Python/Torch 角度看都是合法的。

所以：

> `num_cpu_blocks == 0` 在当前实现中不是“启动即失败”，而是“允许启动，但第一次真的需要 swap-out 时才失败”。

---

## 7. CPU blocks 的真实语义是什么

这个问题不理清的话，容易误判行为。

当前实现里，所谓 “CPU blocks” 不是运行时一块块临时 malloc 的内存，而是：

- 启动时预分配好的 CPU KV swap tensor 的逻辑槽位。

具体来看。

### 7.1 `k_swap` / `v_swap` 是启动时一次性分配的

在：

- `swiftllm/worker/model_instance.py:147-156`

创建了：

- `self.k_swap = torch.zeros(kvswap_shape, dtype=torch.float16, device="cpu")`
- `self.v_swap = torch.zeros(kvswap_shape, dtype=torch.float16, device="cpu")`

其中 `kvswap_shape[0] = num_cpu_blocks`。

这说明 CPU swap 空间是固定容量。

---

### 7.2 `cpu_block_manager` 管的是逻辑 block id

在：

- `swiftllm/worker/block_manager.py`

`BlockManager` 管理的是：

- `num_seq_allocated_blocks`
- `block_table`
- `is_block_free`

它不是直接管理原始 CPU 指针，而是管理：

- 哪个 seq 拿到了哪些 block id
- 哪些 block id 当前空闲

真正的数据位置由：

- `k_swap[block_id]`
- `v_swap[block_id]`

来表示。

---

### 7.3 C++ 层通过 block id 计算偏移

在：

- `csrc/src/block_swapping.cpp:33`

先算出每个 block 对应多少字节：

- `block_size_in_bytes = getTensorSizeInBytes(k_cache) / k_cache.size(0)`

之后在 swap-out 时：

- 写入地址 = `k_swap.data_ptr() + start_target_block_id * block_size_in_bytes`
- 读取地址 = `k_cache.data_ptr() + start_source_block_id * block_size_in_bytes`

对应代码：

- `csrc/src/block_swapping.cpp:65-80`

这说明：

> CPU block id 最终就是 `k_swap/v_swap` 第 0 维上的槽位编号。

因此 `num_cpu_blocks` 不是一个“建议值”，而是实际物理容量上限。

---

## 8. 真正的容量检查发生在哪里

真正的检查不在 scheduler，也不在 C++，而在 worker 里的 `BlockManager`。

### 8.1 `_swap()` 中先确定源和目标 manager

在 `swap-out` 情况下：

- `src_block_manager = self.gpu_block_manager`：`swiftllm/worker/model_instance.py:339`
- `dst_block_manager = self.cpu_block_manager`：`swiftllm/worker/model_instance.py:340`

然后：

- 先根据源端已分配 blocks 计算目标需要的长度：`swiftllm/worker/model_instance.py:341-342`
- 再向目标 block manager 申请对应 blocks：`swiftllm/worker/model_instance.py:344`

---

### 8.2 申请目标 blocks 时会进入 `_allocate_blocks()`

调用链是：

- `dst_block_manager.allocate_blocks_for_seqs(...)`：`swiftllm/worker/model_instance.py:344`
- `BlockManager.allocate_blocks_for_seqs(...)`：`swiftllm/worker/block_manager.py:62-80`
- `BlockManager._allocate_blocks(...)`：`swiftllm/worker/block_manager.py:43-53`

关键判断是：

- `if num_blocks > self.num_free_blocks:`：`swiftllm/worker/block_manager.py:48`
- `raise RuntimeError(...)`：`swiftllm/worker/block_manager.py:49`

异常文本为：

- `No enough free blocks available on {self.device_name} (...)`

因此当 CPU blocks 不足时：

- 报错位置是 Python 层 `BlockManager._allocate_blocks()`
- 报错类型是 `RuntimeError`

---

## 9. 为什么说当前失败路径比“只是晚报错”更严重

这部分是我觉得你最值得关注的点。

问题不只是：

- scheduler 不提前检查 CPU 容量
- worker 晚一点才抛错

而是：

- `_swap()` 的执行顺序本身会让失败带副作用。

### 9.1 `_swap()` 当前顺序

在：

- `swiftllm/worker/model_instance.py:341-345`

顺序是：

1. `seq_lengths = src_block_manager.get_num_allocated_blocks(seq_ids) * block_size`
2. `src_block_ids = src_block_manager.gather_allocated_blocks_and_free(seq_ids)`
3. `dst_block_ids = dst_block_manager.allocate_blocks_for_seqs(seq_ids, seq_lengths)`
4. `swiftllm_c.swap_blocks(...)`

也就是说：

- 先释放源 blocks
- 再申请目标 blocks
- 最后才真正复制数据

---

### 9.2 `gather_allocated_blocks_and_free()` 会立刻把源 blocks 标成 free

在：

- `swiftllm/worker/block_manager.py:89-97`

这一步会：

- gather 出当前 seq 占用的 block ids
- 然后 `self.num_free_blocks += len(gathered_block_ids)`
- 并且在 Triton/kernel 层把 block table 和 free 状态一起更新

这意味着，在第 2 步之后：

- GPU block manager 已经认为这些 blocks 空出来了
- seq 对这些 GPU blocks 的映射也已经被移除

---

### 9.3 如果第 3 步失败，会出现什么情况

如果 CPU blocks 不够，那么：

- 第 3 步 `allocate_blocks_for_seqs()` 会抛 `RuntimeError`
- 第 4 步 `swiftllm_c.swap_blocks(...)` 根本不会执行

这时系统状态变成：

- 源 GPU blocks 已经被逻辑释放
- 目标 CPU blocks 没申请成功
- 数据也还没从 GPU 复制到 CPU

也就是说，这不是一种“状态保持不变”的失败。

更准确地说，它是：

> 在逻辑状态已经部分改变之后，中途失败。

这意味着当前 `swap-out` **不具备原子性**，也没有回滚语义。

如果上层继续运行或试图恢复，这里的状态一致性会变得很脆弱。

所以相比“最后抛个异常”，更大的问题其实是：

- 异常之前，系统的 block 管理状态已经可能不一致。

---

## 10. `block_swapping.cpp` 在这里承担什么职责

这一层的职责要分清，不然容易误以为 C++ 会帮忙兜底。

在：

- `csrc/src/block_swapping.cpp:22-85`

`swap_blocks()` 做的事情是：

1. 读取 `source_block_ids` 和 `target_block_ids`
2. 尝试把连续 block 合并成 segment
3. 对 K/V cache 分别做 `cudaMemcpyAsync`

对于 swap-out：

- `cudaMemcpyDeviceToHost`：`csrc/src/block_swapping.cpp:65-80`

对于 swap-in：

- `cudaMemcpyHostToDevice`：`csrc/src/block_swapping.cpp:49-64`

它没有做的事情包括：

- 不检查 CPU block 容量
- 不检查 target ids 是否越界
- 不检查 source/target id 数量是否匹配
- 不检查 `cudaMemcpyAsync` 的返回值

所以它不是：

- 容量检查层
- 安全兜底层
- 事务边界层

它只是一个“按已给定 block ids 做数据搬运”的执行器。

因此当前系统的正确性，其实主要依赖：

- scheduler 不要下发不可执行的 swap
- block manager 保证 block id 合法
- `_swap()` 顺序本身不要把系统带进中间态

而不是依赖 C++ 层兜底。

---

## 11. 对你问题的逐条回答

### 问题 1：当要 swap to cpu 时，如果 `num_cpu_blocks` 过小，或者 `num_cpu_blocks == 0`，scheduler 是否还会执行 swap to cpu？

**回答：会。**

原因是：

- scheduler 的 swap-out 条件只看 GPU 侧和 batch 限制：`swiftllm/server/scheduler.py:103-110`
- `self.num_free_cpu_blocks` 虽然在 `swiftllm/server/scheduler.py:52` 初始化了，但并没有参与判断

所以只要：

- `len(running_q) > max_batch_size`
- 或 `num_decoding_gpu_blocks > num_gpu_blocks`

scheduler 就仍然可能把请求加入 `newly_swapped_out`。

---

### 问题 2：当 `num_cpu_blocks` 不够时，是否不会提前报错，而是直接运行到执行出错为止？

**回答：是的，基本就是这样。**

更准确一点说：

- 不会在配置层报错
- 不会在 engine 初始化时报错
- 不会在 scheduler 决策时报错
- 会在 worker 的 CPU block 分配阶段报错

真正失败点是：

- `swiftllm/worker/block_manager.py:48-49`

也就是 `BlockManager._allocate_blocks()` 中：

- `if num_blocks > self.num_free_blocks: raise RuntimeError(...)`

---

### 问题 3：是否“当 CPU 不够时也不会输出错误，而是直接让其运行到执行出现错误为止”？

**回答：是的，而且还附带一个更隐蔽的问题：执行出错前状态可能已经被修改。**

因为在 `_swap()` 里：

- 先 `gather_allocated_blocks_and_free(seq_ids)`：`swiftllm/worker/model_instance.py:343`
- 再 `allocate_blocks_for_seqs(...)`：`swiftllm/worker/model_instance.py:344`

所以如果 CPU 分配失败：

- GPU blocks 已经被逻辑释放
- CPU blocks 还没申请成功
- 真正复制也还没发生

因此当前失败路径不是简单的“报错退出”，而是“报错前已经部分改写状态”。

---

## 12. 风险总结

我把当前问题概括成 4 个层次。

### 12.1 调度语义不闭环

scheduler 只知道 GPU 侧压力，不知道 CPU 侧是否有足够落点。

因此它能下发不可执行的 swap-out。

---

### 12.2 错误暴露过晚

`num_cpu_blocks` 配置不合理不会被尽早发现，而是要等到真实业务运行到某次 swap-out 才暴露。

这会让问题变成一种运行期故障，而不是启动期或配置期故障。

---

### 12.3 swap-out 失败不是无副作用失败

由于 `_swap()` 先释放源、后申请目标，失败时系统可能已经进入中间态。

这是当前实现最危险的点之一。

---

### 12.4 C++ 层不是最后一道防线

`block_swapping.cpp` 只是 memcpy 执行器，不会替你补做容量与一致性保护。

所以根因还是在 Python 调度/执行语义这一层。

---

## 13. 建议你后续重点验证的点

虽然这次我只做代码阅读，没有实际运行，但如果你后面要进一步验证，建议重点看下面几件事。

### 13.1 验证 scheduler 是否会在 `num_cpu_blocks == 0` 时仍返回 `cur_swap_out`

可以构造：

- `running_q` 超过 `max_batch_size`
- 或 decoding 总 block 数超过 `num_gpu_blocks`

观察 `scheduler.get_next_batch()` 的输出。

预期是：

- 仍然会出现 `cur_swap_out`

---

### 13.2 验证真实异常点是否在 `BlockManager._allocate_blocks()`

跟踪：

- `ModelInstance._swap()`
- `dst_block_manager.allocate_blocks_for_seqs()`
- `BlockManager._allocate_blocks()`

预期是：

- 报错发生在 Python 层 block manager
- 而不是 `block_swapping.cpp`

---

### 13.3 验证失败前 GPU block manager 是否已经把 blocks 标记为空闲

在 `_swap()` 过程中记录：

- `gpu_block_manager.num_free_blocks`
- `cpu_block_manager.num_free_blocks`
- 对应 seq 的 block table / allocated block 数

预期是：

- 在 CPU 分配失败前，GPU 侧 free blocks 已经增加

---

### 13.4 验证异常向上冒泡的位置

`Engine._main_event_loop()` 在执行：

- `await self._run_on_model_async(self.model.swap_out_seqs, ...)`：`swiftllm/server/engine.py:128-133`

这里没有专门包住 swap-out 异常的恢复逻辑。

因此也值得确认：

- 这个异常最终如何影响整个 event loop
- 是否会直接中断服务循环

---

## 14. 最终结论

最后用一句话概括这次调研结果：

> 当前 SwiftLLM 的 `swap to cpu` 路径中，scheduler 的 swap-out 决策没有纳入 CPU 容量约束，因此即便 `num_cpu_blocks` 很小甚至为 0，仍可能生成 swap-out；真正的容量检查发生在 worker 的 `BlockManager._allocate_blocks()`，所以错误会延迟到执行阶段才暴露；并且由于 `_swap()` 先释放源 GPU blocks、后申请目标 CPU blocks，当前失败路径不是原子性的，存在状态一致性风险。

如果你只想把它压缩成最短版，也可以写成这 3 句：

1. **会调度 swap to cpu。** 因为 scheduler 只看 GPU/batch 压力，不看 CPU blocks 是否够。
2. **不会提前报错。** 真正报错在 worker 的 CPU block 分配阶段。
3. **而且失败前已经可能改状态。** `_swap()` 先释放 GPU blocks，再申请 CPU blocks，所以失败不是无副作用失败。
