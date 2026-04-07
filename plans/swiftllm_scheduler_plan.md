## Scheduler decode block 边界例子

用户在阅读 swiftllm/server/scheduler.py 的 Scheduler.get_next_batch() 时，对 decode 分支的 block 资源判断有疑问。

核心点是：为什么代码按 prompt_len + current_output_len 统计，而不是再额外 +1 去预判“这一轮将生成的新 token”。


设 `block_size = 16`，某请求 `prompt_len = 15`。

1. **prefill 结束后**
   - 采样出第 1 个输出 token。
   - 此时 `output_len = 1`，所以当前总长度是 `15 + 1 = 16`。

2. **下一轮 decode 开始前**
   - scheduler 看到的就是这个更新后的长度 `16`。
   - 所需 block 数是 `ceil(16 / 16) = 1`。
   - 本轮 decode 处理的输入是“刚生成出来的最后一个 token”，其位置是 `16 - 1 = 15`。
   - 因而这一轮只需要 1 个 block，不会报错。

3. **这一轮 decode 结束后**
   - 又新生成了 1 个 token。
   - 此时 `output_len = 2`，当前总长度变成 `15 + 2 = 17`。

4. **再下一轮 decode 开始前**
   - scheduler 重新统计，得到所需 block 数 `ceil(17 / 16) = 2`。
   - 如果 GPU block 不够，就会在这一轮开始前执行 swap out / preempt。
   - 如果够，就正常进入下一轮 decode。

## 结论
因此，这里的资源检查并不是漏掉了“下一轮会多一个 token”。恰恰相反：
- 上一轮刚生成的 token，已经被 append 到 `current_output_len` 中；
- scheduler 在下一轮开始前按“当前长度”重算，正好能捕获跨 block 边界的增长；
- 不需要再人为对长度额外 `+1`，否则会变成过度保守。

## 关于 prefill admission 的补充问答

### 问题
在 `scheduler.py:get_next_batch()` 的 prefill 逻辑里，为什么：
- `cur_num_tokens_sum + cur_seq.prompt_len <= max_tokens_in_batch`
  没有把 `running_q` 中 decode request 的 1 个输入 token 算进去？
- 这个调度算法是不是允许 prefill request 和 decode request 一起参与 inference？

### 回答
1. 这里的 `max_tokens_in_batch` 在当前实现里，约束的是**本轮新启动的 prefill batch 真正送入 forward 的 token 总数**，不是系统里所有活跃 request 的总 token 数。
2. 原因是：只要 prefill 分支成功，`scheduler.py` 就直接 `return cur_batch, [], []`；随后 `engine.py` 只会对 `cur_batch` 构造本轮 `input_ids` 并调用 `model.forward`。
3. 因此，这一轮 forward 是**纯 prefill batch**，实际 token 数就是 `sum(cur_batch 中各请求的 prompt_len)`，不包含 `running_q` 里 decode request 的 1-token 输入，所以这里不需要把它们加进 `max_tokens_in_batch`。
4. 当前调度算法允许 **prefill 和 decode 在系统状态/资源占用层面并存**，因为 admission 时仍会检查：
   - `len(self.running_q) + len(cur_batch) + 1 <= max_batch_size`
   - `cur_batch_block_needed + cur_seq_block_needed + self.num_decoding_gpu_blocks <= num_gpu_blocks`
5. 但它们**不会在当前 scheduler 主路径里混到同一个 `model.forward` batch**。只有当 prefill 分支没启动时，decode 分支才会返回 `self.running_q` 去做 decode forward。
6. 底层 `model_instance.py` / `attention.py` 的确支持 mixed batch（前半 prefill、后半 decode），但当前 scheduler 并没有启用这种 piggyback 调度。

### 一句话结论
所以：
- 你看到的 `max_tokens_in_batch` 没算 decode 的 1-token，并不是漏算；
- 因为当前实现里，启动 prefill 的这一轮根本不会把 running decode 一起送去 forward；
- 当前实现是“资源上并存、执行上分轮”，而不是“prefill + decode 同轮混跑”。
