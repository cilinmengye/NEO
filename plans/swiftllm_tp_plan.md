# Problem 1 （2026-03-29）

为了对swiftLLM添加Tensor Parallel功能，我对swiftLLM worker的整体架构进行了巨大变化，并对swiftLLM server的接口进行更改; 原先代码可以查看@/home/yxlin/github/swift/base-swiftLLM, 现在我修改的代码可以查看@/home/yxlin/github/swift/swiftLLM。我尽量复用了原先swiftLLM编写的kernel代码@/home/yxlin/github/swift/swiftLLM/swiftllm/worker/kernels

整体架构更改风格我模仿的是@/home/yxlin/github/swift/nano-vllm, nano-vllm整体架构实现得非常优雅，整体框架我都是按照其来更改的。

相比原版swiftLLM, 我作出了如下修改:
* 我对swiftLLM作出了模型分层的架构, 新增加了@/home/yxlin/github/swift/swiftLLM/swiftllm/worker/layers 和 @/home/yxlin/github/swift/swiftLLM/swiftllm/worker/models两个目录
* 我重写了加载模型参数的代码，使得加载模型参数代码更通用，更简洁。具体代码在@/home/yxlin/github/swift/swiftLLM/swiftllm/worker/loader.py
* 我将engine config, model config, infer state设置为全局参数。具体代码在@/home/yxlin/github/swift/swiftLLM/swiftllm/engine_config.py, @/home/yxlin/github/swift/swiftLLM/swiftllm/worker/mconfigs/llamaconfig.py, @/home/yxlin/github/swift/swiftLLM/swiftllm/worker/infer_state.py
* 我新增加了ModelInstance, 主要是为了分离和简化ModelRunner类的功能, ModelInstance专注于对模型状态的管理, ModelRunner专注于对多进程通信和管理。具体代码在@/home/yxlin/github/swift/swiftLLM/swiftllm/worker/model_runner.py, @/home/yxlin/github/swift/swiftLLM/swiftllm/worker/model_instance.py
* 我在server端添加了LLMEngine类，负责进行多进程运行模型的启动。具体代码在@/home/yxlin/github/swift/swiftLLM/swiftllm/server/llm_engine.py


但是我感觉我代码有许多地方做的还不够好，我现在想要测试我修改后的代码， 测试代码在@/home/yxlin/github/swift/swiftLLM/examples/offline.py，结果出现了如下报错：

python offline.py --model-path /mnt/hdd/data/yxlin/huggingface/Meta-Llama-3.1-8B
[rank0]: Traceback (most recent call last):
[rank0]:   File "/home/yxlin/github/swift/swiftLLM/examples/offline.py", line 41, in <module>
[rank0]:     llm_engine = LLMEngine(engine_config=engine_config)
[rank0]:                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/home/yxlin/github/swift/swiftLLM/swiftllm/server/llm_engine.py", line 51, in __init__
[rank0]:     self.model_runner = ModelRunner(engine_config=engine_config, rank=0, event=self.events)
[rank0]:                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/home/yxlin/github/swift/swiftLLM/swiftllm/worker/model_runner.py", line 71, in __init__
[rank0]:     self.modelinstance = ModelInstance(self.engine_config, rank)
[rank0]:                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/home/yxlin/github/swift/swift_env/lib/python3.11/site-packages/torch/utils/_contextlib.py", line 116, in decorate_context
[rank0]:     return func(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/home/yxlin/github/swift/swiftLLM/swiftllm/worker/model_instance.py", line 36, in __init__
[rank0]:     self.model = LlamaForCausalLM(self.model_config)
[rank0]:                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/home/yxlin/github/swift/swiftLLM/swiftllm/worker/models/llama.py", line 189, in __init__
[rank0]:     self.model = LlamaModel(config)
[rank0]:                  ^^^^^^^^^^^^^^^^^^
[rank0]:   File "/home/yxlin/github/swift/swiftLLM/swiftllm/worker/models/llama.py", line 158, in __init__
[rank0]:     [LlamaDecoderLayer(config, i) for i in range(config.num_layers)]
[rank0]:   File "/home/yxlin/github/swift/swiftLLM/swiftllm/worker/models/llama.py", line 158, in <listcomp>
[rank0]:     [LlamaDecoderLayer(config, i) for i in range(config.num_layers)]
[rank0]:      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/home/yxlin/github/swift/swiftLLM/swiftllm/worker/models/llama.py", line 128, in __init__
[rank0]:     self.mlp = LlamaMLP(config, layer_id)
[rank0]:                ^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/home/yxlin/github/swift/swiftLLM/swiftllm/worker/models/llama.py", line 29, in __init__
[rank0]:     self.down_proj = RowParallelLinear(
[rank0]:                      ^^^^^^^^^^^^^^^^^^
[rank0]:   File "/home/yxlin/github/swift/swiftLLM/swiftllm/worker/layers/linear.py", line 82, in __init__
[rank0]:     super().__init__(divide(input_size, tp_size), output_size)
[rank0]:   File "/home/yxlin/github/swift/swiftLLM/swiftllm/worker/layers/linear.py", line 32, in __init__
[rank0]:     self.weight = nn.Parameter(torch.empty(output_size, input_size))
[rank0]:                                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/home/yxlin/github/swift/swift_env/lib/python3.11/site-packages/torch/utils/_device.py", line 79, in __torch_function__
[rank0]:     return func(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^
[rank0]: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 224.00 MiB. GPU 0 has a total capacity of 23.52 GiB of which 190.69 MiB is free. Including non-PyTorch memory, this process has 23.32 GiB memory in use. Of the allocated memory 22.86 GiB is allocated by PyTorch, and 1.19 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
[rank0]:[W329 15:29:07.410584226 ProcessGroupNCCL.cpp:1168] Warning: WARNING: process group has NOT been destroyed before we destruct ProcessGroupNCCL. On normal program exit, the application should call destroy_process_group to ensure that any pending NCCL operations have finished in this process. In rare cases this process can exit before this point and block the progress of another member of the process group. This constraint has always been present,  but this warning has only been added since PyTorch 2.4 (function operator())

为什么会出现这样的原因？原先的swiftLLM在使用model /mnt/hdd/data/yxlin/huggingface/Meta-Llama-3.1-8B进行运行时都不会有如上报错，我认为是我的代码哪里实现的不正确了。

请你为我排查问题，同时如何我更改的代码你认为哪里还有不正确的地方，或者是架构上有问题的地方，你都需要提出来。

你需要详细阅读我更改后的代码@/home/yxlin/github/swift/swiftLLM，并和更改前的代码@/home/yxlin/github/swift/base-swiftLLM进行对比，已知更改前的代码功能上是正确的。同时我想让代码的整体框架和实现思路都尽量按照@/home/yxlin/github/swift/nano-vllm的实现风格。所以也需要你仔细阅读@/home/yxlin/github/swift/nano-vllm下的代码。已知nano-vllm功能上也是正确的。

我需要将先做调研，并将调查到的内容，更改代码方案都以中文详细地**追加到**@/home/yxlin/github/swift/swiftLLM/plan.md，即不能删除文档中原来有的内容。

而且你只需要先将内容写入文档中，你先不需要更改具体代码文件。

---

## Claude 调研补充（2026-03-29）

下面是在详细阅读当前 `swiftLLM`、对照 `base-swiftLLM` 与 `nano-vllm` 后得到的调查结论，以及建议的后续代码修改方案。这里严格采用**追加**方式保留你原文内容，不删除前面的任何描述。

### 一、我对当前问题的总体判断

这次问题并不是单点 bug，而是由一组相互关联的问题叠加导致的：

1. **当前 offline OOM 的直接触发原因**是：模型模块参数在 GPU 上按默认 `float32` 构造，而不是按模型推理 dtype（通常是 `float16/bfloat16`）构造，导致建模阶段显存接近翻倍。
2. **即使修掉上述 OOM**，当前 `profile_num_blocks()` 路径仍然会因为“对所有 token 都做 `lm_head`”而制造额外显存压力，并且 profile 行为与真实推理路径不一致。
3. **当前 forward / sampler 语义本身也不正确**：上层要的是每条序列一个 next token，但现在的实现会对展平后的所有 token 逐行 argmax，返回数量和语义都不对。
4. **当前 TP 相关逻辑还有若干 correctness bug**，包括 shard 起始位置写错、lm_head 并行输出语义不完整、多进程共享内存 RPC 协议错误等。
5. **online engine 侧也还没完全对齐新架构**，`LLMEngine / ModelRunner / Engine` 的接口闭环还需要进一步收口。

因此，我建议后续修改按“先修阻塞项，再修 correctness，再修集成”的顺序推进，而不是一开始继续扩大重构范围。

---

### 二、当前 offline OOM 的直接原因

#### 1. 模型参数是在 GPU 上按默认 `float32` 创建，导致建模期显存翻倍

关键代码路径如下：

- `examples/offline.py`
  - 构造 `EngineConfig`
  - 实例化 `LLMEngine`
- `swiftllm/server/llm_engine.py`
  - 创建主进程 `ModelRunner`
- `swiftllm/worker/model_runner.py`
  - 调用 `torch.cuda.set_device(rank)`
  - 调用 `torch.set_default_device("cuda")`
  - 然后创建 `ModelInstance`
- `swiftllm/worker/model_instance.py`
  - 创建 `LlamaForCausalLM`
- `swiftllm/worker/models/llama.py`
  - 构造 `LlamaModel` / `LlamaDecoderLayer` / `LlamaMLP`
- `swiftllm/worker/layers/linear.py`
  - `LinearBase.__init__` 中执行：
    - `self.weight = nn.Parameter(torch.empty(output_size, input_size))`

问题在于：

- 当前只设置了默认 device 为 `cuda`；
- 但没有像 `nano-vllm/nanovllm/engine/model_runner.py` 那样，在模型构造前同步设置 `torch.set_default_dtype(hf_config.torch_dtype)`；
- 因而 `linear.py`、`embed_head.py`、`layernorm.py` 中这些未显式指定 dtype 的参数，极大概率是按默认 `float32` 直接分配到 GPU 上。

对于 8B 模型，这会让模型建参阶段的显存占用接近翻倍。你当前报错发生在 `RowParallelLinear` 的 `down_proj` 初始化处，本质上就是：在模型还没完成构造时，显存已经被 GPU 上的大量 `float32` 参数耗尽了。

#### 2. 为什么原版 `base-swiftLLM` 不会在这里 OOM

结合对 `base-swiftLLM` 的阅读，我认为原因不是“旧版模型天然更省”，而是：

- 旧版并不是用当前这种“在 GPU 上直接构造整套 `nn.Module` 参数矩阵”的方式组织权重；
- 你现在的新版是更加接近 `nano-vllm` 风格的模块化架构，但在迁移过程中漏掉了一个非常关键的步骤：**构造期 dtype 管理**。

所以本质上不是“模块化一定更耗显存”，而是“模块化以后，你需要像 `nano-vllm` 那样显式管理 default dtype/device，否则参数会用错误 dtype 落到 GPU 上”。

---

### 三、即使修掉建模期 OOM，当前 profile 路径仍然有问题

#### 1. `profile_num_blocks()` 当前会走一条过重的 fake prefill 路径

`swiftllm/worker/model_instance.py` 中：

- `profile_num_blocks()` 会构造一个 synthetic max prefill batch：
  - `max_batch_size`
  - `max_tokens_in_batch`
- 然后调用：
  - `self.forward(..., ignore_kvcache=True)`

这本来是为了测量模型在高负载下的显存峰值，进而估算还能分配多少 KV blocks。这个思路本身是合理的。

#### 2. 当前 `LlamaForCausalLM.forward()` 会对所有 token 做 `lm_head`

`swiftllm/worker/models/llama.py` 中当前逻辑是：

- `hidden_states = self.model(input_ids)`
- `return self.lm_head(hidden_states)`

这意味着：

- 如果 synthetic prefill batch 一共展平出 `num_tokens=32768` 个 token；
- 当前实现就会对这 32768 行 hidden states 全部做 vocab projection；
- 这与真实推理的需求不一致，因为最终只需要**每个序列最后一个 token**的 logits 来做 next-token sampling。

因此即使把构造期 dtype 问题修掉，这里仍会：

- 无意义地产生超大的 logits tensor；
- 抬高 profile 阶段的峰值显存；
- 让 `profile_num_blocks()` 的结果偏离真实部署场景。

---

### 四、当前 forward / sampler 语义不正确

#### 1. 当前实现返回的是“每个 token 一个 argmax”，不是“每条序列一个 next token”

在 `swiftllm/worker/model_instance.py` 中：

- `logits = self.model(input_ids)`
- `output_tokens = self.sampler(logits)`

而 `sampler()` 当前是：

- `output_tokens = torch.argmax(logits, dim=1)`

问题是：

- 若 `logits.shape[0] == num_tokens`，那输出 token 个数就是 `num_tokens`；
- 但上层 offline / online engine 都期望的是 `batch_size` 个 token；
- 即每条请求只产生一个 next token。

这说明当前问题不只是“显存占用大”，而是**接口语义本身已经和 `base-swiftLLM` 偏离了**。

#### 2. 正确语义应该是什么

这里 `base-swiftLLM/swiftllm/worker/layers/post_layer.py` 给了一个非常清晰的参考：

- 对 prefill 请求：取每个请求最后一个 prompt token 的 hidden state；
- 对 decode 请求：取当前 decode token 的 hidden state；
- 然后只对这 `batch_size` 个位置做 final norm + lm_head + argmax。

`nano-vllm` 里对应的架构表达则是：

- `forward()` 只返回 hidden states；
- `compute_logits()` 再对需要采样的位置做输出层计算。

所以你的新架构如果想继续向 `nano-vllm` 对齐，一个关键点就是：

> **把 transformer 主体输出和最终 logits 计算解耦，只在 last-token hidden states 上做 final norm + lm_head。**

---

### 五、当前 TP 实现中存在的 correctness bug

#### 1. `ColParallelLinear` / `RowParallelLinear` 的 shard 起始位置写错

在 `swiftllm/worker/layers/linear.py` 中：

- `ColParallelLinear.load_weight()` 当前使用：
  - `start_idx = self.tp_size * shard_size`
- `RowParallelLinear.load_weight()` 当前使用：
  - `start_idx = self.tp_size * shard_size`

这里显然应该依赖 `tp_rank` 而不是 `tp_size`。正确写法应与 `nano-vllm/nanovllm/layers/linear.py` 对齐：

- `start_idx = self.tp_rank * shard_size`

否则会出现：

- rank0 也不会从 0 开始取 shard；
- TP>1 时所有 rank 都会拿错分片；
- 结果不是“可能不准”，而是**一定错误**。

#### 2. 当前 lm_head 的 TP 语义不完整

在 `swiftllm/worker/models/llama.py` 中：

- 当前 `lm_head = ColParallelLinear(config.hidden_size, config.vocab_size)`

在 `swiftllm/worker/model_instance.py` 中：

- 当前每个 rank 都直接本地 `argmax`。

这在 TP>1 下是错误的，因为：

- 每个 rank 只持有 vocab 的一部分 logits；
- 本地 argmax 只是 shard 内部最大值，不是全 vocab 最大值；
- 所以最终 token id 会错。

`nano-vllm/nanovllm/layers/embed_head.py` 的正确思路是：

- 本地只算本 rank 的 vocab shard logits；
- `tp_size > 1` 时 gather 到 rank0；
- 由 rank0 拼接出完整 vocab logits；
- 只在 rank0 上做采样并向上返回 token ids。

这条语义对于当前 `swiftLLM` 也是最自然的第一版实现路径。

---

### 六、当前多进程共享内存 RPC 存在直接 bug

在 `swiftllm/worker/model_runner.py` 中有两个很直接的问题：

#### 1. `pickle.dump` 用错了，应该是 `pickle.dumps`

当前写法：

- `data = pickle.dump([method_name, *args])`

这里 `dump` 是把对象写到文件句柄，不会返回要写入共享内存的 bytes。正确语义应该是：

- `data = pickle.dumps([method_name, *args])`

#### 2. `read_shm()` 和 `loop()` 的返回契约不一致

当前：

- `read_shm()` 中：
  - `method_name, *args = pickle.loads(...)`
  - `return method_name, *args`
- 但 `loop()` 中写的是：
  - `method_name, args = self.read_shm()`

也就是：

- 一边返回“展开后的多元组”；
- 另一边按“二元组 `(method_name, args)`”接收；
- 参数一多就会立刻错位。

这一块建议直接对齐 `nano-vllm/nanovllm/engine/model_runner.py`，不要保留现在这套半成品协议。

---

### 七、当前 online engine 侧还有接口错误

在 `swiftllm/server/engine.py` 中：

#### 1. `get_model_config` 漏了括号

当前：

- `model_config = LlamaModelConfig.get_model_config`

这里拿到的是函数对象，不是 config 实例。应为：

- `model_config = LlamaModelConfig.get_model_config()`

#### 2. `Scheduler` 构造参数传错

当前：

- `self.scheduler = Scheduler(self.model, self.engine_config, model_config.num_gpu_blocks)`

但 `swiftllm/server/scheduler.py` 中 `Scheduler` 的签名是：

- `Scheduler(model_config, engine_config, num_gpu_blocks)`

因此这里传入的第一个参数应该是 `model_config`，不是 `self.model`。

#### 3. `LLMEngine / ModelRunner` 还需要补齐对外 API

当前 online 路径期望底层模型对象能提供：

- `forward(...)`
- `swap_in_seqs(...)`
- `swap_out_seqs(...)`
- `free_seqs_resources(...)`

但你现在新增的 `LLMEngine` / `ModelRunner` 还没有完全按这组接口闭环，所以即使 offline 初始化修好，online 路径也会继续在接口层断掉。

---

### 八、我对当前整体架构的评价

#### 1. 可以保留的方向

我认为下面这几条方向是对的，不建议推翻：

- `ModelInstance` 与 `ModelRunner` 分层：
  - `ModelInstance` 专注模型状态与执行；
  - `ModelRunner` 专注多进程通信与驱动；
  - 这是合理的职责拆分。
- `name_to_module -> load_weight` 的 loader 框架：
  - 当前 loader 的总体设计方向是好的；
  - 它确实比旧版把权重全部堆在 CPU dict 上更通用，也更贴近现代模块化设计。

#### 2. 需要继续向 `nano-vllm` 收口的地方

真正还没收住的点主要有：

- 模型构造期的 default dtype / default device 管理；
- 输出头语义（只针对 last token 计算 logits）；
- TP shard load correctness；
- TP 下 logits 聚合与 rank0 sampling；
- `LLMEngine / Engine / ModelRunner` 的接口闭环。

也就是说：

> 你的重构方向本身没错，问题主要不是“分层太激进”，而是“关键执行细节还没有完全对齐到一个已经被验证过的实现模式”。

---

### 九、建议的代码修改顺序

我建议后续修改严格按下面顺序推进，而不是并行大改：

#### Phase 1：先修启动期 OOM 与 TP RPC 基础设施

**涉及文件：**
- `swiftllm/worker/mconfigs/llamaconfig.py`
- `swiftllm/worker/model_runner.py`

**要做的事：**
1. 在 `LlamaModelConfig` 中解析并保存 HuggingFace 配置里的 `torch_dtype`；
2. 在 `ModelRunner` 中参考 `nano-vllm`：
   - 保存旧 default dtype；
   - 在构造模型前设置 `torch.set_default_dtype(model_config.torch_dtype)`；
   - 再设置 `torch.set_default_device("cuda")`；
   - 模型构造完成后恢复 default dtype/device；
3. 修共享内存 RPC：
   - `pickle.dump -> pickle.dumps`
   - `read_shm()` 返回 `(method_name, args)`
   - 与 `loop()` / `call()` 的契约完全对齐。

**这一步的目标：**
- 先让单卡 offline 初始化不再在建模阶段 OOM；
- 先让 TP 主控平面具备基本可用性。

#### Phase 2：修输出语义，解决 profile OOM 与 batch 输出错误

**涉及文件：**
- `swiftllm/worker/models/llama.py`
- `swiftllm/worker/model_instance.py`

**要做的事：**
1. 将 `LlamaForCausalLM` 改成两段式：
   - `forward()` 只返回 hidden states；
   - `compute_logits()` 再对最后 token 做 final norm + lm_head。
2. last-token 索引逻辑参考：
   - `base-swiftLLM/swiftllm/worker/layers/post_layer.py`
   - `nano-vllm/nanovllm/models/qwen3.py`
3. `sampler()` 只消费 `batch_size` 行 logits，并返回 `batch_size` 个 token。

**这一步的目标：**
- 让 profile 路径与真实推理路径语义一致；
- 让 forward / sampler 的返回值与上层 engine 契约重新对齐。

#### Phase 3：修 TP 输出头与 shard correctness

**涉及文件：**
- `swiftllm/worker/layers/linear.py`
- `swiftllm/worker/layers/embed_head.py`
- `swiftllm/worker/models/llama.py`
- `swiftllm/worker/model_instance.py`

**要做的事：**
1. 修 `ColParallelLinear` / `RowParallelLinear` 的 shard 起始位置：
   - `self.tp_size * shard_size -> self.tp_rank * shard_size`
2. 在 `embed_head.py` 中实现真正的 `ParallelLMHead`；
3. `tp_size > 1` 时 gather logits 到 rank0，并只在 rank0 上采样。

**这一步的目标：**
- 让 TP=2 的输出和 TP=1 对齐；
- 让 `lm_head` 的并行语义真正正确。

#### Phase 4：收口 online engine 接口

**涉及文件：**
- `swiftllm/server/llm_engine.py`
- `swiftllm/server/engine.py`
- `swiftllm/worker/model_runner.py`

**要做的事：**
1. 修 `get_model_config()` 调用错误；
2. 修 `Scheduler(...)` 参数错误；
3. 为 `LLMEngine / ModelRunner` 补齐：
   - `forward`
   - `swap_in_seqs`
   - `swap_out_seqs`
   - `free_seqs_resources`
4. 如有需要保留 `step()` 作为 `forward()` 的别名，兼容当前 offline 用法。

**这一步的目标：**
- 让 offline 和 online 两条路径共享一套一致的底层接口。

---

### 十、建议复用的参考实现

#### 1. 从 `base-swiftLLM` 复用的点

重点参考：

- `base-swiftLLM/swiftllm/worker/layers/post_layer.py`
  - 这里已经很好地实现了“每个请求只取最后一个 token 的 hidden state，再做输出层”的语义。
- `base-swiftLLM/swiftllm/worker/model.py`
  - 可以用来对照 server-facing forward 接口的返回契约。

#### 2. 从 `nano-vllm` 复用的点

重点参考：

- `nano-vllm/nanovllm/engine/model_runner.py`
  - 默认 dtype/device 管理；
  - 共享内存 RPC 协议；
  - rank0 sampling 语义。
- `nano-vllm/nanovllm/layers/linear.py`
  - TP shard 切分逻辑。
- `nano-vllm/nanovllm/layers/embed_head.py`
  - `ParallelLMHead` 的 gather-to-rank0 设计。
- `nano-vllm/nanovllm/models/qwen3.py`
  - `forward()` 与 `compute_logits()` 的接口拆分方式。

---

### 十一、本轮不建议优先大改的部分

为了尽快收住当前最关键的问题，我不建议本轮优先做下面这些事：

1. **不建议重写整个 loader 框架**
   - 当前 loader 的主方向是对的；
   - 真正的问题在于各模块如何消费分片权重，而不是 loader 这个分发框架本身。

2. **不建议本轮优先动 attention kernel 或 scheduler 策略**
   - 当前的最短闭环是：
     - 模型先能构造；
     - offline 先能稳定跑；
     - TP 输出先正确；
     - online 接口再打通。
   - 先不要把改动范围扩展到 kernel 优化与调度策略优化。

---

### 十二、建议的验证方案

#### 1. offline 单卡回归

目标：确认最直接的 OOM 已消失，且输出语义恢复正确。

建议验证：
- 使用 `examples/offline.py`；
- `tensor_parallel_size=1`；
- 加载 `Meta-Llama-3.1-8B`；
- 确认 `LLMEngine -> ModelRunner -> ModelInstance` 初始化成功；
- 确认 `profile_num_blocks()` 得到 `num_gpu_blocks > 0`；
- 跑一个纯 prefill batch，确认返回 token 数量等于 `batch_size`，而不是 `num_tokens`。

#### 2. TP=2 correctness 回归

目标：确认 shard load、lm_head、sampling 都是正确的。

建议验证：
- 在 greedy 采样下，分别运行：
  - `tensor_parallel_size=1`
  - `tensor_parallel_size=2`
- 使用同一个模型、同一组 prompt；
- 对比逐 token 输出是否一致；
- 覆盖：
  - 纯 prefill
  - 纯 decode
  - prefill + decode 混合 batch

#### 3. online engine 集成回归

目标：确认 `Engine.initialize()` 与请求处理路径可用。

建议验证：
- 调用 `Engine.initialize()`；
- 确认 `Scheduler` 正常拿到 `model_config` 与 `num_gpu_blocks`；
- 跑一个 `add_request_and_wait()`；
- 再跑一个 `add_request_and_stream()`。

#### 4. swap 路径回归

目标：确认 RPC 与多进程调用闭环完整。

建议验证：
- 通过缩小 `num_gpu_blocks` 或提高并发请求，主动触发：
  - `swap_out_seqs`
  - `swap_in_seqs`
  - `free_seqs_resources`
- 重点检查 rank0 到子进程的 RPC 是否能正常序列化、反序列化、解包与调用。

---

### 十三、后续代码注释建议

由于你明确要求后续修改代码时必须写出详细注释，我建议把下面这些关键点明确写入代码注释中，而不是只靠提交说明：

1. **为什么要在 `ModelRunner` 中显式设置 default dtype/device**
   - 注释要说明：这一步是为了防止模块参数在 GPU 上以错误 dtype 建参，从而导致构造期显存翻倍。

2. **为什么 `LlamaForCausalLM` 不能对所有 token 都做 `lm_head`**
   - 注释要说明：server-facing 语义是“一条请求返回一个 next token”，因此只应对最后 token hidden states 计算 logits。

3. **为什么 TP 下 `lm_head` 必须 gather 到 rank0 再采样**
   - 注释要说明：每个 rank 只持有部分 vocab logits，本地 argmax 不是全 vocab argmax。

4. **为什么 shard 起始位置必须依赖 `tp_rank`**
   - 注释要明确：分片位置由当前 rank 决定，而不是由 world size 决定。

5. **为什么共享内存 RPC 必须保持 `(method_name, args)` 的固定契约**
   - 注释要说明：主进程和子进程必须使用一致的序列化/解包协议，否则多参数调用会直接错位。

---

### 十四、结论

综合来看：

- 当前这次 OOM 的最直接原因是**模型构造期 dtype 管理缺失**；
- 但继续往下排查后，可以确认当前代码里还同时存在：
  - profile 路径过重；
  - forward / sampler 语义不正确；
  - TP shard load 错误；
  - TP lm_head 聚合语义缺失；
  - 共享内存 RPC 错误；
  - online engine 接口未完全闭环。

因此我认为：

> 当前最合理的策略不是继续扩大重构面，而是按上面的四个 phase，先把关键执行路径收口到一个“单卡能稳定起、TP 结果正确、online 接口闭环”的最小正确版本。

如果后续进入具体改代码阶段，我会按这个顺序推进，并把关键原因、设计取舍与修复要点直接写入对应代码注释里，保证后续你自己继续演进时不需要重新做这轮调查。

# Problem 2

在初步解决Problem 1中OOM的严重问题后，我再次运行了@/home/yxlin/github/swift/swiftLLM/examples/offline.py, 结果出现了如下问题:

[rank0]: Traceback (most recent call last):
[rank0]:   File "/home/yxlin/github/swift/swiftLLM/examples/offline.py", line 60, in <module>
[rank0]:     prompt_phase_outputs = llm_engine.step(
[rank0]:                            ^^^^^^^^^^^^^^^^
[rank0]:   File "/home/yxlin/github/swift/swiftLLM/swiftllm/server/llm_engine.py", line 101, in step
[rank0]:     return self.forward(
[rank0]:            ^^^^^^^^^^^^^
[rank0]:   File "/home/yxlin/github/swift/swiftLLM/swiftllm/server/llm_engine.py", line 83, in forward
[rank0]:     return self.model_runner.call(
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/home/yxlin/github/swift/swiftLLM/swiftllm/worker/model_runner.py", line 143, in call
[rank0]:     return method(*args)
[rank0]:            ^^^^^^^^^^^^^
[rank0]:   File "/home/yxlin/github/swift/swiftLLM/swiftllm/worker/model_runner.py", line 122, in run
[rank0]:     return self.modelinstance.forward(
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/home/yxlin/github/swift/swift_env/lib/python3.11/site-packages/torch/utils/_contextlib.py", line 116, in decorate_context
[rank0]:     return func(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/home/yxlin/github/swift/swiftLLM/swiftllm/worker/model_instance.py", line 299, in forward
[rank0]:     hidden_states = self.model(input_ids)
[rank0]:                     ^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/home/yxlin/github/swift/swift_env/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
[rank0]:     return self._call_impl(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/home/yxlin/github/swift/swift_env/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1562, in _call_impl
[rank0]:     return forward_call(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/home/yxlin/github/swift/swiftLLM/swiftllm/worker/models/llama.py", line 210, in forward
[rank0]:     return self.model(input_ids)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/home/yxlin/github/swift/swift_env/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
[rank0]:     return self._call_impl(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/home/yxlin/github/swift/swift_env/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1562, in _call_impl
[rank0]:     return forward_call(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/home/yxlin/github/swift/swiftLLM/swiftllm/worker/models/llama.py", line 176, in forward
[rank0]:     hidden_states = layer(
[rank0]:                     ^^^^^^
[rank0]:   File "/home/yxlin/github/swift/swift_env/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
[rank0]:     return self._call_impl(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/home/yxlin/github/swift/swift_env/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1562, in _call_impl
[rank0]:     return forward_call(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/home/yxlin/github/swift/swiftLLM/swiftllm/worker/models/llama.py", line 143, in forward
[rank0]:     hidden_states = self.self_attn(hidden_states)
[rank0]:                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/home/yxlin/github/swift/swift_env/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
[rank0]:     return self._call_impl(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/home/yxlin/github/swift/swift_env/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1562, in _call_impl
[rank0]:     return forward_call(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/home/yxlin/github/swift/swiftLLM/swiftllm/worker/models/llama.py", line 108, in forward
[rank0]:     o = self.attn(q, k, v, hidden_state)    # 因为后续 hidden_state 不会再被使用了, 所以为避免
[rank0]:         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/home/yxlin/github/swift/swift_env/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
[rank0]:     return self._call_impl(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/home/yxlin/github/swift/swift_env/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1562, in _call_impl
[rank0]:     return forward_call(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/home/yxlin/github/swift/swiftLLM/swiftllm/worker/layers/attention.py", line 55, in forward
[rank0]:     store_kvcache(
[rank0]:   File "/home/yxlin/github/swift/swiftLLM/swiftllm/worker/kernels/kvcache_mgmt.py", line 92, in store_kvcache
[rank0]:     assert k.is_contiguous()
[rank0]:            ^^^^^^^^^^^^^^^^^
[rank0]: AssertionError

我应该如何解决他？我需要你先做调研，并将调查到的内容，更改代码方案都以中文详细地**追加到**@/home/yxlin/github/swift/swiftLLM/plan.md，即不能删除文档中原来有的内容。

## Claude 调研补充（2026-03-29，Problem 2）

### 一、问题现象的直接结论

这次报错并不是 KV Cache kernel 自己突然写错了，而是 **新的 fused QKV 投影路径改变了 `q / k / v` 的内存布局，但下游 kernel 仍然沿用了旧版 `base-swiftLLM` 的 contiguous 契约**。

最直接的失败点是：

- `swiftllm/worker/models/llama.py` 中 `LlamaAttention.forward()` 通过 `self.qkv_proj(hidden_state)` 得到一个合并后的 `qkv` 大张量；
- 然后执行：
  - `q, k, v = qkv.split([...], dim=-1)`
  - `q = q.view(...)`
  - `k = k.view(...)`
  - `v = v.view(...)`
- 接着 `swiftllm/worker/layers/attention.py` 立即把 `k / v` 交给 `store_kvcache()`；
- 但 `swiftllm/worker/kernels/kvcache_mgmt.py` 里明确写着：
  - `assert k.is_contiguous()`
  - `assert v.is_contiguous()`

所以这次 Problem 2 的本质不是“KV Cache 算法错了”，而是：

> **packed QKV + split/view 之后得到的 `k / v` 不再满足旧 kernel 假定的连续内存布局。**

---

### 二、为什么当前 packed QKV 写法会触发这个问题

当前新版 `swiftLLM` 在 attention 中做了 fused QKV projection，这是合理的优化方向：一次 GEMM 同时生成 Q / K / V，减少 kernel launch 和访存开销。

但问题在于：

1. `qkv = self.qkv_proj(hidden_state)` 得到的是一个大的合并输出；
2. `split(..., dim=-1)` 返回的是这个大张量上的 **view**，而不是新的独立连续张量；
3. 后面再 `view(-1, num_heads, head_dim)`，只是重新解释 shape，并不会自动修复底层 stride / storage 布局；
4. 因此传给 `store_kvcache()` 的 `k / v` 很可能仍然不是 `is_contiguous()`。

而当前 `kvcache_mgmt.py` 的 Triton 包装层索引方式，是按“输入张量就是连续的 `[tokens, heads, head_dim]`”这个前提写的，所以它会直接断言失败。

换句话说：

- **旧 kernel 契约没变；**
- **但是新的上游张量布局已经变了；**
- 二者之间没有重新收口，这就是 Problem 2 的根因。

---

### 三、为什么 `base-swiftLLM` 不会出现这个问题

对照 `base-swiftLLM/swiftllm/worker/layers/transformer_layer.py` 可以看到，旧版是分别计算：

- `q = linear(input_embds, self.weight.q_proj)`
- `k = linear(input_embds, self.weight.k_proj)`
- `v = linear(input_embds, self.weight.v_proj)`

也就是说，Q / K / V 各自来自**独立 GEMM 输出**，再做 `view(...)` 时更自然地满足旧版 kernel 对连续内存的预期。

因此：

- 旧版 kernel 本身并没有问题；
- 真正变化的是当前重构版把“三次独立投影”改成了“一次 fused QKV 投影”，但没有同步调整 kernel 契约或上游张量收口逻辑。

---

### 四、为什么 `nano-vllm` 只能作为参考，不能直接照搬

`nano-vllm` 也使用 packed QKV + split 的结构，但它的 attention / KV cache 路径接受的是 **stride-compatible** 的布局，而不是当前 `swiftLLM` 这里这么强的 `is_contiguous()` 要求。

这意味着：

- `nano-vllm` 可以作为“packed QKV 架构为什么合理”的参考；
- 但**不能直接推出**当前 `swiftLLM` 的 Triton kernel 已经能吃这种非连续 view；
- 如果要像 `nano-vllm` 一样放宽要求，就必须重新审视：
  - `store_kvcache()` 的 Triton 索引逻辑；
  - `paged_attention()` 的 Triton 索引逻辑；
  - 包括 decode 路径切片后的 stride 是否也都成立。

这已经不是最小修复，而是另一个更大的 kernel 契约改造问题。

---

### 五、为什么这次不能只修 `k / v`

这次表面上 first crash 出现在：

- `assert k.is_contiguous()`

但进一步对照当前代码可以发现，decode 路径里：

- `swiftllm/worker/layers/attention.py` 会调用
  - `paged_attention(q[infer_state.num_prefill_tokens:, :, :], ...)`
- 而 `swiftllm/worker/kernels/paged_attn.py` 也要求：
  - `assert q.is_contiguous()`

所以如果这次只在局部把 `k / v` 修成连续：

- prompt phase 可能先不崩；
- 但后面 decode 或 mixed batch 仍可能因为 `q.is_contiguous()` 再次失败。

因此更稳妥的结论是：

> **在 fused QKV 拆分后的统一入口，一次性把 `q / k / v` 都收口到现有 kernel 契约。**

---

### 六、推荐修改方案

我建议采用**最小正确修复**，而不是扩大重构范围：

#### 方案主线

保留现有：
- `QKVParallelLinear`
- fused QKV projection
- 当前 `store_kvcache()` / `paged_attention()` Triton kernel

只在以下位置补上张量布局收口：

- `swiftllm/worker/models/llama.py`
- `LlamaAttention.forward()`

#### 具体改法

在这里：

```python
qkv = self.qkv_proj(hidden_state)
q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
q = q.view(-1, self.num_heads, self.head_dim)
k = k.view(-1, self.num_kv_heads, self.head_dim)
v = v.view(-1, self.num_kv_heads, self.head_dim)
```

后面补充显式的 contiguous 收口，使得：

- `q` 满足 `paged_attention()` 的输入契约；
- `k / v` 满足 `store_kvcache()` 的输入契约。

#### 为什么这条路最合适

因为它：

1. **改动最小**：直接对准当前 regression point；
2. **不推翻已经完成的 QKV 合并结构**；
3. **不需要重写 Triton kernel**；
4. **能同时覆盖 prompt 和 decode 两类 contiguous 风险**；
5. 更符合当前阶段“先恢复 correctness，再评估性能回退”的目标。

---

### 七、本轮不建议优先做的事

#### 1. 不建议立刻退回三次独立 Q/K/V 投影

虽然这样理论上最接近 `base-swiftLLM`，但这等于把刚引入的 fused QKV 结构部分回退，改动面明显更大，也不符合“先做最小修复”的策略。

#### 2. 不建议本轮直接修改 Triton kernel 契约

也就是不建议现在立刻把：

- `store_kvcache()`
- `paged_attention()`

都改成像 `nano-vllm` 那样只依赖 stride，而不是 `is_contiguous()`。

原因是这会把 scope 从“修一个已知回归”扩大成“重审整套 kernel 输入布局设计”，风险和验证成本都更高。

#### 3. 不建议顺手扩大到 attention 调度或别的重构

这次的最短闭环很明确：

1. 让 prompt phase 不再死于 `k.is_contiguous()`；
2. 让 decode / mixed batch 不再死于 `q.is_contiguous()`；
3. 保持当前 kernel 逻辑不动；
4. 再观察是否有可接受的显存 / 性能回退。

---

### 八、建议修改的代码文件

本轮核心修改应优先集中在：

- `swiftllm/worker/models/llama.py`

本轮文档补充应追加到：

- `swiftLLM/plan.md`

如实现时需要补充保护性解释注释，可考虑少量触及：

- `swiftllm/worker/layers/attention.py`
- `swiftllm/worker/kernels/kvcache_mgmt.py`
- `swiftllm/worker/kernels/paged_attn.py`

但我当前**不建议修改这些 kernel 文件的实际逻辑**。

---

### 九、验证方案

#### 1. offline 单卡回归

- 重新运行 `examples/offline.py`
- 确认 prompt phase 不再在：
  - `store_kvcache()`
  - `assert k.is_contiguous()`
  处失败

#### 2. decode / mixed batch 回归

- 继续观察后续 decode 阶段；
- 确认不会转而在：
  - `paged_attention()`
  - `assert q.is_contiguous()`
  处失败

#### 3. 静态检查

- 跑 `python -m compileall swiftllm`
- 确认没有语法错误

#### 4. 观察修复成本

- 重点看显存峰值是否明显上升；
- 如果 contiguous materialization 带来不可接受的显存 / 性能回退，再进入第二轮方案讨论：
  - 退回三投影 Q/K/V；或
  - 重构 kernel 契约，使其原生兼容 packed stride 布局。

---

### 十、后续实现时必须写进代码注释的要点

用户已经明确要求后续修改代码时保留详细注释。因此在真正改代码时，我建议至少把下面这些点写进注释：

1. **为什么 fused QKV + split 会破坏旧 contiguous 假设**；
2. **为什么这次选择在 `LlamaAttention.forward()` 做局部收口**；
3. **为什么这次同时处理 `q / k / v`，而不是只修 `k / v`**；
4. **为什么本轮不直接修改 Triton kernel 契约**；
5. **如果后续要进一步优化，应该优先考虑哪两条方向：**
   - 三投影回退；
   - stride-based kernel 契约重构。

我的建议是：

> 先把当前 packed QKV regression 用最小改动修正确，再决定是否值得继续为性能去重构 kernel 契约。


# Problem 3

上述问题顺利解决了，但是我再次运行了@/home/yxlin/github/swift/swiftLLM/examples/offline.py, 结果出现了如下问题:

Process SpawnProcess-1:
Traceback (most recent call last):
  File "/home/yxlin/.local/share/uv/python/cpython-3.11.14-linux-x86_64-gnu/lib/python3.11/multiprocessing/process.py", line 314, in _bootstrap
    self.run()
  File "/home/yxlin/.local/share/uv/python/cpython-3.11.14-linux-x86_64-gnu/lib/python3.11/multiprocessing/process.py", line 108, in run
    self._target(*self._args, **self._kwargs)
  File "/home/yxlin/github/swift/swiftLLM/swiftllm/worker/model_runner.py", line 92, in __init__
    self.modelinstance = ModelInstance(self.engine_config, rank)
                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/yxlin/github/swift/swift_env/lib/python3.11/site-packages/torch/utils/_contextlib.py", line 116, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/yxlin/github/swift/swiftLLM/swiftllm/worker/model_instance.py", line 51, in __init__
    num_gpu_blocks = self.profile_num_blocks()
                     ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/yxlin/github/swift/swift_env/lib/python3.11/site-packages/torch/utils/_contextlib.py", line 116, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/yxlin/github/swift/swiftLLM/swiftllm/worker/model_instance.py", line 83, in profile_num_blocks
    _ = self.forward(input_ids, seq_ids, [], ignore_kvcache=True)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/yxlin/github/swift/swift_env/lib/python3.11/site-packages/torch/utils/_contextlib.py", line 116, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/yxlin/github/swift/swiftLLM/swiftllm/worker/model_instance.py", line 299, in forward
    hidden_states = self.model(input_ids)
                    ^^^^^^^^^^^^^^^^^^^^^
  File "/home/yxlin/github/swift/swift_env/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/yxlin/github/swift/swift_env/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1562, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/yxlin/github/swift/swiftLLM/swiftllm/worker/models/llama.py", line 227, in forward
    return self.model(input_ids)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/yxlin/github/swift/swift_env/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/yxlin/github/swift/swift_env/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1562, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/yxlin/github/swift/swiftLLM/swiftllm/worker/models/llama.py", line 193, in forward
    hidden_states = layer(
                    ^^^^^^
  File "/home/yxlin/github/swift/swift_env/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/yxlin/github/swift/swift_env/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1562, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/yxlin/github/swift/swiftLLM/swiftllm/worker/models/llama.py", line 160, in forward
    hidden_states = self.self_attn(hidden_states)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/yxlin/github/swift/swift_env/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/yxlin/github/swift/swift_env/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1562, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/yxlin/github/swift/swiftLLM/swiftllm/worker/models/llama.py", line 125, in forward
    o = self.attn(q, k, v, hidden_state)    # 因为后续 hidden_state 不会再被使用了, 所以为避免
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/yxlin/github/swift/swift_env/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/yxlin/github/swift/swift_env/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1562, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/yxlin/github/swift/swiftLLM/swiftllm/worker/layers/attention.py", line 70, in forward
    o[:infer_state.num_prefill_tokens, :] = vllm_flash_attn.flash_attn_varlen_func(
    ~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/yxlin/github/swift/swift_env/lib/python3.11/site-packages/torch/utils/_device.py", line 79, in __torch_function__
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
RuntimeError: The expanded size of the tensor (32768) must match the existing size (16384) at non-singleton dimension 0.  Target sizes: [32768, 4096].  Tensor sizes: [16384, 4096]
[rank0]: Traceback (most recent call last):
[rank0]:   File "/home/yxlin/github/swift/swiftLLM/examples/offline.py", line 41, in <module>
[rank0]:     llm_engine = LLMEngine(engine_config=engine_config)
[rank0]:                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/home/yxlin/github/swift/swiftLLM/swiftllm/server/llm_engine.py", line 51, in __init__
[rank0]:     self.model_runner = ModelRunner(engine_config=engine_config, rank=0, event=self.events)
[rank0]:                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/home/yxlin/github/swift/swiftLLM/swiftllm/worker/model_runner.py", line 92, in __init__
[rank0]:     self.modelinstance = ModelInstance(self.engine_config, rank)
[rank0]:                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/home/yxlin/github/swift/swift_env/lib/python3.11/site-packages/torch/utils/_contextlib.py", line 116, in decorate_context
[rank0]:     return func(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/home/yxlin/github/swift/swiftLLM/swiftllm/worker/model_instance.py", line 51, in __init__
[rank0]:     num_gpu_blocks = self.profile_num_blocks()
[rank0]:                      ^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/home/yxlin/github/swift/swift_env/lib/python3.11/site-packages/torch/utils/_contextlib.py", line 116, in decorate_context
[rank0]:     return func(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/home/yxlin/github/swift/swiftLLM/swiftllm/worker/model_instance.py", line 83, in profile_num_blocks
[rank0]:     _ = self.forward(input_ids, seq_ids, [], ignore_kvcache=True)
[rank0]:         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/home/yxlin/github/swift/swift_env/lib/python3.11/site-packages/torch/utils/_contextlib.py", line 116, in decorate_context
[rank0]:     return func(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/home/yxlin/github/swift/swiftLLM/swiftllm/worker/model_instance.py", line 299, in forward
[rank0]:     hidden_states = self.model(input_ids)
[rank0]:                     ^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/home/yxlin/github/swift/swift_env/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
[rank0]:     return self._call_impl(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/home/yxlin/github/swift/swift_env/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1562, in _call_impl
[rank0]:     return forward_call(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/home/yxlin/github/swift/swiftLLM/swiftllm/worker/models/llama.py", line 227, in forward
[rank0]:     return self.model(input_ids)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/home/yxlin/github/swift/swift_env/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
[rank0]:     return self._call_impl(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/home/yxlin/github/swift/swift_env/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1562, in _call_impl
[rank0]:     return forward_call(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/home/yxlin/github/swift/swiftLLM/swiftllm/worker/models/llama.py", line 193, in forward
[rank0]:     hidden_states = layer(
[rank0]:                     ^^^^^^
[rank0]:   File "/home/yxlin/github/swift/swift_env/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
[rank0]:     return self._call_impl(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/home/yxlin/github/swift/swift_env/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1562, in _call_impl
[rank0]:     return forward_call(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/home/yxlin/github/swift/swiftLLM/swiftllm/worker/models/llama.py", line 160, in forward
[rank0]:     hidden_states = self.self_attn(hidden_states)
[rank0]:                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/home/yxlin/github/swift/swift_env/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
[rank0]:     return self._call_impl(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/home/yxlin/github/swift/swift_env/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1562, in _call_impl
[rank0]:     return forward_call(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/home/yxlin/github/swift/swiftLLM/swiftllm/worker/models/llama.py", line 125, in forward
[rank0]:     o = self.attn(q, k, v, hidden_state)    # 因为后续 hidden_state 不会再被使用了, 所以为避免
[rank0]:         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/home/yxlin/github/swift/swift_env/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
[rank0]:     return self._call_impl(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/home/yxlin/github/swift/swift_env/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1562, in _call_impl
[rank0]:     return forward_call(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/home/yxlin/github/swift/swiftLLM/swiftllm/worker/layers/attention.py", line 70, in forward
[rank0]:     o[:infer_state.num_prefill_tokens, :] = vllm_flash_attn.flash_attn_varlen_func(
[rank0]:     ~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/home/yxlin/github/swift/swift_env/lib/python3.11/site-packages/torch/utils/_device.py", line 79, in __torch_function__
[rank0]:     return func(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^
[rank0]: RuntimeError: The expanded size of the tensor (32768) must match the existing size (16384) at non-singleton dimension 0.  Target sizes: [32768, 4096].  Tensor sizes: [16384, 4096]

我应该如何解决他？我需要你先做调研，并将调查到的内容，更改代码方案都以中文详细地**追加到**@/home/yxlin/github/swift/swiftLLM/plan.md，即不能删除文档中原来有的内容。

---

## Claude 补充调研（2026-03-30）

### 一、这次报错的直接根因

这次 `RuntimeError: Target sizes: [32768, 4096], Tensor sizes: [16384, 4096]` 的根因，不是 `profile_num_blocks()` 把 token 数算错了，也不是 `flash_attn_varlen_func(...)` 本身突然返回了错误 token 数；核心问题是：

- 当前 `LlamaAttention.forward()` 前半段已经进入 **Tensor Parallel 语义**；
- `QKVParallelLinear` 在 TP 下只会产出 **当前 rank 的 local Q/K/V shard**；
- 但 `swiftllm/worker/layers/attention.py` 里的 `Attention.forward()` 仍沿用了旧版“attention 结果直接写回 full hidden width buffer”的契约；
- 因此 prefill 路径里这句
  - `flash_attn_varlen_func(...).reshape(-1, o.shape[-1])`
  实际上是在把 **local hidden shard** 错误地按 **full hidden width** 重排。

当 `tensor_parallel_size = 2` 时，这个错误会被直接放大成：

- 正确语义应该是：`[32768, 2048]`
- 但当前代码却尝试按 full hidden width 重排成：`[16384, 4096]`
- 接下来再赋值给目标切片 `o[:32768, :]`，就会立刻触发 shape mismatch。

因此，这次 Problem 3 的本质不是“prefill token 数异常”，而是：

> **attention 前半段已经切到了 TP local shard 语义，但 attention 输出缓冲区仍停留在旧版 full hidden 语义。**

### 二、为什么 `profile_num_blocks()` 会稳定暴露这个问题

`swiftllm/worker/model_instance.py` 中的 `profile_num_blocks()` 会构造一个 synthetic prefill batch：

- `max_batch_size = 16`
- `max_tokens_in_batch = 2048 * 16 = 32768`

也就是说，它会稳定地跑一条“大 batch、纯 prefill、TP=2”的路径。这个路径并没有偏离真实逻辑，而是把 attention 输出 contract 的错误更早、更稳定地暴露出来。

因此不建议通过下面这些办法“绕过”问题：

- 把 `tensor_parallel_size` 改回 1
- 降低 `max_batch_size`
- 降低 `max_tokens_in_batch`
- 给 `profile_num_blocks()` 单独加特判逻辑

因为这些做法只是在绕过暴露点，并没有修复 attention 的真实 shape contract。

### 三、当前 TP attention 的正确边界应该是什么

结合当前线性层实现可以确认：

1. `QKVParallelLinear`
   - 是 **列并行**；
   - 输出的是当前 rank 的 local Q/K/V shard。

2. `Attention.forward()`
   - 输入既然已经是 local `q / k / v`；
   - 那么它在 `o_proj` 之前产出的 attention 中间结果，也必须仍然是 **local hidden shard**。

3. `RowParallelLinear(o_proj)`
   - 才是“从 local hidden shard 聚合回 full hidden”的边界；
   - 也就是说 full hidden width 语义应当在 `o_proj` 之后恢复，而不是在 flash attention / paged attention 的输出阶段提前假定存在。

换句话说，TP 下更合理的 contract 应该是：

- `q / k / v`：`[num_tokens, local_heads, head_dim]`
- `attention output`（pre-`o_proj`）：`[num_tokens, local_hidden_size]`
- `o_proj(RowParallelLinear)` 输出：`[num_tokens, hidden_size]`

### 四、为什么不能继续复用 `hidden_state` 作为 pre-`o_proj` 输出 buffer

当前 `swiftllm/worker/models/llama.py` 中有这样一段旧逻辑：

- `o = self.attn(q, k, v, hidden_state)`

它默认 `hidden_state` 既可作为 attention 输入来源，也可复用成 attention 输出 buffer。这个思路在非 TP / full hidden attention 时代成立，但现在已经不成立了，因为：

- `hidden_state.shape[-1] == hidden_size`，它表示的是 **full hidden width**；
- 但 `self.attn(...)` 在 TP 下的中间输出语义应该是 **local hidden shard**；
- 继续复用 `hidden_state`，就等价于继续把 full hidden buffer 强行塞给 local attention 输出使用；
- 这正是 Problem 3 的契约错位来源。

因此，这一轮修复不应再复用 `hidden_state` 作为 pre-`o_proj` buffer，而应显式分开：

- `hidden_state`：full hidden 输入
- `attn_out_local`：local hidden shard 输出
- `o_proj(attn_out_local)`：恢复 full hidden

### 五、除 prefill 以外，还需要同步核对 decode 路径

虽然这次 traceback 首先出现在 prefill 的 `flash_attn_varlen_func(...)` 写回阶段，但 decode 路径也必须遵守同一套 local/full hidden contract。

原因是：

- `paged_attention(...)` 的输入 `q` 也是 local heads；
- 如果 wrapper 仍按 `model_config.num_q_heads` / `model_config.num_kv_heads` 这样的全局 head 数去组织 grid 和中间缓冲区，那么 decode 路径会继续把 TP local shard 当成 full heads 来解释；
- 同样地，KV cache 的写入 wrapper 若仍按全局 KV head 数解释本 rank 的 `k / v`，也会让 TP 语义继续错位。

因此，本轮推荐修复不仅要改 prefill 路径的 reshape，还要把 attention 相关 wrapper 的 head 数来源统一收口到 **运行时张量的 local shape**，让：

- prefill attention
- decode paged attention
- KV cache 写入

三者在 TP 下都遵守同一套 local shard 契约。

### 六、推荐的最小改法

本轮推荐做 **局部 contract 修正**，而不是大改架构：

1. 修改 `swiftllm/worker/models/llama.py`
   - 不再把 `hidden_state` 传入 `self.attn(...)` 作为输出 buffer；
   - 让 `self.attn(q, k, v)` 直接返回 `attn_out_local`；
   - 再由 `self.o_proj(attn_out_local)` 把 local hidden 聚合回 full hidden。

2. 修改 `swiftllm/worker/layers/attention.py`
   - 在 `Attention.forward()` 内部显式创建 local attention 输出缓冲区；
   - prefill 路径把 `flash_attn_varlen_func(...)` 的结果 reshape 成 `[num_prefill_tokens, local_hidden_size]`；
   - decode 路径把 `paged_attention(...)` 的输出也写入同一个 local hidden buffer；
   - 最终返回整个 `attn_out_local`。

3. 修改 attention 相关 kernel wrapper 的 head 数来源
   - `paged_attention(...)` 的 Python wrapper 不再盲目依赖 `model_config.num_q_heads` / `model_config.num_kv_heads` 这些全局值来解释当前 rank 的输入张量；
   - `store_kvcache(...)` wrapper 也应按当前 `k / v` 的 local head shape 传递 head 数；
   - 这样可以在不重写 Triton kernel 主体的前提下，让 TP 下的 wrapper 与 local shard 语义重新对齐。

### 七、本轮明确不做的事

这一轮不建议做下面这些事：

- 不修改 `profile_num_blocks()` 的业务目标；
- 不为了绕过报错而给 profile 路径单独加特判；
- 不重写 scheduler；
- 不重写 KV cache 分配策略；
- 不回退 fused / packed QKV；
- 不直接重构底层 Triton kernel 主体；
- 不通过把 TP 改回 1 来规避问题。

原因很简单：

> 当前真正出错的是 attention 的 shape contract，而不是 profile、scheduler 或 KV cache 机制本身的业务目标。

### 八、计划修改的文件

本轮实际修复时，预计至少会涉及：

- `swiftllm/worker/models/llama.py`
- `swiftllm/worker/layers/attention.py`

为了让 decode / KV cache 路径也遵守同一套 TP local shard 语义，还需要同步核对并按最小改动修正：

- `swiftllm/worker/kernels/paged_attn.py`
- `swiftllm/worker/kernels/kvcache_mgmt.py`

### 九、实现时必须写进代码注释的要点

用户已明确要求代码里保留详细中文注释。因此在真正改代码时，必须至少解释清楚下面几点：

1. 为什么 TP attention 的中间输出是 **local hidden shard**，而不是 full hidden；
2. 为什么 `flash_attn_varlen_func(...)` 的输出不能再按 `o.shape[-1]` 这种 full-width 目标去 reshape；
3. 为什么不能继续复用 `hidden_state` 作为 pre-`o_proj` attention 输出 buffer；
4. 为什么 `o_proj(RowParallelLinear)` 才是从 local hidden 回到 full hidden 的边界；
5. 为什么本轮不修改 `profile_num_blocks()`、scheduler 或 KV cache 设计，而只修 attention shape contract；
6. 为什么 decode / KV cache wrapper 也必须同步使用 local head 数，而不是继续假定全局 head 数。

### 十、验证方案

修复完成后，建议至少做下面几步验证：

1. `python -m compileall swiftllm`
2. 重新运行：
   - `python /home/yxlin/github/swift/swiftLLM/examples/offline.py --model-path /mnt/hdd/data/yxlin/huggingface/Meta-Llama-3.1-8B`
3. 保持：
   - `tensor_parallel_size = 2`
4. 重点确认：
   - `profile_num_blocks()` 不再在 synthetic prefill 阶段触发 shape mismatch；
   - prompt phase 的 KV cache 写入与 prefill attention 能正常工作；
   - decode phase 的 paged attention 也能在 TP=2 下跑通；
   - `tensor_parallel_size = 1` 与 `tensor_parallel_size = 2` 均至少可以稳定完成同一组 prompts 的生成。
