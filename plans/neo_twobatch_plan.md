# NEO 双 sub-batch pipeline 代码级讲解

这份说明只聚焦 **worker 侧在 `len(batches) == 2` 时的双 sub-batch pipeline**。重点不是调度器怎样决定“要不要切成两个 sub-batch”，而是：**一旦 worker 已经拿到两个 `SubBatch`，代码到底怎样让它们在同一串 transformer layer 上交错前进。**

本文主要对应以下代码：

- `swiftllm/worker/model.py:278-294`：`_forward_pipeline()` 主入口
- `swiftllm/worker/layers/transformer_layer.py:451-501`：启动 / 稳态 / 收尾
- `swiftllm/worker/layers/transformer_layer.py:158-370`：QKV 传输、attention、swap、postproj/preproj 细节
- `swiftllm/structs.py:254-293`：`SubBatch` 在 forward 前整理出的关键字段
- `swiftllm/worker/buffer.py:14-37`：共享 attention/residual buffer
- `swiftllm/worker/block_swapper.py:21-115`：GPU KV、CPU swap、CPU QKV/O buffer

---

## 1. 先抓住整条主调用链

worker 真正跑双 batch 的主线非常短，核心就在 `swiftllm/worker/model.py`。

### 1.1 入口：`_forward_batches()`

`swiftllm/worker/model.py:298-330`

它做三件事：

1. `_prepare_inputs(batches)`：把每个 `SubBatch` 需要的 GPU 侧输入张量准备好。
2. `pre_layer.forward(...)`：先把两个 sub-batch 的 token 一起变成 embedding。
3. 根据 `len(batches)` 分流：
   - `len == 1` 走 `_forward_sequential()`
   - `len == 2` 走 `_forward_pipeline()`

也就是说，**双 batch pipeline 只发生在 transformer body 内部**。`pre_layer` 和最后的 `post_layer` 都不属于双 batch 交错主体。

### 1.2 双 batch 主体：`_forward_pipeline()`

`swiftllm/worker/model.py:278-294`

```python
q1, k1, v1 = self.transformer_layers[-1].forward_first_stage(embeddings, batches)
for layer in self.transformer_layers[:-1]:
    q1, k1, v1 = layer.forward_double(q1, k1, v1, batches)
embeddings = self.transformer_layers[-1].forward_last_stage(q1, k1, v1, batches)
```

主结构可以直接读成三段：

1. **启动阶段**：`forward_first_stage()`
2. **稳态阶段**：对 `transformer_layers[:-1]` 逐个执行 `forward_double()`
3. **收尾阶段**：`forward_last_stage()`

这已经揭示了 NEO 双 batch pipeline 的本质：

- 不是“两份完整模型并行跑”
- 而是**同一串 layer object**，让两个 sub-batch 处在错位状态中向前推进
- 一个 batch 在做某层 `attention`
- 另一个 batch 同时在做该层 `postproj` 和下一层 `preproj`

---

## 2. 看懂后文前，先记住最小状态模型

如果你只想理解“两个 sub-batch 到底怎样交错”，真正需要记住的状态并不多。

## 2.1 `SubBatch` 里最关键的字段

定义在 `swiftllm/structs.py:254-293`。

### `iter_width`

`iter_width = self.perfdata.s`，表示这个 sub-batch 本轮 forward 会处理多少个 token 位置。

对于一个混合 batch：

- prefill request 会贡献它的整段 prompt token
- decode request 只贡献 1 个 token

所以 `iter_width` 不是“请求数”，而是**本轮 transformer 真正看到的 token 数**。

### `num_prefs / num_prgds / num_gdecs / num_cdecs`

这些字段把一个 `SubBatch` 内部划成几段：

- `num_prefs = num_cprfs + num_gprfs`
  - 前面这段是 **prefill 区域**
- `num_prgds = num_prefs + num_gdecs`
  - 到这里为止是会在 GPU paged attention 中参与的请求
- `num_cdecs`
  - 最后这段是 **CPU decode 区域**

简单说：一个 `SubBatch` 并不是“纯 prefill”或“纯 decode”，而可能同时包含：

1. CPU-prefill 请求
2. GPU-prefill 请求
3. GPU-decode 请求
4. CPU-decode 请求

### `sum_pref_toks / sum_prgd_toks`

- `sum_pref_toks`：prefill 区域总 token 数
- `sum_prgd_toks`：prefill + GPU decode 这两段合起来的 token 边界

这两个边界在 `swiftllm/worker/layers/transformer_layer.py:_attention()` 里直接决定切片：

- `q[:sum_pref_toks]` / `k[:sum_pref_toks]` / `v[:sum_pref_toks]`：prefill attention
- `q[sum_pref_toks:sum_prgd_toks]`：GPU decode paged attention
- `q[-num_cdecs:]`：CPU decode 需要传去 CPU 的尾部 QKV

所以从 worker 视角，一个 `SubBatch` 的 token 排列可以近似理解成：

```text
|<------ prefill tokens ------>|<-- gpu decode one-token rows -->|<-- cpu decode one-token rows -->|
0                         sum_pref_toks                    sum_prgd_toks                      iter_width
```

## 2.2 每个 batch 在流水里真正携带的三种“计算状态”

在理解 pipeline 时，把每个 batch 抽象成三种状态最有用：

### 状态 A：`Embeddings(Bx, Li)`

表示 batch `Bx` 当前拿着某层的输入 embedding，下一步应该做 `_preproj()`。

### 状态 B：`QKV(Bx, Li)`

表示 batch `Bx` 已经完成 `_preproj()`，已经有了该逻辑层 `Li` 的 `q / k / v`，下一步应该做 `_attention()`。

### 状态 C：`AttnOut(Bx, Li)`

表示 batch `Bx` 已经做完该逻辑层 `Li` 的 `_attention()`，attention 输出已经落在 `batch.attn_out_buf` 中，下一步应该做 `_postproj()`。

这三种状态正对应代码里的三段：

- `_preproj()`：`Embeddings -> QKV`
- `_attention()`：`QKV -> AttnOut`
- `_postproj()`：`AttnOut -> Embeddings(next layer)`

后文整个 pipeline 都可以用这三种状态来追踪。

## 2.3 `attn_out_buf` / `residual_buf` 为什么能跨阶段保留

这是一个很容易误读的点。

在 `swiftllm/worker/model.py:249-253` 里，`_prepare_inputs()` 先给每个 batch 建了 `torch.zeros(...)`。但这不是最终真正使用的 backing storage。

真正决定 buffer 布局的是 `swiftllm/worker/buffer.py:26-37`：

```python
batch.attn_out_buf = self.attn_out_buf[offs: offs + batch.iter_width]
batch.residual_buf = self.residual_buf[offs: offs + batch.iter_width]
```

也就是说：

- `batch.attn_out_buf`
- `batch.residual_buf`

最终都只是 **共享大 buffer 的切片 view**。

这很重要，因为 pipeline 里会出现这种情况：

- batch0 刚完成某层 attention，输出先留在 `attn_out_buf`
- 过一会儿另一个 batch 在同一 layer object 上执行别的动作
- 再之后 batch0 才回来做 `_postproj()`

如果 `attn_out_buf` 不是稳定保留的 slice，这个 pipeline 根本立不住。

---

## 3. 为什么偏偏是 `transformer_layers[-1]` 负责启动和收尾

这是整段代码最容易第一眼看错的地方。

`swiftllm/worker/model.py:166-178` 初始化 layer object 的方式是：

```python
LlamaTransformerLayer(
    ...,
    self.weight.layers[layer_id],
    self.weight.layers[layer_id + 1 - self.model_config.num_layers],
    ...,
    layer_id
)
```

这意味着每个 `LlamaTransformerLayer` 对象同时带两份权重：

- `self.weight`：当前逻辑层 `Li`
- `self.next_layer_weight`：下一逻辑层 `L(i+1)`，并且带有环绕

所以最后一个对象 `transformer_layers[-1]` 的特殊性是：

- `self.weight = L(N-1)`
- `self.next_layer_weight = L0`

于是它天然站在“最后一层和第一层的接缝处”。

### 3.1 它为什么能负责启动

启动阶段要先让某个 batch 进入 **`preproj(L0)`**。而最后一个对象因为握着 `next_layer_weight = L0`，所以它可以通过：

- `forward_first_stage()`
- 内部调用 `_preproj(..., layer_off=1)`

直接做出逻辑层 `L0` 的 QKV。

### 3.2 它为什么能负责收尾

收尾阶段又需要有人完成最后逻辑层 `L(N-1)` 的 `_postproj()`。最后一个对象刚好又握着：

- `self.weight = L(N-1)`

因此 `forward_last_stage()` 自然也应该落在它身上。

### 3.3 这不代表“最后一层先执行”

一定要把这个误区拆开：

- `transformer_layers[-1]` **不是**“逻辑最后一层先跑”
- 它只是一个**包装对象**，恰好同时持有 `L(N-1)` 和 `L0` 的衔接信息
- 启动时它是借 `next_layer_weight` 做 `L0`
- 收尾时它是借 `self.weight` 做 `L(N-1)`

所以看起来是“最后一个对象先被调用”，但逻辑层序仍然是正常的 `L0 -> L1 -> ... -> L(N-1)`。

---

## 4. `forward_double()` 的核心不变量

要看懂稳态阶段，最好的办法不是死背 stage 0 / stage 1，而是先抓住每次 `forward_double()` 的输入输出不变量。

`swiftllm/worker/layers/transformer_layer.py:430-448`

### 进入一次 `forward_double()` 之前

在某个逻辑层边界 `Li` 上，总有如下状态：

- `B0 = AttnOut(B0, Li)`
  - 即 batch0 已经完成层 `Li` 的 attention，输出留在 `attn_out_buf`
- `B1 = QKV(B1, Li)`
  - 即 batch1 已经完成层 `Li` 的 preproj，正拿着 `q/k/v`

### 退出一次 `forward_double()` 之后

状态整体前推一层，变成：

- `B0 = AttnOut(B0, L(i+1))`
- `B1 = QKV(B1, L(i+1))`

也就是说，**一次 `forward_double()` 会把两个 batch 都推进一层，但两者保持错位**：

- 一个永远停在 `AttnOut`
- 另一个永远停在 `QKV`

这就是 pipeline 能持续滚动的关键。

---

## 5. `forward_double()` 内部到底怎么交错

### 5.1 第一半：`_forward_pipeline_stage(..., cur_stage=0)`

对应 `swiftllm/worker/layers/transformer_layer.py:397-427`。

核心代码：

```python
self._transfer_qkv(q1, k1, v1, batches[cur_stage^1], cur_stage=cur_stage)
self._swap_out_blocks(batches[cur_stage])
e0 = self._postproj(batches[cur_stage])
q0, k0, v0 = self._preproj(e0, batches[cur_stage], layer_off=1)
self._attention(q1, k1, v1, batches[cur_stage^1], cur_stage=cur_stage)
```

当 `cur_stage=0` 时：

- `batches[cur_stage] = batches[0] = B0`
- `batches[cur_stage^1] = batches[1] = B1`

所以这一半做的是：

#### 对 `B0`

1. `_swap_out_blocks(B0)`
2. `_postproj(B0)` —— 用当前层 `Li` 的 `self.weight`
3. `_preproj(B0, layer_off=1)` —— 用下一层 `L(i+1)` 的 `next_layer_weight`

结果：`B0` 从 `AttnOut(B0, Li)` 推进成 `QKV(B0, L(i+1))`

#### 对 `B1`

1. `_transfer_qkv(q1, k1, v1, B1)`
2. `_attention(q1, k1, v1, B1, cur_stage=0)`

结果：`B1` 从 `QKV(B1, Li)` 推进成 `AttnOut(B1, Li)`

所以第一半结束后，两个 batch 的角色临时互换为：

- `B0 = QKV(B0, L(i+1))`
- `B1 = AttnOut(B1, Li)`

### 5.2 第二半：`_forward_pipeline_stage(..., cur_stage=1)`

接着 `forward_double()` 立刻做第二半：

```python
q1, k1, v1 = self._forward_pipeline_stage(q0, k0, v0, batches, cur_stage=1)
```

此时：

- 输入的 `q0/k0/v0` 就是刚才 `B0` 新产生的 `QKV(B0, L(i+1))`
- `cur_stage=1` 时：
  - `batches[cur_stage] = batches[1] = B1`
  - `batches[cur_stage^1] = batches[0] = B0`

于是第二半变成：

#### 对 `B1`

1. `_swap_out_blocks(B1)`
2. `_postproj(B1)` —— 仍然是当前 layer object 的 `self.weight`，也就是 `Li`
3. `_preproj(B1, layer_off=1)` —— 变成 `L(i+1)`

结果：`B1` 从 `AttnOut(B1, Li)` 推进成 `QKV(B1, L(i+1))`

#### 对 `B0`

1. `_transfer_qkv(q0, k0, v0, B0)`
2. `_attention(q0, k0, v0, B0, cur_stage=1)`

这里注意：`_attention()` 内部会计算：

```python
cur_layer_id = (self.layer_id + cur_stage) % self.model_config.num_layers
```

所以当 `cur_stage=1` 时，attention 真正执行的是 `L(i+1)`。

结果：`B0` 从 `QKV(B0, L(i+1))` 推进成 `AttnOut(B0, L(i+1))`

### 5.3 于是整个 `forward_double()` 的输出就是

- `B0 = AttnOut(B0, L(i+1))`
- `B1 = QKV(B1, L(i+1))`

也就是回到了和进入前同形的错位状态，只是逻辑层整体前进了一层。

这就是所谓的**稳态不变量**。

---

## 6. 一个 layer object 为什么能同时做 `postproj(i)` 和 `preproj(i+1)`

这也是整段代码最容易忽略的精妙点。

### `_postproj()` 用的是当前层权重

`swiftllm/worker/layers/transformer_layer.py:358-370`

```python
o = linear(batch.attn_out_buf, self.weight.o_proj)
...
ug = linear(o, self.weight.up_gate_proj)
...
embeddings = linear(..., self.weight.down_proj)
```

所以 `_postproj()` 明确使用 `self.weight`，对应当前逻辑层 `Li`。

### `_preproj(..., layer_off=1)` 用的是下一层权重

`swiftllm/worker/layers/transformer_layer.py:199-255`

```python
weight = self.weight if not layer_off else self.next_layer_weight
```

当 `layer_off=1` 时，`_preproj()` 就切到了 `next_layer_weight`，也就是 `L(i+1)`。

### 不只是权重变了，KV 写入层号也偏移了

同一个函数里还有：

```python
gpu_layer = (self.layer_id + layer_off) % self.model_config.num_layers
```

所以 `layer_off=1` 不只是“线性层权重换成下一层”，连 **prefill KV 写入 GPU KV cache 的逻辑层号** 也一起偏移到下一层。

因此这个 layer object 确实不是单纯“处理一层”。在 pipeline 模式下，它更像是当前层与下一层之间的**桥接节点**：

- 用当前层权重吃掉 `AttnOut(Li)`
- 再立刻用下一层权重产出 `QKV(L(i+1))`

---

## 7. 用一个 3 层例子完整走一遍：`B0/B1 × L0/L1/L2`

下面用最小但完整的例子讲清整个流水。

假设模型只有 3 个逻辑层：

- `L0`
- `L1`
- `L2`

两个 sub-batch：

- `B0`
- `B1`

我们约定记号：

- `QKV(Bx, Lk)`：`Bx` 已完成 `Lk` 的 `_preproj()`
- `AttnOut(Bx, Lk)`：`Bx` 已完成 `Lk` 的 `_attention()`，结果在 `attn_out_buf`
- `Final(Bx)`：`Bx` 已完成最后层 postproj，得到 transformer body 输出

## 7.1 启动阶段：`forward_first_stage()`

对应 `swiftllm/worker/layers/transformer_layer.py:451-474`。

核心代码：

```python
q0, k0, v0 = self._preproj(embeddings[0], batches[0], layer_off=1)
...
self._transfer_qkv(q0, k0, v0, batches[0], cur_stage=1)
q1, k1, v1 = self._preproj(embeddings[1], batches[1], layer_off=1)
...
self._attention(q0, k0, v0, batches[0], cur_stage=1)
return q1, k1, v1
```

这里调用者是 `transformer_layers[-1]`，也就是那个持有：

- `self.weight = L2`
- `self.next_layer_weight = L0`

的对象。

所以两次 `_preproj(..., layer_off=1)` 实际上都在做 **`preproj(L0)`**。

### 启动时发生了什么

1. 先把 `B0` 的 `embeddings0` 做成 `QKV(B0, L0)`
2. 然后启动 `B0` 的 `_attention(..., cur_stage=1)`
   - 因为 `cur_layer_id = (layer_id + 1) % 3 = 0`
   - 所以这里跑的确实是 `attention(L0)`
3. 与此同时，把 `B1` 的 `embeddings1` 也做成 `QKV(B1, L0)`
4. 但 `B1` 还没做 attention，`forward_first_stage()` 直接把它的 `q1/k1/v1` 返回给上层

### 启动结束后的状态

- `B0 = AttnOut(B0, L0)`
- `B1 = QKV(B1, L0)`

这正是后续稳态循环的起始不变量。

---

## 7.2 稳态阶段一：`layer0.forward_double()`

现在进入 `transformer_layers[:-1]` 的第一个对象，假设它对应逻辑桥接 `L0 -> L1`。

### 第一半（`cur_stage=0`）

- `B0`：`postproj(L0) -> preproj(L1)`
- `B1`：`attention(L0)`

阶段结束时：

- `B0 = QKV(B0, L1)`
- `B1 = AttnOut(B1, L0)`

### 第二半（`cur_stage=1`）

- `B1`：`postproj(L0) -> preproj(L1)`
- `B0`：`attention(L1)`

阶段结束时：

- `B0 = AttnOut(B0, L1)`
- `B1 = QKV(B1, L1)`

可以看到，经过一次 `forward_double()`，两个 batch 都从 `L0` 边界推进到了 `L1` 边界。

---

## 7.3 稳态阶段二：`layer1.forward_double()`

同理，第二个对象对应桥接 `L1 -> L2`。

### 第一半

- `B0`：`postproj(L1) -> preproj(L2)`
- `B1`：`attention(L1)`

中间状态：

- `B0 = QKV(B0, L2)`
- `B1 = AttnOut(B1, L1)`

### 第二半

- `B1`：`postproj(L1) -> preproj(L2)`
- `B0`：`attention(L2)`

阶段结束时：

- `B0 = AttnOut(B0, L2)`
- `B1 = QKV(B1, L2)`

这就是最后一个逻辑层边界。

---

## 7.4 收尾阶段：`forward_last_stage()`

对应 `swiftllm/worker/layers/transformer_layer.py:477-501`。

这里再次回到 `transformer_layers[-1]`，它的 `self.weight` 正是 `L2`。

核心代码：

```python
self._transfer_qkv(q1, k1, v1, batches[1], cur_stage=0)
self._swap_out_blocks(batches[0])
e0 = self._postproj(batches[0])
self._attention(q1, k1, v1, batches[1], cur_stage=0)
e1 = self._postproj(batches[1])
return torch.cat((e0, e1))
```

### 收尾时发生了什么

- `B0` 现在已经是 `AttnOut(B0, L2)`，所以它只差最后一个 `_postproj(L2)`，直接得到 `Final(B0)`
- `B1` 现在还是 `QKV(B1, L2)`，所以它还要：
  1. `_attention(L2)`
  2. `_postproj(L2)`
  才能得到 `Final(B1)`

### 收尾结束后的状态

- `Final(B0)`
- `Final(B1)`

最后 `torch.cat((e0, e1))` 把两段输出拼回去，交给 `post_layer.forward(...)`。

---

## 8. 真正的异构重叠发生在哪里

前面讲的是“两个 batch 的状态怎样交错”。但 NEO 的 worker 代码还有另一层交错：**GPU attention、CPU decode、QKV 传输、block swap** 之间的重叠。

下面只讲和双 batch 理解直接相关的重叠关系。

## 8.1 `_transfer_qkv()`：把 CPU decode 需要的尾部 QKV 送到 CPU

`swiftllm/worker/layers/transformer_layer.py:158-179`

只有当 `batch.num_cdecs > 0` 时，这一步才有意义。

它做的事情是：

1. 先 `self._comm_wait_compute()`
   - `cpu_communication_stream.wait_stream(default_stream)`
   - 表示通信流必须等默认计算流上的 QKV 先算完
2. 在 `cpu_communication_stream` 上，把：
   - `q[-num_cdecs:]`
   - `k[-num_cdecs:]`
   - `v[-num_cdecs:]`
   拷到 `Swapper` 里 pinned CPU buffer：
   - `q_cpu`
   - `k_cpu`
   - `v_cpu`
3. 记录 `qkvtr_e` 事件

这说明：**CPU decode 不是另起一个 batch**，而是同一 sub-batch 尾部那一小段 decode token，走了 CPU attention 路径。

## 8.2 `_swap_out_blocks()`：把需要下放的 prefilling KV block 异步搬到 CPU

`swiftllm/worker/layers/transformer_layer.py:181-197`

只有 `batch.num_cprfs > 0` 才会触发。

它在 `cpu_communication_stream` 上调用：

```python
self.swapper.swap_blocks(...)
```

而 `Swapper` 的底层资源在 `swiftllm/worker/block_swapper.py:21-115` 里：

- GPU KV cache：`k_cache / v_cache`
- CPU KV swap 区：`k_swap / v_swap`
- CPU QKV/O buffer：`q_cpu / k_cpu / v_cpu / o_cpu`
- GPU / CPU block table：`gpu_block_table / cpu_block_table`

所以这里不是抽象意义的“swap 一下”，而是真的在 GPU KV 与 CPU KV 区域之间移动 block。

## 8.3 `_attention()`：把 prefill、GPU decode、CPU decode 三条路径拼起来

`swiftllm/worker/layers/transformer_layer.py:258-355`

它内部其实按顺序组织了三段 attention：

### 第 1 段：prefill attention

- 如果 GPU 架构是 Ampere+，用 flash attention
- 否则走自定义 `prefill_attention`

处理切片：

- `q[:sum_pref_toks]`
- `k[:sum_pref_toks]`
- `v[:sum_pref_toks]`

输出写到：

- `o[:sum_pref_toks]`

### 第 2 段：GPU decode paged attention

如果 `batch.num_gdecs > 0`，调用：

```python
paged_attention(..., cur_layer_id, ...)
```

处理切片：

- `q[sum_pref_toks:sum_prgd_toks]`

访问的是 GPU KV cache：

- `self.swapper.k_cache`
- `self.swapper.v_cache`
- `self.swapper.gpu_block_table`

### 第 3 段：CPU decode attention

如果 `batch.num_cdecs > 0`：

1. 先等 `qkvtr_e.synchronize()`
   - 保证尾部 QKV 已经从 GPU 拷到 CPU pinned buffer
2. 调用：

```python
torch.ops.pacpu.paged_attention_cpu(...)
```

它消费的是：

- `q_cpu / k_cpu / v_cpu`
- `k_swap / v_swap`
- `cpu_block_table`

输出落在：

- `o_cpu`

3. 再在 `cpu_communication_stream` 上把 `o_cpu` 异步拷回：

```python
o[-batch.num_cdecs:, :].copy_(oc, non_blocking=True)
```

也就是写回到当前 batch 的 `attn_out_buf` 尾部区域。

### 最后的 join 点

函数尾部统一执行：

```python
self._compute_wait_comm()
```

即：

- 默认计算流等待 `cpu_communication_stream`

所以后续 `_postproj()` 看到的 `attn_out_buf`，已经同时包含：

- prefill attention 的输出
- GPU decode attention 的输出
- CPU decode attention 回传后的输出

## 8.4 `_preproj()` 里为什么还要插一个等待

`swiftllm/worker/layers/transformer_layer.py:234-253`

在 prefill KV 要写入 GPU KV cache 前，代码会先：

```python
self._compute_wait_comm()
```

注释写得很直白：

> if swapping is too slow, there's risk that we writes to what's being swapped out

也就是说，**store_kvcache 和 swap_blocks 在物理存储上可能打架**。所以这里先把 communication stream 上尚未完成的 swap 等掉，再执行 `store_kvcache(...)`，避免覆盖或竞争。

---

## 9. `cur_stage` 和 `events[0]/events[1]` 为什么不是 batch 编号

这是第二个非常容易误读的点。

很多人第一次看会下意识觉得：

- `events[0]` 对应 batch0
- `events[1]` 对应 batch1

但这不对。

### 9.1 `cur_stage` 表示的是 pipeline stage slot

看 `_forward_pipeline_stage()` 就很清楚：

- `cur_stage=0` 时
  - `batches[cur_stage] = B0`
  - `batches[cur_stage^1] = B1`
- `cur_stage=1` 时
  - `batches[cur_stage] = B1`
  - `batches[cur_stage^1] = B0`

说明 `cur_stage` 只是“这一次半阶段里谁在做 postproj/preproj，谁在做 attention”的**槽位选择器**。

### 9.2 `cur_stage` 还会直接影响逻辑层号

更关键的是 `_attention()` 里这句：

```python
cur_layer_id = (self.layer_id + cur_stage) % self.model_config.num_layers
```

所以 `cur_stage` 根本不只是 profiling index，它还参与决定：

- 当前 `_attention()` 正在执行逻辑层 `Li`
- 还是逻辑层 `L(i+1)`

这也是为什么：

- 同一个 layer object 在 `cur_stage=0` 和 `cur_stage=1` 两次调用中
- attention 会分别落在相邻两层的逻辑编号上

### 9.3 `events[0]/events[1]` 只是两个 stage slot 的性能槽

因此：

- `events[0] / events[1]` 记录的是两次半阶段的事件
- 不是固定绑定到某个 batch
- 某个 batch 在第一半和第二半里会交换角色
- 所以也不可能和某个固定 event slot 一一对应

---

## 10. 把整条流水压缩成一句话

如果要用一句话概括 NEO 的 worker 双 sub-batch pipeline，可以这样说：

> 它让一个 batch 始终停在“当前层 attention 已完成、等待 postproj”的状态，另一个 batch 始终停在“当前层 preproj 已完成、等待 attention”的状态；每次 `forward_double()` 先让前者做 `postproj(i)->preproj(i+1)`、同时让后者做 `attention(i)`，再交换角色，于是两个 batch 以错位方式沿着同一串 layer object 持续前推。

---

## 11. 最容易误读的点集中纠正

### 误解 1：`transformer_layers[-1]` 说明最后一层先跑

不对。

它只是那个同时持有：

- `L(N-1)` 的 `self.weight`
- `L0` 的 `next_layer_weight`

的包装对象，所以负责启动和收尾。

### 误解 2：双 batch pipeline 是两份模型并行跑

不对。

真正情况是：**同一串 layer object 上，两个 batch 维持错位状态滚动前进。**

### 误解 3：`events[0] / events[1]` 就是 batch0 / batch1

不对。

它们是 pipeline 的两个 stage slot，不是 batch id。

### 误解 4：一个 layer object 只负责一个逻辑层

在顺序模式里近似可以这么看；但在 pipeline 模式里不行。

因为它会在同一次 `forward_double()` 中串起：

- `postproj(i)`
- `preproj(i+1)`
- `attention(i)` 或 `attention(i+1)`

### 误解 5：CPU decode 是额外拆出来的第三个 batch

不对。

CPU decode 仍然属于当前 sub-batch，只是它尾部那段 token：

- 先把 QKV 拷到 CPU
- 在 CPU 上做 `paged_attention_cpu`
- 再把输出拷回 `attn_out_buf`

### 误解 6：`attn_out_buf` / `residual_buf` 是每个 batch 各自新分配的独立张量

最终不是。

真实 forward 中，它们会在 `ModelForwardBuffers.alloc_for_batches()` 里被改成共享大 buffer 的 slice view。

### 误解 7：QKV 传输、swap、GPU attention、CPU attention 是完全串行的

不对。

这段代码里有明确的重叠和同步边界：

- `_comm_wait_compute()`
- `qkvtr_e.synchronize()`
- `_compute_wait_comm()`

也正因为有这些边界，CPU 通信流和默认 GPU 计算流之间才能既重叠又不踩数据。

---

## 12. 如果顺着源码自己追，建议的阅读顺序

建议按下面顺序跳源码：

1. `swiftllm/worker/model.py:298-330` 看 `_forward_batches()`
2. `swiftllm/worker/model.py:278-294` 看 `_forward_pipeline()`
3. `swiftllm/worker/layers/transformer_layer.py:451-474` 看 `forward_first_stage()`
4. `swiftllm/worker/layers/transformer_layer.py:430-448` 看 `forward_double()`
5. `swiftllm/worker/layers/transformer_layer.py:397-427` 看 `_forward_pipeline_stage()`
6. `swiftllm/worker/layers/transformer_layer.py:477-501` 看 `forward_last_stage()`
7. 再回头补：
   - `_preproj()`：199-255
   - `_attention()`：258-355
   - `_postproj()`：358-370
   - `_transfer_qkv()`：158-179
   - `_swap_out_blocks()`：181-197

这样读最不容易迷路。

---

## 13. 最后的直觉图

可以把整个 worker 双 batch pipeline 记成下面这张“错位推进图”：

```text
启动后：
B0 = AttnOut(L0)
B1 = QKV(L0)

一次 forward_double(L0->L1) 后：
B0 = AttnOut(L1)
B1 = QKV(L1)

再一次 forward_double(L1->L2) 后：
B0 = AttnOut(L2)
B1 = QKV(L2)

最后 forward_last_stage：
B0: postproj(L2) -> Final
B1: attention(L2) -> postproj(L2) -> Final
```

所以它的本质不是“两个 batch 一起做同一层”，而是：

- **一个 batch 永远比另一个 batch 多走半步**
- 这个“半步”正好就是 `attention` 与 `postproj->next preproj` 之间的错位
- 于是 GPU 计算、CPU 通信、CPU decode 能在同一条 layer 流水上被塞得更满
