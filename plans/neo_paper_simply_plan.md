# NEO 论文阅读笔记

## 论文信息
- **标题**: NEO: Saving GPU Memory Crisis with CPU Offloading for Online LLM Inference
- **作者**: Xuanlin Jiang, Yang Zhou, Shiyi Cao, Ion Stoica, Minlan Yu
- **会议**: MLSys 2025
- **主题方向**: 大模型推理系统、在线 LLM serving、CPU-GPU 协同推理、系统优化

---

## 一、背景动机

这篇论文关注的是 **在线 LLM 推理（online LLM inference）** 的系统问题。

当前大模型在线服务通常依赖 **request batching** 提高吞吐，但现实里批大小经常被 **GPU 显存** 限制住。原因主要有两个：

1. **模型参数本身就很大**，例如 70B 模型要占用大量显存。
2. **KV cache 持续增长**，并且其大小会随着输入长度和输出长度线性增加。

结果就是：虽然 GPU 算力很强，但因为显存不够，batch size 起不来，GPU 计算资源被浪费，形成论文所说的 **GPU memory crisis**。

### 现有方法的问题
已有工作大致有两类：

- **量化 / 稀疏化**：能省显存，但可能损伤精度。
- **CPU offloading**：把权重、KV cache 或部分计算搬到 CPU。

但过去的 offloading 方法大多存在几个问题：

- 往往牺牲 latency 来换 throughput；
- 频繁在 CPU/GPU 间搬运 KV cache，容易被 **PCIe 带宽** 卡住；
- 一些方法假设请求长度固定，依赖静态 profiling，不适合真实在线 workload；
- FastDecode 虽然面向在线场景，但 CPU 资源需求过高，成本不现实。

### 为什么需要这项研究
论文的目标很明确：

> 只利用 GPU 主机自带的本地 CPU 和内存，在不牺牲精度、尽量不增加延迟的前提下，提高在线 LLM 推理吞吐。

这个目标非常实际，因为公有云里 GPU 实例通常已经绑定了一定的 CPU 和内存资源，这些资源常常没被充分利用。

---

## 二、摘要分析

### 1. 主要研究问题
如何在在线 LLM 推理中，利用本地 CPU 缓解 GPU 显存瓶颈，从而提升 batch size 和吞吐，同时保持低延迟。

### 2. 关键术语
- Online LLM inference
- KV cache
- CPU offloading
- Asymmetric GPU-CPU pipelining
- Load-aware scheduling
- Selective batching
- Paged attention

### 3. 技术领域
属于 **大模型推理系统（LLM systems）/ MLSys / 推理优化** 方向。

### 4. 论文提出的解决方案
提出 **NEO**，核心思想是：

- 不把所有解码 attention 都搬到 CPU；
- 而是只把 **一部分请求** 的 decoding attention 和 KV cache 放到 CPU；
- 通过 **asymmetric pipelining** 和 **load-aware scheduling** 动态平衡 CPU 与 GPU 负载；
- 在不同 iteration 中自适应决定哪些请求在 GPU 跑、哪些请求 offload 到 CPU。

### 5. 主要贡献
可以概括为 4 点：

1. 提出一种适合在线场景的 **部分 offloading** 设计，而不是全量 offload。
2. 提出 **非对称 GPU-CPU pipeline**，减少 pipeline bubble，平衡 CPU/GPU。
3. 提出 **负载感知调度器**，适配真实动态 workload。
4. 在 T4 / A10G / H100 与 7B / 8B / 70B 模型上验证，吞吐显著提升且延迟基本不变。

---

## 三、摘要中文翻译

> 在线 LLM 推理支撑了许多令人兴奋的应用，例如智能聊天机器人和自主智能体。现代 LLM 推理引擎广泛依赖请求 batching 来提升推理吞吐，以便在昂贵的 GPU 加速器上实现成本可接受的部署。然而，GPU 显存受限在实践中严重限制了可达到的 batch size，导致大量 GPU 计算资源被浪费。
>
> 我们提出了 NEO，一个面向在线 LLM 推理的系统，它将部分 attention 计算与 KV cache 状态从 GPU 卸载到本地主机 CPU，从而有效提升 GPU batch size，并进一步提高推理吞吐。为此，NEO 提出了非对称的 GPU-CPU 流水线和负载感知调度，以平衡 GPU 和 CPU 的负载并充分利用两者的计算与内存资源。
>
> 我们在多种 workload（如代码生成、文本摘要）、多种 GPU（T4、A10G、H100）以及多种 LLM 模型（7B、8B、70B）上评估了 NEO。实验表明，在保持相同延迟的情况下，NEO 相比纯 GPU 方案在 T4、A10G 和 H100 上分别实现了最高 7.5×、26% 和 14% 的吞吐提升；当使用更强 CPU 时，在 A10G 上最高可获得 79.3% 的吞吐提升。为促进后续研究，我们开源了代码。

---

## 四、方法论分析

## 1. 方法总览

NEO 的整体架构包括：

- 一个运行在 CPU 上的 **scheduler**
- 分布在 CPU/GPU 上的 **executors**
- 三类队列：
  - prefilling waitqueue
  - GPU decoding runqueue
  - CPU decoding runqueue

调度器每个 iteration 都需要决定：

1. 哪些请求进入 GPU batch；
2. 哪些请求进入 CPU batch；
3. 是否需要 swap-in / swap-out KV cache；
4. 当前这一轮应该采用：
   - **GPU-only**
   - 或 **two-batch asymmetric pipelining**

---

## 2. 核心方法一：Asymmetric Pipelining

这是论文最关键的设计。

### 2.1 为什么不是简单 offloading
如果只是把 decoding attention 和 KV cache 丢给 CPU，GPU 做别的部分，那么 CPU 之外的时段会空转，资源利用率低。

### 2.2 为什么不是 symmetric pipelining
已有工作常做法是把一个 decoding batch 均分成两个子 batch，然后让 GPU 的 linear ops 和 CPU 的 attention ops 对称重叠。

但论文指出这样有三个问题：

1. **GPU 显存浪费**：大量 GPU 显存闲置，没有充分保留 KV cache。
2. **CPU 容易成为瓶颈**：CPU attention 往往比 GPU linear 阶段更慢。
3. **现实请求不规则**：请求长度不同，很难切成“对称”的两个 batch，bubble 很多。

### 2.3 NEO 的非对称设计
NEO 采取 **partial offloading**：

- 一部分请求仍然是 **GPU-request**
- 一部分请求变成 **CPU-request**

KV cache 也分成：

- **GPU-cache**
- **CPU-cache**

并且优先让更多请求保留在 GPU-cache 里，以最大化 GPU 显存利用。

### 2.4 两个非对称子 batch
NEO 每轮形成两个子 batch：

- **batch-0**：主要在 GPU 上跑，包含
  - prefill requests
  - GPU decoding requests
  - 少量 CPU decoding requests
- **batch-1**：主要在 CPU 上跑，包含
  - 大多数 CPU decoding requests

这样会形成一种非对称结构：

- batch-0 有较长的 GPU linear stage
- batch-1 有较长的 CPU attention stage

二者互补，从而更容易 overlap。

### 2.5 这个设计的直觉
可以把它理解成：

> GPU 继续负责它擅长的大量 dense computation，CPU 只接手“更偏 memory-bandwidth-bound 的 decoding attention”，并且只接一部分，不把自己压爆。

这是 NEO 相比 FastDecode 更关键的点。

---

## 3. 核心方法二：Load-Aware Scheduling

NEO 不只是提出 pipeline，还提出了在线调度器。

### 3.1 为什么需要动态调度
真实 workload 中：

- 输入长度不同
- 输出长度不同
- 请求到达时间不同

因此固定的离线最优策略很快失效。

### 3.2 调度原则
论文给出几个原则：

- **Greedy**：同时估计 GPU-only 和 asymmetric pipelining 两种方案，选吞吐更高的。
- **Balancing**：尽量让 CPU busy time 和 GPU busy time 接近。
- **Hiding CPU**：避免 CPU 忙而 GPU 闲。
- **Maximizing GPU**：尽量让 GPU 装更多请求。

### 3.3 调度目标
令 iteration 时间为 $T$，batch size 为 $x$，目标是最小化：

$$
T / x
$$

因为 transformer layer 时间占 iteration 的主要部分，所以近似只优化 layer 时间。

论文给出估计式：

$$
T \approx T_{tr} = L \times \left( \max\{T_{po0}+T_{pr0}, T_{ca1}\} + \max\{T_{po1}+T_{pr1}+T_{ga0}, T_{ca0}\} \right)
$$

其中：

- $T_{po}$：post-projection 时间
- $T_{pr}$：pre-projection 时间
- $T_{ga}$：GPU attention 时间
- $T_{ca}$：CPU attention 时间

并定义：

$$
T_{lx} = T_{pox} + T_{prx}
$$

调度时要尽量满足：

$$
T_{l0} \ge T_{ca1}
$$

以及

$$
T_{l1} + T_{ga0} \ge T_{ca0}
$$

直觉上就是：**让 CPU 的活尽量被 GPU 这边的活“遮住”。**

### 3.4 调度过程
每轮 iteration 大致做 6 步：

1. 初始化两个空 batch
2. 先放 GPU decoding requests 到 batch-0，并考虑 swap in/out
3. 再放 prefilling requests 到 batch-0
4. 扫描 CPU decoding queue，把请求分配到 batch-0 或 batch-1
5. 如有必要减少 prefilling requests，避免破坏平衡
6. 估计 GPU-only 与 two-batch 的 token rate，选更优者

论文说调度开销 **< 3 ms / iteration**，而单轮 iteration 通常是数百毫秒，所以额外开销较小。

---

## 五、实现细节

NEO 基于 **SwiftLLM** 实现，后者是一个简化版 vLLM。

### 1. CPU attention kernel
作者自己实现了 **PACPU (Paged-Attention-for-CPU)**：

- C++ torch extension
- 底层用 **ISPC** 生成 SIMD CPU kernel
- 支持 paged attention 所需的 block table / 非连续 tensor

优化重点是 memory-bound attention：

- 单核内用 SIMD load/store
- 多核间沿 request 维度拆任务
- 最后聚合 partial outputs

### 2. 降低 kernel launch overhead
由于 Python GIL 和 Triton-JIT 的额外开销，作者把大量 GPU kernel 改写为 **CUDA C++**，把 kernel launch 开销从每层 1.2ms 降到 0.6ms。

### 3. 多 GPU 支持
为了支持 70B 模型，SwiftLLM 被改造成支持：

- model sharding
- tensor parallelism
- Ray actors
- NCCL 通信

---

## 六、数据集

论文使用了三类 workload：

### 1. Azure Code trace (AC)
- 来自 Azure 生产环境的代码生成推理 trace
- 更贴近真实 coding assistant 场景
- 输入输出一般较长

### 2. OpenAI Summarization Comparisons (OSC)
- 文本摘要/聊天场景数据
- 相对更短请求
- 用在 T4 场景更合适

### 3. Synthetic workloads
- 人工控制 input/output 长度
- 用于系统性测试不同长度下的吞吐变化

### 为什么选这些数据集
因为论文关注的是 **真实在线 serving 场景**，所以必须覆盖：

- 长请求
- 短请求
- 真实 trace
- 可控合成 workload

这样才能说明方法不是只对某个特殊 case 有效。

---

## 七、实验设置

### 硬件
- AWS g4dn.4xlarge：T4
- AWS g5.2/4/8/16xlarge：A10G
- 本地 8×H100 服务器（2×H100 setting）

### 模型
- Llama-2-7B
- Llama-3.1-8B
- Llama-3.1-70B

### 基线
- **vLLM**
- **SwiftLLM**（NEO 的 GPU-only 版本）
- **FastDecode+**
  - 作者自实现的 FastDecode 风格 baseline
  - offload 所有 decoding attention

---

## 八、实验指标

论文主要看两个指标：

### 1. Average per-token latency
把单请求总 latency 除以输出 token 数，然后再求平均。它反映在线服务场景里的交互体验。

### 2. Throughput / Relative throughput
反映系统吞吐能力。很多图中以相对于 baseline 的比值表示。

### 总体关注点
核心就是：

- **在相同 latency 下，吞吐能否更高**
- 或者 **在相同负载下，latency 能否不变甚至更低**

---

## 九、实验结果解读

### 1. 与 vLLM 比：在线 latency-load 曲线
论文在 3 个场景下比较 NEO 和 vLLM：

- 2xH100 + 70B + AC
- A10G + 8B + AC
- T4 + 7B + OSC

结果：

- H100：在 2 sec latency 下，吞吐提升 **14.3%**
- A10G：在 2 sec latency 下，吞吐提升 **6.4%**
- T4：在 1 sec latency 下，吞吐提升 **563%**，接近 **6×**

#### 如何理解
- **低端 GPU（T4）收益最大**，因为它最受显存限制。
- **高端 GPU（H100）也有收益**，但没那么夸张，因为其原始 batch size 已经较高。

论文还给出摘要里的更广义结果：

- T4：最高 **7.5×**
- A10G：最高 **26%**
- H100：最高 **14%**

这些更像不同 workload / 长度配置下的最大值。

### 2. 延迟分布
NEO 与 vLLM 的 latency distribution 基本接近。这说明吞吐提升不是靠“暴力增大 batch 导致延迟恶化”换来的。

### 3. 与 FastDecode+ 比较
NEO 在 latency 和 throughput 上都优于 FastDecode+。

原因在于：

- FastDecode+ 会把所有 decoding attention 都扔给 CPU；
- 当 CPU queue 很重、prefill queue 很轻时，系统被 CPU 卡死；
- NEO 则可动态回退到 GPU-only，或者只部分 offload。

这说明 **“全部 offload” 并不是最优策略**，关键是动态平衡。

### 4. 不同输入输出长度
论文发现：

- 当输出长度较短时，NEO 可能略差于 baseline；
- 随着输出长度增加，NEO 收益上升；
- 达到 CPU/GPU 平衡点后，收益最大；
- 再继续增加输出长度，收益会下降，并逐渐接近 baseline。

这很符合系统直觉：如果输出太短，offloading 的管理成本不值得；如果输出适中，CPU 可以很好接住那部分 attention；如果输出过长，CPU 又容易成为瓶颈。

### 5. CPU 容量敏感性
在 A10G 上，换不同 EC2 instance：

- g5.2xlarge：最高 **12.2%**
- g5.4xlarge：最高 **13.3%**
- g5.8xlarge：最高 **29.7%**
- g5.16xlarge：最高 **79.3%**

论文结论是：

> **CPU memory bandwidth 比 CPU core 数更关键。**

因为 decoding attention 在 CPU 上主要是 **memory-bandwidth-bound**，不是 compute-bound。

这是一个很重要的系统洞察。

---

## 十、创新点

我认为这篇论文的创新点主要有 4 个。

### 创新点 1：不是“全 offload”，而是“部分 offload”
这是最核心的思想。作者意识到 CPU 不是越用越好，而是要 **控制 offload 比例**，让 CPU 不成为新瓶颈。

### 创新点 2：非对称流水线
不是把 batch 平分，而是构造两个功能互补的子 batch：

- 一个偏 GPU-heavy
- 一个偏 CPU-heavy

这种不对称设计更适合真实不规则 workload。

### 创新点 3：在线负载感知调度
不是固定策略，而是每轮 iteration 动态决策：

- 是否启用 offloading
- offload 哪些请求
- 放入哪个 batch

### 创新点 4：系统视角很强
它不是单纯优化某个 kernel，而是把：

- 内存层次
- CPU/GPU 协同
- 调度策略
- workload 动态性

放在一起联合设计。

---

## 十一、架构理解（通俗版）

可以把 NEO 想成这样：

- GPU 很强，但“桌面”太小，放不下太多请求的 KV cache；
- CPU 有很大的“仓库”，但干活慢；
- 所以 NEO 不让 CPU 接全部工作，只让它接一部分最适合它的活 —— decoding attention；
- 同时 GPU 继续做 dense computation 和 prefill；
- 调度器不断看两边谁忙谁闲，动态调整请求分流。

一句话总结：

> **NEO 的本质是：用 CPU 内存和带宽换取 GPU 可容纳 batch size，从而提高整体在线 serving 吞吐。**

---

## 十二、局限性与潜在弱点

### 1. 依赖 CPU 内存带宽
论文已经明确说明：收益很大程度上取决于 **CPU memory bandwidth**。如果宿主机 CPU 带宽较弱，收益会明显下降。

### 2. 对短输出 workload 不一定有利
如果请求输出太短，offload 带来的调度和 swap 开销可能抵不过收益。

### 3. Profiling 误差会影响调度
NEO 依赖离线 profiling + 线性插值估计时间。但真实 workload 的动态性会导致估计不准，进而造成 suboptimal scheduling。

### 4. 目前实现仍弱于生产级 vLLM 某些场景
作者承认在 2-GPU 设置下 SwiftLLM 比 vLLM 吞吐低 **8.8%**，说明底层工程成熟度还有限。

### 5. 不一定提升 perf-TCO
在云上，因为 CPU 资源已打包进实例价格，看起来“免费”；但在自建集群里，未必比纯 GPU 更优。

### 6. 能耗未充分评估
NEO 提升吞吐的一个代价是更多使用 CPU，能耗会增加。只是从云用户账单视角，它可能仍然划算。

---

## 十三、未来工作 / 改进空间

论文和我的理解中，未来可扩展方向包括：

### 1. 与 chunked prefill 结合
作者认为可以结合 Sarathi-Serve 风格的 chunked prefill，进一步做 finer-grained 的 CPU/GPU 平衡。

### 2. Offload 更多模块
当前主要 offload decoding attention。极端 workload 下，也许可以把部分 dense ops 也放到 CPU，但这需要进一步验证。

### 3. 支持 remote CPU workers
本论文只用 host CPU。未来可扩展到远程 CPU worker，但网络传输和成本会更复杂。

### 4. 集成到 vLLM / SGLang
如果能把 NEO 思想整合进生产级 serving engine，会更有实际价值。

### 5. 更准确的在线 cost model
当前主要依赖离线 profiling 和简单插值，未来可尝试在线学习式调度模型。

### 6. 更系统的能耗 / 成本分析
尤其是 throughput-per-watt、perf-per-dollar、不同云实例的最优配置。

---

## 十四、与相关工作的对比

### 1. 与 GPU-only 推理系统
代表：vLLM、SGLang、Orca、DistServe、NanoFlow

这些工作主要优化 GPU 内部效率，例如：

- continuous batching
- paged attention
- chunked prefill
- disaggregated prefill/decode

NEO 与它们不同的地方在于：

> 它不是只优化 GPU 内部，而是把 **CPU 也纳入推理数据面**。

### 2. 与离线 offloading 工作
代表：FlexGen、PowerInfer、TwinPilots、HeteGen、InstInfer

这些方法更多服务于：

- 长上下文
- 离线场景
- latency 不敏感场景

NEO 的区别是：

> 它强调 **online inference**，要求低延迟，不接受粗暴的吞吐换延迟。

### 3. 与 FastDecode
这是最接近的工作。二者都把 decoding attention 放到 CPU。

但 NEO 比 FastDecode 更进一步：

- **部分 offloading**，而不是全 offload
- **非对称 pipeline**
- **动态 load-aware scheduling**
- 更适配真实不规则 workload

所以它可以看作是：

> **FastDecode 路线的一个更实用、更系统化的演进版本。**

---

## 十五、研究价值评价

### 理论价值
这篇论文最重要的理论启发是：

> 在 LLM serving 中，瓶颈不只是 FLOPS，更是 memory hierarchy 与 resource balance。

它提醒我们不要只盯着 GPU 算力，而要看：

- 显存大小
- CPU 内存容量
- CPU 内存带宽
- PCIe 通信
- 请求长度分布
- 调度策略

### 实际应用价值
很高。特别适合：

- 显存受限 GPU（如 T4、A10G）
- 成本敏感的在线 LLM 服务
- 云上部署场景

### 领域影响
这篇工作属于一个很重要的研究路线：

> **heterogeneous LLM serving / CPU-GPU collaborative inference**

未来随着 GPU 算力增长快于显存增长，这条路线很可能越来越重要。

---

## 十六、个人总结

如果只用一句话总结这篇论文：

> **NEO 通过“只把合适的一部分 decoding attention 和 KV cache 放到 CPU”，并结合非对称流水线与动态调度，在不明显增加延迟的前提下，缓解 GPU 显存瓶颈并显著提高在线 LLM 推理吞吐。**

### 我认为它最有价值的点
不是“CPU offloading”本身，而是这三个判断：

1. **不是所有东西都值得 offload**
2. **CPU/GPU 协同必须动态平衡**
3. **在线推理不能接受简单的 latency-for-throughput tradeoff**

### 一句话评价
这是一篇很典型的 **MLSys/系统论文**：问题抓得准，设计有清晰动机，实验覆盖较充分，系统 insight 也比较扎实。
