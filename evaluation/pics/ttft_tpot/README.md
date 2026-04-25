基于 reproduce-fig6b.py 脚本在AC数据集上测量 vllm vs NEO 在 TTFT, TPOT, Avg token latency 上差异

结果表示 vllm TTFT 比 NEO 高， TPOT 比 NEO 低。相关原因解释在 `NEO/plans/neo_ttft_tpot_result_plan.md`

> NEO 通过把一部分 decode attention 与 KV cache 放到 CPU 侧，扩大了可接纳的并发请求集合，显著降低高负载下的新请求排队时间，因此 TTFT 和端到端平均 token latency 更低；但 CPU decode、GPU/CPU 协同、KV swap 与 pipeline 失衡会拉长 decode 阶段的 inter-token interval，因此 TPOT 可能高于 vLLM。vLLM 在本实验中启用了 chunked prefill，decode 侧调度较强(即 decode request 优先)，所以高负载下 TPOT 可以优于 NEO，但它仍受 GPU KV cache 容量与 preemption/recompute/排队影响，TTFT 会快速恶化。

chunked-prefill 即使在不考虑 Pipeline Parallelism 情况下

chunked-prefill批处理的优势在于：

prefill 阶段可以搭载（piggyback）在 decode 阶段未被充分利用的算力上，提升整体算力利用率。
decode 阶段可以和 prefill 阶段共享一次权重读取，减少内存带宽压力，提高带宽利用率。
这样，GPU 的计算单元和内存带宽都能被更充分利用，整体吞吐和 QPS 明显提升。

下图展示了在 A6000 GPU 上运行 LLaMA-13B 模型，不同 batch 组合方式下每个 token 的处理时间（单位：毫秒）：

仅包含 prompt 的请求（prompt 长度为 1024，batch 大小为 4）；
仅包含 decode 的请求（batch 大小为 4，序列长度为 1024）；
一个混合 batch，包括 1 个长度为 1021 的 prefill 请求和 3 个 decode 请求。
结果表明，混合 batch 能将每个 token 的解码时间显著降低一个数量级，大幅提升整体推理效率；同时，prefill 阶段的耗时几乎没有变化。

![fig](https://pica.zhimg.com/v2-170293ee9e310c6afbcddf8d00d62458_1440w.jpg)

图片来源：SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked-Prefills

但是 vllm 开了 chunked-prefill 在吞吐量上也没比 NEO 高多少。