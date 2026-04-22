"""
Llama transformer layer 及其 worker-side timing 工具。

这个文件里不仅实现了一层 transformer 的 forward，还承担了 NEO worker 侧最细粒度的
性能测量职责：

- 线性 / post-layer 相关时间；
- GPU prefill attention 时间；
- GPU decode attention 时间；
- CPU decode attention 时间；
- pipeline 下 Python launch / 调度额外开销。

这些 layer-level timing 最终会被 `worker/model.py` 中的 `ModelPerfResult` 汇总，并由
server 侧 `ModelProfiler` 消费。
"""

import time
import torch
import torch.distributed as dist
import vllm_flash_attn_2_cuda as flash_attn_cuda
# import vllm_flash_attn

# pylint: disable=no-name-in-module
from swiftllm_c import \
    fused_add_rmsnorm_inplace, \
    silu_and_mul_inplace, \
    rotary_embedding_inplace, \
    store_kvcache#, paged_attention
    # linear, \

from swiftllm.model_config import LlamaModelConfig
from swiftllm.engine_config import EngineConfig
from swiftllm.worker.weight import LlamaTransformerLayerWeight
from swiftllm.worker.block_swapper import Swapper
from swiftllm.structs import SubBatch

from swiftllm.worker.kernels.linear import linear
# from swiftllm.worker.kernels.rotary_emb import rotary_embedding_inplace
# from swiftllm.worker.kernels.kvcache_mgmt import store_kvcache
# from swiftllm.worker.kernels.silu_and_mul import silu_and_mul_inplace
# from swiftllm.worker.kernels.rmsnorm import fused_add_rmsnorm_inplace
from swiftllm.worker.kernels.paged_attn import paged_attention
from swiftllm.worker.kernels.prefill_attn import prefill_attention

class TransformerEvents:
    """
    单层 transformer 的性能监控与同步观察点。

    这里混合使用了两种时间来源：

    - CUDA event：记录 GPU 段的 elapsed time；
    - `time.perf_counter()`：记录 CPU decode 与 Python launch overhead。

    此外 `qkvtr_e` 还承担同步作用：它标记“CPU decode 所需的 Q/K/V 已经搬到 host buffer，
    CPU 自定义算子可以开始”的边界。
    """
    def __init__(self, engine_config: EngineConfig):
        self.engine_config = engine_config
        # 当前 stage 的起点；`linr_time` 会从这里算到 `linr_e`。
        self.stage_s = torch.cuda.Event(enable_timing=True)
        # 线性部分结束点：包括 pre/post projection、RMSNorm、MLP 等非 attention 段。
        self.linr_e = torch.cuda.Event(enable_timing=True)
        # GPU prefill attention 结束点。
        self.pref_e = torch.cuda.Event(enable_timing=True)
        # GPU decode attention 结束点。
        self.gdec_e = torch.cuda.Event(enable_timing=True)
        # QKV 异步搬到 CPU buffer 完成的观察点；CPU decode 会显式等待它。
        self.qkvtr_e = torch.cuda.Event()
        # launch overhead 前半段的 CPU wall-clock 起点。
        self.lnch_s = 0.0
        # 启动 CPU decode 之前、等待 QKV ready 前的 CPU wall-clock 分界点。
        self.lnch_m = 0.0
        # CPU decode 真正进入 `paged_attention_cpu(...)` 的 wall-clock 起点。
        self.cdec_s = 0.0
        # CPU decode 算子返回时的 wall-clock 终点。
        self.cdec_e = 0.0
        # launch overhead 后半段的 CPU wall-clock 终点。
        self.lnch_e = 0.0

    @property
    def linr_time(self) -> float:
        """
        当前 stage 中线性 / 非 attention 部分的 GPU 时间。

        区间是 `stage_s -> linr_e`。在不同执行形态下，这段可能对应：
        - sequential：本层 preproj 到 postproj/MLP；
        - pipeline：某个 stage 里 postproj+下一层 preproj 的组合。
        """
        return self.stage_s.elapsed_time(self.linr_e)

    @property
    def pref_time(self) -> float:
        """
        GPU prefill attention 时间。

        区间是 `linr_e -> pref_e`，表示 attention 段里 prefill 部分的完成时间。
        """
        return self.linr_e.elapsed_time(self.pref_e)

    @property
    def gdec_time(self) -> float:
        """
        GPU decode attention 时间。

        区间是 `pref_e -> gdec_e`，即 prefill 之后、CPU decode 之前的 GPU decode 段。
        """
        return self.pref_e.elapsed_time(self.gdec_e)

    @property
    def cdec_time(self) -> float:
        """
        CPU decode wall-clock 时间。

        这里不用 CUDA event，因为 `torch.ops.pacpu.paged_attention_cpu(...)` 是同步 CPU C++ op。
        因此直接用 `cdec_s -> cdec_e` 的 wall-clock 区间表示。
        """
        return self.cdec_e - self.cdec_s

    @property
    def lnch_time(self) -> float:
        """
        pipeline 中 CPU 侧 launch / Python 调度额外开销。

        它不包含真正的 CPU decode 算子时间，而是把 CPU decode 之前和之后的两段 Python /
        launch 开销拼起来：

        - `lnch_s -> lnch_m`
        - `cdec_e -> lnch_e`
        """
        return self.lnch_e - self.cdec_e + self.lnch_m - self.lnch_s

    def pf_record(self, name: str):
        """
        在性能监控开启时记录指定 CUDA event。
        """
        if self.engine_config.monitor_performance:
            getattr(self, name).record()

    def pf_time(self, name: str):
        """
        在性能监控开启时记录指定 CPU wall-clock 时间戳，单位毫秒。

        CPU decode 与 launch overhead 之所以不用 CUDA event，是因为这两段跨越了 Python 调度、
        CPU 自定义算子与异步回拷，直接用 wall-clock 更符合当前代码路径的实际边界。
        """
        if self.engine_config.monitor_performance:
            setattr(self, name, time.perf_counter() * 1e3) # ms

    def pf_time_nocpu(self):
        """
        在本层没有 CPU decode request 时写入占位时间戳。

        这样 `cdec_time` 会自然变成 0，而 `lnch_time` 仍能保持连续口径；否则后续聚合
        `ModelPerfResult` 时，这几个字段会因为缺少时间戳而断裂。
        """
        if self.engine_config.monitor_performance:
            self.lnch_m = self.cdec_s = self.cdec_e = time.perf_counter()

class LlamaTransformerLayer:
    """
    Llama 的一层 transformer。

    除了执行本层 forward 外，它还维护两套 `TransformerEvents`：
    - `events[0]`
    - `events[1]`

    在 sequential 路径里，两套槽位会被同一 batch 的不同线性边界复用；
    在 pipeline 路径里，它们则对应两个 pipeline stage，而不是简单对应“batch0 / batch1”。
    """
    def __init__(
        self,
        model_config: LlamaModelConfig,
        engine_config: EngineConfig,
        weight: LlamaTransformerLayerWeight,
        next_layer_weight: LlamaTransformerLayerWeight | None,
        cpu_communication_stream: torch.cuda.Stream,
        layer_id: int
    ):
        self.model_config = model_config
        self.engine_config = engine_config
        self.weight = weight
        self.next_layer_weight = next_layer_weight
        self.cpu_communication_stream = cpu_communication_stream
        self.layer_id = layer_id

        self.events = [TransformerEvents(engine_config) for _ in range(2)]

        # Set after KV cache initialization
        self.swapper = None


    def set_swapper(self, swapper: Swapper):
        """
        注入当前层共享的 swapper / KV cache 管理器。
        """
        self.swapper = swapper


    def _comm_wait_compute(self):
        """
        让 CPU communication stream 等待默认 CUDA stream。

        用于“先算后拷”路径：只有默认 stream 上的计算结果 ready 后，通信 stream 才能安全地
        发起 host/device 拷贝或 swap。
        """
        self.cpu_communication_stream.wait_stream(torch.cuda.default_stream())


    def _compute_wait_comm(self):
        """
        让默认 CUDA stream 等待 CPU communication stream。

        这对应“先拷后算”路径，例如 CPU decode 的输出回拷完成前，后续 post-projection
        不能继续消费 attention 输出。
        """
        torch.cuda.default_stream().wait_stream(self.cpu_communication_stream)


    def _maybe_allreduce(self, x: torch.Tensor):
        if self.model_config.world_size > 1:
            dist.all_reduce(x)


    def _transfer_qkv(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        batch: SubBatch,
        cur_stage: int = 0
    ):
        """
        把当前 batch 尾部的 CPU decode Q/K/V 异步搬到 host buffer。

        这是 CPU decode 的前置准备阶段：
        - 先让 communication stream 等待默认 stream 上的 QKV 计算完成；
        - 再把最后 `num_cdecs` 条 request 的 Q/K/V 拷到 CPU buffer；
        - 最后在 `qkvtr_e` 处打点，表示 CPU 侧 attention 的输入已经就绪。

        `_attention()` 中的 `qkvtr_e.synchronize()` 会把这里作为“CPU op 可以真正开始”的
        显式同步边界。
        """
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


    def _swap_out_blocks(
        self,
        batch: SubBatch,
    ):
        """
        把 CPU-prefill 产生的新 KV blocks 从 GPU swap 到 CPU。

        这里假设“要被换出的新 prefill KV”已经在当前层的最后阶段准备好了，因此可以在
        communication stream 上异步发起 swap，不阻塞默认计算流。
        """
        if batch.num_cprfs > 0:
            with torch.cuda.stream(self.cpu_communication_stream):
                self.swapper.swap_blocks(
                    batch.src_blk_ids,
                    batch.dst_blk_ids,
                    is_swap_out=True,
                    gpu_layer=self.model_config.num_layers if self.engine_config.extra_layer_for_cprf else self.layer_id,
                    cpu_layer=self.layer_id
                )


    def _preproj(
        self,
        embeddings: torch.Tensor,
        batch: SubBatch,
        layer_off: int = 0
    ) -> tuple[torch.Tensor]:
        """
        执行 pre-projection：RMSNorm、QKV 线性映射、RoPE，以及需要时的 KV cache 存储。

        在 pipeline 路径里，`layer_off=1` 表示这里算出来的是“下一层要用的 QKV”，因此权重
        取 `next_layer_weight`。
        """
        weight = self.weight if not layer_off else self.next_layer_weight

        self._maybe_allreduce(embeddings)
        fused_add_rmsnorm_inplace(
            embeddings,
            batch.residual_buf,
            weight.attn_norm,
            self.model_config.rms_norm_eps
        )

        # Calculate QKV
        q = linear(embeddings, weight.q_proj)		# [iter_width, hidden_size / ws]
        k = linear(embeddings, weight.k_proj)		# [iter_width, num_kv_heads / ws * head_dim]
        v = linear(embeddings, weight.v_proj)		# [iter_width, num_kv_heads / ws * head_dim]
        q = q.view(batch.iter_width, -1, self.model_config.head_dim)
        k = k.view(batch.iter_width, -1, self.model_config.head_dim)
        v = v.view(batch.iter_width, -1, self.model_config.head_dim)

        # Rotary emb
        rotary_embedding_inplace(
            q,
            k,
            batch.position_sin,
            batch.position_cos
        )

        # 这里只为 prefill request 存 KV；decode request 的 KV 不在这里显式写入。
        if batch.num_prefs > 0 and self.swapper is not None:
            gpu_layer = (self.layer_id + layer_off) % self.model_config.num_layers
            itm_layer = self.model_config.num_layers if self.engine_config.extra_layer_for_cprf else gpu_layer
            # 如果前面还有未完成的 swap，先等通信 stream，避免写到仍在被 swap 的区域。
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

        return q, k, v


    def _attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        batch: SubBatch,
        cur_stage: int = 0 # also as the offset of layer_id
    ):
        """
        执行 attention 段，并把结果写入当前 batch 的 `attn_out_buf`。

        时序上分三段：
        1. prefill attention（GPU）
        2. GPU paged attention（GPU decode）
        3. `torch.ops.pacpu.paged_attention_cpu(...)`（CPU decode）

        `TransformerEvents` 会在这三段之间依次打点，从而把 layer-level timing 分解成
        `pref_time / gdec_time / cdec_time / lnch_time`。
        """

        events = self.events[cur_stage]
        o = batch.attn_out_buf.view(batch.iter_width, -1, self.model_config.head_dim)
        cur_layer_id = (self.layer_id + cur_stage) % self.model_config.num_layers

        if batch.num_prefs > 0:
            # prefill attention 先执行；Ampere 及以上优先用 flash-attn，否则退回自定义 kernel。
            if torch.cuda.get_device_properties(0).major >= 8:
                # flash_attn.forward(
                # pylint: disable=c-extension-no-member
                flash_attn_cuda.varlen_fwd(
                    q[:batch.sum_pref_toks],
                    k[:batch.sum_pref_toks],
                    v[:batch.sum_pref_toks],
                    o[:batch.sum_pref_toks],
                    batch.pref_st_locs_we,
                    batch.pref_st_locs_we,
                    None,
                    None,  # block table
                    None,  # alibi slopes
                    batch.max_pref_toks,
                    batch.max_pref_toks,
                    0.0,
                    self.model_config.softmax_scale,
                    False,
                    True,  # causal
                    -1,    # window size 0
                    -1,    # window size 1
                    0.0,   # softcap
                    False, # return softmax
                    None
                )
            else:
                prefill_attention(
                    q, k, v, o[:batch.sum_pref_toks],
                    self.model_config, self.engine_config, batch
                )
        # `linr_e -> pref_e` 对应 prefill attention 时间。
        events.pf_record("pref_e")

        # GPU decode attention 紧接着 prefill 之后执行；它在同一 attention 段内，但统计上单独切出来。
        if batch.num_gdecs > 0:
            # with torch.cuda.stream(self.decoding_piggyback_stream):
            #     torch.cuda.current_stream().wait_event(self.events[cur_stage].stage_s)
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
        # `pref_e -> gdec_e` 对应 GPU decode attention 时间。
        events.pf_record("gdec_e")

        if batch.num_cdecs > 0:
            oc = self.swapper.o_cpu[:batch.num_cdecs]
            # CPU launch overhead 的前半段：从 stage 开始到真正等待 CPU decode 输入 ready 之前。
            events.pf_time("lnch_m")
            # 显式等待 `_transfer_qkv()` 完成，确保 CPU 侧拿到的 Q/K/V 已就绪。
            self.events[cur_stage].qkvtr_e.synchronize()
            # CPU decode 的真正起点。
            events.pf_time("cdec_s")
            # 这是同步 CPU C++ op；返回时 CPU attention 已经完成。
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
            events.pf_time("cdec_e")
            # CPU 算子产出的 attention output 再异步回拷到 GPU，供后续 post-projection 消费。
            with torch.cuda.stream(self.cpu_communication_stream):
                o[-batch.num_cdecs:, :].copy_(oc, non_blocking=True)
        else:
            # 没有 CPU decode 时也写入占位时间戳，保持 `cdec_time / lnch_time` 统计口径一致。
            events.pf_time_nocpu()
        # 默认 CUDA stream 在继续 post-projection 前，必须等 CPU decode 输出回拷完成。
        self._compute_wait_comm() # Wait for CPU decoding to finish


    def _postproj(
        self,
        batch: SubBatch
    ) -> torch.Tensor:
        """
        执行 attention 之后的输出投影、FFN 与残差更新。
        """
        o = linear(batch.attn_out_buf, self.weight.o_proj)
        self._maybe_allreduce(o)
        fused_add_rmsnorm_inplace(o, batch.residual_buf, self.weight.ffn_norm, self.model_config.rms_norm_eps)
        ug = linear(o, self.weight.up_gate_proj)
        del o
        silu_and_mul_inplace(ug)
        embeddings = linear(ug[:, :ug.shape[1] // 2], self.weight.down_proj)
        del ug
        return embeddings


    def forward(self, batch: SubBatch, embeddings: torch.Tensor) -> torch.Tensor:
        """
        sequential 单 batch 路径下执行本层 forward。

        timing 上大致是：
        - `stage_s -> linr_e`：preproj + postproj/MLP 等线性段；
        - `linr_e -> pref_e -> gdec_e`：attention 内的 GPU 段；
        - `cdec_s -> cdec_e`：CPU decode；
        - `lnch_s/lnch_m/lnch_e`：围绕 CPU decode 的 launch / Python overhead。
        """
        self.events[0].pf_record("stage_s")
        self.events[0].pf_time("lnch_s")
        q, k, v = self._preproj(embeddings, batch)
        self.events[0].pf_record("linr_e")
        self._transfer_qkv(q, k, v, batch)
        self._attention(q, k, v, batch)
        del q, k, v
        # sequential 路径里，events[1] 主要补“attention 之后那段 linear 边界”，
        # 使 `ModelPerfResult` 能把本层完整 linear 开销拼出来。
        self.events[1].pf_record("stage_s")
        self._swap_out_blocks(batch)
        embeddings = self._postproj(batch)
        self.events[0].pf_time("lnch_e")
        self.events[1].pf_record("linr_e")
        return embeddings


    def _forward_pipeline_stage(
        self,
        q1: torch.Tensor,  # [num_tokens, num_q_heads, head_dim]
        k1: torch.Tensor,  # [num_tokens, num_kv_heads, head_dim]
        v1: torch.Tensor,  # [num_tokens, num_kv_heads, head_dim]
        batches: list[SubBatch],
        cur_stage: int,
    ) -> tuple[torch.Tensor]:
        """
        执行 pipeline 中的一个 steady-state stage。

        这一 stage 会同时交织两类工作：

        - `batches[cur_stage]`：做 postproj，再为下一层做 preproj；
        - `batches[cur_stage ^ 1]`：消费上一阶段留下的 QKV，执行 attention。

        因而这里的 `events[cur_stage]` 记录的是“第 `cur_stage` 个 pipeline stage”的 timing，
        不是简单记录某一个固定 batch 的 timing。
        """
        self.events[cur_stage].pf_record("stage_s")
        self.events[cur_stage].pf_time("lnch_s")
        self._transfer_qkv(q1, k1, v1, batches[cur_stage^1], cur_stage=cur_stage)
        self._swap_out_blocks(batches[cur_stage])
        e0 = self._postproj(batches[cur_stage])
        q0, k0, v0 = self._preproj(e0, batches[cur_stage], layer_off=1)
        del e0
        self.events[cur_stage].pf_record("linr_e")
        self._attention(q1, k1, v1, batches[cur_stage^1], cur_stage=cur_stage)
        self.events[cur_stage].pf_time("lnch_e")

        return q0, k0, v0


    def forward_double(
        self,
        q1: torch.Tensor,  # [num_tokens, num_q_heads, head_dim]
        k1: torch.Tensor,  # [num_tokens, num_kv_heads, head_dim]
        v1: torch.Tensor,  # [num_tokens, num_kv_heads, head_dim]
        batches: list[SubBatch]
    ) -> tuple[torch.Tensor]:
        """
        对双 sub-batch 执行本层的 steady-state pipeline。

        它连续跑两个 `_forward_pipeline_stage()`：
        - 先推进 stage 0；
        - 再推进 stage 1。

        因此一层内部的两套 `TransformerEvents` 正好对应这两个 stage 槽位。
        """
        q0, k0, v0 = self._forward_pipeline_stage(q1, k1, v1, batches, cur_stage=0)
        q1, k1, v1 = self._forward_pipeline_stage(q0, k0, v0, batches, cur_stage=1)

        return q1, k1, v1


    def forward_first_stage(
        self,
        embeddings: torch.Tensor,
        batches: list[SubBatch]
    ) -> tuple[torch.Tensor]:
        """
        执行双 sub-batch pipeline 的 first stage。

        这里会：
        - 先对 batch0 做 preproj 并直接开始 attention；
        - 同时为 batch1 预先算出下一步要用的 QKV。

        因此它负责把 pipeline 从“纯 embeddings 输入”推进到“后续各层 steady-state 可接续”的
        状态。
        """
        embeddings = torch.split(embeddings, [batch.iter_width for batch in batches])
        q0, k0, v0 = self._preproj(embeddings[0], batches[0], layer_off=1)
        # 第一段 attention 开始前，必须先确认之前异步发起的 swap 已经完成。
        self._compute_wait_comm() # Here we must make sure all swaps are done before the first attention

        self.events[1].pf_record("stage_s")
        self.events[1].pf_time("lnch_s")
        self._transfer_qkv(q0, k0, v0, batches[0], cur_stage=1)
        q1, k1, v1 = self._preproj(embeddings[1], batches[1], layer_off=1)
        self.events[1].pf_record("linr_e")
        self._attention(q0, k0, v0, batches[0], cur_stage=1)
        self.events[1].pf_time("lnch_e")

        return q1, k1, v1


    def forward_last_stage(
        self,
        q1: torch.Tensor,  # [num_tokens, num_q_heads, head_dim]
        k1: torch.Tensor,  # [num_tokens, num_kv_heads, head_dim]
        v1: torch.Tensor,  # [num_tokens, num_kv_heads, head_dim]
        batches: list[SubBatch]
    ) -> torch.Tensor:
        """
        执行双 sub-batch pipeline 的 last stage，并返回拼接后的输出。

        这里负责收尾：
        - batch0 只剩 postproj / FFN；
        - batch1 还要完成 attention，再接 postproj / FFN。

        因此它与 `forward_first_stage()` 一起夹住中间的 `forward_double()` steady-state 段，
        共同构成完整的双 sub-batch pipeline。
        """
        self.events[0].pf_record("stage_s")
        self.events[0].pf_time("lnch_s")
        self._transfer_qkv(q1, k1, v1, batches[1], cur_stage=0)
        self._swap_out_blocks(batches[0])
        e0 = self._postproj(batches[0])
        self.events[0].pf_record("linr_e")
        self._attention(q1, k1, v1, batches[1], cur_stage=0)
        self.events[0].pf_time("lnch_e")

        e1 = self._postproj(batches[1])

        return torch.cat((e0, e1))
