"""
LlamaModel worker 侧执行与性能采样逻辑。

这个模块不只是“跑一次 forward”这么简单。在正常在线路径里，它是被 `Executor`
驱动的模型 worker；而在启动阶段 profiling 路径里，它还负责：

- 按 iteration 记录 model-level timing 边界；
- 汇总各层 `TransformerEvents` 产出的 layer-level timing；
- 在监控开启时把一次 iteration 物化成 `ModelPerfResult`；
- 把这些原始测量结果返回给 `ModelProfiler`，供后者构造 profile tables。

因此这里是 server 侧 `ModelProfiler -> Executor -> worker model` 性能采样链路里的
数据面终点。
"""

import json
import itertools

import numpy as np
import torch
import torch.distributed as dist
import ray

from swiftllm.engine_config import EngineConfig
from swiftllm.model_config import LlamaModelConfig
from swiftllm.worker.weight import load_weights
from swiftllm.worker.buffer import ModelForwardBuffers
from swiftllm.worker.block_swapper import Swapper
from swiftllm.structs import Request, SubBatch

from .layers.pre_layer import LlamaPreLayer
from .layers.transformer_layer import LlamaTransformerLayer
from .layers.post_layer import LlamaPostLayer

class ModelEvents:
    """
    一次 model forward 的全局 timing 边界。

    这里记录的是 model 级别的 CUDA event，不是某一层内部的 event：

    - `frwd_s / frwd_e`：整次 forward 的起止；
    - `fstg_s`：pre-layer 结束、进入 transformer body 之前；
    - `mnbd_s / mnbd_e`：transformer main body 的起止；
    - `lstg_e`：transformer body 结束、post-layer 开始之前。

    `ModelPerfResult` 会把这些 event 与各层 `TransformerEvents` 一起消费，形成 profiler
    真正需要的聚合指标。
    """

    def __init__(self, engine_config: EngineConfig):
        self.engine_config = engine_config
        # 整次 model forward 的起点。
        self.frwd_s = torch.cuda.Event(enable_timing=True)
        # pre-layer（embedding / input 准备）结束点。
        self.fstg_s = torch.cuda.Event(enable_timing=True)
        # transformer main body 起点。
        self.mnbd_s = torch.cuda.Event(enable_timing=True)
        # transformer main body 终点。
        self.mnbd_e = torch.cuda.Event(enable_timing=True)
        # last-stage / post-layer 之前的边界。
        self.lstg_e = torch.cuda.Event(enable_timing=True)
        # 整次 model forward 的终点。
        self.frwd_e = torch.cuda.Event(enable_timing=True)

    def pf_record(self, name:str):
        """
        在性能监控开启时记录指定 CUDA event。

        启动阶段 `ModelProfiler._run_test_case()` 会先通过 `Executor` 打开
        `engine_config.monitor_performance`；只有在该开关打开时，这些 event 才会真正被记录。
        在线正常推理路径不会因此额外打点。
        """
        if self.engine_config.monitor_performance:
            getattr(self, name).record()


class ModelPerfResult:
    """
    一次 profiling iteration 的原始性能测量结果。

    它不是在线调度器直接查询的 predictor，而是 worker 在一次 forward 结束后，把：

    - model-level timing（来自 `ModelEvents`）
    - layer-level timing（来自每层 `TransformerEvents`）

    汇总成的一份可序列化结果。随后 `ModelProfiler` 会对多个 repeat 的
    `ModelPerfResult` 做平均，得到 `avg_linr_time`、`avg_pref_time`、`avg_gdec_time`、
    `avg_cdec_time`、`avg_lnch_time`，再回填给 `TablePerfPredictor`。
    """

    # pylint: disable=too-many-instance-attributes
    # 这些字段正好对应 `ModelProfiler` 要回填的 5 类 profile table / 常量：
    # linear、GPU prefill、GPU decode、CPU decode、launch overhead。
    fields_to_dump = [
        "avg_linr_time",
        "avg_pref_time",
        "avg_gdec_time",
        "avg_cdec_time",
        "avg_lnch_time"
    ]
    def __init__(
        self,
        layers: list[LlamaTransformerLayer],
        model_events: ModelEvents,
        use_pipline: bool
    ):
        # 先同步，确保所有 CUDA event 和跨 stream 的异步拷贝都已完成，
        # 这样下面读取 elapsed_time / CPU wall-clock 片段时口径才完整。
        torch.cuda.synchronize() # Ensure all events are recorded
        if use_pipline:
            # pipeline / double sub-batch 模式下，layer.events[0/1] 分别对应两个 pipeline stage。
            # `linr_times` 使用当前 stage 的线性段；而 pref/gdec/cdec 对应的是同一 stage 内
            # 被 attention 消费的“另一侧 sub-batch”，因此这里按 `i ^ 1` 取值。
            self.linr_times = np.array([[layer.events[i].linr_time for layer in layers[:-1]] for i in range(2)])
            self.pref_times = np.array([[layer.events[i^1].pref_time for layer in layers] for i in range(2)])
            self.gdec_times = np.array([[layer.events[i^1].gdec_time for layer in layers] for i in range(2)])
            self.cdec_times = np.array([[layer.events[i^1].cdec_time for layer in layers] for i in range(2)])
            self.lnch_times = np.array([[layer.events[i].lnch_time for layer in layers] for i in range(2)])
        else:
            # sequential 单 batch 模式下，每层会同时用到 events[0] 与 events[1]：
            # - events[0] 覆盖真正的 attention / CPU decode / launch timing；
            # - events[1] 主要补 post-proj 之后那段 linear 边界。
            # 因此 linear 时间要把两套槽位加起来，而 pref/gdec/cdec/lnch 则直接取 stage 0。
            self.linr_times = np.array([sum(layer.events[i].linr_time for i in range(2)) for layer in layers])
            self.pref_times = np.array([layer.events[0].pref_time for layer in layers])
            self.gdec_times = np.array([layer.events[0].gdec_time for layer in layers])
            self.cdec_times = np.array([layer.events[0].cdec_time for layer in layers])
            self.lnch_times = np.array([layer.events[0].lnch_time for layer in layers])

        # 以下是 model-level 切片时间：
        # - prlr_time: pre-layer / 输入准备阶段
        # - fstg_time: first-stage 到 main body 入口之间
        # - mnbd_time: transformer main body 本体
        # - lstg_time: main body 结束到 last-stage 边界
        # - polr_time: post-layer / sampling 阶段
        self.prlr_time = model_events.frwd_s.elapsed_time(model_events.fstg_s)
        self.fstg_time = model_events.fstg_s.elapsed_time(model_events.mnbd_s)
        self.mnbd_time = model_events.mnbd_s.elapsed_time(model_events.mnbd_e)
        self.lstg_time = model_events.mnbd_e.elapsed_time(model_events.lstg_e)
        self.polr_time = model_events.lstg_e.elapsed_time(model_events.frwd_e)

        # `avg_*` 是 profiler 真正消费的聚合结果。
        # 对 sequential 情况它们是一维数组；对 pipeline 情况则会保留双 stage 维度。
        self.avg_linr_time = self.linr_times.mean(-1)
        self.avg_pref_time = self.pref_times.mean(-1)
        self.avg_gdec_time = self.gdec_times.mean(-1)
        self.avg_cdec_time = self.cdec_times.mean(-1)
        self.avg_lnch_time = self.lnch_times.mean(-1)

    def __repr__(self):
        return json.dumps({
            field: getattr(self, field).tolist() for field in self.fields_to_dump
        }, indent=2)

    @staticmethod
    def mean(results: list["ModelPerfResult"], name: str) -> float:
        """
        对多次 profiling repeat 的同名字段做平均。

        `ModelProfiler` 在 warmup 之后会重复执行同一个 profiling case，多次 iteration 的
        原始结果就存放在 `results` 里。这里返回的是该 case 在字段 `name` 上的平均值，
        不是在线运行过程中的滚动统计。
        """
        ret = np.array([getattr(result, name) for result in results]).mean(0).tolist()
        return ret

    @staticmethod
    def mean_all(results: list["ModelPerfResult"]) -> dict[str, float]:
        """
        对所有 profiler 关心的聚合字段统一求平均。
        """
        return {
            field: ModelPerfResult.mean(results, field) for field in ModelPerfResult.fields_to_dump
        }



class LlamaModel:
    """
    Llama worker 模型。

    它既是实际执行 forward 的 worker，也是 profiling 模式下性能数据的生产者：

    - `Executor.do_one_iteration()` 驱动这里执行一次 iteration；
    - 若 `monitor_performance=True`，本模块会记录 event / timestamp；
    - iteration 结束后把结果整理成 `ModelPerfResult`；
    - `ModelProfiler` 再通过 `Executor.turn_off_perf_monitor_and_flush_results()` 把这些
      结果取回并做平均。

    To initialize, please:
    - call __init__()
    - call load_weights()
    - call init_kvcache_and_swap()
    """

    @torch.inference_mode()
    def __init__(
        self,
        engine_config: EngineConfig,
        model_config: LlamaModelConfig,
        rank: int
    ):
        """
        初始化 worker 侧 LlamaModel。

        这里会完成：
        - 权重加载；
        -各层与 buffer 初始化；
        - CPU/GPU 通信 stream 建立；
        - profiling 结果缓存与 model-level events 初始化。

        KV cache / swap 相关结构则要等 `num_gpu_blocks / num_cpu_blocks` 确定后，
        由 `init_kvcache_and_swap()` 再初始化。
        """
        self.engine_config = engine_config
        self.model_config = model_config

        model_config.rank = rank
        model_config.world_size = engine_config.tensor_parallel_degree

        # CPU kernel library & stream
        if engine_config.library_path:
            torch.ops.load_library(engine_config.library_path)
        self.cpu_communication_stream = torch.cuda.Stream()

        # Load weights
        self.weight = load_weights(
            model_config,
            torch.float16,
            engine_config.model_path,
            engine_config.use_dummy
        )

        # Initialize buffers
        self.buffer = ModelForwardBuffers(engine_config, model_config)

        # Initialize layers
        self.pre_layer = LlamaPreLayer(self.model_config, self.weight)
        self.transformer_layers = [
            LlamaTransformerLayer(
                self.model_config,
                self.engine_config,
                self.weight.layers[layer_id],
                self.weight.layers[layer_id + 1 - self.model_config.num_layers],
                self.cpu_communication_stream,
                layer_id
            )
            for layer_id in range(self.model_config.num_layers)
        ]
        self.post_layer = LlamaPostLayer(self.model_config, self.weight)

        # Initialize rotary embeddings
        self.cos_cached = None
        self.sin_cached = None
        self._init_to_get_rotary()

        # Swapper
        self.swapper = None

        # profiling 模式下，worker 会把每次 iteration 的 `ModelPerfResult` 先缓存到这里，
        # 等 `turn_off_perf_monitor_and_flush_results()` 时统一返回给 server 侧 profiler。
        self.perf_results = []
        # 一次 model forward 的全局 timing 边界。
        self.events = ModelEvents(engine_config)


    @torch.inference_mode()
    def init_kvcache_and_swap(self, engine_config: EngineConfig):
        """
        根据 profiler 已测出的 block 数初始化 GPU/CPU KV cache 与 swapper。
        """
        self.engine_config.num_cpu_blocks = engine_config.num_cpu_blocks
        self.engine_config.num_gpu_blocks = engine_config.num_gpu_blocks
        self.swapper = Swapper(self.engine_config, self.model_config)

        for layer in self.transformer_layers:
            layer.set_swapper(self.swapper)


    def _init_to_get_rotary(self):
        rope_scaling_factor = self.model_config.rope_scaling_factor
        base = self.model_config.rope_theta
        # max_position_embeddings = self.model_config.max_position_embeddings
        # max_seq_len = max_position_embeddings * rope_scaling_factor
        max_seq_len = self.engine_config.max_seq_len

        inv_freq = 1.0 / (base ** (torch.arange(0, self.model_config.head_dim, 2, device="cuda", dtype=torch.float32) / self.model_config.head_dim))
        t = torch.arange(max_seq_len + 128, device="cuda", dtype=torch.float32) / rope_scaling_factor
        freqs = torch.outer(t, inv_freq)

        self.cos_cached = torch.cos(freqs).to(torch.float16)
        self.sin_cached = torch.sin(freqs).to(torch.float16)


    def _prepare_inputs(self, batches: list[SubBatch]):
        """
        为 runtime forward 构造 GPU 侧输入结构。

        调度器与 block manager 已经提前决定好 batch 形状和 block 映射；这里主要补齐
        worker 真正执行时需要的 tensor 化字段，例如：

        1. `prgd_seq_ids / prgd_seq_lens`
        2. rotary embedding 所需的位置编码
        3. attention / residual 中间 buffer
        """
        for batch in batches:
            batch.prgd_seq_ids = torch.tensor(batch.seq_ids_list[:batch.num_prgds], dtype=torch.int32, device='cuda')
            batch.prgd_seq_lens = torch.tensor(batch.seq_lens_list[:batch.num_prgds], dtype=torch.int32, device='cuda')
            batch.pref_st_locs_we = torch.tensor(
                [0] + list(itertools.accumulate(batch.seq_lens_list[:batch.num_prefs])),
                dtype=torch.int32, device='cuda'
            )

            position_indices = torch.tensor(
                sum([list(range(req.prompt_len)) for req in batch.all_reqs[:batch.num_prefs]], []) + \
                    [req.seq_len - 1 for req in batch.all_reqs[batch.num_prefs:]],
                dtype=torch.int32, device='cuda'
            )

            batch.position_cos = self.cos_cached[position_indices]
            batch.position_sin = self.sin_cached[position_indices]

            batch.attn_out_buf = torch.zeros(
                (batch.iter_width, self.model_config.hidden_size // self.model_config.world_size), dtype=torch.float16, device='cuda'
            )
            batch.residual_buf = torch.zeros(
                (batch.iter_width, self.model_config.hidden_size), dtype=torch.float16, device='cuda'
            )

            batch.last_token_indices = torch.cat([
                batch.pref_st_locs_we[1:] - 1,
                torch.arange(batch.sum_pref_toks, batch.iter_width, dtype=torch.int32, device='cuda')
            ])

        self.buffer.alloc_for_batches(batches)


    def _forward_sequential(self, batch: SubBatch, embeddings: torch.Tensor) -> torch.Tensor:
        """
        顺序执行单个 sub-batch 的 transformer body。

        在 sequential 模式下，`mnbd_s -> mnbd_e` 覆盖的是整段 transformer layers 的总时间；
        每层内部再通过 `TransformerEvents` 把 linear / prefill / GPU decode / CPU decode /
        launch overhead 继续细分。
        """
        # 在进入第一层 attention 前，先确保前面通过通信 stream 发起的 swap 已经可见。
        torch.cuda.current_stream().wait_stream(self.cpu_communication_stream)
        self.events.pf_record("mnbd_s")
        for layer in self.transformer_layers:
            embeddings = layer.forward(batch, embeddings)
        self.events.pf_record("mnbd_e")
        return embeddings


    def _forward_pipeline(self, batches: list[SubBatch], embeddings: torch.Tensor) -> torch.Tensor:
        """
        以 double sub-batch pipeline 方式执行 transformer body。

        形态上分三段：
        1. `forward_first_stage()`：先把 batch0 的 attention 跑起来，同时为 batch1 做 preproj；
        2. 中间各层 `forward_double()`：交织推进两个 sub-batch；
        3. `forward_last_stage()`：收尾并产出拼接后的 embeddings。

        model-level 的 `mnbd_s / mnbd_e` 只包住中间 steady-state pipeline body；
        first / last stage 则由 `fstg_time / lstg_time` 单独体现。
        """
        assert len(batches) == 2

        q1, k1, v1 = self.transformer_layers[-1].forward_first_stage(embeddings, batches)
        self.events.pf_record("mnbd_s")

        # 每轮循环都会：
        # - 把 batch0 的 attn_out_buf 推进到更新后的版本；
        # - 把 batch1 的 q/k/v 推进到下一层所需的更新版本。
        for layer in self.transformer_layers[:-1]:
            q1, k1, v1 = layer.forward_double(q1, k1, v1, batches)
        self.events.pf_record("mnbd_e")

        embeddings = self.transformer_layers[-1].forward_last_stage(q1, k1, v1, batches)
        return embeddings


    @torch.inference_mode()
    def _forward_batches(self, batches: list[SubBatch]) -> list[int]:
        """
        执行一次 worker-side model iteration，并在需要时产出 `ModelPerfResult`。

        要求进入本函数前：
        - 请求 blocks 已经分配完成；
        - block tables 已经由 block manager / swapper 设置好。

        返回本轮 forward 生成出的 output tokens。
        """
        self._prepare_inputs(batches)
        self.events.pf_record("frwd_s")

        # pre-layer 对输入 token 做 embedding / 输入准备，`frwd_s -> fstg_s` 对应模型级前置阶段。
        embeddings = self.pre_layer.forward(sum([Request.get_input_tokens(b.all_reqs) for b in batches], []))
        self.events.pf_record("fstg_s")

        if len(batches) == 1:
            embeddings = self._forward_sequential(batches[0], embeddings)
        elif len(batches) == 2:
            embeddings =  self._forward_pipeline(batches, embeddings)
        else:
            raise ValueError("Invalid number of batches")
        self.events.pf_record("lstg_e")

        output_tokens = self.post_layer.forward(batches, embeddings, self.buffer.cur_residual_buf)
        self.events.pf_record("frwd_e")

        if self.engine_config.monitor_performance:
            # profiling case 的原始测量结果在 iteration 全部结束后才统一 materialize。
            # 这些结果稍后会被 `Executor -> ModelProfiler` flush 回 server 侧。
            self.perf_results.append(ModelPerfResult(self.transformer_layers, self.events, False))

        return output_tokens


    def do_one_iteration(
        self,
        batches: list[SubBatch],
        mappings: tuple[tuple[list[int], list[int]], tuple[list[int], list[int]]],
        swappings: tuple[list[int], list[int]],
        is_swap_out: bool = False
    ) -> list[int]:
        """
        执行一次完整 iteration。

        顺序是：
        1. 按 `mappings` 更新 block tables；
        2. 按 `swappings` 发起 swap in/out；
        3. 在这些 runtime 状态准备完成后，真正执行 `_forward_batches()`。

        也就是说，worker 侧被测到的性能数据反映的是“带着真实 block table / swap 状态”的
        一次 forward，而不是脱离 runtime 上下文的裸算子测试。
        """

        if self.swapper is not None:
            self.swapper.set_block_tables(mappings)

        if swappings[0]:
            with torch.cuda.stream(self.cpu_communication_stream):
                for layer_id in range(self.model_config.num_layers):
                    self.swapper.swap_blocks(*swappings, is_swap_out, layer_id, layer_id)

        return self._forward_batches(batches)


    def turn_on_perf_monitor(self):
        """
        打开 worker 侧性能监控。

        server 侧 `ModelProfiler._run_test_case()` 会先通过 `Executor.turn_on_perf_monitor()`
        调到这里。此后本 worker 在每次 iteration 中都会记录 event / timestamp，并把结果
        追加到 `self.perf_results`。
        """
        self.engine_config.monitor_performance = True


    def turn_off_perf_monitor_and_flush_results(self):
        """
        关闭性能监控并返回本轮累计的 profiling 结果。

        这是 `ModelProfiler` 消费 worker 侧测量数据的出口：profiler 在一个 test case
        结束后通过 `Executor` 调到这里，把本轮 repeat 期间累积的 `ModelPerfResult` 全部取回，
        随后再做 `ModelPerfResult.mean(...)` 聚合。
        """
        self.engine_config.monitor_performance = False
        ret = self.perf_results
        self.perf_results = []
        return ret


@ray.remote(num_cpus=8, num_gpus=1)
class RemoteLlamaModel(LlamaModel):
    """
    供 RayExecutor 调用的远程 worker 版本。
    """

    @torch.inference_mode()
    def __init__(
        self,
        engine_config: EngineConfig,
        model_config: LlamaModelConfig,
        rank: int
    ):
        """
        初始化远程 worker，并先建立 TP 所需的分布式通信组。
        """

        dist.init_process_group(
            backend="nccl",
            world_size=engine_config.tensor_parallel_degree,
            rank=rank
        )
        super().__init__(engine_config, model_config, rank)
