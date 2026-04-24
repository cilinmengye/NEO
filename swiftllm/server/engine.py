"""
服务端主引擎。

这个模块把 SwiftLLM/NEO 在线服务的几个核心组件串起来：
- `Executor`：真正执行模型 forward；
- `ModelProfiler`：启动阶段测容量边界、构建 profile tables；
- `BlockManager`：准备/更新 KV cache 与 swap 状态；
- `Scheduler`：基于 `ModelProfiler.pp` 做 batch picking 与 mode selection；
- `TokenizationEngine`：异步批量分词。

需要注意的是：性能预测器主要参与“调度形成 batch”的阶段；真正进入 worker 执行时，
`batches` 已经是 scheduler 决策后的结果，worker 不会再直接查询 predictor 本体。
"""

import time
import sys
import asyncio
import functools
import logging
from typing import AsyncGenerator

from swiftllm.engine_config import EngineConfig
from swiftllm.model_config import LlamaModelConfig
from swiftllm.server.executor import SingleProcExecutor, RayExecutor
from swiftllm.server.profiler import ModelProfiler
from swiftllm.structs import Request, RawRequest, StepOutput, SubBatch

from swiftllm.server.tokenization_engine import TokenizationEngine
from swiftllm.server.scheduler import Scheduler
from swiftllm.server.block_manager import BlockManager

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

class Engine:
    """
    基础引擎。

    这是更偏离线/同步的执行封装：负责初始化模型、profiler、block manager，并提供单步
    `step()` 接口。它本身不维护在线请求事件循环。
    """

    def __init__(self, engine_config: EngineConfig):
        # 全局运行配置。
        self.engine_config = engine_config
        # 从模型路径加载出的结构配置，后续 profiler/scheduler/worker 都会依赖它。
        self.model_config = LlamaModelConfig.load_from_model_path(engine_config.model_path)
        self.initialized = False

        assert engine_config.max_batch_size <= engine_config.max_tokens_in_batch, \
            f"max_batch_size {engine_config.max_batch_size} exceeds max_tokens_in_batch {engine_config.max_tokens_in_batch}"
        assert engine_config.max_batch_size <= engine_config.max_seqs_in_block_table, \
            f"max_batch_size {engine_config.max_batch_size} exceeds max_seqs_in_block_table {engine_config.max_seqs_in_block_table}"
        assert engine_config.tensor_parallel_degree >= 1, "Tensor parallel degree should be positive"

        # 以下字段会在 `initialize()` 中真正创建。
        self.executor = None
        self.event_loop = None
        self.profiler = None
        self.block_manager = None
        # 单卡走 `SingleProcExecutor`；多卡 TP 走 `RayExecutor`。
        self.executor_class = SingleProcExecutor if engine_config.tensor_parallel_degree == 1 else RayExecutor


    def initialize(self):
        """
        初始化基础执行引擎。

        顺序上先创建 `executor`，再创建 `ModelProfiler` 并调用 `profile_num_blocks()`：
        这里先测的是 GPU/CPU KV cache 的容量边界，而不是 profile tables。本步骤得到的
        `num_gpu_blocks/num_cpu_blocks` 会直接影响后续 block manager 与 scheduler 的容量约束。
        """
        logger.info("Initializing model...")
        self.executor = self.executor_class(self.engine_config, self.model_config)

        logger.info("Profiling model...")
        self.profiler = ModelProfiler(self.executor)
        self.profiler.profile_num_blocks()

        logger.info("Initializing block manager...")
        self.block_manager = BlockManager(self.engine_config, self.model_config)

        logger.info("Initializing KV cache and swap...")
        self.executor.init_kvcache_and_swap()

        logger.info("Model initialized")
        self.initialized = True


    def step(self, batches: list[SubBatch], cur_swap_out: list[Request]=None, cur_swap_in: list[Request]=None):
        """
        执行一步同步 forward。

        传入的 `batches` 已经是外部调度器/调用方做完决策后的结果；这里仅负责：
        1. 让 `BlockManager.prepare()` 生成 forward 所需结构；
        2. 调用 `executor.do_one_iteration()` 真正执行；
        3. 让 block manager 更新状态并释放结束请求。
        """
        forward_args = self.block_manager.prepare(batches, cur_swap_out or [], cur_swap_in or [])
        output_token_ids = self.executor.do_one_iteration(batches, *forward_args)
        self.block_manager.update_and_free(batches, output_token_ids)



class AsyncEngine(Engine):
    """
    在线服务使用的异步引擎。

    它在 `Engine` 的基础上补上：
    - `Scheduler`：在线 batch picking / mode selection；
    - `TokenizationEngine`：异步分词；
    - 两个长期运行的事件循环：tokenization loop 与 main forward loop。
    """

    def __init__(self, engine_config: EngineConfig):
        super().__init__(engine_config)

        # 以下字段会在 `initialize_async()` 中创建。
        self.scheduler = None
        self.tokenization_engine = None

        # 原始字符串请求会先暂存在这里，等待 tokenization loop 批量分词。
        self.untokenized_raw_requests: list[tuple[Request, str]] = []


    async def _run_on_model_executor_async(self, func, *args, **kwargs):
        """
        把同步的模型执行函数包装成异步调用。

        主事件循环本身是 asyncio 驱动的，而底层模型执行仍是同步计算，因此这里通过
        `run_in_executor()` 把阻塞计算转移到线程池/执行器上下文中。
        """
        func_partial = functools.partial(func, *args, **kwargs)
        return await self.event_loop.run_in_executor(None, func_partial)


    async def initialize_async(self):
        """
        初始化异步在线引擎。

        关键接入顺序是：
        1. 先复用 `Engine.initialize()` 完成 executor / block 容量 profiling；
        2. 再调用 `self.profiler.init_profile_tables(...)` 构造 profile tables；
        3. 把最终生成的 `self.profiler.pp` 注入 `Scheduler`。

        因此，运行时 scheduler 使用的是启动阶段离线测得并回填好的 `TablePerfPredictor`，
        而不是在线重新 profile。
        """
        self.event_loop = asyncio.get_event_loop()

        super().initialize()

        logger.info("Initializing performance table...")
        self.profiler.init_profile_tables(self.block_manager)

        logger.info("Initializing scheduler...")
        self.scheduler = Scheduler(self.engine_config, self.model_config, self.profiler.pp)

        logger.info("Initializing tokenization engine...")
        # pylint: disable=no-member
        self.tokenization_engine = TokenizationEngine.remote(self.engine_config)

        logger.info("Engine initialized")
        self.initialized = True


    def _check_request_len(self, request: Request):
        assert request.prompt_len + request.max_output_len <= self.engine_config.max_seq_len, \
            f"Request length {request.prompt_len + request.output_len} exceeds max_seq_len {self.engine_config.max_seq_len}"


    def _admit_request(self, request: Request, raw_request: RawRequest):
        if isinstance(raw_request.prompt, str):
            self.untokenized_raw_requests.append((request, raw_request.prompt))
            return

        request.prompt_token_ids = raw_request.prompt
        request.prompt_len = len(raw_request.prompt)
        self._check_request_len(request)
        self.scheduler.on_requests_arrival([request])


    async def add_request_and_stream(self, raw_request: RawRequest) -> AsyncGenerator[StepOutput, None]:
        """
        以 streaming 模式提交请求，并逐 token 异步返回结果。

        若请求是原始字符串 prompt，它会先进入 `untokenized_raw_requests`，等待 tokenization
        loop 批量处理；之后 scheduler 才会在某一轮把它接纳进 batch。
        """
        request = Request(raw_request)
        self._admit_request(request, raw_request)
        while True:
            step_output = await request.output_q.get()
            yield step_output
            request.output_q.task_done()
            if step_output.request.is_finished():
                break


    async def add_request_and_wait(self, raw_request: RawRequest) -> tuple[Request, list[int]]:
        """
        以 non-streaming 模式提交请求，并等待其完整结束。

        - 若 `prompt` 还是字符串，则先走异步 tokenization；
        - 若已经是 token ids，则直接进入 scheduler。
        """
        request = Request(raw_request)
        self._admit_request(request, raw_request)

        await request.finished_event.wait()
        return (request, request.output_token_ids)


    async def _tokenize_raw_request_event_loop(self):
        """
        原始字符串请求的分词事件循环。

        它持续把 `untokenized_raw_requests` 中累积的请求拿出来批量分词，并在完成后把结果
        交给 `Scheduler.on_requests_arrival()`；真正的调度与执行仍发生在主事件循环中。
        """
        while True:
            if not self.untokenized_raw_requests:
                # 没有新请求时短暂让出事件循环。
                await asyncio.sleep(0.002)
                continue

            # 批量取出当前所有待分词请求，减少小 batch tokenization 开销。
            cur_untokenized_raw_requests = self.untokenized_raw_requests
            self.untokenized_raw_requests = []

            prompts = [prompt for _, prompt in cur_untokenized_raw_requests]
            assert all(isinstance(prompt, str) for prompt in prompts), "untokenized_raw_requests must contain only string prompts"
            prompt_token_ids = await self.tokenization_engine.batched_tokenize.remote(prompts)

            new_requests = []
            for (request, _), prompt_token_id in zip(cur_untokenized_raw_requests, prompt_token_ids):
                request.prompt_token_ids = prompt_token_id
                request.prompt_len = len(prompt_token_id)
                assert request.prompt_len + request.max_output_len <= self.engine_config.max_seq_len, \
                    f"Request length {request.prompt_len + request.output_len} exceeds max_seq_len {self.engine_config.max_seq_len}"
                new_requests.append(request)

            # 分词结束后，请求正式进入 scheduler 的 waiting 队列。
            self.scheduler.on_requests_arrival(new_requests)
            await asyncio.sleep(0.001)  # yield the event loop


    async def _main_event_loop(self):
        """
        在线服务主事件循环。

        它反复执行：
        1. 向 scheduler 询问下一轮 batch / swap 决策；
        2. 让 block manager 准备 forward 参数；
        3. 调用 executor 真正执行；
        4. 更新 block 状态并清理结束请求。

        这里拿到的 `batches` 已经包含 scheduler 基于 predictor 做出的决策结果；worker 执行
        阶段只消费这些结构，不会再直接访问 `PerfPredictor`。
        """
        while True:
            # 向 scheduler 拉取下一轮已经决策好的 batch 以及 swap in/out 计划。
            batches, cur_swap_out, cur_swap_in = self.scheduler.get_next_batch()
            if not (len(batches) or len(cur_swap_in) or len(cur_swap_out)):
                # 没有任何工作可做时，让出事件循环。
                await asyncio.sleep(0.001)
                continue

            # 根据 scheduler 输出的 batch 结果，构造真正的 runtime forward 参数。
            forward_args = self.block_manager.prepare(batches, cur_swap_out, cur_swap_in)

            # 真正执行模型 forward。
            if any(b.num_prefs for b in batches):
                logger.info(f"Forwarding batches with sizes {[(b.num_cprfs, b.num_gprfs, b.num_gdecs, b.num_cdecs) for b in batches]}, "
                            f"swap out: {len(cur_swap_out)}, swap in: {len(cur_swap_in)}")
            output_token_ids = await self._run_on_model_executor_async(self.executor.do_one_iteration, batches, *forward_args)

            # 用本轮输出更新请求状态、释放结束请求，并同步回 scheduler 队列。
            finished_reqs = self.block_manager.update_and_free(batches, output_token_ids)
            self.scheduler.remove_finished_requests(finished_reqs)


    async def start_all_event_loops(self):
        """
        启动在线服务需要的两个长期事件循环。
        """
        assert self.initialized, "Engine not initialized. Please call `initialize()` before starting the event loop."
        await asyncio.gather(
            self._tokenize_raw_request_event_loop(),
            self._main_event_loop()
        )
