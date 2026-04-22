"""
SwiftLLM 的调度器。

这个模块负责两类事情：
- 维护 waiting / GPU decoding / CPU decoding 三类请求队列；
- 在每轮迭代里结合 KV cache 容量、batch budget 与性能预测器，决定：
  - 哪些请求继续留在 GPU decode；
  - 哪些请求需要 swap out 到 CPU；
  - 哪些 CPU decode 请求可以 swap in；
  - 新到达的 prefill 请求落到 GPU 还是 CPU；
  - 当前轮次是走 sequential 还是双 sub-batch pipelined。

这里的“智能”主要不是来自复杂搜索，而是来自 `PerfPredictor` 驱动的启发式估时。
`Scheduler` 不会真正执行模型，只会构造带有 `BatchPerfData` 的 `SubBatch`，并根据
预测的 `gpu_time / cpu_time` 做调度决策。
"""

import sys
import math
import logging
from collections import deque

from swiftllm.engine_config import EngineConfig
from swiftllm.model_config import LlamaModelConfig
from swiftllm.utils import cdiv
from swiftllm.structs import Request, SubBatch
from swiftllm.perfpredictor import PerfPredictor

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

class RequestIdManager:
    """
    维护 block table request id 的分配器。

    scheduler 在真正把某条请求 launch 进 runtime 前，必须先给它分配一个稳定的
    `request_id`，供 block manager / block table / worker 侧索引使用。
    """
    def __init__(self, max_id: int):
        # request id 的合法范围是 [0, max_id)。
        self.max_id = max_id
        # 用一个 deque 维护当前可分配的空闲 id。
        self.available_ids = deque(range(max_id))

    def get_id(self) -> int:
        """
        取出一个可用 request id。

        若空闲 id 耗尽，说明 block table 容量不足，调度器无法再接纳新请求。
        """
        if not self.available_ids:
            raise RuntimeError("No more available request ids. Please try to increase `max_seqs_in_block_table`")
        return self.available_ids.popleft()

    def get_num_available_ids(self) -> int:
        """
        返回当前剩余可分配的 request id 数量。
        """
        return len(self.available_ids)

    def free_id(self, req_id: int):
        """
        归还一个 request id。
        """
        self.available_ids.append(req_id)

    def free_ids(self, req_ids: list[int]):
        """
        批量归还多个 request id。
        """
        self.available_ids.extend(req_ids)

class ScheduleBudget:
    """
    调度预算。

    这里维护的是一轮调度中还能再接纳多少 request / token，目的是在 batch 形成阶段就
    约束住上界，避免后续 forward 时出现超出 `max_batch_size` 或 `max_tokens_in_batch`
    的情况。
    """
    def __init__(self, max_batch_size: int, max_tokens_in_batch: int):
        # 剩余还能容纳多少条 request。
        self.remaining_batch_size = max_batch_size
        # 剩余还能容纳多少 token 宽度。
        self.remaining_tokens_in_batch = max_tokens_in_batch

    @property
    def overspent(self) -> bool:
        """
        当前预算是否已经被透支。
        """
        return self.remaining_batch_size < 0 or self.remaining_tokens_in_batch < 0

    def check_and_substract(self, num_tokens) -> bool:
        """
        检查预算是否足够，并在足够时扣减预算。

        `num_tokens` 表示这条候选请求会占用的 token 宽度：
        - 对 prefill，是整条 prompt 长度；
        - 对 decode，是 1（每轮只前进一步）。
        """
        if self.remaining_batch_size >= 1 and \
            self.remaining_tokens_in_batch >= num_tokens:
            self.remaining_batch_size -= 1
            self.remaining_tokens_in_batch -= num_tokens
            return True
        return False

    def add(self, num_tokens) -> bool:
        """
        回退一次预算扣减。

        调度器在“试探性加入某条请求后又放弃”时，会调用它把 budget 恢复回来。
        """
        self.remaining_batch_size += 1
        self.remaining_tokens_in_batch += num_tokens

class Scheduler:
    """
    SwiftLLM/NEO 的严格 FCFS 调度器。

    它维护 waiting / decoding 队列，并结合 `PerfPredictor` 进行 load-aware scheduling：
    - 为 `SubBatch` 构造调度期 `perfdata`；
    - 估计 CPU decode 能否被另一侧 GPU 工作覆盖；
    - 比较 sequential candidate 与 pipelined candidate 的预测吞吐；
    - 决定本轮真正交给 engine/block manager/worker 的 batch 形态。
    """

    def __init__(
        self,
        engine_config: EngineConfig,
        model_config: LlamaModelConfig,
        predictor: PerfPredictor
    ):
        self.engine_config = engine_config
        self.model_config = model_config

        # 运行时共享的性能预测器。
        # 它通常来自 `ModelProfiler.pp`，后续所有 `SubBatch(self.predictor)` 都会把同一个
        # profile-table 模型注入调度试探路径。
        self.predictor = predictor

        # 以下三个队列都按到达时间排序：
        # - waiting_q: 还未进入任何一次 forward 的新请求；
        # - gpu_decoding_q: 当前驻留在 GPU 上继续 decode 的请求；
        # - cpu_decoding_q: 已被换出到 CPU 侧 decode 的请求。
        self.waiting_q: deque[Request] = deque()
        self.gpu_decoding_q: list[Request] = []
        self.cpu_decoding_q: deque[Request] = deque()

        # `num_itm_blocks` 是“中间态/可转移”视角下能容纳的新 prefill block 上界。
        self.num_itm_blocks = engine_config.num_gpu_blocks
        # `num_gpu_blocks` 则是当前真的允许 GPU decode 占用的 block 数；
        # 当禁用 partial offloading 时，这里会退化成 0，表示不再做 GPU/CPU 部分卸载协同。
        self.num_gpu_blocks = engine_config.num_gpu_blocks * (not engine_config.disable_partial_offl)

        # 当前 GPU decoding 队列占据的 GPU block 数。
        # 这个值逻辑上应始终等于所有 `gpu_decoding_q` 请求所需 block 数之和。
        self.num_decoding_gpu_blocks = 0

        # request id 分配器：只有请求真正被 launch 时才会拿到 id。
        self.request_id_manager = RequestIdManager(self.engine_config.max_seqs_in_block_table)

    def _get_block_needed(self, request: Request) -> int:
        """
        计算一条请求当前需要多少个 KV cache blocks。

        block 数按当前总序列长度 `seq_len` 向上取整。
        这是 scheduler 判断 swap in/out 与 prefill admission 是否可行的基础容量单位。
        """
        return cdiv(request.seq_len, self.engine_config.block_size)

    def on_requests_arrival(self, requests: list[Request]):
        """
        接收一批已完成 tokenization 的新请求。

        这些请求先进入 `waiting_q`，真正何时被 launch 由后续调度轮次决定。
        """
        self.waiting_q.extend(requests)

    def _get_remains(self, batches: list[SubBatch]) -> float:
        """
        估计两个 pipeline sub-batch 的“CPU decode 可被遮蔽余量”。

        注意这里的返回值不是“剩余 token 数”或“剩余 block 数”，而是一个基于预测时间的
        overlap 裕量：

        对第 `j` 个 sub-batch，公式是：
        - 另一侧 sub-batch 的 `linr_T`
        - 加上本侧 sub-batch 的 `pref_T`
        - 加上本侧 sub-batch 的 `gdec_T`
        - 再减去本侧 sub-batch 的 `cpu_time`

        直觉上，它在回答：当前这侧 sub-batch 的 CPU decode 工作，是否还能被另一侧/本侧
        的 GPU 工作盖住。若结果为负，表示 CPU 侧已经比可重叠的 GPU 工作更长，pipeline
        可能失衡。
        """
        assert len(batches) == 2
        return [
            batches[j^1].perfdata.linr_T +
            batches[j].perfdata.pref_T +
            batches[j].perfdata.gdec_T -
            batches[j].perfdata.cpu_time
            for j in range(2)
        ]

    def _decide_mode_and_gen_batch(
        self,
        gpu_prefill_reqs: list[Request],
        cpu_prefill_reqs: list[Request],
        budget: ScheduleBudget
    ) -> list[SubBatch]:
        """
        在给定 decode / prefill 候选集合的前提下，决定本轮 batch 形态。

        这个函数比较的是两类候选：
        - `gpu_only_batch`：single-batch / sequential candidate；它不是字面意义上的
          “只用 GPU 的 batch”，而是“不拆成两个 pipeline sub-batch”的单批方案；
        - `batches[0], batches[1]`：双 sub-batch pipeline candidate。

        这里并不会穷举最优解，而是基于 predictor 提供的 `gpu_time / cpu_time` 做一套
        启发式构造，然后比较两种方案的预测吞吐。

        Returns:
            `[gpu_only_batch]` 表示本轮走 sequential；
            `[batch0, batch1]` 表示本轮走 pipelined。
        """
        assert not self.engine_config.always_use_gpu, "This function is not designed for GPU-only mode"
        # `batches[0]/batches[1]` 是 pipeline 方案下的两个 sub-batch。
        batches = [SubBatch(self.predictor) for _ in range(2)]
        # `gpu_only_batch` 是 sequential 候选，不是“纯 GPU request”语义。
        gpu_only_batch = SubBatch(self.predictor)

        # Step 1: 先把所有 prefill 与 GPU decode 都放进第一个 sub-batch。
        # 这是两个候选方案共同的初始骨架：
        # - pipeline 方案先从 `batches[0]` 开始；
        # - sequential 方案则全部装进 `gpu_only_batch`。
        for req in gpu_prefill_reqs:
            batches[0].add_pref(req, is_gpu=True)
            gpu_only_batch.add_pref(req, is_gpu=True)

        for req in cpu_prefill_reqs:
            batches[0].add_pref(req, is_gpu=False)
            gpu_only_batch.add_pref(req, is_gpu=False)

        for req in self.gpu_decoding_q:
            batches[0].add_gdec(req)
            gpu_only_batch.add_gdec(req)

        # 没有任何可做的工作时，直接返回空 batch。
        if not batches[0] and self.num_gpu_blocks > 0:
            return []

        # Step 2: 只对 sequential candidate 做一次 prefill 裁剪。
        # 目的是避免单 batch 的 iteration width 过大；`linr_S_threshold` 是经验阈值，
        # 并不是 profile table 的一部分。
        while gpu_only_batch.get_num_prefs():
            req, is_gpu = gpu_only_batch.pop_pref()
            if is_gpu or gpu_only_batch.perfdata.s < self.predictor.linr_S_threshold:
                gpu_only_batch.add_pref(req, is_gpu)
                break

        # Step 3: 试探性把 CPU decoding 请求分配到两个 pipeline sub-batch 中。
        # `min_out_cpu_len` 记录“已经被判定放不下的最短 CPU decode 长度”；
        # 由于队列按 FCFS/长度演化顺序处理，一旦某个更短的也放不下，更长的就没必要再试。
        min_out_cpu_len = 1e9
        # 从 batch1 开始尝试加 cdec，让第二个 sub-batch 先承担 CPU decode。
        next_batch_idx = 1
        for req in self.cpu_decoding_q:
            # decode 每轮只占一个 token 宽度预算。
            if not budget.check_and_substract(1):
                break
            if req.seq_len >= min_out_cpu_len:
                budget.add(1)
                continue
            # `remains[i]` 表示把这条请求放进对应 batch 后，该 batch 的 CPU decode
            # 还能否被另一侧 GPU 工作遮蔽。
            batches[next_batch_idx].add_cdec(req)
            remains = self._get_remains(batches)
            assert all(not math.isnan(r) for r in remains), remains
            if min(remains) < 0 and self.num_gpu_blocks > 0:
                # 当前候选会让某一侧 CPU 工作溢出可遮蔽窗口，因此跳过这条请求。
                min_out_cpu_len = req.seq_len
                budget.add(1)
                batches[next_batch_idx].pop_cdec()
                continue
            # 下一条优先放到“剩余可遮蔽空间更大”的那一侧。
            next_batch_idx = remains[1] > remains[0]

        # 没有形成第二个 sub-batch 时，pipeline 方案没有意义，退回 sequential。
        if not batches[1] and self.num_gpu_blocks > 0:
            return [gpu_only_batch] # This is to prevent division by zero

        # 完全没有 GPU blocks 可用时，不再比较 sequential/pipeline rate，直接返回非空 batch。
        if self.num_gpu_blocks == 0:
            ret = []
            if len(batches[0]) > 0:
                ret.append(batches[0])
            if len(batches[1]) > 0:
                ret.append(batches[1])
            return ret


        # Step 4: 再次收缩 pipeline 方案中 batch0 的 prefill 数量。
        # 这里的目的是避免 batch0 的 prefill 过多，导致另一侧 CPU decode 长时间空转/等待。
        while batches[0].get_num_prefs():
            req, is_gpu = batches[0].pop_pref()
            if is_gpu or batches[0].perfdata.s < self.predictor.linr_S_threshold or min(self._get_remains(batches)) < 0:
                batches[0].add_pref(req, is_gpu)
                break

        # Step 5: 比较 sequential 与 pipelined 的预测吞吐。
        # 这里用每层 GPU 时间乘以层数，近似得到一轮 iteration 的总 GPU 主路径时间。
        seqential_time = gpu_only_batch.perfdata.gpu_time * self.model_config.num_layers
        pipelined_time = (batches[0].perfdata.gpu_time + batches[1].perfdata.gpu_time) * self.model_config.num_layers
        seqential_rate = len(gpu_only_batch) / seqential_time
        pipelined_rate = sum(len(batches[i]) for i in range(2)) / pipelined_time
        # print(f"Sequential time: {seqential_time}, Pipelined time: {pipelined_time}")
        # print(f"Sequential rate: {seqential_rate}, Pipelined rate: {pipelined_rate}")
        if seqential_rate < pipelined_rate:
            return batches
        else:
            return [gpu_only_batch]
        # return [gpu_only_batch]

    def _get_next_batch_new(self) -> tuple[list[SubBatch], list[Request], list[Request]]:
        """
        生成下一轮要执行的 batch，并给出 swap in/out 决策。

        返回值为 `(new_batches, newly_swapped_out, newly_swapped_in)`。

        这个函数前半段主要在整理候选集合与容量边界；真正的 mode decision
        （sequential vs pipelined）是在 `_decide_mode_and_gen_batch()` 里完成的。
        """
        # 预算的主要作用是提前限制 batch 大小，避免后续真正执行时 CUDA OOM。
        budget = ScheduleBudget(
            self.engine_config.max_batch_size,
            self.engine_config.max_tokens_in_batch
        )

        # 当前轮调度要回答三个问题：
        # 1. 新来的 prefill 去 GPU 还是 CPU；
        # 2. 哪些正在 GPU decode 的老请求需要 swap out；
        # 3. 哪些 CPU decode 请求可以 swap in。

        # 注意结合实际计算时的语义，pref_to_cpu 并不是说在 CPU 中执行
        # prefill request, 而是 prefill 计算本身仍经过 GPU，但其 生成出的 KV 
        # 会在本轮后被 swap 到 CPU（或先写入 intermediate 区，再转去 CPU）。
        pref_to_cpu = []
        pref_to_gpu = []
        swpout_reqs = []
        swpin_reqs = []

        # 这些阈值会在每轮重新计算，因为可用 block / 队列状态会持续变化。
        swap_out_threshold = self.num_gpu_blocks
        # swap_in 阈值略保守，避免刚换入就又因为贴着上界而触发换出。
        swap_in_threshold = round(swap_out_threshold * 0.95)
        cpu_threshold = self.engine_config.num_cpu_blocks - self.engine_config.num_gpu_blocks

        # Step 1: 先尽量保留当前 GPU decoding 请求。
        # 调度器默认认为：已经在 GPU 上继续 decode 的请求优先级更高，除非容量或预算不够，
        # 否则不会轻易打断它们。
        # 减少 budget
        gpu_block_needed = sum(self._get_block_needed(req) for req in self.gpu_decoding_q)
        budget.remaining_batch_size -= len(self.gpu_decoding_q)
        budget.remaining_tokens_in_batch -= len(self.gpu_decoding_q)

        # Step 2: 如有必要，先做 swap out。
        # 这一步只是在整理 decode 常驻集：当 GPU block 或 batch budget 超限时，
        # 把队尾请求移到 `cpu_decoding_q`，并记录到 `swpout_reqs`。
        # 增加 budget
        while budget.overspent or gpu_block_needed > swap_out_threshold:
            # 严格 FCFS 下，优先抢占最后启动的那个请求。
            victim = self.gpu_decoding_q.pop()
            self.cpu_decoding_q.appendleft(victim)
            swpout_reqs.append(victim)
            gpu_block_needed -= self._get_block_needed(victim)
            budget.add(1)

        # Step 3: 如果没有发生 swap out，再尝试 swap in 一部分 CPU decode 请求。
        # swap in/out 不会在同一轮同时发生；只有上一步没把 GPU 挤爆，才说明还有余量把
        # 某些 CPU decode 拉回 GPU。
        # 减少 budget
        while self.cpu_decoding_q:
            candidate = self.cpu_decoding_q[0]
            cur_block_needed = self._get_block_needed(candidate)
            if gpu_block_needed + cur_block_needed > swap_in_threshold or \
            not budget.check_and_substract(1):
                break
            gpu_block_needed += cur_block_needed
            swpin_reqs.append(candidate)
            self.cpu_decoding_q.popleft()
            self.gpu_decoding_q.append(candidate)
        assert not swpout_reqs or not swpin_reqs

        # Step 4: 试探性接纳 waiting_q 中的新 prefill 请求。
        # 这里先不真正修改 waiting_q，只是为了算出“理论上最多能接纳多少条”，
        # 并把候选拆成 `pref_to_gpu` / `pref_to_cpu` 两组，供后面的 mode decision 使用。
        # 减少budget
        itm_block_needed = 0
        # CPU decode 已经占掉的 block 数也要算进来，因为 CPU prefill 同样会消耗 CPU 侧容量。
        cpu_block_needed = sum(self._get_block_needed(req) for req in self.cpu_decoding_q) # for bounding new prefillings
        for i, candidate in enumerate(self.waiting_q):
            cur_block_needed = self._get_block_needed(candidate)

            # admission 需要同时满足：
            # - ITM / 中间态 block 不超上限；
            # - CPU 侧 block 不超上限；
            # - block table 里还有足够 request id；
            # - batch budget 仍允许加入这条 prefill。
            if  itm_block_needed + cur_block_needed > self.num_itm_blocks or \
                cpu_block_needed + cur_block_needed > cpu_threshold or \
                self.request_id_manager.get_num_available_ids() < i or \
                not budget.check_and_substract(candidate.prompt_len):
                break
            # 新 prefill 往往是后续 swap 压力的主要来源，因此这里采用简单启发式：
            # 1. 能进 GPU 就优先进 GPU；
            # 2. 一旦某个更早到达的请求只能去 CPU，为了严格 FCFS，后面的请求也都去 CPU。
            if not pref_to_cpu and gpu_block_needed + cur_block_needed <= self.num_gpu_blocks:
                gpu_block_needed += cur_block_needed
                pref_to_gpu.append(candidate)
            else:
                cpu_block_needed += cur_block_needed
                itm_block_needed += cur_block_needed
                pref_to_cpu.append(candidate)

        # Step 5: 基于上面整理出的候选集合，真正做 mode decision 并形成 batch。
        batches = self._decide_mode_and_gen_batch(pref_to_gpu, pref_to_cpu, budget)

        # Step 6: 把“候选 admission”回切成“本轮实际 launch 的 prefill 数量”。
        # 因为 `_decide_mode_and_gen_batch()` 可能会裁掉一部分 prefill，所以这里只取最终 batch
        # 里真正保留下来的那些请求，并在这一刻才从 waiting_q 弹出、分配 request id。
        real_num_prefs = sum(b.get_num_prefs() for b in batches)
        pref_to_gpu = pref_to_gpu[:real_num_prefs]
        pref_to_cpu = pref_to_cpu[:real_num_prefs - len(pref_to_gpu)]
        for _ in range(real_num_prefs):
            candidate = self.waiting_q.popleft()
            candidate.request_id = self.request_id_manager.get_id()

        if pref_to_gpu or pref_to_cpu:
            logger.info(
                "Gdecs: %d, Cdecs: %d, Pr2gs: %d, Pr2cs: %d, Waiting: %d",
                len(self.gpu_decoding_q), len(self.cpu_decoding_q), len(pref_to_gpu), len(pref_to_cpu), len(self.waiting_q)
            )

        # 真正更新 runtime 队列：被接纳的 prefill 从 waiting 进入 gpu/cpu decoding 队列，
        # 后续轮次就会把它们当作 decode 请求继续推进。
        self.gpu_decoding_q.extend(pref_to_gpu)
        self.cpu_decoding_q.extend(pref_to_cpu)

        return batches, swpout_reqs, swpin_reqs

    def _get_next_batch_old(self) -> tuple[list[SubBatch], list[Request], list[Request]]:
        """
        GPU-only 旧调度路径。

        当 `always_use_gpu=True` 时使用这条逻辑：不会做新的 load-aware pipeline mode
        decision，而是退化到更直接的 GPU decode / prefill 调度。
        """
        # 先统计当前 GPU decoding 队列占用的 block 数。
        self.num_decoding_gpu_blocks = sum(self._get_block_needed(req) for req in self.gpu_decoding_q)
        newly_swapped_out = []
        while len(self.gpu_decoding_q) > self.engine_config.max_batch_size or \
            self.num_decoding_gpu_blocks > self.num_gpu_blocks:
            # 超限时抢占最后一个运行中的请求，并把它换出到 CPU 队列。
            victim = self.gpu_decoding_q.pop()
            self.num_decoding_gpu_blocks -= self._get_block_needed(victim)
            newly_swapped_out.append(victim)
        newly_swapped_out.reverse()   # Keep it in the order of arrival time
        self.cpu_decoding_q.extendleft(newly_swapped_out)

        if not self.cpu_decoding_q:
            cur_batch = SubBatch()
            cur_batch_block_needed = self.num_decoding_gpu_blocks
            for req in self.gpu_decoding_q:
                cur_batch.add_gdec(req)

            # 若没有 CPU decode 队列阻塞，则尝试继续吸收新的 GPU prefill。
            while self.waiting_q:
                cur_seq: Request = self.waiting_q[0]
                cur_seq_block_needed = self._get_block_needed(cur_seq)
                if  len(cur_batch)+1 <= self.engine_config.max_batch_size and \
                    cur_batch_block_needed + cur_seq_block_needed <= self.num_gpu_blocks and \
                    cur_batch.perfdata.s + cur_seq.prompt_len <= self.engine_config.max_tokens_in_batch:
                    cur_batch.add_pref(cur_seq, True)
                    cur_batch_block_needed += cur_seq_block_needed
                    self.waiting_q.popleft()
                else:
                    # Strict FCFS
                    break
            if len(cur_batch):
                # 真正 launch 这批 prefill 前，给它们分配 request id。
                for req in cur_batch.gprf_reqs:
                    req.request_id = self.request_id_manager.get_id()
                if len(cur_batch.gprf_reqs) > 0:
                    logger.info(f"Waiting: {len(self.waiting_q)}, Prefs: {len(cur_batch.gprf_reqs)}, Gdecs: {len(self.gpu_decoding_q)}, Cdecs: {len(self.cpu_decoding_q)}")
                self.gpu_decoding_q.extend(cur_batch.gprf_reqs)
                self.num_decoding_gpu_blocks = cur_batch_block_needed
                return [cur_batch], [], []

        newly_swapped_in = []
        if not newly_swapped_out:
            # 本轮没有发生 swap out 时，才尝试把一部分 CPU decode 拉回 GPU。
            while self.cpu_decoding_q:
                cur_seq = self.cpu_decoding_q[0]
                num_cur_seq_blocks = self._get_block_needed(cur_seq)
                if len(self.gpu_decoding_q) + 1 <= self.engine_config.max_batch_size and \
                    self.num_decoding_gpu_blocks + num_cur_seq_blocks <= self.num_gpu_blocks:
                    self.gpu_decoding_q.append(cur_seq)
                    self.num_decoding_gpu_blocks += num_cur_seq_blocks
                    self.cpu_decoding_q.popleft()
                    newly_swapped_in.append(cur_seq)
                else:
                    break

        cur_batch = SubBatch(self.predictor)
        for req in self.gpu_decoding_q:
            cur_batch.add_gdec(req)
        return ([cur_batch] if cur_batch else []), newly_swapped_out, newly_swapped_in

    def get_next_batch(self) -> tuple[list[SubBatch], list[Request], list[Request]]:
        """
        获取下一轮要执行的 batch（或两个 sub-batches）。

        返回值统一为：
        `(new_batch(es), newly_swapped_out_reqs, newly_swapped_in_reqs)`。
        """
        if self.engine_config.always_use_gpu:
            return self._get_next_batch_old()

        return self._get_next_batch_new()

    def remove_finished_requests(self, reqs: list[Request]):
        """
        从 decoding 队列中移除本轮刚结束的请求，并归还它们的 request id。
        """
        def not_finished_func(req: Request) -> bool:
            return not req.is_finished()
        self.gpu_decoding_q = list(filter(not_finished_func, self.gpu_decoding_q))
        self.cpu_decoding_q = deque(filter(not_finished_func, self.cpu_decoding_q))

        self.request_id_manager.free_ids([req.request_id for req in reqs])
