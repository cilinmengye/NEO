"""
Llama 模型性能 profiling 模块。

这个模块的职责不是在线调度，而是在引擎启动阶段构造一组人工 batch，实际跑模型并
记录各类时间，然后把结果整理成 `TablePerfPredictor` 需要的 profile tables。

运行时 scheduler 只做查表 / 插值；不会在在线请求路径里重新做 profiling。
"""

import time
import os
import sys
import json
import math
import logging

import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from swiftllm.perfpredictor import TablePerfPredictor
from swiftllm.structs import create_request, SubBatch
from swiftllm.utils import GB

from swiftllm.worker.model import ModelPerfResult
from swiftllm.server.block_manager import BlockManager
from swiftllm.server.executor import Executor

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

class ModelProfiler:
    """
    启动阶段的模型 profiler。

    它负责两件事：
    - 测出 runtime 可用的 GPU / CPU KV cache block 数；
    - 为 `TablePerfPredictor` 构造 linr / pref / gdec / cdec 等 profile table。

    `executor` 负责真正执行 profiling case；`pp` 是最终回填完成、并注入 scheduler
    的预测器；`bm` 在存在时用于让 profiling case 经过更真实的 prepare / update 路径。
    """
    @torch.inference_mode()
    def __init__(self, executor: Executor):
        self.engine_config = executor.engine_config
        self.model_config = executor.model_config
        self.executor = executor
        # `pp` 会在 `init_profile_tables()` 中创建并逐张表回填。
        self.pp = None
        # `bm` 为可选；没有它时 profiling 直接走 `set_model_forward_args()`。
        self.bm = None
        os.makedirs(self.engine_config.profile_result_path, exist_ok=True)

    def init_profile_tables(self, block_manager: BlockManager):
        """
        初始化所有 profile tables。

        顺序是：
        1. 先创建空的 `TablePerfPredictor`；
        2. 再逐张 profile 并把结果写回 `self.pp.*_T_list(s)`；
        3. 初始化结束后，运行时调度只会查询 `self.pp`，不会重新 profile。
        """
        # Validate necessary constraints
        engine_config = self.executor.engine_config

        self.bm = block_manager
        self.pp = TablePerfPredictor(engine_config)

        self.pp.linr_S_list, self.pp.linr_T_list = self._profile_linr(self.pp.linr_S_list)
        self.pp.pref_S_list, self.pp.pref_T_list = self._profile_pref(self.pp.pref_S_list)
        self.pp.gdec_N_list, self.pp.gdec_T_list = self._profile_gdec(self.pp.gdec_N_list)
        self.pp.cdec_S_list, self.pp.cdec_N_lists, self.pp.cdec_T_lists = self._profile_cdec(self.pp.cdec_S_list, self.pp.cdec_N_lists)


    def _run_test_case_seq(
        self,
        pref_lens: list[int],
        gdec_lens: list[int],
        cdec_lens: list[int],
        nwarmup = 2,
        nrepeat = 3
    ):
        """
        跑一个单 batch / sequential profiling case。
        """
        return self._run_test_case([pref_lens], [gdec_lens], [cdec_lens], nwarmup, nrepeat)


    def _run_test_case_pip_same(
        self,
        pref_lens: list[int],
        gdec_lens: list[int],
        cdec_lens: list[int],
        nwarmup = 2,
        nrepeat = 3
    ):
        """
        跑一个双 sub-batch 且两侧形状相同的 profiling case。

        这个 helper 主要用于估计 pipeline 相关固定开销，例如 launch time。
        """
        return self._run_test_case([pref_lens] * 2, [gdec_lens] * 2, [cdec_lens] * 2, nwarmup, nrepeat)


    @torch.inference_mode()
    def _run_test_case(
      self,
      pref_lens: list[list[int]],
      gdec_lens: list[list[int]],
      cdec_lens: list[list[int]],
      nwarmup = 2,
      nrepeat = 3
    ) -> ModelPerfResult:
        """
        构造并执行一个人工 profiling case，返回性能统计结果。

        参数里的 `pref_lens / gdec_lens / cdec_lens` 描述的是“每个 sub-batch 内各类 request
        的长度分布”，而不是真实用户请求。函数内部会据此直接伪造 `SubBatch`。
        """
        # print(f"Running test case with pref_lens={pref_lens}, gdec_lens={gdec_lens}, cdec_lens={cdec_lens}")

        nbatches = len(pref_lens)
        # 当前 profiling 只支持 sequential(1 batch) 或双 batch pipeline(2 batches) 两种形态。
        assert nbatches in (1, 2), "Only support 1 or 2 batches"

        batches = []
        offs = 0
        for i in range(nbatches):
            # profiling 时直接构造人工 SubBatch，而不是走 scheduler 的真实 admission 流程。
            batch = SubBatch()
            npref = len(pref_lens[i])
            ngdec = len(gdec_lens[i])
            ncdec = len(cdec_lens[i])

            for j in range(npref):
                batch.add_pref(create_request([10] * pref_lens[i][j], offs + j, [], True), is_gpu=True)

            for j in range(ngdec):
                # decode case 需要已有历史 token，因此这里用 `output_token_ids=[10]` 伪造“已经 decode 过一步”的状态。
                batch.add_gdec(create_request([10] * (gdec_lens[i][j] - 1), offs + npref + j, [10], True))

            for j in range(ncdec):
                batch.add_cdec(create_request([10] * (cdec_lens[i][j] - 1), offs + npref + ngdec + j, [10], True))

            offs += npref + ngdec + ncdec
            batches.append(batch)

        if self.bm is None:
            # 没有 block manager 时，直接把调度期 SubBatch 转成 forward 所需字段。
            for batch in batches:
                batch.set_model_forward_args(self.model_config)
            args = (([], []), ([], [])), ([], [])
        else:
            # 有 block manager 时，profiling case 也走 prepare 路径，这样更接近真实 runtime。
            args = self.bm.prepare(batches, [], [])

        for i in range(-nwarmup, nrepeat):
            if i == 0:
                # 先做若干轮 warmup，再打开 perf monitor，仅统计正式重复段。
                self.executor.turn_on_perf_monitor()
            # Directly call this since we already allocated the blocks
            output_tokens = self.executor.do_one_iteration(batches, *args)

        if self.bm is not None:
            # 所有 profiling request 都是 quick-stop，因此跑完一轮后会全部结束，可以立即回收。
            self.bm.update_and_free(batches, output_tokens)

        res = self.executor.turn_off_perf_monitor_and_flush_results()
        return res

    def _profile_linr(
        self,
        S_list: list[int]
    ) -> list[float]:
        """
        Profile linear/post-layer 部分。

        输入维度 `S` 是 iteration width；测试 case 用一条长度为 `S` 的 prefill request 构造，
        最终读取 `avg_linr_time` 作为表值。
        """
        result_path = self.engine_config.profile_result_path + "linr.json"

        if os.path.exists(result_path):
            with open(result_path, "r") as f:
                table = json.load(f)
            # 只要缓存表覆盖了当前需要的最大上界，就直接复用，避免重复 profile。
            if table["S_list"][-1] >= S_list[-1]:
                return table["S_list"], table["T_list"]
        else:
            table = {
                "S_list": [],
                "T_list": []
            }

        print(f"Profiling linear part with S_list={S_list} ...")

        T_list = []
        # 倒序 profile 可以更早暴露大 batch / 大宽度下的 OOM 或性能异常。
        for S in tqdm(list(reversed(S_list))):
            if S in table["S_list"]:
                T_list.append(table["T_list"][table["S_list"].index(S)])
                continue
            res = self._run_test_case_seq(
                pref_lens=[S],
                gdec_lens=[],
                cdec_lens=[]
            )
            T_list.append(ModelPerfResult.mean(res, "avg_linr_time"))
        T_list = list(reversed(T_list))

        with open(result_path, "w") as f:
            json.dump({
            "S_list": S_list,
            "T_list": T_list
            }, f, indent=2)

        plt.figure(figsize=(16, 12))
        plt.plot(S_list, T_list)
        plt.xlim(0)
        plt.ylim(0)
        plt.xlabel("S")
        plt.ylabel("T_l(ms)")
        plt.savefig(self.engine_config.profile_result_path + "linr.png")
        plt.close()

        return S_list, T_list

    def _profile_pref(
        self,
        S_list: list[int]
    ) -> list[list[float]]:
        """
        Profile GPU prefill attention 部分。

        输入维度 `S` 是单条 prefill request 的 prompt 长度；测试 case 同样构造一条长度为
        `S` 的 prefill request，并读取 `avg_pref_time`。
        """
        result_path = self.engine_config.profile_result_path + "pref.json"

        if os.path.exists(result_path):
            with open(result_path, "r") as f:
                table = json.load(f)
            if table["S_list"][-1] >= S_list[-1]:
                return table["S_list"], table["T_list"]

        print(f"Profiling prefill part with S_list={S_list}...")

        T_list = []
        for S in tqdm(S_list):
            res = self._run_test_case_seq(
                pref_lens=[S],
                gdec_lens=[],
                cdec_lens=[]
            )
            T_list.append(ModelPerfResult.mean(res, "avg_pref_time"))

        plt.figure(figsize=(16, 12))
        plt.plot(S_list, T_list)
        plt.xlim(0)
        plt.ylim(0)
        plt.xlabel("S")
        plt.ylabel("T(ms)")
        plt.savefig(self.engine_config.profile_result_path + "pref.png")
        plt.close()

        with open(result_path, "w") as f:
            json.dump({
            "S_list": S_list,
            "T_list": T_list
            }, f, indent=2)

        return S_list, T_list

    def _profile_gdec(
        self,
        N_list: list[int]
    ) -> list[float]:
        """
        Profile GPU decode attention 部分。

        输入维度 `N` 是 sub-batch 内所有 GPU decode request 的累计 token 数；这里会把
        `N` 尽量切成若干条 decode request，使其总 token 数恰好为 `N`，再读取 `avg_gdec_time`。
        """
        result_path = self.engine_config.profile_result_path + "gdec.json"

        if os.path.exists(result_path):
            with open(result_path, "r") as f:
                res = json.load(f)
            if res["N_list"][-1] >= N_list[-1]:
                return res["N_list"], res["T_list"]

        print(f"Profiling GPU attention part with N_list={N_list} ...")

        T_list = []
        L = self.engine_config.max_seq_len
        for N in tqdm(N_list):
            res = self._run_test_case_seq(
                pref_lens=[],
                # 用若干条 decode request 拼出总 token 数 N；每条长度不超过 max_seq_len。
                gdec_lens=[L] * ((N - 1) // L) + [(N - 1) % L + 1],
                cdec_lens=[]
            )
            T_list.append(ModelPerfResult.mean(res, "avg_gdec_time"))

        with open(result_path, "w") as f:
            json.dump({
            "N_list": N_list,
            "T_list": T_list
            }, f, indent=2)

        plt.figure(figsize=(16, 12))
        plt.plot(N_list, T_list)
        plt.xlim(0)
        plt.ylim(0)
        plt.xlabel("N")
        plt.ylabel("T(ms)")
        plt.savefig(self.engine_config.profile_result_path + "gdec.png")
        plt.close()

        return N_list, T_list

    def _profile_cdec(
        self,
        S_list: list[int],
        N_lists: list[list[int]]
    ) -> list[list[float]]:
        """
        Profile CPU decode attention 的二维代价表。

        - `S_c`：CPU decode request 数
        - `N_c`：这些 request 的累计 token 数

        由于每个 `S_c` 下合法的 `N_c` 取值范围不同，这里先按每行各自的 `N_lists[i]` 做
        profile，再投影到 `self.pp.cdec_N_list_agg` 这个统一网格上。
        """
        result_path = self.engine_config.profile_result_path + "cdec.json"

        if os.path.exists(result_path):
            with open(result_path, "r") as f:
                table = json.load(f)
            if table["S_list"][-1] >= S_list[-1] and table["N_lists"][-1][-1] >= N_lists[-1][-1]:
                return table["S_list"], table["N_lists"], table["T_lists"]

        print(f"Profiling CPU attention part with S_list={S_list}, N_lists={N_lists} ...")

        T_lists = []
        block_size = self.engine_config.block_size
        for i, S in enumerate(tqdm(S_list)):
            T_lists.append([])
            for N in self.pp.cdec_N_list_agg:
                # 这个 N 小于当前行允许的最小累计 token 数，先记成 0.0 占位，后续再补表。
                if N < N_lists[i][0]:
                    T_lists[-1].append(0.0)
                    continue
                # 这个 N 超出当前行允许的最大累计 token 数，先记成 inf 占位，后续再补表。
                if N > N_lists[i][-1]:
                    T_lists[-1].append(float("inf"))
                    continue
                assert N % block_size == 0, "N must be divisible by block size"
                # NB 表示总共跨了多少个 KV blocks；后面会把这些 block 尽量平均分到 S 条 cdec request 上。
                NB = N // block_size
                res = self._run_test_case_seq(
                    # Divide N into S segments as even as possible
                    pref_lens=[],
                    gdec_lens=[],
                    cdec_lens=[NB // S * block_size] * (S - NB % S) + [(NB // S + 1) * block_size] * (NB % S),
                )
                T_lists[-1].append(ModelPerfResult.mean(res, "avg_cdec_time"))

        nS = len(S_list)
        nN = len(self.pp.cdec_N_list_agg)
        for i in range(nS):
            for j in reversed(range(nN)):
                if T_lists[i][j] == 0.0:
                    # 第一轮补左下区域：沿着左侧与上侧已知值做双线性风格外推，
                    # 使聚合二维表在统一网格上保持可查询。
                    assert i > 0 and j < nN - 1
                    T_lists[i][j] = T_lists[i - 1][j] + T_lists[i][j + 1] - T_lists[i - 1][j + 1]

        for i in reversed(range(nS)):
            for j in range(nN):
                if T_lists[i][j] == float("inf"):
                    # 第二轮补右上区域：同理，用右下邻域的已知值把超上界位置补齐。
                    assert i < nS - 1 and j > 0
                    T_lists[i][j] = T_lists[i + 1][j] + T_lists[i][j - 1] - T_lists[i + 1][j - 1]

        T_array = np.array(T_lists)

        plt.figure(figsize=(16, 12))
        ax = plt.axes(projection='3d')
        ax.plot_surface(
            np.outer(S_list, np.ones(nN)),
            np.outer(np.ones(nS), self.pp.cdec_N_list_agg),
            T_array,
            label = "CPU"
        )

        with open(result_path, "w") as f:
            json.dump({
            "S_list": S_list,
            "N_lists": N_lists,
            "T_lists": T_lists
            }, f, indent=2)

        ax.set_xlim(0)
        ax.set_ylim(0)
        ax.set_xlabel("S_c")
        ax.set_ylabel("N_c")
        ax.set_zlabel("T(ms)")
        plt.savefig(self.engine_config.profile_result_path + "cdec.png")
        plt.close()

        return S_list, N_lists, T_lists

    def _profile_lnch(
        self,
        S_list: list[int]
    ) -> list[float]:
        """
        Profile pipeline 中的 kernel launch / 调度固定开销。

        该函数仍然保留在代码里，但当前 `TablePerfPredictor` 默认使用的是固定常数
        `lnch_T = 0.8`，而不是这里动态 profile 的结果。
        """
        result_path = self.engine_config.profile_result_path + "lnch.json"

        if os.path.exists(result_path):
            with open(result_path, "r") as f:
                res = json.load(f)
            if res["S_list"] == S_list:
                return res["T_list"]

        print(f"Profiling kernel launch time with S_list={S_list} ...")

        T_list = []
        for S in tqdm(S_list):
            res = self._run_test_case_pip_same(
                pref_lens=[S // 2 - 2 * (S // 10)],
                gdec_lens=[10] * (S // 10),
                cdec_lens=[10] * (S // 10)
            )
            T_list.append(ModelPerfResult.mean(res, "avg_lnch_time"))

        with open(result_path, "w") as f:
            json.dump({
            "S_list": S_list,
            "T_list": T_list
            }, f, indent=2)

        plt.figure(figsize=(16, 12))
        plt.plot(S_list, T_list)
        plt.xlim(0)
        plt.ylim(0)
        plt.xlabel("S")
        plt.ylabel("T(ms)")
        plt.savefig(self.engine_config.profile_result_path + "lnch.png")
        plt.close()

        T_mean = np.array(T_list).mean()

        return T_mean

    @torch.inference_mode()
    def profile_num_blocks(self):
        """
        估算 runtime 可分配的 GPU / CPU KV cache block 数。

        这里测的不是单个算子时间，而是容量边界：
        - 先根据单个 KV slot 大小推导 GPU / CPU block 的字节数；
        - CPU blocks 可直接由 `swap_space` 推出；
        - GPU blocks 则通过一个“最大压力 prefill batch”的峰值显存占用反推。

        Finally, we set the number of GPU blocks in the engine configuration.
        """
        engine_config = self.engine_config
        model_config = self.model_config
        gpu_block_size_bytes = engine_config.block_size * model_config.get_kvslot_size(engine_config.extra_layer_for_cprf)
        cpu_block_size_bytes = engine_config.block_size * model_config.get_kvslot_size(False)
        engine_config.num_cpu_blocks = engine_config.swap_space * GB // cpu_block_size_bytes

        if self.engine_config.num_gpu_blocks_override == -1:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            ws = self.engine_config.tensor_parallel_degree

            # 构造一个尽可能重的 prefill batch，目的是逼近 runtime 下 KV/cache/memory pressure 的上界。
            N = engine_config.max_tokens_in_batch
            S = engine_config.max_batch_size
            self._run_test_case_seq(
                pref_lens=[N // S] * (S - N % S) + [N // S + 1] * (N % S),
                gdec_lens=[],
                cdec_lens=[],
                nrepeat=1, nwarmup=0
            )
            torch.cuda.synchronize()

            # peak_memory = torch.cuda.max_memory_allocated()
            # total_memory = torch.cuda.get_device_properties(0).total_memory
            peak_memory = 0
            for i in range(ws):
                free_memory, total_memory = torch.cuda.mem_get_info(i)
                single_peak_memory = total_memory - free_memory
                peak_memory = max(peak_memory, single_peak_memory)
            useable_memory = total_memory * engine_config.gpu_mem_utilization
            print(f"[Engine.profiler] GPU total memory: {total_memory/GB:.2f} GB, runtime peak memory: {peak_memory/GB:.2f} GB")
            if useable_memory < peak_memory:
                raise RuntimeError(
                    f"Peak memory {peak_memory/GB:.2f} GB exceeds usable memory {useable_memory/GB:.2f} GB "
                    f"({total_memory/GB:.2f} GB * {engine_config.gpu_mem_utilization})"
                )

            torch.cuda.empty_cache()
            # 用“可用于 KV cache 的显存 / 单 block 字节数”反推出可分配 GPU blocks。
            engine_config.num_gpu_blocks = math.floor((useable_memory - peak_memory) / gpu_block_size_bytes)
        else:
            engine_config.num_gpu_blocks = engine_config.num_gpu_blocks_override

        assert engine_config.num_gpu_blocks * self.engine_config.block_size >= self.engine_config.max_tokens_in_batch, \
            f"Number of GPU blocks {self.engine_config.num_gpu_blocks} is not enough to hold the maximum batch size"

        num_gpu_blocks = engine_config.num_gpu_blocks
        num_cpu_blocks = engine_config.num_cpu_blocks
        logger.info(f"[Engine.profiler] Number of GPU blocks: {num_gpu_blocks} ({num_gpu_blocks * gpu_block_size_bytes/GB:.2f} GB)")
        logger.info(f"[Engine.profiler] Number of CPU blocks: {num_cpu_blocks} ({num_cpu_blocks * cpu_block_size_bytes/GB:.2f} GB)")
