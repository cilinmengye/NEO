"""
性能预测模块。

这个模块本身不参与 worker 的实际计算，而是给调度器提供 batch 级别的时间估计：

- linear part 时间
- GPU prefill attention 时间
- GPU decode attention 时间
- CPU decode attention 时间
- pipeline 中 CPU 侧的 launch / 调度额外开销

运行时的核心用法是：Scheduler 在试探性地向 SubBatch 中加入 / 移除请求时，
通过 BatchPerfData 调用这里的 predictor，实时估计当前候选 batch 的 gpu_time / cpu_time，
再据此决定：

- CPU decode 能否被另一侧 GPU 工作覆盖；
- prefill 是否需要裁掉一部分；
- 当前轮次应该走 sequential 还是 pipelined。
"""

from swiftllm.engine_config import EngineConfig

class PerfPredictor:
    """
    性能预测器接口。

    注意这里定义的是“估时接口”，不是执行算子。调用方不会通过它触发真实的
    prefill / decode 计算，而只是查询当前 batch 形状对应的预测时间。
    """
    def __init__(
        self, *args
    ):
        raise NotImplementedError

    def get_linr_T(self, S: int) -> float:
        """
        返回 linear part 的预测时间。

        参数 `S` 表示当前 sub-batch 的 iteration width。调度器会把 prefill token
        数与 decode request 数折算到同一个 `S` 上，再用它估计 post-layer 线性部分
        的时间。
        """
        raise NotImplementedError

    def get_pref_T(self, S: int) -> float:
        """
        返回 GPU prefill attention 的预测时间。

        参数 `S` 表示单条 prefill request 的 prompt 长度。BatchPerfData 会把同一
        sub-batch 内各条 prefill request 的预测时间逐条累加。
        """
        raise NotImplementedError

    def get_gdec_T(self, N: int) -> float:
        """
        返回 GPU decode attention 的预测时间。

        参数 `N` 表示当前 sub-batch 内 GPU decode 的累计 token 数。这里预测的是
        整个 sub-batch 的聚合 decode 代价，而不是单条请求的独立代价。
        """
        raise NotImplementedError

    def get_cdec_T(self, S: int, N: int) -> float:
        """
        返回 CPU decode attention 的预测时间。

        这里的 CPU decode 是一个二维代价函数：

        - `S`：当前被分配到 CPU decode 的 request 数
        - `N`：这些 request 的累计 token 数

        调度器会用这个值和另一侧 sub-batch 的 GPU 工作量对比，估计 CPU decode
        是否能被 pipeline 隐藏掉。
        """
        raise NotImplementedError

    def get_lnch_T(self) -> float:
        """
        返回 pipeline 中 CPU 侧固定 launch / 调度额外开销。

        当前实现里它不是按 batch 动态变化的，而是一个常量项。
        """
        raise NotImplementedError

class ZeroPerfPredictor(PerfPredictor):
    """
    一个始终返回 0 的占位 predictor。

    它的用途不是正式调度，而是让某些“只想构造 batch / 复用 SubBatch 数据结构，
    但并不关心估时”的路径可以正常运行。比如 profiler 在构造人工测试 batch 时，
    就会先使用默认的 ZeroPerfPredictor。
    """
    def __init__(
        self, *args
    ):
        pass

    def get_linr_T(self, S: int) -> float:
        return 0.0

    def get_pref_T(self, S: int) -> float:
        return 0.0

    def get_gdec_T(self, N: int) -> float:
        return 0.0

    def get_cdec_T(self, S: int, N: int) -> float:
        return 0.0

    def get_lnch_T(self) -> float:
        return 0.0

class TablePerfPredictor(PerfPredictor):
    """
    基于 profile table 的性能预测器。

    它会先在启动阶段由 ModelProfiler 生成多张离线 profile table，运行时再通过
    查表 + 插值来估算未直接采样过的 batch 形状。

    因此它并不是在线学习模块；真实在线调度时这里只做只读查询，不会重新 profile。
    """
    def __init__(
        self,
        engine_config: EngineConfig
    ):
        # Linr: linear/post-layer 部分按 iteration width S 建一维表。
        # 前 1~511 全保留，之后按 2 的幂稀疏采样，以降低 profile 点数。
        self.linr_S_list = list(range(1, 512)) + [
            2 ** i for i in range(
                9,
                (engine_config.max_tokens_in_batch - 1).bit_length()
            )
        ] + [engine_config.max_tokens_in_batch]
        self.linr_T_list = None
        # lower-bound 查找表：给定任意整数 S，可 O(1) 找到第一个 >= S 的采样点下标。
        self.linr_S_lb_idx = self._get_lb_idx_list(self.linr_S_list)
        # 调度 heuristic：当 sequential candidate 的宽度过大时，会用它裁掉一部分 prefill。
        # 它不是 profile table 的一部分，而是 scheduler 里的经验阈值。
        self.linr_S_threshold = 128 # NOTE: This is a heuristic value

        # Pref: GPU prefill attention 按单条 prompt 长度 S 建一维表。
        # 这里用 3*2^(i-2) 和 2^i 交错采样，在稀疏覆盖大范围时比纯 2 的幂更平滑。
        self.pref_S_list = sum([[2 ** (i-2) * 3, 2 ** i] for i in range(
            (engine_config.block_size - 1).bit_length(),
            (engine_config.max_tokens_in_batch - 1).bit_length()
        )], []) + [engine_config.max_tokens_in_batch]
        self.pref_T_list = None
        self.pref_S_lb_idx = self._get_lb_idx_list(self.pref_S_list)

        # Gdec: GPU decode attention 按 sub-batch 内累计 decode token 数 N 建一维表。
        self.gdec_N_list = sum([[2 ** (i-2) * 3, 2 ** i] for i in range(
            (engine_config.block_size - 1).bit_length(),
            (engine_config.max_gpu_tokens - 1).bit_length()
        )], []) + [engine_config.max_gpu_tokens]
        self.gdec_T_list = None
        self.gdec_N_lb_idx = self._get_lb_idx_list(self.gdec_N_list)

        # Cdec: CPU decode attention 是二维代价函数。
        # 第一维 S_c 表示 CPU decode request 数。
        self.cdec_S_list = [2 ** i for i in range(
            0,
            (engine_config.max_batch_size - 1).bit_length()
        )] + [engine_config.max_batch_size]
        # 第二维 N_c 表示这些 CPU decode request 的累计 token 数。
        # 每一行的合法范围都不同，所以先按每个 S 单独保存一组 N 采样点。
        self.cdec_N_lists = [
            [S * engine_config.block_size] +
            [2 ** i for i in range(
                (S * engine_config.block_size).bit_length(),
                (min(S * engine_config.max_seq_len, engine_config.max_cpu_tokens) - 1).bit_length()
            )] +
            [min(S * engine_config.max_seq_len, engine_config.max_cpu_tokens)]
            for S in self.cdec_S_list
        ]
        # 为了运行时统一查表 / 插值，再把所有行出现过的 N 采样点并成一个全局网格。
        self.cdec_N_list_agg = sorted(list(set(sum(self.cdec_N_lists, []))))

        # cdec_T_lists[i][j] 对应第 i 个 S_c 与聚合网格中第 j 个 N_c 的估计时间。
        # 初始化时先占位，真正的二维表由 ModelProfiler 回填。
        self.cdec_T_lists = [None]
        self.cdec_S_lb_idx = self._get_lb_idx_list(self.cdec_S_list)
        self.cdec_N_lb_idx = self._get_lb_idx_list(self.cdec_N_list_agg)

        # Lnch: pipeline 中 CPU 侧额外开销。
        # 当前实现使用经验常数 0.8ms，而不是启动时重新 profile 的结果。
        self.lnch_T = 0.8
        # self.lnch_T = self._profile_lnch(lnch_S_list)

    def _get_lb_idx_list(self, input_list: list[int]) -> list[int]:
        """
        预计算 lower-bound 查找表。

        返回的列表满足：给定整数 `x`，`ret[x]` 是最小的 `j`，使得
        `input_list[j] >= x`。

        这样运行时在 `_interp_1d()` 中就不需要每次做二分搜索，而是可以用 `x` 直接
        O(1) 查到应落在哪个采样区间。
        """
        return sum(
            [[i+1] * (input_list[i+1] - input_list[i]) for i in range(len(input_list) - 1)],
            [0] * (input_list[0] + 1)
        )

    def _interp(self, x: int, x0: int, x1: int, y0: float, y1: float) -> float:
        """
        对两个采样点做一次线性插值。

        `x0/x1` 是相邻采样点，`y0/y1` 是对应 profile 时间。调用方保证 `x` 落在
        区间 `[x0, x1]` 内。
        """
        return y0 + (y1 - y0) * (x - x0) / (x1 - x0)

    def _interp_1d(self, x, xs: list[int], ys: list[float], x_lb_idx: list[int]) -> float:
        """
        在一维 profile table 上查询 / 插值。

        这个函数服务于 `linr/pref/gdec` 三类一维代价模型：

        - 如果 `x` 正好命中采样点，直接返回该点的 profile 时间；
        - 否则用 lower-bound 查找表找到左右相邻采样点，再做线性插值；
        - `x == 0` 时直接返回 0，表示该类工作量为空。
        """
        assert x <= xs[-1], f"x={x} exceeds the maximum {xs[-1]}"
        if x == 0:
            return 0.0
        idx = x_lb_idx[x]
        if idx == 0 or x == xs[idx]:
            return ys[idx]
        return self._interp(x, xs[idx-1], xs[idx], ys[idx-1], ys[idx])

    def get_linr_T(self, S: int) -> float:
        """
        查询 linear/post-layer 部分的预测时间。

        这里的 `S` 是当前 sub-batch 的总 iteration width，而不是单条请求长度。
        """
        return self._interp_1d(S, self.linr_S_list, self.linr_T_list, self.linr_S_lb_idx)

    def get_pref_T(self, S: int) -> float:
        """
        查询单条 prefill request 的 GPU prefill attention 预测时间。

        BatchPerfData 会把多条 prefill request 的结果累加为整个 sub-batch 的 `pref_T`。
        """
        return self._interp_1d(S, self.pref_S_list, self.pref_T_list, self.pref_S_lb_idx)

    def get_gdec_T(self, N: int) -> float:
        """
        查询 GPU decode attention 的聚合预测时间。

        `N` 是当前 sub-batch 内所有 GPU decode request 的累计 token 数。
        """
        return self._interp_1d(N, self.gdec_N_list, self.gdec_T_list, self.gdec_N_lb_idx)

    def get_cdec_T(self, S: int, N: int) -> float:
        """
        查询 CPU decode attention 的二维预测时间。

        查询顺序分两步：

        1. 先在 `S_c` 维度找到当前 CPU decode request 数所在的邻近两行；
        2. 再在每一行上按 `N_c` 做一维查询 / 插值；
        3. 最后沿 `S_c` 方向再做一次插值。

        因此虽然代码看起来分步实现，本质上是在做 `(S_c, N_c)` 上的双线性插值。
        """
        assert S < len(self.cdec_S_lb_idx), f"CPU batch size {S} exceeds the maximum {len(self.cdec_S_lb_idx)}"
        if S == 0:
            return 0.0
        s_idx = self.cdec_S_lb_idx[S]
        if s_idx == 0 or S == self.cdec_S_list[s_idx]:
            return self._interp_1d(N, self.cdec_N_list_agg, self.cdec_T_lists[s_idx], self.cdec_N_lb_idx)
        s1 = self.cdec_S_list[s_idx]
        s0 = self.cdec_S_list[s_idx - 1]
        ts1 = self._interp_1d(N, self.cdec_N_list_agg, self.cdec_T_lists[s_idx], self.cdec_N_lb_idx)
        ts0 = self._interp_1d(N, self.cdec_N_list_agg, self.cdec_T_lists[s_idx - 1], self.cdec_N_lb_idx)
        return self._interp(S, s0, s1, ts0, ts1)

    def get_lnch_T(self) -> float:
        """
        返回固定 launch 开销。

        调度器会把它并入 `BatchPerfData.cpu_time`，作为 pipeline 中 CPU 侧额外的固定成本。
        """
        return self.lnch_T
