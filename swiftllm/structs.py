import asyncio
import dataclasses
from swiftllm.perfpredictor import PerfPredictor, ZeroPerfPredictor
from swiftllm.model_config import LlamaModelConfig

@dataclasses.dataclass
class StepOutput:
    """
    一次 decoding step 的输出。
    """
    token_id: int
    request: "Request"


class RawRequest:
    """
    用户提交到引擎的原始请求。
    """
    prompt: str | list[int]
    max_output_len: int

    def __init__(self, prompt: str | list[int], max_output_len: int):
        self.prompt = prompt
        self.max_output_len = max_output_len


class Request:
    """
    系统中的请求对象。

    一个请求会经历 waiting / processing / finished 等状态；这里保存的是调度器、
    block manager 与执行器在整个生命周期中都会访问的核心字段。
    """

    prompt_token_ids: list[int]     # 到达后由 tokenizer 生成的 prompt token ids
    prompt_len: int     # `prompt_token_ids` 的长度
    output_len: int     # 当前已经生成的输出 token 数
    max_output_len: int     # 最终最多允许生成的输出 token 数

    output_q: asyncio.Queue[StepOutput] # streaming 模式下逐 token 向外回传结果
    finished_event: asyncio.Event       # non-streaming 模式下用于等待请求结束

    request_id: int     # request 在 block table 中的槽位 id
    output_token_ids: list[int]     # 当前已经生成出的输出 token 序列

    @property
    def seq_len(self) -> int:
        return self.prompt_len + self.output_len

    def __init__(self, raw_request: RawRequest):
        # Request 在进入 untokenized_raw_requests 时先创建壳对象；
        # 真正的 `prompt_token_ids` / `prompt_len` 会在 tokenization 完成后再填入。
        self.prompt_token_ids = []
        self.prompt_len = 0
        self.max_output_len = raw_request.max_output_len
        self.output_len = 0
        self.output_q = asyncio.Queue()
        self.finished_event = asyncio.Event()
        self.request_id = -1
        self.output_token_ids = []

    def is_finished(self) -> bool:
        return self.output_len == self.max_output_len

    @staticmethod
    def get_ids(reqs: list["Request"]) -> list[int]:
        """
        提取一组请求的 request id。
        """
        return [req.request_id for req in reqs]

    @staticmethod
    def get_lens(reqs: list["Request"]) -> list[int]:
        """
        提取一组请求当前的总序列长度。
        """
        return [req.seq_len for req in reqs]


    @staticmethod
    def get_input_tokens(reqs: list["Request"]) -> list[list[int]]:
        """
        返回一次 model forward 需要拼接的输入 token。

        当前实现里：
        - prefill 请求输入完整 prompt；
        - decode 请求只输入上一步新生成的最后一个 token。
        """
        return sum([req.prompt_token_ids if req.output_len == 0 else req.output_token_ids[-1:] for req in reqs], [])


    @staticmethod
    def update_output(reqs: list["Request"], output_toks: list[int]) -> list["Request"]:
        """
        用本轮 forward 的输出 token 更新请求状态。

        要求 `reqs` 与 `output_toks` 顺序一一对应；返回本轮刚结束的请求列表。
        """
        assert len(reqs) == len(output_toks), f"Number of requests {len(reqs)} and output tokens {len(output_toks)} do not match"
        finished_reqs = []
        for req, tok in zip(reqs, output_toks):
            req.output_len += 1
            req.output_token_ids.append(tok)
            req.output_q.put_nowait(StepOutput(tok, req))
            if req.is_finished():
                req.finished_event.set()
                finished_reqs.append(req)
        return finished_reqs


    def __getstate__(self):
        """
        为序列化提取最小必要状态。

        这里只保留运行时重建请求所需的字段，而不会把整个对象的异步状态一起序列化。
        """
        return {
            "prompt_token_ids": self.prompt_token_ids if self.output_len == 0 else [],
            "output_token_ids": self.output_token_ids[-1:] if self.output_len > 0 else [],
            "prompt_len": self.prompt_len,
            "output_len": self.output_len,
            "request_id": self.request_id
        }


    def __setstate__(self, state):
        """
        从序列化结果恢复请求状态。
        """
        self.prompt_token_ids = state["prompt_token_ids"]
        self.output_token_ids = state["output_token_ids"]
        self.prompt_len = state["prompt_len"]
        self.output_len = state["output_len"]
        self.request_id = state["request_id"]


def create_request(
    prompt_token_ids: list[int],
    req_id: int,
    output_token_ids: list[int] | None = None,
    quick_stop: bool = False
) -> Request:
    """
    构造一个人工 request。

    这个 helper 主要服务于 profiler / 测试路径：可以直接伪造 prompt、历史输出和
    request id，而不必经过真实的用户请求进入流程。

    `quick_stop=True` 时会把 `max_output_len` 设为“当前输出长度 + 1”，这样 profiling
    case 在跑完一轮后就会自然结束，便于 block manager 回收资源。
    """
    ret = Request(RawRequest("", 0))
    ret.prompt_token_ids = prompt_token_ids
    ret.output_token_ids = output_token_ids or []
    ret.prompt_len = len(ret.prompt_token_ids)
    ret.output_len = len(ret.output_token_ids)
    ret.max_output_len = ret.output_len + 1 if quick_stop else 10 ** 9
    ret.request_id = req_id
    return ret

class BatchPerfData:
    """
    调度阶段维护的 batch 级性能状态。

    它不是 profile table 本身，而是 scheduler 在“试探性 add/pop request”时实时维护的
    一份聚合统计；真正的时间估计由内部持有的 `predictor` 按当前状态查表得到。
    """
    # pylint: disable=too-many-instance-attributes, missing-function-docstring
    def __init__(self, predictor: PerfPredictor):
        # `x`: 当前 sub-batch 的总请求数。
        self.x = 0
        # `s`: 当前 sub-batch 的 iteration width。
        # 对 prefill 来说累加 prompt_len，对 decode 来说每条请求只贡献 1。
        self.s = 0
        # `n_g`: 当前 sub-batch 内 GPU decode 请求的累计 token 数。
        self.n_g = 0
        # `x_c`: 当前 sub-batch 内被分配到 CPU decode 的请求数。
        self.x_c = 0
        # `n_c`: 这些 CPU decode 请求的累计 token 数。
        self.n_c = 0

        # 运行时共享的性能预测器。调度器不会直接执行算子，而是通过它查询估时。
        self.predictor = predictor
        # prefill 时间按 request 粒度逐条累加，因此 add/pop 时可以增量维护。
        self.pref_T = 0
        # GPU decode 时间不是“各 request 独立时间之和”，而是 `n_g` 上的聚合代价，
        # 所以每次 add_gdec 后都重新按新的总 `n_g` 查一次表。
        self.gdec_T = 0
        # pipeline 中 CPU 侧额外 launch / 调度固定开销。
        self.lnch_T = predictor.get_lnch_T()

    def add_pref(self, prompt_len):
        """
        向候选 batch 中加入一条 prefill request。

        prefill attention 的建模粒度是“单条 request 的 prompt 长度”，所以这里直接把
        `get_pref_T(prompt_len)` 累加到 `pref_T` 上。
        """
        self.x += 1
        self.s += prompt_len
        # 累加？而且不管是 cpu prefill 还是 gpu prefill
        self.pref_T += self.predictor.get_pref_T(prompt_len)

    def pop_pref(self, prompt_len):
        """
        从候选 batch 中移除一条 prefill request，并回退对应的聚合状态。
        """
        self.x -= 1
        self.s -= prompt_len
        self.pref_T -= self.predictor.get_pref_T(prompt_len)

    def add_gdec(self, seq_len):
        """
        向候选 batch 中加入一条 GPU decode request。

        这里 `s += 1` 是因为 decode 在 post-layer / linear part 的宽度贡献按“请求数”计；
        `n_g += seq_len` 则用于估计 GPU decode attention 的聚合代价。
        """
        self.x += 1
        self.s += 1
        self.n_g += seq_len
        self.gdec_T = self.predictor.get_gdec_T(self.n_g)

    def add_cdec(self, seq_len):
        """
        向候选 batch 中加入一条 CPU decode request。

        CPU decode 的估时是二维函数 `cdec(S_c, N_c)`，因此这里先只维护 `(x_c, n_c)`；
        真正的 `cdec_T` 在读取属性时再按最新状态查 predictor。
        """
        self.x += 1
        self.s += 1
        self.x_c += 1
        self.n_c += seq_len

    def pop_cdec(self, seq_len):
        """
        从候选 batch 中移除一条 CPU decode request，并回退二维状态 `(x_c, n_c)`。
        """
        self.x -= 1
        self.s -= 1
        self.x_c -= 1
        self.n_c -= seq_len

    @property
    def linr_T(self) -> float:
        """
        当前 iteration width 对应的 linear/post-layer 预测时间。
        """
        return self.predictor.get_linr_T(self.s)

    @property
    def cdec_T(self) -> float:
        """
        当前 CPU decode 状态 `(x_c, n_c)` 对应的预测时间。
        """
        return self.predictor.get_cdec_T(self.x_c, self.n_c)

    @property
    def gpu_time(self) -> float:
        """
        当前 sub-batch 在 GPU 侧的总预测时间。

        scheduler 在比较 sequential / pipelined 方案时，主要就依赖这个聚合值。
        """
        return self.linr_T + self.pref_T + self.gdec_T

    @property
    def cpu_time(self) -> float:
        """
        当前 sub-batch 在 CPU 侧的总预测时间。

        它由 CPU decode 时间与固定 launch 开销组成；调度器会拿它和另一侧 sub-batch
        的 GPU 工作量比较，判断 CPU decode 能否被 pipeline 隐藏。
        """
        return self.cdec_T + self.lnch_T



class SubBatch:
    """
    调度器使用的 sub-batch 抽象。

    它不是“单设备 batch”，而是一次调度候选中的混合执行单元：同一个 `SubBatch` 里
    可以同时包含 CPU/GPU prefill、GPU decode、CPU decode 四类请求。
    """
    # pylint: disable=too-many-instance-attributes, missing-function-docstring
    def __init__(self, predictor: PerfPredictor=ZeroPerfPredictor()):
        # gprf/cprf/gdec/cdec 分别表示：
        # - GPU prefill requests
        # - CPU prefill requests
        # - GPU decode requests
        # - CPU decode requests
        self.gprf_reqs = []

        # 注意结合实际计算时的语义，pref_to_cpu 并不是说在 CPU 中执行
        # prefill request, 而是 prefill 计算本身仍经过 GPU，但其 生成出的 KV 
        # 会在本轮后被 swap 到 CPU（或先写入 intermediate 区，再转去 CPU）。
        self.cprf_reqs = []
        self.gdec_reqs = []
        self.cdec_reqs = []
        # `perfdata` 只服务于调度阶段的估时与试探，不直接参与 worker 真正执行。
        self.perfdata = BatchPerfData(predictor)

    def __len__(self):
        return self.perfdata.x

    def add_pref(self, req: Request, is_gpu: bool):
        """
        加入一条 prefill request，并同步更新 perfdata。
        """
        if is_gpu:
            self.gprf_reqs.append(req)
        else:
            self.cprf_reqs.append(req)
        # 不管是 cpu prefill 还是 gpu prefill 都添加到 perfdata 的 prefill ？
        self.perfdata.add_pref(req.prompt_len)

    def pop_pref(self) -> Request:
        """
        移除最近加入的一条 prefill request，并同步回退 perfdata。

        当前实现优先从 `cprf_reqs` 弹出；只有 CPU prefill 为空时，才从 `gprf_reqs`
        弹出。调度器正是利用这个行为，对候选 batch 做逐条裁剪。
        """
        is_gpu = not self.cprf_reqs
        req = self.gprf_reqs.pop() if is_gpu else self.cprf_reqs.pop()
        self.perfdata.pop_pref(req.prompt_len)
        return req, is_gpu

    def add_gdec(self, req: Request):
        """
        加入一条 GPU decode request，并同步更新 perfdata。
        """
        self.gdec_reqs.append(req)
        self.perfdata.add_gdec(req.seq_len)

    def add_cdec(self, req: Request):
        """
        加入一条 CPU decode request，并同步更新 perfdata。
        """
        self.cdec_reqs.append(req)
        self.perfdata.add_cdec(req.seq_len)

    def pop_cdec(self):
        """
        移除最近加入的一条 CPU decode request，并同步回退 perfdata。
        """
        req = self.cdec_reqs.pop()
        self.perfdata.pop_cdec(req.seq_len)

    def get_num_prefs(self) -> int:
        """
        返回当前 sub-batch 中 prefill request 的总条数。
        """
        return len(self.gprf_reqs) + len(self.cprf_reqs)

    def set_model_forward_args(self, model_config: LlamaModelConfig):
        """
        把调度阶段的 `SubBatch` 转换成执行阶段需要的字段。

        这里是 perf predictor 接入链路里的一个关键边界：
        - 在 batch 形成阶段，调度器依赖 `perfdata` 做估时；
        - 真正进入 worker 执行前，只保留 runtime forward 需要的结构化字段；
        - 因此转换完成后会删除 `perfdata`，说明 predictor 主要服务于调度，而不是算子执行。

        The comments indicate each attribute's usage in the model forward pass.
        """
        # pylint: disable=attribute-defined-outside-init
        self.batch_size = self.perfdata.x # post-layer
        self.iter_width = self.perfdata.s # post-layer
        del self.perfdata

        self.num_cprfs = len(self.cprf_reqs)
        self.num_gprfs = len(self.gprf_reqs)
        self.num_gdecs = len(self.gdec_reqs)

        # 注意结合实际计算时的语义，pref_to_cpu 并不是说在 CPU 中执行
        # prefill request, 而是 prefill 计算本身仍经过 GPU，但其 生成出的 KV 
        # 会在本轮后被 swap 到 CPU（或先写入 intermediate 区，再转去 CPU）。
        self.num_cdecs = len(self.cdec_reqs)
        self.num_prefs = self.num_cprfs + self.num_gprfs
        # num_prgds 可以理解为 prefill + gpu decode 合并计数
        self.num_prgds = self.num_prefs + self.num_gdecs

        self.all_reqs = self.cprf_reqs + self.gprf_reqs + self.gdec_reqs + self.cdec_reqs
        assert all(req.request_id >= 0 for req in self.all_reqs), "Request ID not set"
        del self.cprf_reqs, self.gprf_reqs, self.gdec_reqs, self.cdec_reqs

        self.seq_ids_list = Request.get_ids(self.all_reqs)
        self.seq_lens_list = Request.get_lens(self.all_reqs)

        # Useful for attn kernels
        self.sum_pref_toks = sum(self.seq_lens_list[:self.num_prefs]) # store-pref-KV, pref, gdec
        self.sum_prgd_toks = self.sum_pref_toks + self.num_gdecs # gdec
        self.max_pref_toks = max(self.seq_lens_list[:self.num_prefs], default=0) # store-pref-KV, pref

        # Useful for paged attention
        sum_gdec_toks = sum(self.seq_lens_list[self.num_prefs:self.num_prgds])
        max_gdec_toks = max(self.seq_lens_list[self.num_prefs:self.num_prgds], default=0)
        seq_block_size = 2048
        num_kv_heads = model_config.num_kv_heads
        while num_kv_heads*(sum_gdec_toks/seq_block_size) < 1024 and seq_block_size//2 >= 64 and \
            max_gdec_toks / (seq_block_size//2) <= 128:
            seq_block_size //= 2
        self.seq_block_size = seq_block_size
        self.num_seq_blocks = (max_gdec_toks + seq_block_size - 1) // seq_block_size


    def print_profile(self):
        print(f"cprf lens: {[req.prompt_len for req in self.cprf_reqs]}, gprf lens: {[req.prompt_len for req in self.gprf_reqs]}, "
              f"gdec lens: {[req.seq_len for req in self.gdec_reqs]}, cdec lens: {[req.seq_len for req in self.cdec_reqs]}")
