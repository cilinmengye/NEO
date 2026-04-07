# NEO evaluation 调研报告

## 1. 先回答你的核心问题：NEO 这个项目是不是在做 NEO 和 vLLM 的 evaluation？

**结论：是，但不止于此。**

NEO 的 evaluation 不是单一的 “NEO vs vLLM” 对比，而是围绕论文中的不同图表，分别对比 **不同 baseline**：

1. **Figure 6c / load-latency curve**：主要是 **NEO（`ours`） vs vLLM（`vllm`）** 的在线负载-延迟对比。
   证据在 `NEO/evaluation/reproduce-fig6c.py:46-62`，脚本只启动 `vllm` 和 `ours` 两套服务，并在绘图时使用 `sys_file_names=["vllm", "ours"]`。

2. **Figure 10a / generation throughput**：主要是 **NEO（`ours`） vs 非 CPU offloading baseline（`base`）** 的吞吐对比。
   证据在 `NEO/evaluation/reproduce-fig10a.py:48-67`，脚本只启动 `base` 和 `ours`，最后画的是相对吞吐提升。

3. `NEO/evaluation/server.py:51-64` 里还保留了第三种 `fsdc` 模式，说明作者的 baseline 体系并不只包含 vLLM。
   但当前仓库公开的两个复现实验脚本里，`fsdc` 并没有被实际调用。

所以更准确地说：

- `evaluation` 框架**支持**把 NEO 与 vLLM 做对比；
- 但整个 benchmark **不是只评 NEO vs vLLM**；
- 它是按论文图表分别对比 **vLLM、NEO、自身 non-offloading baseline，以及预留的其他 baseline 变体**。

---

## 2. NEO 项目整体思路是什么？

### 2.1 项目层面的系统思路

根据 `NEO/README.md:1-5`，NEO 是一个 **在线 LLM 推理系统**，核心目标不是提升模型质量，而是提升 **在线 serving 场景中的吞吐与资源利用率**。

README 的核心描述是：

- 将 **部分 attention 计算** 和 **KV cache 状态** 从 GPU 卸载到本地 CPU；
- 缓解 GPU 显存瓶颈；
- 允许更大的 batch size；
- 最终提升 inference throughput。

README 里给出的两个关键机制是：

- **asymmetric GPU-CPU pipelining**
- **load-aware scheduling**

也就是说，NEO 的目标是 **系统性能优化**，不是模型精度优化。

### 2.2 evaluation 层面的实验思路

`NEO/evaluation` 本质上是一个统一的 **serving benchmark harness**，它的通用流程很固定：

1. 选择后端服务（NEO / vLLM / baseline）；
2. 启动服务进程；
3. 通过统一的 OpenAI-compatible completion API 向 `http://localhost:8000/v1/completions` 发请求；
4. 记录每个请求的 `start/end/input_len/output_len`；
5. 将结果写入 `evaluation/results/*.json`；
6. 再由绘图脚本读取这些 JSON，计算指标并生成 PDF 图。

对应代码关系如下：

- `NEO/evaluation/reproduce-fig6c.py` / `NEO/evaluation/reproduce-fig10a.py`：实验入口
- `NEO/evaluation/server.py`：启动/停止不同 serving backend
- `NEO/evaluation/benchmark.py`：构造 workload、发请求、记录 timing
- `NEO/evaluation/api_client.py`：统一 completion API client
- `NEO/evaluation/illustrator.py`：读取结果并计算指标、画图

这套设计说明：**NEO 的 benchmark 不是直接在模型层评估，而是在统一 API 层对不同 serving engine 做系统性能评估。**

---

## 3. benchmark 使用的数据集是什么？分别测什么？

## 3.1 Figure 6c：真实长度分布数据集（OpenAI summarization comparison）

### 数据来源

README 在 `NEO/README.md:57-60` 里说明：

- Hardware: AWS g4.4xlarge
- Model: LLaMa-2-7B
- Workload: **OpenAI summarization comparison**
- 来源：`CarperAI/openai_summarize_comparisons`

### 本地数据文件

对应的数据文件是：

- `NEO/evaluation/data/osc-Llama-2-7b-hf.json`

`NEO/evaluation/benchmark.py:105-118` 的 `prepare_real_test()` 会读取：

- `evaluation/data/{dataset_name}-{config['model']}.json`

对于 `config-t4-7b.json`：

- `dataset_name = "osc"`
- `config['model'] = "Llama-2-7b-hf"`

所以最终就是：

- `evaluation/data/osc-Llama-2-7b-hf.json`

### 这个“数据集”在代码里实际怎么被使用

这点非常关键。

该 JSON 文件里每条样本只有两个字段：

- `prompt`：输入 token 长度
- `max_tokens`：输出 token 长度上限

例如 `NEO/evaluation/data/osc-Llama-2-7b-hf.json:1-20` 可以看到数据格式就是长度统计，而不是原始文本。

而 `prepare_real_test()` 实际构造请求的方法是：

- `prompts = [[10] * data["prompt"] for data in datas]`
- `output_lens = [data["max_tokens"] for data in datas]`

也就是：

- **输入并不是原始摘要文本**；
- 而是用固定 token id `10` 重复若干次，构造出“长度相同”的假 prompt。

所以 Figure 6c 的 workload 更准确地说是：

- **使用 OpenAI summarization comparison 的长度分布轨迹**；
- 但并不保留文本语义；
- 它测的是 **真实长度分布下的系统性能**，不是摘要质量。

### 样本量

`NEO/evaluation/benchmark.py:111-114` 写得很明确：

- 默认只取 `json.load(f)[:100]`

注释说明：

- 如果去掉 `[:100]`，才会用完整数据；
- 完整测试会很久，约 10 小时；
- 论文原始实验使用的是 **2000 requests**。

README 在 `NEO/README.md:111-114` 也明确说了：

- 默认复现脚本只用 **100 requests**；
- 原始实验使用 **2000 requests**；
- 这样做是为了更快验证结果趋势。

### 它测什么

Figure 6c 测的是：

- 在不同请求到达率（req/s）下；
- NEO 和 vLLM 的 **在线延迟表现**。

---

## 3.2 Figure 10a：合成 workload

Figure 10a **没有使用外部真实文本数据集**，而是纯合成 workload。

`NEO/evaluation/reproduce-fig10a.py:21-30` 里给出的关键参数：

- `num_data = 2000`
- `input_len = 1000`
- `output_lens = [50, 100, 200, 300, 400]`

然后 `NEO/evaluation/benchmark.py:91-102` 的 `prepare_mock_test()` 会：

1. 把输入长度在平均值上下 10% 范围内随机扰动；
2. 把输出长度也在平均值上下 10% 范围内随机扰动；
3. 用 `[10] * input_len` 构造固定 token prompt。

`README.md:67-69` 对这个合成 workload 的说明与代码一致：

- 平均输入长度固定为 `1000`
- 平均输出长度从 `{50, 100, 200, 300, 400}` 中选
- 输入、输出长度都在各自均值上下 10% 区间内独立均匀采样

### 它测什么

Figure 10a 测的是：

- 在不同平均输出长度下；
- NEO 相对 baseline 的 **吞吐提升**。

---

## 4. 实验指标是什么？如何计算？

## 4.1 Figure 6c：负载-延迟曲线

### 横轴：request rate

在 `NEO/evaluation/reproduce-fig6c.py:20-22` 中：

- `vllm_rates = [0.2, 0.4, 0.5, 0.6]`
- `ours_rates = [0.5, 1.5, 2.5, 3.1, 3.5, 3.7, 3.9]`

所以横轴是：

- **Request rate (req/s)**

### 请求到达过程

`NEO/evaluation/benchmark.py:49-57` 中：

- `gaps = np.random.exponential(1 / rate, len(prompts)).tolist()`

这表示请求之间的间隔来自 **指数分布**，即近似模拟：

- **Poisson arrival / 泊松到达流**

因此，这不是固定等间隔发请求，而是更接近在线系统中的随机到达过程。

### 纵轴：平均每生成 token 延迟

Figure 6c 最终画图时，调用的是 `illustrator.py` 中的 `draw_one_rl_diagram()`；
它内部通过 `get_lat_avg()` 读取结果文件并计算延迟。

`NEO/evaluation/illustrator.py:7-12` 的公式是：

1. 读取 JSON；
2. 丢弃前 `len(data)//4` 的数据；
3. 对每条请求计算：
   `(end - start) / output_len`
4. 再求平均。

因此，这个指标严格来说是：

- **平均每输出 token 延迟**
- 或更准确说：**平均每生成 token completion latency**

图中的 y 轴标题在 `NEO/evaluation/illustrator.py:39-47` 被写成：

- `Average per token latency (s)`

但从实现上看，分母只用了 `output_len`，没有算 input tokens。
所以如果要非常严谨，应该理解为：

- 这是 **per generated token latency**；
- 不是输入+输出总 token 的统一 per-token latency。

### 一个实现细节

`get_lat_avg()` 的注释写的是：

- `# only take latter half`

但代码实际是：

- `data = data[len(data) // 4:]`

也就是丢掉前 1/4，保留后 3/4，不是严格的“后半段”。这一点如果做严谨汇报，建议按代码实现表述，而不是照搬注释。

---

## 4.2 Figure 10a：吞吐与相对吞吐

### 吞吐测试怎么跑

在 `NEO/evaluation/benchmark.py:29-79` 中，`run_test()` 有两种模式：

- `rate > 0`：latency/load test
- `rate <= 0`：throughput test

而 `reproduce-fig10a.py` 调 `run_test()` 时没有传 `rate`，所以默认：

- `rate = -1`

这意味着：

- 所有请求几乎立刻并发发出；
- 不做 rate 节流；
- 用于测吞吐。

### benchmark.py 中的吞吐计算

`run_test()` 在 throughput 模式下会额外记录一个吞吐日志：

- 先对所有请求结束时间排序；
- 取中间一段；
- 然后用 `(len(req_end_times)-1)/(last-first)` 算 req/s。

对应代码在 `NEO/evaluation/benchmark.py:73-78`。

但要注意：

- **Figure 10a 画图时并不是直接使用这里的吞吐值。**

### illustrator.py 真正用于作图的吞吐定义

Figure 10a 用 `draw_one_ps_diagram()`，它内部调用 `get_tp()`。

`NEO/evaluation/illustrator.py:55-68` 的逻辑是：

1. 读取结果文件；
2. 取所有请求的结束时间并排序；
3. 计算相邻结束时间的差值；
4. 只保留中间稳定区间；
5. 用该区间平均结束间隔的倒数，得到 req/s。

在 `reproduce-fig10a.py:55-67` 中，参数是：

- `interv=[0.3, 0.7]`

注释明确说：

- 忽略最前 30% 和最后 30% 数据；
- 以减少 warm-up 和 cool-down 的影响。

所以 Figure 10a 实际画出来的是：

- **稳定区间内的 request throughput（req/s）**。

### 最终图上的指标

`draw_one_ps_diagram()` 在 `NEO/evaluation/illustrator.py:100-120` 中，最终取的是：

- `ratios = [tp1 / tp0 for tp0, tp1 in tps]`

其中：

- `tp0 = baseline throughput`
- `tp1 = ours throughput`

所以图上的 y 轴不是绝对吞吐，而是：

- **Relative throughput（相对吞吐）**
- 即：`throughput(ours) / throughput(base)`

README 在 `NEO/README.md:63-69` 提到的：

- 12.2%、13.3%、29.7%、79.3% 更高吞吐

本质上就是这种相对吞吐增益的表达。

---

## 4.3 当前公开脚本没有评测哪些指标

从 `NEO/evaluation` 当前代码看，它主要只做 **系统性能 benchmark**，没有直接做以下指标：

- 准确率
- ROUGE / BLEU
- 胜率 / 偏好得分
- TTFT（首 token 延迟）
- P95 / P99 latency
- GPU / CPU 利用率的结构化统计
- 内存峰值统计

虽然 `NEO/evaluation/illustrator.py:124-149` 里有：

- `parse_ours_server_log()`
- `parse_vllm_server_log()`

可以解析 server log 中的一些运行状态，
但在当前公开的两个复现实验脚本里，这些函数**没有进入最终结果图的计算路径**。

所以这套 evaluation：

- **重点是系统吞吐/延迟；**
- **不是模型质量 benchmark。**

---

## 5. 框架启动参数、命令行参数、环境变量、默认值分别是什么？

## 5.1 evaluation 如何启动 vLLM

`NEO/evaluation/server.py:13-49` 中，若 `name[:4] == "vllm"`，则启动命令形如：

```bash
numactl -N 0 -m 0 \
  vllm serve <model_path> \
  --port 8000 \
  --block-size <block_size> \
  --max-model-len <max_model_len> \
  --max-num-seqs <min(chunk_size, max_num_seqs)> \
  --max-num-batched-tokens <chunk_size> \
  --tensor-parallel-size <tensor_parallel_size> \
  --num-gpu-blocks-override <num_gpu_blocks_override> \
  --swap-space <swap_space / tensor_parallel_size> \
  --enforce-eager \
  --disable-sliding-window \
  --disable-async-output-proc \
  --disable-custom-all-reduce \
  --disable-frontend-multiprocessing \
  --tokenizer-pool-size 1 \
  --enable-chunked-prefill \
  --preemption-mode recompute \
  --dtype float16
```

### vLLM 的环境变量

在 `server.py:46-49` 中还额外注入了：

- `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`

### vllm / vllm256 / vllm512 的区别

`server.py:22-24` 的逻辑是：

- 如果名字是 `vllm`，则 chunk size = `num_gpu_blocks_override * block_size`
- 如果名字是 `vllm256`、`vllm512` 这样的形式，就直接从名字后缀里取 chunk size

这和 README 中 `NEO/README.md:52-54` 对：

- `vLLM-256`
- `vLLM-512`

的描述是对应的。

不过当前公开复现脚本 `reproduce-fig6c.py` 实际只调用了：

- `vllm`

没有直接跑 `vllm256` 与 `vllm512` 两个名字。

---

## 5.2 evaluation 如何启动 NEO / baseline

`NEO/evaluation/server.py:51-87` 中，若名字属于：

- `ours`
- `base`
- `fsdc`

则统一启动：

```bash
python -m swiftllm.server.api_server
```

并附带一组 engine 参数。

### 共用参数

对这三种模式，`server.py:66-81` 中共有的参数是：

- `--port 8000`
- `--model-path <config["model_path"]>`
- `--block-size <config["block_size"]>`
- `--max-blocks-per-seq <(max_num_batched_tokens - 1) // block_size + 1>`
- `--max-seqs-in-block-table <config["max_num_seqs"]>`
- `--max-batch-size <config["max_num_seqs"]>`
- `--max-tokens-in-batch <config["max_num_batched_tokens"]>`
- `--tensor-parallel-degree <config["tensor_parallel_size"]>`
- `--num-gpu-blocks-override <derived value>`
- `--swap-space <derived value>`
- `--library-path <NEO/pacpu/build/{library}>`
- `--profile-result-path <NEO/profile_results/>`

### 三种模式的差异

#### `base`

`server.py:53-57`：

- 附加参数：`--always-use-gpu`
- `num_gpu_blocks_override = config["num_gpu_blocks_override"]`
- `swap_space = config["swap_space"] // 8`

这说明 `base` 的含义是：

- **不做 CPU offloading 的 baseline**

这也与 README 对 Figure 10a 的说法一致：

- `non-CPU-offloading baseline`

#### `ours`

`server.py:57-60`：

- 附加参数：`--extra-layer-for-cprf`
- `num_gpu_blocks_override = config["num_gpu_blocks_override"] * nl // (nl + 1)`
- `swap_space = config["swap_space"]`

其中 `nl = config['num_layers']`。

这就是：

- **NEO 的主方法**

#### `fsdc`

`server.py:61-64`：

- 附加参数：`--disable-partial-offl`
- `--extra-layer-for-cprf`
- 派生参数与 `ours` 相同

这是另一个对照变体，但当前两个公开复现实验脚本没有调用它。

---

## 5.3 NEO 服务端 API 入口与 CLI 参数定义

### API 入口

`NEO/swiftllm/server/api_server.py:16-45` 定义了：

- `POST /v1/completions`

请求体里用到的核心字段是：

- `prompt`
- `max_tokens`
- 可选 `stream`

这说明 benchmark 统一通过 OpenAI 风格 completions 接口去驱动后端。

### api_server.py 直接暴露的参数

在 `NEO/swiftllm/server/api_server.py:47-58` 中，显式增加了：

- `--host`，默认 `localhost`
- `--port`，默认 `8000`

其余参数来自 `swiftllm.EngineConfig.add_cli_args()`。

---

## 5.4 EngineConfig 支持的 CLI 参数及默认值

`NEO/swiftllm/engine_config.py:68-165` 给出了所有主要 CLI 参数。

### 必填或核心参数

- `--model-path`：必填
- `--use-dummy`：默认关闭
- `--block-size`：默认 `16`
- `--gpu-mem-utilization`：默认 `0.99`
- `--num-gpu-blocks-override`：默认 `-1`
- `--swap-space`：默认 `20`
- `--max-seqs-in-block-table`：默认 `768`
- `--max-blocks-per-seq`：默认 `512`
- `--max-batch-size`：默认 `512`
- `--max-tokens-in-batch`：默认 `3072`
- `--library-path`：无默认值
- `--profile-result-path`：无默认值
- `--tensor-parallel-degree`：默认 `1`
- `--disable-partial-offl`：默认关闭
- `--always-use-gpu`：默认关闭
- `--extra-layer-for-cprf`：默认关闭

但在 evaluation 中，很多默认值都会被 config 文件覆盖。

---

## 5.5 两个 config 文件里的实际实验配置

## Figure 6c 对应：`NEO/evaluation/configs/config-t4-7b.json`

文件内容在 `config-t4-7b.json:1-13`：

- `model`: `Llama-2-7b-hf`
- `model_path`: `/home/ubuntu/weights/Llama-2-7b-hf`
- `num_layers`: `32`
- `block_size`: `16`
- `max_model_len`: `832`
- `max_num_seqs`: `512`
- `max_num_batched_tokens`: `832`
- `tensor_parallel_size`: `1`
- `gpu_memory_utilization`: `0.99`
- `num_gpu_blocks_override`: `54`
- `swap_space`: `20`
- `library`: `libpacpu-llama2_7b-tp1.so`

## Figure 10a 对应：`NEO/evaluation/configs/config-a10-8b.json`

文件内容在 `config-a10-8b.json:1-13`：

- `model`: `Llama-3-8B`
- `model_path`: `/home/ubuntu/weights/Llama-3-8B`
- `num_layers`: `32`
- `block_size`: `16`
- `max_model_len`: `20000`
- `max_num_seqs`: `1024`
- `max_num_batched_tokens`: `20480`
- `tensor_parallel_size`: `1`
- `gpu_memory_utilization`: `0.99`
- `num_gpu_blocks_override`: `1650`
- `swap_space`: `120`
- `library`: `libpacpu-llama3_8b-tp1.so`

---

## 6. 实验整体流程和实验思路分别是什么？

## 6.1 Figure 6c：NEO vs vLLM 的负载-延迟曲线

### 实验入口

- `NEO/evaluation/reproduce-fig6c.py`

### 实验流程

根据 `reproduce-fig6c.py:30-62`：

1. 读取 `config-t4-7b.json`
2. 启动 `vllm`
3. 对 `vllm_rates = [0.2, 0.4, 0.5, 0.6]` 逐点运行测试
4. 关闭 `vllm`
5. 启动 `ours`
6. 对 `ours_rates = [0.5, 1.5, 2.5, 3.1, 3.5, 3.7, 3.9]` 逐点运行测试
7. 关闭 `ours`
8. 调用 `draw_one_rl_diagram()` 生成 `fig6c.pdf`

### 实验思路

这个实验要回答的问题是：

- 在真实请求长度分布下；
- 当在线请求到达率逐渐上升时；
- NEO 相比 vLLM 能否承受更高负载，同时保持更低或更可接受的 token latency。

因此它的核心设计是：

- 使用真实 workload 的长度分布；
- 但不关心文本内容；
- 用泊松到达流施加不同负载；
- 最终绘制负载-延迟曲线。

### 一个值得注意的点

`vllm` 和 `ours` 的 rate 采样点并不相同，这说明脚本的思路不是“严格同一组 x 值逐点硬比较”，而更像是在各自系统的有效工作区间内取若干代表性点，再画出趋势线。

---

## 6.2 Figure 10a：NEO vs baseline 的吞吐提升

### 实验入口

- `NEO/evaluation/reproduce-fig10a.py`

### 实验流程

根据 `reproduce-fig10a.py:38-67`：

1. 读取 `config-a10-8b.json`
2. 启动 `base`
3. 依次对 `output_lens = [50,100,200,300,400]` 跑 throughput test
4. 关闭 `base`
5. 启动 `ours`
6. 用同样的 output lengths 再跑一遍 throughput test
7. 关闭 `ours`
8. 调用 `draw_one_ps_diagram()` 生成 `fig10a.pdf`

### workload 设计

`reproduce-fig10a.py:21-30` 与 `benchmark.py:91-102` 联合说明：

- 请求总数：`2000`
- 平均输入长度：`1000`
- 平均输出长度：`[50,100,200,300,400]`
- 每条请求的输入长度和输出长度都做 ±10% 均匀扰动
- 所有请求几乎一起发出，用于测满载吞吐

### 实验思路

这个实验想回答的问题是：

- 当输入长度固定、输出长度变化时；
- NEO 相比非 CPU-offloading baseline；
- 吞吐能提高多少。

所以它是一个典型的：

- **控制输入长度**
- **扫描输出长度**
- **比较稳态吞吐增益**

的系统实验。

### README 与脚本之间的关系

README 在 `NEO/README.md:63-69` 说的是论文 Figure 10a 的完整结论：

- 不同 CPU capacity 下 NEO 相对 baseline 的吞吐提升。

但当前公开脚本 `reproduce-fig10a.py:1-8` 明确说明：

- 只复现原图中的一条线；
- 具体是 g5.x16large 对应的那条线。

所以仓库中的脚本更像：

- **代表性复现脚本**，不是完整论文所有曲线的全量复现。

---

## 7. README 与 evaluation 目录之间如何对应？

README 的 “Performance Results + Reproduction” 与 `evaluation` 目录是直接一一对应的。

### Figure 6c

- README：`NEO/README.md:49-60`
- 实现：`NEO/evaluation/reproduce-fig6c.py`

### Figure 10a

- README：`NEO/README.md:61-69`
- 实现：`NEO/evaluation/reproduce-fig10a.py`

### 配置文件说明

README 的 `NEO/README.md:104-107` 提醒用户去改：

- `evaluation/configs/config-t4-7b.json`
- `evaluation/configs/config-a10-8b.json`

这与代码实现完全一致。

### 结果文件说明

README 的 `NEO/README.md:109-114` 说运行后会生成：

- `fig6c.pdf`
- `fig10a.pdf`

这正对应 `illustrator.py` 中的 `plt.savefig(...)`。

因此可以把 README 理解为：

- **用户视角的复现实验说明文档**；

而 `evaluation` 目录是：

- **这些实验说明的具体落地代码**。

---

## 8. 我认为这个项目 evaluation 最重要的几个发现

## 8.1 这是系统 benchmark，不是模型 benchmark

虽然用了 OpenAI summarization comparison 的长度分布，但当前实现：

- 没有使用原始自然语言文本；
- 没有评估摘要质量；
- prompt 只是固定 token 序列；
- `api_client.py:16-22` 里参数是 `temperature=0.0`、`ignore_eos=True`；
- 返回值也只关注 completion token 序列，不关心语义正确性。

所以这个 benchmark 的本质是：

- **serving system benchmark**
- 不是 **task quality benchmark**

## 8.2 Figure 6c 和 Figure 10a 的 baseline 不一样

这是理解 NEO evaluation 时最容易混淆的点：

- **Figure 6c：NEO vs vLLM**
- **Figure 10a：NEO vs non-CPU-offloading baseline（base）**

所以如果问：

> NEO 这个项目是不是将 NEO 和 vLLM 进行 evaluation？

最准确的回答应该是：

- **部分 benchmark 是；**
- **但整个 evaluation 不只围绕 vLLM 展开。**

## 8.3 当前仓库更偏“论文代表性图表的简化复现”

从 README 和脚本可以直接看出：

- Figure 6c 默认只用 100 / 2000 请求；
- Figure 10a 只复现原图中的一条线；
- 主要目标是快速验证趋势，而不是完整复现论文所有实验条件。

## 8.4 vLLM 在这里被当作可替换 serving backend

`NEO/evaluation/server.py:22-49` 用 `vllm serve ...` 启动 vLLM；
而 benchmark client 始终请求统一的：

- `http://localhost:8000/v1/completions`

这说明作者在 benchmark harness 这一层，把：

- NEO
- vLLM
- baseline

都抽象成同一种 completion API backend，再在统一 workload 下比较。
这是整个 evaluation 框架最核心的工程设计。

---

## 9. 关键文件清单

- `NEO/README.md`
  项目定位、论文图表说明、硬件/模型/workload 描述、复现方法。

- `NEO/evaluation/reproduce-fig6c.py`
  Figure 6c 入口；NEO vs vLLM；真实长度分布 workload；负载-延迟曲线。

- `NEO/evaluation/reproduce-fig10a.py`
  Figure 10a 入口；NEO vs base；合成 workload；相对吞吐曲线。

- `NEO/evaluation/server.py`
  benchmark 启动器；负责启动 vLLM / ours / base / fsdc。

- `NEO/evaluation/benchmark.py`
  workload 构造、请求发送、计时、结果落盘。

- `NEO/evaluation/api_client.py`
  completion API 请求封装。

- `NEO/evaluation/illustrator.py`
  读取结果并计算延迟/吞吐，生成 PDF。

- `NEO/evaluation/configs/config-t4-7b.json`
  Figure 6c 的实验配置。

- `NEO/evaluation/configs/config-a10-8b.json`
  Figure 10a 的实验配置。

- `NEO/evaluation/data/osc-Llama-2-7b-hf.json`
  OpenAI summarization comparison 的长度轨迹样本。

- `NEO/swiftllm/server/api_server.py`
  NEO 服务端 API 入口。

- `NEO/swiftllm/engine_config.py`
  NEO engine 的 CLI 参数定义与默认值。

# Server 是否支持 prefill request and decode request in same batch

 /btw 模型架构是否实现了server允许prefill request和decode request在同一个batch中？在vllm中是否实现了?                                          
                                                                                                                                              
    是，NEO 这套实现明确支持 prefill request 和 decode request 出现在同一个 batch / sub-batch 里。                                              
                                                                                                                                                
    依据从上下文里可以直接看出来：                                                                                                              
                                                                                                                                                
    - SubBatch 不是只存一种请求，而是同时区分                                                                                                   
      - gprf_reqs：GPU prefill
      - cprf_reqs：CPU-side prefill 路径
      - gdec_reqs：GPU decode
      - cdec_reqs：CPU decode
    - SubBatch.set_model_forward_args(...) 会把它们合并成
    all_reqs = cprf + gprf + gdec + cdec
    - Request.get_input_tokens(...) 对 prefill 喂整段 prompt，对 decode 只喂最后一个 token，所以同一轮 forward 本身就支持“不同请求处于不同阶段”
    - transformer_layer._attention(...) 里也显式分三段处理：
      a. prefill attention
      b. GPU decode attention
      c. CPU decode attention

    所以从调度器、batch 数据结构、到模型 forward 路径，NEO 都是按“prefill 和 decode 混合批处理”设计的，而不是分成两个完全独立的批次系统。

    对于 vLLM：

    - vLLM 也可以做到，但要看是否开启对应机制。
    - 在你这个仓库的评测脚本里，vLLM baseline 启动时用了
    --enable-chunked-prefill
    - 这正是 vLLM 用来让 prefill 与 decode 共存/交错调度 的关键机制

    所以更准确地说：

    - NEO：明确实现了 mixed prefill+decode batching
    - vLLM：在开启 chunked prefill 时也实现了；你这里的 baseline 就是按这个模式跑的
    - 如果不开 chunked prefill，vLLM 的 prefill/decode 混合能力就没有这么强，行为也不等同于 NEO 这种围绕异构 pipeline 专门设计的实现

    一句话总结：

    ▎ NEO 的“prefill + decode 同批”是系统核心设计的一部分；vLLM 也支持类似能力，但主要依赖 chunked prefill，而且它的目标和实现方式与 NEO
    不完全一样。