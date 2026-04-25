# 复现总结

* 基于Docker成功构建其和NEO论文同**软件实验环境**, 硬件环境主要区别在于GPU(I am 4090), CPU(I am v80)。具体看`NEO/docker` 和 本 README.md `Environment Build`.

* 复现fig6c. 基于4090成功复现出在特定参数配置下，不同请求速率下ouput token latency显示 vllm > neo. 具体看`NEO/evaluation/pics/highgpumempress_conf`。复现的秘诀是尽量让GPU KV Cache block非常紧张，因为vllm开启的参数是`--preemption-mode recompute`, 那么在KV Cache block 紧张的情况下会引发强制，导致重算，让vllm延迟变高。因为NEO实现了offload CPU to compute, 所以对于NEO是存在在相同请求速率下ouput token latency显示 vllm > neo情况的。

   * 论文仓库配置`config-t4-7b.json` 令 `num_gpu_blocks_override * block_size == max_num_batched_tokens == max_model_len`, 而且其把`num_gpu_blocks_override`设置的很低，即尽量让GPU KV Cache block非常紧张。原理就是**人为把 GPU KV/block 容量压到较紧，用较小的 iteration token budget 运行，使系统更容易进入 KV 紧张和 preemption 区；由于vllm设置选的是 recompute，紧张时更可能发生的是丢弃 KV 后重算，而不是换到 CPU。**

* NEO 理论上真正强大之处在于 **吞吐量**，因为其摘要就写着 `However, the limited GPU memory has largely limited the batch size achieved in practice, leaving significant GPU compute resources wasted.We present NEO, an online LLM inference system that offloads part of attention compute and KV cache states from the GPU to the local host CPU, effectively increasing the GPU batch size and thus inference throughput.`

* 复现结果具体可以看`NEO/evaluation/pics`下，我还对 NEO vs vllm TTFT, TPOT 进行了补充实验

# NEO: Saving GPU Memory Crisis with CPU Offloading for Online LLM Inference

Online LLM inference powers many exciting applications such as intelligent chatbots and autonomous agents. Modern LLM inference engines widely rely on request batching to improve inference throughput, aiming to make it cost-efficient when running on expensive GPU accelerators. However, the limited GPU memory has largely limited the batch size achieved in practice, leaving significant GPU compute resources wasted. 

NEO is an online LLM inference system that offloads part of attention compute and KV cache states from the GPU to the local host CPU, effectively increasing the GPU batch size and thus inference throughput. To this end, NEO proposes asymmetric GPU-CPU pipelining and load-aware scheduling to balance GPU and CPU loads and fully utilize their compute and memory resources. Our MLSys'25 paper is [here](https://yangzhou1997.github.io/paper/neo_mlsys25.pdf).

## Requirements

Python >= 3.10
PyTorch >= 2.4

2 versions of g++ (see `pacpu/build.sh` for more details):

- one >= 13 (for compiling CPU kernel)
- the other < 13 (for passing the NVCC version check)

Intel ISPC compiler == 1.23, which can be installed by `sudo snap install ispc --channel latest/edge`

## Installation

1. Clone the NEO repository and `cd` into the repo.

2. Install dependencies by `pip install -r requirements.txt.`

3. Install the swiftLLM library to your local environment by `pip install -e .`

4. Build and install auxiliary GPU operators library by `pip install -e csrc`

5. Build the CPU operator library by 

   ```bash
   cd pacpu
   bash build.sh <model-name> <tensor-parallel-degree> 
   # e.g bash build.sh llama2_7b 1
   cd ..
   ```

## Offline Example

```bash
cd NEO
python examples/example.py --model-path ... --model-name ...
# e.g. python examples/example.py --model-path /home/ubuntu/weights/Llama-2-7b-hf/ --model-name llama2_7b
```

Run `python examples/example.py --help` to see more options.

## Performance Results

### Load-latency Curves

The figure below (Figure 6c in the paper) shows online latencies of NEO and other baselines under different request rates.

vLLM-256 and vLLM-512 designate vLLM with chunked-prefilling at the chunk size of 256 and 512 tokens, respectively.

![image-20250221101244560](docs/load-latency.png)

- Hardware: AWS g4.4xlarge instance, with Tesla T4 GPU, 8 cores of Xeon P-8259CL CPU, and 64 GB main memory.
- Model: LLaMa-2-7B
- Workload: OpenAI summarization comparison ([CarperAI](https://huggingface.co/datasets/CarperAI/openai_summarize_comparisons.))

### Generation Throughput

The figure below (Figure 10a in the paper) shows NEO's throughput gains over the non-CPU-offloading baseline under different workloads. NEO achieves up to 12.2%, 13.3%, 29.7%, and 79.3% higher throughput over the baseline under different CPU capacities.

![image-20250221101309717](docs/cpu-sensitivity.png)

- Hardware: AWS g5.nxlarge instances (n=2,4,8,16), with A10 GPU, 2n cores of EPYC 7R32 CPU, and 16n GB main memory.
- Model: LLaMa-3-8B
- Workload: Synthetic workloads with various input and output lengths. For a pair of input length $l_i$ and output length $l_o$, we synthesize requests with input and output lengths sampled independently and uniformly from $[0.9l_i, 1.1l_i]$ and $[0.9l_o, 1.1l_o]$, respectively. Here we fix $l_i=1000$ and pick $l_o$ from $\{50, 100, 200, 300, 400\}$.

## Reproduction

Below are instructions for reproducing Figure 6c in the paper. Instructions for Figure 10a are the same except for specific details noted in parentheses.

### With an AWS Account

1. Launch a g4dn.4xlarge (g5.16xlarge) instance in us-east-1 region with community AMI neo-ae-g4-image (neo-ae-g5-image).
2. SSH to the instance and run `mamba activate neo` in the shell.
3. run `cd NEO`
4. run `python evaluation/reproduce-fig6c.py`(`python evaluation/reproduce-fig10a.py`)

> NOTE: Although the model weights are pre-packaged in the images, the first time loading them would take about 1 hour. Therefore, it is recommended to download the weights from the internet and replace those embedded in  the image, which usually takes less than 10 min. The following script can be used to retrieve the weights from Huggingface:
>
> ```bash
> cd ~
> rm -r weights/*
> ip install 'huggingface_hub[cli]' 
> huggingface-cli login --token <your huggingface token>
> # For g5 instance:
> huggingface-cli download meta-llama/Llama-3.1-8B --local-dir weights/Llama-3-8B --exclude "*.pth"
> # For g4 instance:
> huggingface-cli download meta-llama/Llama-2-7b-hf --local-dir weights/Llama-2-7b-hf --exclude "*.pth"
> ```
>
> Alternatively, you may use the pre-packaged weights within the image. It is possible to encounter timeout issues during the initial execution of the evaluation script due to prolonged loading times. If this occurs, simply rerunning the script should resolve the issue.

### Without an AWS Account

1. Prepare a machine with 
   - Nvidia Tesla T4 (A10G) GPU;
   - CPU with AVX2 support;
   - At least 30GB (120GB) main memory for CPU KV Cache.
   - Ubuntu >= 22.04
2. Follow the steps in the Installation section to install dependencies.
3. Download LLaMa-2-7B (LLaMa-3-8B) model weights. You can refer to the NOTE above for weight retrieving scripts.
4. Modify `model_path` entry in `evaluation/configs/config-t4-7b.json` ( `evaluation/configs/config-a10-8b.json`) to the actual path to the model weights.
5. run `python evaluation/reproduce-fig6c.py`(`python evaluation/reproduce-fig10a.py`) in top level directory of the NEO repository.

### Expected Results

- The reproduced figure fig6c.pdf (fig10a.pdf) will be produced in `evaluation` directory.
- For Figure 6c, there will be only 2 lines (Neo and vLLM). By default the script only uses a small subset (100 requests) of the original input data (2000 requests) used in the original experiment. This is for the purpose of demonstration and quick verification of the results for faster evaluation. As a result, the latency would be lower than the original figure due to less average queuing latency.
- For Figure 10a, only 2 lines (x16large and baseline) in the original figure will be drawn.

> NOTE: You can change the hyperparameters of the experiments by modifying the corresponding scripts. Please refer to comments in the code for detailed instructions.

# Environment Build

1. docker file: `NEO/docker/Dockerfile.cu124`


2. docker build image

```
docker build -f docker/Dockerfile.cu124 -t neo-cu124-ispc123:dev .
```

3. run docker container

```
docker run --rm -it \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --ulimit nofile=524288:524288 \
  --cap-add SYS_NICE \
  -v /home/yxlin/github/swift/NEO:/workspace/NEO \
  -v /mnt/hdd/data/yxlin/huggingface/:/workspace/huggingface \
  -v /tmp:/tmp \
  -w /workspace/NEO \
  --name neo-cu124-ispc123 \
  neo-cu124-ispc123:dev
```

4. In container, build uv venv

```
cd /workspace/NEO
uv venv .venv --python 3.12.12 --seed
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pip install -e csrc --no-build-isolation
```

5. In container, build vllm uv venv

```
uv venv .venv_vllm --python 3.12.12 --seed
source .venv_vllm/bin/activate
pip install --upgrade pip setuptools wheel
pip install vllm==0.7.3
```

> 如果遇到类似 `TypeError: non-default argument 'vision_config' follows default argument` 的错误
> 可以尝试减低 transformers 版本: `/workspace/NEO/.venv_vllm/bin/pip install "transformers<5"`

<hr>

> ps: 
> * 强制删除容器: `docker rmi neo-cu124-ispc123:dev`
> * 