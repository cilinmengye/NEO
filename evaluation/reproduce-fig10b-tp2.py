"""
My reproduce create, not author create. Reproduction script for Figure 10b in the paper.

DataSet: [AzureLLMInferenceTrace](https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2023.md)
    * Code
        * all line: 8819
        * max token len(prompt + output token len): 7841
    * Conversation
        * all line: 19366
        * max token len(prompt + output token len): 14089
"""

import asyncio
import json
import csv
import os

import matplotlib.pyplot as plt

from server import start_server, stop_server
from benchmark import run_test
from illustrator import get_tp, get_tp_token

cur_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = f"{cur_dir}/data"
res_dir = f"{cur_dir}/results"

with open(f"{cur_dir}/configs/config-4090-7b-tp2.json", "r") as f:
    config = json.load(f)

# 这两个参数显式暴露在脚本顶部，方便先小规模验证，再逐步扩大压测强度。
NUM_THRESHOLD = 200000
MAX_INFLIGHT = 20000


def read_csv(
    file_path: str,
    num_threshold: int,
) -> list:
    res = []
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= num_threshold:
                break
            res.append(row)
    return res


def prepare_ac_test(
    dataset_name: str,
    config: dict,
    server_name: str,
    num_threshold: int,
):
    """
    准备 Azure trace 数据。

    `num_threshold` 现在由外部传入，而不是写死在函数内部，
    这样可以先用小样本验证调度逻辑是否稳定，再扩大到更高负载。
    """
    ac_codepath = f"{data_dir}/AzureLLMInferenceTrace_code.csv"
    ac_convpath = f"{data_dir}/AzureLLMInferenceTrace_conv.csv"

    code_datas = read_csv(ac_codepath, num_threshold)
    conv_datas = read_csv(ac_convpath, num_threshold)

    prompts = [[10] * int(data["ContextTokens"]) for data in code_datas]
    output_lens = [int(data["GeneratedTokens"]) for data in code_datas]
    prompts += [[10] * int(data["ContextTokens"]) for data in conv_datas]
    output_lens += [int(data["GeneratedTokens"]) for data in conv_datas]

    res_file = f"{res_dir}/{server_name}-{dataset_name}"
    return prompts, output_lens, res_file, config["model_path"]


async def one_round(server_name: str):
    start_server(server_name, config)
    try:
        # Figure 10b 仍然是吞吐测试，所以 rate 保持为 -1。
        # 不同点在于：通过 max_inflight 把请求注入方式改成“受控高并发饱和”，
        # 避免客户端自己先被海量并发建连打爆。
        await run_test(
            *prepare_ac_test("ac", config, server_name, NUM_THRESHOLD),
            rate=-1,
            max_inflight=MAX_INFLIGHT,
        )
    finally:
        stop_server()
    await asyncio.sleep(5)


async def main():
    await one_round("vllm")
    await one_round("ours")


def draw_one_bar_diagram(
    name,
    labels,
    values,
    xlabel,
    ylabel,
    colors=None,
):
    if colors is None:
        colors = [
            "tab:blue",
            "tab:orange",
            "tab:green",
            "tab:red",
            "tab:purple",
            "tab:brown",
            "tab:pink",
            "tab:gray",
            "tab:olive",
            "tab:cyan",
        ]

    bar_colors = [colors[i % len(colors)] for i in range(len(labels))]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values, color=bar_colors)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # 在每个柱子顶部添加数值，便于直接读取吞吐结果。
    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}",
            ha="center",
            va="bottom",
        )

    plt.savefig(f"{name}.pdf", format="pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    asyncio.run(main())
    ours_res_file = f"{res_dir}/ours-ac-tp.json"
    vllm_res_file = f"{res_dir}/vllm-ac-tp.json"

    ours_req_per_s = get_tp([ours_res_file], [0.3, 0.7])
    vllm_req_per_s = get_tp([vllm_res_file], [0.3, 0.7])
    draw_one_bar_diagram(
        f"{cur_dir}/fig10b_reqps",
        ["ours", "vllm"],
        [ours_req_per_s[0], vllm_req_per_s[0]],
        "",
        "Throughput (req/s)",
    )

    ours_token_per_s = get_tp_token([ours_res_file])
    vllm_token_per_s = get_tp_token([vllm_res_file])
    draw_one_bar_diagram(
        f"{cur_dir}/fig10b_tokenps",
        ["ours", "vllm"],
        [ours_token_per_s[0], vllm_token_per_s[0]],
        "",
        "Throughput (token/s)",
    )
