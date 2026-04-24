import json
import os
import matplotlib.pyplot as plt
import numpy as np

cur_dir = os.path.dirname(os.path.realpath(__file__))


def _get_successful_records(file):
    with open(file) as f:
        data = json.load(f)

    # 新结果文件会额外写入 ok/error 字段；这里统一只消费成功请求。
    # 旧结果文件没有 ok 字段时，get(..., True) 会保持原有兼容行为。
    return [item for item in data if item.get('ok', True)]


def _get_metric_records(file):
    data = _get_successful_records(file)
    return data[len(data) // 4:]


def _get_tpot_sample(record: dict) -> float | None:
    if record.get('streamed') is not True:
        return None

    token_offsets = record.get('token_offsets')
    if isinstance(token_offsets, list) and len(token_offsets) > 1:
        return sum([
            token_offsets[i] - token_offsets[i - 1]
            for i in range(1, len(token_offsets))
        ]) / (len(token_offsets) - 1)

    chunk_offsets = record.get('chunk_offsets')
    if isinstance(chunk_offsets, list) and len(chunk_offsets) > 1:
        return sum([
            chunk_offsets[i] - chunk_offsets[i - 1]
            for i in range(1, len(chunk_offsets))
        ]) / (len(chunk_offsets) - 1)

    first_token_offset = record.get('first_token_offset')
    events_received = record.get('events_received')
    if first_token_offset is None or not isinstance(events_received, int) or events_received <= 1:
        return None

    remaining = (record['end'] - record['start']) - first_token_offset
    if remaining <= 0:
        return None

    return remaining / (events_received - 1)


def get_metric_avg(file, metric: str = "avg_per_token_latency"):
    data = _get_metric_records(file)

    if metric == "avg_per_token_latency":
        return sum([(x['end'] - x['start']) / x['output_len'] for x in data]) / len(data)

    if metric == "ttft":
        ttfts = []
        for record in data:
            if record.get('streamed') is not True:
                continue
            first_token_offset = record.get('first_token_offset')
            if first_token_offset is not None:
                ttfts.append(first_token_offset)
        if not ttfts:
            raise RuntimeError(
                f"No TTFT samples available in {file}; rerun benchmark with collect_stream_metrics=True"
            )
        return sum(ttfts) / len(ttfts)

    if metric == "tpot":
        tpots = []
        for record in data:
            sample = _get_tpot_sample(record)
            if sample is not None:
                tpots.append(sample)
        if not tpots:
            raise RuntimeError(
                f"No TPOT samples available in {file}; need token_offsets or chunk stream events"
            )
        return sum(tpots) / len(tpots)

    raise ValueError(f"Unknown metric: {metric}")


def get_lat_avg(file):
    """
    end - start 是单个请求的 completion time / latency:
    此指标能回答:
        单请求慢不慢
        平均请求时延是多少
    """
    return get_metric_avg(file)


def _get_metric_ylabel(metric: str) -> str:
    if metric == "avg_per_token_latency":
        return "Average per token latency (s)"
    if metric == "ttft":
        return "TTFT (s)"
    if metric == "tpot":
        return "TPOT (s)"
    raise ValueError(f"Unknown metric: {metric}")


def draw_one_rl_diagram(
    title: str,
    data_name: str,
    sys_file_names: list[str],
    sys_legend_names: list[str],
    rate_lists: list[list[float]],
    ylim: float,
    markers: list[str],
    set_ylabel: bool = False,
    metric: str = "avg_per_token_latency",
):
    lats = []
    max_rate = max([max(rate_list) for rate_list in rate_lists])
    for sys_file_name, rate_list in zip(sys_file_names, rate_lists):
        lats.append([])
        for rate in rate_list:
            rate_str = str(rate).replace(".", "_")
            lats[-1].append(get_metric_avg(f"{cur_dir}/results/{sys_file_name}-{data_name}-lat-{rate_str}.json", metric))

    # ax.set_title(title, y=-0.3, fontsize="x-large")

    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    for i, sys_legend_name in enumerate(sys_legend_names):
        ax.plot(rate_lists[i], lats[i], label=sys_legend_name, marker=markers[i])

    ax.set_xlabel("Ruquest rate (req/s)", fontsize="large")
    if set_ylabel:
        ax.set_ylabel(_get_metric_ylabel(metric), fontsize="large")
    ax.set_xlim(0, max_rate)
    ax.set_xticks([0.5 * x for x in range(round(max_rate * 2 + 1))])
    # ax.set_ylim(-ylim / 50, ylim)
    # ax.set_yticks([ylim / 5 * x for x in range(6)])
    ax.set_xticklabels([f"{x:.1f}" for x in ax.get_xticks()], fontsize="large")
    # ax.set_yticklabels([f"{y:.2f}" for y in ax.get_yticks()], fontsize="large")
    ax.grid(True)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend()
    plt.savefig(f"{cur_dir}/{title}.pdf", bbox_inches='tight')
    return handles, labels


def get_tp(filenames: list[str], interv: tuple[float, float]):
    tps = []
    for i, filename in enumerate(filenames):
        data = _get_successful_records(filename)

        times = sorted([d['end'] for d in data])
        data = [times[j] - times[j-1] for j in range(1, len(times))]

        ndata = len(data)
        nwarmup = round((ndata + 1) * interv[0])
        ncooldown = round((ndata + 1) * interv[1])
        tps.append(1 / np.mean(data[nwarmup: ncooldown]))
    return tps


def get_tp_token(filenames: list[str]):
    tps = []
    for i, filename in enumerate(filenames):
        data = _get_successful_records(filename)

        first_start = min([d['start'] for d in data])
        last_end = max([d['end'] for d in data])
        total_time = last_end - first_start
        total_tokens = sum([d['output_len'] + d['input_len'] for d in data])
        tps.append(total_tokens / total_time)

    return tps


def draw_one_ps_diagram(
    title: str,
    base_sys_name: str,
    interv: list[float],
    num_datas: list[int],
    sys_file_names: list[str],
    legend_names: list[str | None],
    input_lens: list[int],
    output_lens: list[int],
    markers: list[str],
    show_ylabels: bool = False,
    show_legend: bool = True,
):
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    for i in range(len(num_datas)):
        tps = []
        for out_len in output_lens:
            file_names = [f'{cur_dir}/results/{sys_name}-{num_datas[i]}-{input_lens[i]}-{out_len}-tp.json' for sys_name in [base_sys_name, sys_file_names[i]]]
            tp_pair = get_tp(file_names, interv)
            tps.append(tp_pair)

        ratios = [tp1 / tp0 for tp0, tp1 in tps]
        ax.plot(output_lens, ratios, label=f'{legend_names[i]}', marker=markers[i])

    # draw y = 1 line
    ax.plot([output_lens[0], output_lens[-1]], [1, 1], 'r--', label='baseline')

    ax.set_xlabel('Avg. output length', fontsize='large')
    if show_ylabels:
        ax.set_ylabel('Relative throughput', fontsize='large')
    ax.set_xticklabels([f'{x:.0f}' for x in ax.get_xticks()], fontsize='large')
    ax.set_yticklabels([f'{x:.2f}' for x in ax.get_yticks()], fontsize='large')
    handles, labels = ax.get_legend_handles_labels()
    if show_legend:
        ax.legend()
    fig.savefig(f'{cur_dir}/{title}.pdf', bbox_inches='tight')
    return handles, labels


def parse_ours_server_log(file):
    # Get sizes list from lines like below
    # INFO:swiftllm.server.engine:Forwarding batches with sizes [(0, 1, 14, 0)], swap out: 0, swap in: 4
    with open(file) as f:
        lines = f.readlines()

    sizes = []
    for line in lines:
        if 'Forwarding batches with sizes' in line:
            sizes.append(eval(line.split('[')[1].split(']')[0]))

    return sizes


def parse_vllm_server_log(file):
    # Get Rumnings list from lines like below
    # INFO 10-28 08:28:24 metrics.py:351] Avg prompt throughput: 5806.9 tokens/s, Avg generation throughput: 53.0 tokens/s, Running: 12 reqs, Swapped: 0 reqs, Pending: 1466 reqs, GPU KV cache usage: 93.6%, CPU KV cache usage: 0.0%.
    with open(file) as f:
        lines = f.readlines()

    runnings = []
    for line in lines:
        if 'Running' in line and 'Avg prompt throughput' in line:
            runnings.append(int(line.split('Running: ')[1].split(' reqs')[0]))

    return runnings
