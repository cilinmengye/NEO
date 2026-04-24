import os
import asyncio
import time
import logging
import json
import random

import aiohttp
import numpy as np
from tqdm import tqdm

# pylint: disable=import-error
from api_client import request_completions, request_completions_stream, AIOHTTP_TIMEOUT

cur_dir = os.path.dirname(os.path.abspath(__file__))
res_dir = f"{cur_dir}/results"
os.makedirs(res_dir, exist_ok=True)

logger = logging.getLogger(__name__)
logging.basicConfig(filename=f"{cur_dir}/evaluation.log", level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

api_url = "http://localhost:8000/v1/completions"


def _results_match_streaming_mode(all_results: list[dict], collect_stream_metrics: bool) -> bool:
    if not collect_stream_metrics:
        return True

    for result in all_results:
        if not result.get("ok", True):
            continue

        if result.get("streamed") is not True:
            return False

        if result.get("first_token_offset") is None:
            return False

        if "token_offsets" in result:
            continue

        if result.get("stream_observation") != "chunk":
            return False

        chunk_offsets = result.get("chunk_offsets")
        if isinstance(chunk_offsets, list) and chunk_offsets:
            continue

        if not isinstance(result.get("events_received"), int) or result["events_received"] <= 0:
            return False

    return True



async def request_completions_task(
    session: aiohttp.ClientSession,
    prompt: list[int],
    output_len: int,
    model_path: str,
    collect_stream_metrics: bool = False,
):
    """
    发送单个请求并记录起止时间。

    这里统一把成功/失败都收敛成结构化结果，而不是直接把异常抛给
    `asyncio.gather()`；这样吞吐测试在部分请求失败时仍能产出完整结果，
    并且不会因为首个失败立刻中断整轮测试。
    """
    start = time.perf_counter()
    try:
        stream_result = None
        if collect_stream_metrics:
            stream_result = await request_completions_stream(session, api_url, prompt, output_len, model_path, start_time=start)
        else:
            await request_completions(session, api_url, prompt, output_len, model_path)
        end = time.perf_counter()
        result = {
            "input_len": len(prompt),
            "output_len": output_len,
            "start": start,
            "end": end,
            "ok": True,
            "error": None,
        }
        if stream_result is not None:
            result.update(stream_result)
        return result
    except Exception as exc:  # pylint: disable=broad-except
        end = time.perf_counter()
        return {
            "input_len": len(prompt),
            "output_len": output_len,
            "start": start,
            "end": end,
            "ok": False,
            "error": repr(exc),
        }


async def _run_rate_test(
    session: aiohttp.ClientSession,
    prompts: list[list[int]],
    output_lens: list[int],
    model_path: str,
    rate: float,
    collect_stream_metrics: bool = False,
):
    """
    保持原有 latency / rate test 语义：
    - 请求之间仍按指数分布 gap 发送；
    - 但所有请求共享同一个 session，避免重复创建连接管理对象。
    """
    tasks = []
    np.random.seed(0)
    gaps = np.random.exponential(1 / rate, len(prompts)).tolist()
    for prompt, output_len in tqdm(zip(prompts, output_lens), total=len(prompts)):
        tasks.append(
            asyncio.create_task(
                request_completions_task(
                    session,
                    prompt,
                    output_len,
                    model_path,
                    collect_stream_metrics=collect_stream_metrics,
                )
            )
        )
        await asyncio.sleep(gaps.pop(0))

    return await asyncio.gather(*tasks)


async def _run_throughput_test(
    session: aiohttp.ClientSession,
    prompts: list[list[int]],
    output_lens: list[int],
    model_path: str,
    max_inflight: int,
):
    """
    受控吞吐测试。

    核心思想：
    - 不对请求到达率做 sleep 节流；
    - 但限制同时在途的请求数；
    - 某个请求一完成，就立刻补发下一个请求。

    这样压测的是“持续饱和状态下的可持续吞吐”，而不是客户端能否
    在一个瞬间创建几十万条 TCP 连接。
    """
    total = len(prompts)
    if total == 0:
        return []

    max_inflight = max(1, min(max_inflight, total))
    results = [None] * total
    next_idx = 0
    pending: dict[asyncio.Task, int] = {}

    def _submit_one(index: int):
        pending[
            asyncio.create_task(
                request_completions_task(session, prompts[index], output_lens[index], model_path)
            )
        ] = index

    # 先把 in-flight 窗口填满，后续每完成一个再补一个。
    while next_idx < max_inflight:
        _submit_one(next_idx)
        next_idx += 1

    with tqdm(total=total) as pbar:
        while pending:
            done, _ = await asyncio.wait(
                pending.keys(),
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in done:
                index = pending.pop(task)
                results[index] = task.result()
                pbar.update(1)

                if next_idx < total:
                    _submit_one(next_idx)
                    next_idx += 1

    return results


async def run_test(
    prompts: list[list[int]],
    output_lens: list[int],
    res_prefix: str,
    model_path: str,
    rate: float = -1,  # -1 means throughput test
    max_inflight: int | None = None,
    collect_stream_metrics: bool = False,
):
    if rate > 0:
        res_file = f"{res_prefix}-lat-{str(rate).replace('.', '_')}.json"
    else:
        res_file = f"{res_prefix}-tp.json"

    should_rerun = False
    if os.path.exists(res_file):
        logger.info("Test result file already exists: %s", res_file)
        with open(res_file, "r") as f:
            all_results = json.load(f)
        if not _results_match_streaming_mode(all_results, collect_stream_metrics):
            logger.info("Cached result file %s is incompatible with collect_stream_metrics=%s, rerunning", res_file, collect_stream_metrics)
            should_rerun = True
    else:
        should_rerun = True

    if should_rerun:
        logger.info("Running test, saving results to %s", res_file)

        # throughput 模式下默认只保留一个保守的在途窗口，防止客户端自己先被打爆。
        inflight = None if rate > 0 else (20000 if max_inflight is None else max_inflight)

        # connector limit 与吞吐模式的 in-flight 上限对齐，这样连接池行为和调度逻辑一致。
        connector_limit = 0 if inflight is None else inflight
        connector = aiohttp.TCPConnector(limit=connector_limit, limit_per_host=connector_limit)

        async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT, connector=connector) as session:
            if rate > 0:
                all_results = await _run_rate_test(
                    session,
                    prompts,
                    output_lens,
                    model_path,
                    rate,
                    collect_stream_metrics=collect_stream_metrics,
                )
            else:
                all_results = await _run_throughput_test(session, prompts, output_lens, model_path, inflight)

        with open(res_file, "w") as f:
            # 保持旧字段兼容，同时追加 ok/error 便于排查部分失败。
            json.dump(all_results, f, indent=4)

        successful_count = sum(result["ok"] for result in all_results)
        failed_count = len(all_results) - successful_count
        logger.info(
            "Finished test with %d successes and %d failures",
            successful_count,
            failed_count,
        )
        if failed_count:
            first_error = next(result["error"] for result in all_results if not result["ok"])
            logger.warning("First request error: %s", first_error)

    successful_results = [result for result in all_results if result.get("ok", True)]
    if not successful_results:
        raise RuntimeError(f"No successful requests recorded in {res_file}")

    # if collect_stream_metrics:
    #     for result in successful_results:
    #         if result.get("tokens_received") != result["output_len"]:
    #             raise RuntimeError(
    #                 f"Streaming token count mismatch in {res_file}: expected {result['output_len']}, got {result.get('tokens_received')}"
    #             )

    times = [(result["start"], result["end"]) for result in successful_results]

    if rate > 0:
        comp_times = [end - start for start, end in times]
        pertok_times = [
            (result["end"] - result["start"]) / (result["input_len"] + result["output_len"])
            for result in successful_results
        ]
        average_completion_time = sum(comp_times) / len(comp_times)
        average_pertok_time = sum(pertok_times) / len(pertok_times)
        logger.info("Average completion time: %.3f s", average_completion_time)
        logger.info("Average per-token completion time: %.3f s", average_pertok_time)
    else:
        n = len(times)
        if n < 2:
            raise RuntimeError(f"Need at least 2 successful requests to compute throughput: {res_file}")

        req_end_times = sorted([end for _, end in times])
        req_end_times = req_end_times[n // 10: n - n // 10 * 3 + 1]
        throughput = (len(req_end_times) - 1) / (req_end_times[-1] - req_end_times[0])
        logger.info("Throughput: %.3f req/s", throughput)


def _get_rand_array(n: int, avg_val: int, ratio: float):
    """
    Get a random array with average value `avg_val`,

    all values are uniformly distributed in the range of [avg_val * (1 - ratio), avg_val * (1 + ratio)]
    """
    delta = int(avg_val * ratio)
    return [avg_val + random.randint(-delta, delta) for _ in range(n)]


def prepare_mock_test(
    nreqs: int,
    input_len: int,
    output_len: int,
    server_name: str,
    config: dict
) -> tuple[list[list[int]], list[int], str]:
    input_lens = _get_rand_array(nreqs, input_len, 0.1)
    output_lens = _get_rand_array(nreqs, output_len, 0.1)
    prompts = [[10] * input_len for input_len in input_lens]
    res_file = f"{res_dir}/{server_name}-{nreqs}-{input_len}-{output_len}"
    return prompts, output_lens, res_file, config['model_path']


def prepare_real_test(
    dataset_name: str,
    config: dict,
    server_name: str
) -> tuple[list[list[int]], list[int], str]:
    input_file = f"{cur_dir}/data/{dataset_name}-{config['model']}.json"
    with open(input_file, "r") as f:
        # Remove the [:100] to use the full dataset. However, it may take a long time (~10h) to run the full test of fig6c.
        datas = json.load(f)[:1000]
        prompts = [[10] * data["prompt"] for data in datas]
        output_lens = [data["max_tokens"] for data in datas]

    res_file = f"{res_dir}/{server_name}-{dataset_name}"
    return prompts, output_lens, res_file, config['model_path']
