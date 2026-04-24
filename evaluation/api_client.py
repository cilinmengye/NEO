import json
import time

from fastapi import HTTPException
import aiohttp

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)


def _build_completion_payload(
    prompt: str | list[int],
    output_len: int,
    model_path: str,
) -> dict:
    return {
        "model": model_path,
        "prompt": prompt,
        "max_tokens": output_len,
        "temperature": 0.0,
        "ignore_eos": True,
    }


def _get_sse_event_metrics(event: dict) -> tuple[int, bool, str | None]:
    token_count = 0
    has_output = False
    for choice in event.get("choices", []):
        token_ids = choice.get("token_ids")
        if isinstance(token_ids, list):
            token_count += len(token_ids)
            has_output = True
            continue
        if token_ids is not None:
            token_count += 1
            has_output = True
            continue

        text = choice.get("text")
        if isinstance(text, list):
            if any(piece for piece in text):
                has_output = True
            continue
        if text:
            has_output = True
            continue

        delta = choice.get("delta")
        if isinstance(delta, dict) and delta.get("content"):
            has_output = True

    if token_count > 0:
        return token_count, has_output, "token"
    if has_output:
        return 0, True, "chunk"
    return 0, False, None


async def request_completions(
    session: aiohttp.ClientSession,
    api_url: str,
    prompt: str | list[int],
    output_len: int,
    model_path: str
):
    """
    发送一次 completion 请求。

    这里不再在函数内部创建 ClientSession，原因是：
    - throughput 模式下会同时存在大量请求；
    - 如果每个请求都单独创建 session/connector，会制造额外的建连风暴；
    - 由上层复用一个共享 session，才能真正复用连接池并限制并发建连数量。
    """
    payload = _build_completion_payload(prompt, output_len, model_path)

    async with session.post(url=api_url, json=payload) as response:
        if response.status != 200:
            raise HTTPException(status_code=response.status, detail=await response.text())
        data = json.loads(await response.text())

    return data["choices"][0]["text"]


async def request_completions_stream(
    session: aiohttp.ClientSession,
    api_url: str,
    prompt: str | list[int],
    output_len: int,
    model_path: str,
    start_time: float | None = None,
) -> dict:
    payload = _build_completion_payload(prompt, output_len, model_path)
    payload["stream"] = True
    start_time = time.perf_counter() if start_time is None else start_time
    token_offsets = []
    chunk_offsets = []
    first_token_offset = None
    events_received = 0
    stream_observation = None
    saw_sse = False

    async with session.post(url=api_url, json=payload) as response:
        if response.status != 200:
            raise HTTPException(status_code=response.status, detail=await response.text())

        async for raw_line in response.content:
            line = raw_line.decode("utf-8").strip()
            if not line:
                continue

            if line.startswith("data:"):
                saw_sse = True
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                num_tokens, has_output, observation = _get_sse_event_metrics(json.loads(data))
            elif ":" in line:
                continue
            else:
                num_tokens = 1
                has_output = True
                observation = "token"

            if not has_output:
                continue

            events_received += 1
            offset = time.perf_counter() - start_time
            if first_token_offset is None:
                first_token_offset = offset

            if observation == "chunk":
                if stream_observation != "chunk":
                    token_offsets.clear()
                    stream_observation = "chunk"
                chunk_offsets.append(offset)
                continue

            if stream_observation == "chunk":
                chunk_offsets.append(offset)
                continue

            stream_observation = "token"
            token_offsets.extend([offset] * num_tokens)

    result = {
        "streamed": True,
        "stream_observation": stream_observation or ("chunk" if saw_sse else "token"),
        "events_received": events_received,
        "first_token_offset": first_token_offset,
    }
    if result["stream_observation"] == "token":
        result["tokens_received"] = len(token_offsets)
        result["token_offsets"] = token_offsets
    elif chunk_offsets:
        result["chunk_offsets"] = chunk_offsets
    return result
