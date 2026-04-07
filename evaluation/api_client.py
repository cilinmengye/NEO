import os
import json
import asyncio

from fastapi import HTTPException
import aiohttp

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)


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
    payload = {
        "model": model_path,
        "prompt": prompt,
        "max_tokens": output_len,
        "temperature": 0.0,
        "ignore_eos": True
    }

    async with session.post(url=api_url, json=payload) as response:
        if response.status != 200:
            raise HTTPException(status_code=response.status, detail=await response.text())
        data = json.loads(await response.text())

    return data["choices"][0]["text"]
