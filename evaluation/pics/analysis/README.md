此 ours-server.log 我是在 config-4090-8b.json 配置：
```
{
    "model": "Llama-3-8B",
    "model_path": "/workspace/huggingface/Meta-Llama-3.1-8B",
    "num_layers": 32,
    "block_size": 16,
    "max_model_len": 26400,
    "max_num_seqs": 1024,
    "max_num_batched_tokens": 26400,
    "tensor_parallel_size": 1,
    "gpu_memory_utilization": 0.95,
    "num_gpu_blocks_override": 1650,
    "swap_space": 60,
    "library": "libpacpu-llama3_8b-tp1.so"
}
```

运行数据集 AC， 数据量 4000， 执行脚本 reproduce-fig6b.py ，在 rate = [3.1, 3.5] 下 运行 NEO 采集到的信息

并且我改动了 `scheduler.py`:
```
        # if pref_to_gpu or pref_to_cpu:
        logger.info(
                "Gdecs: %d, Cdecs: %d, Pr2gs: %d, Pr2cs: %d, Waiting: %d",
                len(self.gpu_decoding_q), len(self.cpu_decoding_q), len(pref_to_gpu), len(pref_to_cpu), len(self.waiting_q)
            )
```

且改动了 `engine.py`
```
            # if any(b.num_prefs for b in batches):
            logger.info(f"Forwarding batches with sizes {[(b.num_cprfs, b.num_gprfs, b.num_gdecs, b.num_cdecs) for b in batches]}, "
                            f"swap out: {len(cur_swap_out)}, swap in: {len(cur_swap_in)}")
            output_token_ids = await self._run_on_model_executor_async(self.executor.do_one_iteration, batches, *forward_args)
```

目的是让其每次 iteration 我都知道其具体调度信息

观察 ours-server.log 可以发现，因为 NEO 秉持 GPU decode request 优先的原则，所以其大部分时候 GPU 都在执行 decode request

因为此原因，大部分时候 CPU 也是没有参与运算的，即大部分时候都是只有一个 gpu-only batch, 且这个 batch 都在执行 GPU decode request

这会严重拖慢 TTFT (Time to First Token)