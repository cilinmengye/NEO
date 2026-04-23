当前实验环境是在4090下使用llama-3-8B, 选择AC数据集，选择其中4000条request在不同rate下测试vllm 和 NEO 延时

可以看到在 GPU KV Cache 给得较为大方的情况下（配置大概给了3.32 GB），在请求压力比较大的情况下，NEO 延时 还是比 vllm 好的，而且甚至 NEO 
吞吐量也比 vllm 大

但是可以通过观察 our-server.log 可以发现 NEO 依旧有较多的 Cdec 待处理在请求压力较大时。