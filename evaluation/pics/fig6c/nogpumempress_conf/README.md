出现此种结果的原因是**大部分请求都只在 GPU 上运行了**，可以从 our-server.log 看出，基本上都只其一个 gpu-only batch 到 GPU 上执行。

而且 Waiting queue 大多数时候都是 0, 或者根本不大。说明 GPU 显存基本上没有压力。