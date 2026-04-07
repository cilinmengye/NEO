# NEO 环境调研与 `csrc` 编译报错分析

## 1. 问题概述

在 `/home/yxlin/github/swift/NEO` 中执行：

```bash
pip install -e csrc --no-build-isolation
```

会在构建 `swiftllm_c` editable wheel 时失败。失败发生在：

- `csrc/src/small_kernels.cu`
- `csrc/src/linear.cu`

而下面两个文件的编译没有报错：

- `csrc/src/block_swapping.cpp`
- `csrc/src/entrypoints.cpp`

报错关键信息是：

- 编译 `.cu` 文件时调用的是 `/usr/local/cuda-12.4/bin/nvcc`
- 报错来自 `/usr/local/cuda-12.4/include/crt/host_config.h:143`
- 错误内容：`gcc versions later than 13 are not supported`

因此，这不是一个随机编译失败，而是一个非常明确的 **CUDA 12.4 与宿主 GCC 版本不兼容** 问题。

---

## 2. 直接结论

### 结论 1：为什么 `block_swapping.o`、`entrypoints.o` 不报错？

因为它们对应的源文件是 `.cpp`：

- `csrc/src/block_swapping.cpp`
- `csrc/src/entrypoints.cpp`

在当前构建系统里，这类文件走的是普通 C++ 编译器（`c++/g++`），**不会经过 `nvcc` 的 host compiler 版本检查**。

### 结论 2：为什么 `linear.o`、`small_kernels.o` 会报错？

因为它们对应的源文件是 `.cu`：

- `csrc/src/linear.cu`
- `csrc/src/small_kernels.cu`

`.cu` 文件在当前构建系统里会交给 `nvcc` 编译，而 `nvcc` 会进一步调用宿主 C++ 编译器（host compiler）。当前机器上默认的 `gcc/g++` 版本过高，超过了 CUDA 12.4 支持范围，因此在编译 `.cu` 文件时直接被 `nvcc` 拒绝。

### 结论 3：根因不在 `.cu` 代码逻辑本身，而在构建配置没有正确指定兼容的 host compiler

也就是说：

- 不是 `small_kernels.cu` 写错了
- 不是 `linear.cu` 语法有问题
- 而是 `csrc/setup.py` 没有像 `pacpu/build.sh` 那样，把 **普通 C++ 编译器** 和 **NVCC host compiler** 分开指定

---

## 3. 证据链

## 3.1 `csrc/setup.py` 的构建方式决定了 `.cpp` 和 `.cu` 会走不同编译路径

文件：`/home/yxlin/github/swift/NEO/csrc/setup.py`

关键内容如下：

```python
ext_modules = [
    cpp_extension.CUDAExtension(
        "swiftllm_c",
        [
            "src/entrypoints.cpp",
            "src/block_swapping.cpp",
            "src/small_kernels.cu",
            # "src/attention.cu",
            "src/linear.cu",
        ],
        extra_compile_args={
            'cxx': ['-O3'],
            'nvcc': ['-O3', '--use_fast_math']
        }
    ),
]
```

这段配置说明：

- 该扩展使用的是 `torch.utils.cpp_extension.CUDAExtension`
- `.cpp` 和 `.cu` 文件被放在同一个 extension 中统一构建
- 但实际编译时，仍然会按后缀分流：
  - `.cpp` -> `cxx`
  - `.cu` -> `nvcc`

这正好对应你日志里的现象：

- `block_swapping.cpp` / `entrypoints.cpp` 是 `c++` 在编
- `small_kernels.cu` / `linear.cu` 是 `nvcc` 在编

### 关键问题

`csrc/setup.py` 当前只有：

- `cxx: ['-O3']`
- `nvcc: ['-O3', '--use_fast_math']`

但是 **没有** 下面这些内容：

- 没有显式指定 `-ccbin`
- 没有处理 `CUDAHOSTCXX`
- 没有自动探测兼容版本的 `g++`
- 没有在默认 GCC 过新时提前给出友好报错

因此 `nvcc` 很可能直接拿系统默认 `g++` 当 host compiler 使用。

---

## 3.2 报错本身已经明确指向 GCC 版本过高

你的日志中最关键的部分是：

```text
/usr/local/cuda-12.4/include/crt/host_config.h:143:2: 错误：#error -- unsupported GNU version! gcc versions later than 13 are not supported!
```

这说明：

- 问题发生在 CUDA 头文件的 host compiler 版本检查阶段
- 并不是在编译具体 kernel 逻辑时才失败
- 因此这不是算法实现错误，而是工具链不匹配

也就是说，只要 `nvcc` 看到的宿主 GCC 版本大于 13，它就会在很早的阶段直接失败。

---

## 3.3 仓库 README 已经明确说明“需要两个版本的 g++”

文件：`/home/yxlin/github/swift/NEO/README.md`

README 里写的是：

```text
2 versions of g++ (see `pacpu/build.sh` for more details):
- one >= 13 (for compiling CPU kernel)
- the other < 13 (for passing the NVCC version check)
```

这段说明和当前报错完全吻合。

它已经明确表达了两个事实：

1. 项目作者知道 NVCC 对 GCC 版本有限制
2. 该项目原本就不是“单一 g++ 版本”环境，而是“双编译器”环境

所以，当前 `csrc` 编译失败，并不是你环境特殊，而是 `csrc` 这条构建路径没有把 README 中提到的双编译器要求真正落实到构建脚本里。

---

## 3.4 `pacpu/build.sh` 已经实现了正确的双编译器做法

文件：`/home/yxlin/github/swift/NEO/pacpu/build.sh`

内容：

```bash
Torch_DIR=$(python -c 'import torch;print(torch.utils.cmake_prefix_path)')/Torch
CUDA_HOST_COMPILER_PATH=$(which g++-11)
CXX_COMPILER_PATH=$(which g++-13)

mkdir -p build
cmake -B build -S . -DTorch_DIR=$Torch_DIR -DModel=$1 -DTP=$2 -DCMAKE_CUDA_HOST_COMPILER=${CUDA_HOST_COMPILER_PATH} -DCMAKE_CXX_COMPILER=${CXX_COMPILER_PATH}
cmake --build build
```

这个脚本的设计非常关键：

- `CUDA_HOST_COMPILER_PATH=$(which g++-11)`
- `CXX_COMPILER_PATH=$(which g++-13)`

并且在 CMake 中分开传入：

- `-DCMAKE_CUDA_HOST_COMPILER=...`
- `-DCMAKE_CXX_COMPILER=...`

这说明作者已经在 `pacpu` 中采用了：

- **较旧的 g++** 给 NVCC 做 host compiler
- **较新的 g++** 负责普通 C++ 编译

也就是说，`pacpu` 已经解决了你现在在 `csrc` 遇到的问题。

因此最自然的修复思路不是去修改 `.cu` 源码，而是：

> 把 `pacpu` 这套 split-compiler 思路复用到 `csrc/setup.py`。

---

## 4. 本地环境调研结果

调研发现当前环境中：

- `nvcc` 存在：`/usr/local/cuda-12.4/bin/nvcc`
- 默认 `gcc` 存在：`/usr/bin/gcc`
- 默认 `g++` 存在：`/usr/bin/g++`

但在 PATH 中没有找到：

- `g++-11`
- `g++-12`
- `g++-13`
- `gcc-11`
- `gcc-12`
- `gcc-13`

这点非常重要。

### 这意味着什么？

这意味着当前机器上，至少从 PATH 来看：

- 只有系统默认 `gcc/g++`
- 缺少 README 和 `pacpu/build.sh` 所预期的多版本 GCC 环境

因此：

1. `csrc/setup.py` 没法自动使用 `g++-11` 之类的兼容 host compiler
2. `pacpu/build.sh` 其实也大概率会因为 `which g++-11` / `which g++-13` 失败而出问题

换句话说，这不是只有 `csrc` 的问题，**当前机器的编译器准备本身也不满足该仓库推荐环境**。

---

## 5. 为什么 `.cpp` 能过，而 `.cu` 不能过：更直白的解释

可以把这次构建理解成两套不同的门禁：

### 5.1 `.cpp` 文件

`.cpp` 文件直接交给普通 C++ 编译器。

只要：

- 代码语法正确
- 头文件能找到
- ABI 没冲突

那么它就可以编过去。

因此：

- `block_swapping.cpp`
- `entrypoints.cpp`

即便包含 CUDA runtime 相关 API，也依然只是 host 侧 C++ 封装代码，所以普通 C++ 编译器可以处理。

### 5.2 `.cu` 文件

`.cu` 文件交给 `nvcc`。

而 `nvcc` 在处理 `.cu` 时，不只是编 device code，还会调用宿主 C++ 编译器处理 host 部分。因此，`nvcc` 会首先检查你提供的 host compiler 是否在支持范围内。

当前情况是：

- `nvcc = CUDA 12.4`
- host compiler = 系统默认 GCC（版本过新）

于是还没进入真正的 kernel 编译逻辑，就先在版本检查那里失败了。

所以：

- 不是 `.cu` 比 `.cpp` “更难编”
- 而是 `.cu` 走了 `nvcc` 这条路径，而 `nvcc` 对宿主 GCC 版本有限制

---

## 6. 推荐解决方案

## 6.1 最推荐：补齐双编译器环境，并让 `csrc` 显式使用它

这是最符合当前仓库设计的方案。

### 做法

- 安装一个兼容 CUDA 12.4 的较旧 GCC/G++，优先考虑：
  - `g++-11`
  - `g++-12`
  - `g++-13`
- 同时保留较新的 GCC/G++ 给普通 C++ 路径使用

对于 `csrc`：

- `nvcc` 使用旧版 `g++` 做 host compiler
- `.cpp` 继续走普通 C++ 编译器

### 优点

- 和 `pacpu/build.sh` 保持一致
- 不需要修改 `.cu` 源码
- 不需要全局降级系统默认编译器
- 是最稳妥、最符合项目预期的做法

### 缺点

- 需要机器上存在多个 GCC 版本
- 需要在构建脚本中明确指定编译器

---

## 6.2 短期 workaround：安装前手工设置编译器环境变量

如果只是想先快速验证问题，可以采用临时方案：

- 手工设置 `CUDAHOSTCXX`
- 必要时同时设置 `CC` / `CXX`

让 `csrc` 在安装时显式使用兼容的旧 GCC。

### 优点

- 最快验证问题是否只是编译器配置
- 暂时不需要改代码

### 缺点

- 完全依赖人工记忆
- 新环境、CI、他人复现时仍容易踩坑
- 不是长期方案

---

## 6.3 永久方案：修改 `csrc/setup.py`

### 推荐改动方向

在 `csrc/setup.py` 中补充以下逻辑：

1. 优先读取用户显式设置的：
   - `CUDAHOSTCXX`
   - `CC`
   - `CXX`
2. 如果没有设置 `CUDAHOSTCXX`，自动探测：
   - `g++-11`
   - `g++-12`
   - `g++-13`
3. 将探测结果显式传给 NVCC（例如通过 `-ccbin` 或等效机制）
4. 如果系统默认 GCC 过新，且又没找到兼容 host compiler，则在构建一开始就快速报错，并明确告诉用户需要安装哪个版本的 GCC

### 优点

- 用户体验最好
- 能把仓库 README 里的编译器要求真正落实到安装流程中
- 可以避免用户在 `pip install -e csrc` 时再次遇到相同问题

### 缺点

- 需要改 `csrc/setup.py`
- 需要补充一些探测与报错逻辑

---

## 6.4 配套方案：同步更新 README

当前 README 只写了“需要两个版本的 g++”，但没有把 `csrc` 这条安装路径讲清楚。

建议在 `README.md` 的 Installation 部分补充：

- `csrc` 为什么会依赖双编译器
- `pacpu/build.sh` 是参考实现
- 如果机器默认 GCC 太新，如何为 `csrc` 指定兼容的 host compiler

这样可以减少后续重复踩坑。

---

## 7. 不推荐作为默认解法的方案

## 7.1 给 NVCC 加 `--allow-unsupported-compiler`

这个方法可以绕过版本检查，但不推荐作为正式方案。

### 原因

- 它只是跳过检查，不代表工具链真的兼容
- 可能导致：
  - 编译期失败
  - 链接期失败
  - 运行时错误
  - 更隐蔽的错误结果

### 适用场景

- 临时实验
- 快速验证

### 不适合

- 正式环境
- 稳定复现
- 仓库默认安装方式

---

## 7.2 全局切换系统默认 GCC 到旧版本

不推荐。

### 原因

- 会影响整机其他项目
- 与 `pacpu` 对较新 C++ 编译器的需求矛盾
- 侵入性太强

---

## 7.3 修改 `small_kernels.cu` / `linear.cu` 本身

当前不推荐。

### 原因

现有证据已经足够说明问题出在构建工具链，而不是源代码逻辑本身。除非在修复编译器问题之后，仍然出现新的真实编译错误，否则不应优先动 `.cu` 源码。

---

## 8. 推荐实施顺序

### 第一步：先补环境

先确认机器上真正存在以下编译器中的至少一组：

- `g++-11` 或 `g++-12` 或 `g++-13`
- 同时保留较新的 `g++` 供普通 C++ 编译使用

否则即使改了 `csrc/setup.py`，也没有可用的兼容 host compiler 可供选择。

### 第二步：修改 `csrc/setup.py`

目标是让其具备：

- 自动识别兼容 `CUDAHOSTCXX`
- 或自动探测兼容版本 `g++`
- 并显式传给 `nvcc`

### 第三步：更新 `README.md`

补齐 `csrc` 的安装说明与双编译器解释。

### 第四步：重新验证

重新执行：

```bash
pip install -e csrc --no-build-isolation
```

重点观察：

- `small_kernels.cu` / `linear.cu` 是否仍然走到了不兼容的 host compiler
- `block_swapping.cpp` / `entrypoints.cpp` 是否继续正常编译

---

## 9. 关键文件清单

本次调研中最关键的文件如下：

- `/home/yxlin/github/swift/NEO/csrc/setup.py`
  - `swiftllm_c` 的构建入口
  - 问题核心：没有显式指定 NVCC host compiler

- `/home/yxlin/github/swift/NEO/README.md`
  - 已经写明需要两个版本的 `g++`
  - 但对 `csrc` 的具体安装路径说明还不够落地

- `/home/yxlin/github/swift/NEO/pacpu/build.sh`
  - 已经实现 split-compiler 策略
  - 是 `csrc` 修复的最佳参考

- `/home/yxlin/github/swift/NEO/csrc/src/small_kernels.cu`
  - 当前报错触发点之一
  - 适合作为修复后的验证目标

- `/home/yxlin/github/swift/NEO/csrc/src/linear.cu`
  - 当前报错触发点之一
  - 适合作为修复后的验证目标

---

## 10. 最终建议

### 总结一句话

这次问题的本质是：

> `csrc` 在编译 `.cu` 文件时走了 `nvcc`，而当前 `nvcc` 默认使用了过新的 GCC 作为 host compiler；`.cpp` 文件不经过这条检查路径，所以不会报错。

### 最合理的解决方向

不是去改 `.cu` 源码，而是：

1. 准备仓库要求的双 GCC 环境
2. 让 `csrc/setup.py` 显式使用兼容的 NVCC host compiler
3. 同步把这一要求写清楚到 README 中

### 当前额外发现

当前机器 PATH 中并没有发现：

- `g++-11`
- `g++-12`
- `g++-13`
- `gcc-11`
- `gcc-12`
- `gcc-13`

这说明本机目前并不满足仓库 README / `pacpu/build.sh` 预期的双编译器环境。因此在实施修复前，**先补齐编译器版本本身** 很可能是必要前提。

---

## 11. 后续可直接执行的动作建议

如果下一步要真正落地修复，我建议按这个顺序进行：

1. 安装兼容版本的 GCC/G++（至少一个给 NVCC 用）
2. 修改 `csrc/setup.py`，增加 `CUDAHOSTCXX` / `-ccbin` 处理逻辑
3. 更新 `README.md`
4. 重新执行 `pip install -e csrc --no-build-isolation` 验证

这个顺序的优点是：

- 只改与问题直接相关的部分
- 改动面最小
- 和仓库现有 `pacpu/build.sh` 设计保持一致
- 最容易得到稳定、可复现的结果

---

## 12. 本次实际修复与验证结果（追加）

### 12.1 已完成的代码修改

我已经修改了 `/home/yxlin/github/swift/NEO/csrc/setup.py`，把原本完全依赖系统默认编译器的行为改成了“标准环境变量优先，自定义变量回退”的模式。

当前实现逻辑是：

1. 优先读取标准环境变量：
   - `CC`
   - `CXX`
   - `CUDAHOSTCXX`
2. 如果标准变量未设置，则回退到你在 `~/.bashrc` 中提供的变量：
   - `CC <- GCC15`
   - `CXX <- GXX15`
   - `CUDAHOSTCXX <- GXX12`
3. 对 `.cu` 编译路径，显式向 `nvcc` 追加：
   - `-ccbin=<CUDAHOSTCXX>`
4. 如果没有解析到可用的 `CUDAHOSTCXX`，则在构建一开始就报出清晰错误，而不是等 `nvcc` 用错默认 GCC 后再失败。

这一步已经把“`nvcc` 错误使用 GCC 15 作为 host compiler”这个原始问题修掉了。

### 12.2 实际验证到的结果

我随后执行了：

```bash
source ~/.bashrc
export CC="${CC:-${GCC15}}"
export CXX="${CXX:-${GXX15}}"
export CUDAHOSTCXX="${CUDAHOSTCXX:-${GXX12}}"
pip install -e csrc --no-build-isolation
```

验证结果表明：

- 原先的 `unsupported GNU version! gcc versions later than 13 are not supported` 报错已经消失
- 构建命令中可以明确看到：
  - `.cu` 文件现在使用的是 `-ccbin=/usr/local/bin/g++`
- `.cpp` 文件仍然由 `/usr/bin/g++` 正常编译

这说明：

- **原始 bug 已经被定位并绕过**：`nvcc` 不再误用系统默认的 GCC 15
- 但构建流程又暴露出了 **第二层环境兼容性问题**

### 12.3 新出现的第一处阻塞错误

在修复 host compiler 选择之后，新的首个失败点变成了：

```text
/usr/include/bits/mathcalls.h: error: exception specification is incompatible with that of previous function "cospi" / "sinpi" / "rsqrt"
/usr/local/include/c++/12.1.0/ext/concurrence.h: error: too many initializer values
/usr/local/include/c++/12.1.0/bits/std_mutex.h: error: too many initializer values
```

这组错误说明当前问题已经不再是“GCC 版本过高”，而是进入了下一层：

1. **CUDA 12.4 与 Fedora 43 / glibc 2.42 的头文件兼容问题**
   - `cospi` / `sinpi` / `rsqrt` 系列声明同时出现在：
     - glibc 的 `/usr/include/bits/mathcalls.h`
     - CUDA 的 `/usr/local/cuda-12.4/include/crt/math_functions.h`
   - 两边声明的 exception specification 不一致，于是 `nvcc` 直接报错

2. **自编译 `/usr/local/bin/g++`(12.1.0) 与当前系统 glibc / pthread 头的混搭问题**
   - `ext/concurrence.h` 和 `std_mutex.h` 中的 `too many initializer values` 很像是：
     - `/usr/local/include/c++/12.1.0/...`
     - Fedora 43 的系统头文件
   - 之间存在结构定义或初始化器布局不匹配

也就是说，现在的阻塞已经从“编译器版本选择错误”切换成了“CUDA 12.4 所在工具链与当前 Fedora 43 用户态头文件不兼容”。

### 12.4 对当前环境的最终判断

截至这一步，可以确认两件事：

#### 已解决

- `csrc/setup.py` 未显式指定 NVCC host compiler 的问题
- `nvcc` 错误使用 GCC 15 导致的版本检查失败

#### 尚未解决

- CUDA 12.4 与 Fedora 43 / glibc 2.42 的头文件冲突
- `/usr/local/bin/g++` 12.1.0 与当前系统头文件/线程库头的兼容性问题

因此，**当前仓库内可修的第一层 bug 已经修到位，但构建仍被更底层的系统工具链兼容问题卡住**。

### 12.5 当前最合理的下一步建议

基于本次实测，后续建议按优先级这样处理：

1. **保留当前 `csrc/setup.py` 的 host compiler 修复**
   - 因为它确实修掉了原始问题
   - 即使后续要更换环境，这部分逻辑仍然是合理的

2. **不要把 `/usr/local/bin/g++` 12.1.0 当作最终长期方案**
   - 它虽然能绕过 GCC>13 检查
   - 但会引入和当前 Fedora 43 系统头/库的兼容问题

3. **优先考虑在受 CUDA 12.4 支持的环境中构建 `csrc`**
   - 例如：
     - 更旧、与 CUDA 12.4 官方支持矩阵一致的 Linux 发行版环境
     - 或 NVIDIA 官方 CUDA 12.4 devel 容器
   - 这是当前最稳妥、最小风险的办法

4. **不建议把正式修复建立在下面这些手段上**
   - `--allow-unsupported-compiler`
   - 直接修改 `.cu` 源码
   - 依赖脆弱的宏去压掉 glibc / CUDA 头文件冲突
   - 直接修改 `/usr/local/cuda-12.4/include` 下的系统 CUDA 头文件

### 12.6 这一轮修复的实际结论

这一轮工作已经得到一个非常明确的结论：

> 原始 bug 的确是 `csrc/setup.py` 没有为 `nvcc` 正确指定 host compiler；这个问题已经通过新增 `CUDAHOSTCXX` / `-ccbin` 逻辑得到修复。当前残留失败不再是同一个 bug，而是 CUDA 12.4 与 Fedora 43/glibc 2.42 以及 `/usr/local` GCC 12.1 工具链组合之间的更深层兼容性问题。

因此，如果后续目标是“真正把 `pip install -e csrc` 编译成功”，下一步的重点已经不在仓库源码，而在**切换到更受支持的 CUDA 构建环境**。
