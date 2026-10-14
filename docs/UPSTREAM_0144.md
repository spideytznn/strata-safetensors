# 上游 0.1.44.1 合并：2026-10-14

合并上游 `296b59d2`（`v0.1.44.1`），保留直接 safetensors 加载和当前桌面路径。
上游原始 NVFP4 说明另存为 [NVFP4_UPSTREAM.md](NVFP4_UPSTREAM.md)；
它的 GGUF/pack 转换、量化权重、MTP 和性能数字不代表本分支的桌面配置。

## 桌面运行选择

- W4A4/CUTLASS auto prefill，8192 行工作区，BF16 KV Tensor Core。
- 原有 FP32 激活专家 decode、CPU AVX-512 和热专家调度。
- 原始 BF16/F32 MTP dense 张量、FP8 MTP 专家、完整原始输出头；`spec=3`。
- 单路及两路 MTP，原有图片槽位隔离、会话分支保护、12 GiB 内存预算和 unlimited 槽位。
- `STRATA_ROUTE_TAIL_SKIP=0`，计算全部路由专家。
- `STRATA_ADAPT_SIGNAL=copy` 显式保留升级前的调度信号选择；同时包含上游 WDDM 提交刷新修复。
- 端口 8880，Host 检查保持关闭，CPU 保持物理核布局。

上游 `Split<40>`、UE4M3 位运算转换和 8-lane down 内核已合入。
UE4M3 位运算转换是共享函数，当前 FP32 激活专家 decode 的 `nvfp4_value`
也调用它，因此会使用这项优化。`Split<40>` 和 8-lane down 属于 W4A8/q8_1
专家路径；当前桌面使用 `STRATA_NVFP4_F32=1`，没有启用这两个分支。
原有 FP32 激活权重复用继续保留。
不能把上游 3060 的专家内核数字当成本机整机 decode 收益。

## MTP 和 arena

MTP 的草稿、验证及回滚机制沿用 Strata；本分支主要区别是保留原始检查点权重、
完整输出头和原生并发适配。合入上游 GDN/QSA 批量状态提交、MTP 计时工具，
保留原生图片 M-RoPE 和可选 canonical rollback 清理。
单路批量提交与两路 slot commit 是不同路径；不能把单路提交优化外推到两路。
本次不以张量精度推断端到端接受率或速度，也未证明与上游 MTP 的质量相同。

上游 `ArenaExpertSource` 把专家集中加载到 RAM，并为搬运注册一部分页。
桌面 safetensors 使用自己的 RAM 专家源：所有专家保留锁定 RAM 副本，
GPU 热专家是额外副本。概念上也是常驻 RAM，但没有使用 `ArenaExpertSource`，
所以不会自动启用它新增的异步 arena tier 或按需 pin 策略。
通用 GGUF arena 路径保留上游功能；按需 pin 开启时等待最终 VRAM 估计后同步打开 arena，
避免与本分支提前加载 arena 的线程冲突。默认的提前加载保留。

## 合并验证范围

CUDA/SM120 构建、服务端 mock-engine 回归、会话缓存/快照和小型算子检查。
独立输出程序，运行中的旧程序不会被覆盖或重启。
具体结果见 `bench/results/upstream-0144-merge-20261014/`。
没有全模型速度、接受率、质量或长上下文配对测量；HIP、SYCL、Metal 和 Linux 未验证。
