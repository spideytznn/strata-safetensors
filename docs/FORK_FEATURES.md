# 当前分支与官方的维护边界

核对日期：2026-10-12。官方基线为 v0.1.43
`7695859aa13afb248bdb4d36f9b2e95707db079d`。
清理工作分支为 `cleanup/desktop-native-20261012`，验收后同步到项目主分支。
此次以桌面实际使用的两个入口为范围，清理未使用的分支扩展；
官方原有功能继续保留。具体删除与验收见 [桌面范围整理](DESKTOP_SCOPE.md)。

## 保留的差异

| 范围 | 原生适配 | 主要入口 |
|---|---|---|
| 权重与布局 | 直接加载原始 NVFP4 safetensors；校验张量类型、范围和布局，保留专家编码与缩放、BF16 dense 权重。 | [模型绑定](../src/core/safetensors_model.cpp)、[读取器](../src/weights/safetensors.cpp) |
| 常驻与读取保护 | 全部专家保留锁定 RAM 副本，GPU 热专家是额外驻留；批量加载、映射、关闭启动句柄、封闭驻留后专家/MTP 源读取。 | [加载](../src/core/safetensors_model.cpp)、[源接口](../include/strata/weights/safetensors.hpp) |
| 原始专家计算 | NVFP4 的 FP32 激活 CPU/AVX-512 与 GPU 路径、行展开和权重复用；接入既有 CPU/GPU 调度。 | [CPU](../src/kernels/cpu/native_expert.cpp)、[GPU](../src/kernels/cuda/native_mmvq.cu) |
| Prefill | W4A8 专家批计算、原始 BF16 dense 路径及激活余项补偿；专用工作区优先分配并预算后续分配。 | [prefill](../src/prefill/prefill.cpp)、[MoE MMQ](../src/prefill/moe_mmq.cu) |
| Ngram | 原始 FP8 ngram、分段张量地址和边界检查，按需读取。 | [读取器](../src/ngram/ple_reader.cpp) |
| 单路 MTP | 加载原始 BF16/F32 draft 张量和 FP8 专家，使用完整原始输出头；复用官方验证、回滚和提交。 | [加载](../src/core/safetensors_mtp.cpp)、[draft](../src/core/mtp.cpp) |
| 缓存与两路 | 原生 KV/GDN/PLE 会话状态、A/B/A 和磁盘恢复；逐请求采样、两路热专家晋升，无 MTP 时独立伸缩 KV 与设备槽复制。 | [快照](../src/core/conversation_snapshot.cpp)、[并发说明](NATIVE_CONCURRENCY.md) |
| Tokenizer 与服务 | HF tokenizer、模板与 prompt 前缀复用；reasoning 历史保留；接入官方视觉，图片独占门。 | [tokenizer](../tools/safetensors_tokenizer.py)、[服务](../serve/server.py) |
| 构建与回归 | 原生构建、两个桌面配置，以及布局、读取保护、输出/主状态、缓存、取消、并发和视觉验收。 | [构建](../tools/build_safetensors_engine.ps1)、[验收](../tools/native_runtime_acceptance.py) |

这些是原生适配与具体扩展，并不表示 CPU/GPU 协同、热专家晋升、MTP、INT8 KV、
会话缓存、视觉、网页或兼容 API 是本项目独创。它们来自官方和继承分支。
v0.1.43 的非有限值保护、请求计时修复、交换重叠及后端修复继续保留。

## 当前本机入口

文件默认值不等于网页持久化设置或请求参数；后两者可以覆盖采样值。

| 桌面入口 | 配置 | MTP | 显存预留 | 文件温度 |
|---|---|---|---|---|
| `Start-Strata-Safetensors.bat` | [fast-v0142](../config/native/rtx5090-262k-fast-v0142.json) | native / spec 3 / min-p 0.5 | 700 MiB | 1.0 |
| `Start-Strata-Safetensors-Parallel.bat` | [parallel](../config/native/rtx5090-262k-parallel.json) | 完全不加载 | 1536 MiB | 0.7 |

二者已切换为验收后的 `strata-desktop-cleanup.exe`，旧程序保留供回退。
二者均保留 262144 总窗口、4096 prefill、INT8 KV、GPU 视觉、会话缓存、
Host 头不限与 `127.0.0.1:8880` 监听。Q4KL/Q6KL/官方 GGUF 桌面入口属于其他仓库。
`--spec 3` 是一行目标加最多两行草稿；未设最少草稿数，可按置信度截断。

`STRATA_ROUTE_TAIL_SKIP=0`。`STRATA_ADAPT_OVERLAP=1`、
`STRATA_ADAPT_OVERLAP_RESIDENT=1`、`STRATA_ADAPT_NOWAIT=0`、
`STRATA_ADAPT_LAG=1` 保持现有设置；两路 batch 交换边界仍同步等待。
没有据此新增吞吐承诺。

## 已移除与未验证

- 移除闲置 FP4 MMA/W4A4 实验、原生双路 MTP、Q8 MTP 投影、原生精简 draft 词表及 BF16 batch draft 实验。
- 移除本分支旧 NVFP4 GGUF 转换、重数量化、便携包发布/更新代码及闲置配置/入口。
  官方安装器、打包工具及对应测试/词表恢复官方版本。
- 连续批处理、共享前缀页、ragged prefill 和统一调度实验此前已回退，此次未重新加入。
- 未承诺通用模型、原生 HIP/SYCL/Linux/多 GPU、vLLM 同级吞吐、全面质量一致或 MTP 必然加速。
  原始编码保留不等于逐位等同未量化 BF16；有限回归也不等于广泛质量评估。

来源：[Niko1221/Strata](https://github.com/Niko1221/Strata)、
[sergqwer/strata-nvfp4](https://github.com/sergqwer/strata-nvfp4)。
原始代码起点为 `d167eb89301a02e0d6299f49d35494ba5096bc10`；保留作者与许可证。
