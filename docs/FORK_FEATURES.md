# 当前分支与官方的维护边界

核对日期：2026-10-14。官方基线为 v0.1.43.1
`a0116b2782a5b82622358c04a06496485d12b2d6`。
维护分支为 `main`；当前仅保留新版 prefill / BF16 KV 单路与双路桌面入口。
已合并上游 0.1.44.1，保留原始权重路径，尾部专家跳过关闭；见 [升级说明](UPSTREAM_0144.md)。
此次以桌面实际使用的两个入口为范围，清理未使用的分支扩展；
官方原有功能继续保留。具体删除与验收见 [桌面范围整理](DESKTOP_SCOPE.md)。

## 保留的差异

| 范围 | 原生适配 | 主要入口 |
|---|---|---|
| 权重与布局 | 直接加载原始 NVFP4 safetensors；校验张量类型、范围和布局，保留专家编码与缩放、BF16 dense 权重。 | [模型绑定](../src/core/safetensors_model.cpp)、[读取器](../src/weights/safetensors.cpp) |
| 常驻与读取保护 | 全部专家保留锁定 RAM 副本，GPU 热专家是额外驻留；批量加载、映射、关闭启动句柄、封闭驻留后专家/MTP 源读取。 | [加载](../src/core/safetensors_model.cpp)、[源接口](../include/strata/weights/safetensors.hpp) |
| 原始专家计算 | NVFP4 的 FP32 激活 CPU/AVX-512 与 GPU 路径、行展开和权重复用；接入既有 CPU/GPU 调度。 | [CPU](../src/kernels/cpu/native_expert.cpp)、[GPU](../src/kernels/cuda/native_mmvq.cu) |
| Prefill | W4A4/CUTLASS auto 专家批计算、BF16 舍入的 dense prefill；8192 行专用工作区优先分配。 | [prefill](../src/prefill/prefill.cpp)、[MoE MMQ](../src/prefill/moe_mmq.cu) |
| Ngram | 原始 FP8 ngram、分段张量地址和边界检查，按需读取。 | [读取器](../src/ngram/ple_reader.cpp) |
| MTP | 加载原始 BF16/F32 draft 张量和 FP8 专家，使用完整原始输出头；复用官方验证、回滚和提交。双路共享权重、独立草稿 KV，每路最多一枚草稿。 | [加载](../src/core/safetensors_mtp.cpp)、[draft](../src/core/mtp.cpp) |
| 缓存与两路 | 原生 KV/GDN/PLE 会话状态、A/B/A 和磁盘恢复；逐请求采样、两路热专家晋升、独立伸缩 KV 与设备槽复制；单 GPU target-only 或 batch MTP 支持带图历史双路与图片身份匹配。 | [快照](../src/core/conversation_snapshot.cpp)、[并发说明](NATIVE_CONCURRENCY.md) |
| Tokenizer 与服务 | HF tokenizer、模板与 prompt 前缀复用；reasoning 历史保留；接入官方视觉，按引擎能力保留旧图片独占门；监控按请求记录读取进度。 | [tokenizer](../tools/safetensors_tokenizer.py)、[服务](../serve/server.py) |
| 构建与回归 | 原生构建、两个桌面配置，以及布局、读取保护、输出/主状态、缓存、取消、并发和视觉验收。 | [构建](../tools/build_safetensors_engine.ps1)、[验收](../tools/native_runtime_acceptance.py) |

这些是原生适配与具体扩展，并不表示 CPU/GPU 协同、热专家晋升、MTP、INT8 KV、
会话缓存、视觉、网页或兼容 API 是本项目独创。它们来自官方和继承分支。
v0.1.43 的非有限值保护、请求计时修复、交换重叠及后端修复继续保留。
v0.1.43.1 的槽位拒绝重试、工具结束符解析、取消图片准备、会话停车进度、
交换信号和 prefill 借用修复已融合，见 [升级记录](UPSTREAM_V01431.md)。

## 当前本机入口

文件默认值不等于网页持久化设置或请求参数；后两者可以覆盖采样值。

| 桌面入口 | 配置 | MTP | 显存预留 | 文件温度 |
|---|---|---|---|---|
| `Start-Strata-Safetensors.bat` | [single](../config/native/rtx5090-262k-prefill-tc-bf16.json) | native / spec 3 / min-p 0.5 | 1536 MiB | 0.7 |
| `Start-Strata-Safetensors-Parallel.bat` | [parallel](../config/native/rtx5090-262k-prefill-tc-bf16-parallel.json) | native / batch-mtp / spec 3 / min-p 0.5 | 1536 MiB | 0.7 |

均使用 `strata-classic-bf16-kv-slotcommit.exe`，262144 总窗口、8192 prefill、BF16 KV Tensor Core、
GPU 视觉、会话缓存、Host 头不限及 `127.0.0.1:8880`。
`STRATA_ROUTE_TAIL_SKIP=7` 是用户保留的桌面调度设置；通用默认仍为 0。
PCIe share 为 0.25，保留交换重叠、连续 prefill 和会话分支保护。
当前配置未启用试验原生 W4A4 decode 或 MTP 投影；prefill 与 KV 是独立选择。

## 已移除与未验证

- 当前重新保留仅供 prefill 的 W4A4/CUTLASS 实现；闲置 W4A4 decode、FP8 KV、
  Q8 MTP 投影、精简 draft 词表和旧试验配置不在当前维护路径。
  原生双路 MTP 后按用户要求重新适配官方 slot verifier：共享权重、独立 KV 与图片位置，
  每路最多一枚草稿，支持逐请求置信度截断及可选异步草稿提交；
  桌面双路根据本日温度 0.7 / 思考模式对照启用共享 MTP，异步草稿默认不开。
- 移除本分支旧 NVFP4 GGUF 转换、重数量化、便携包发布/更新代码及闲置配置/入口。
  官方安装器、打包工具及对应测试/词表恢复官方版本。
- 连续批处理、共享前缀页、ragged prefill 和统一调度实验此前已回退，此次未重新加入。
- 未承诺通用模型、原生 HIP/SYCL/Linux/多 GPU、vLLM 同级吞吐、全面质量一致或 MTP 必然加速。
  原始编码保留不等于逐位等同未量化 BF16；有限回归也不等于广泛质量评估。

来源：[Niko1221/Strata](https://github.com/Niko1221/Strata)、
[sergqwer/strata-nvfp4](https://github.com/sergqwer/strata-nvfp4)。
原始代码起点为 `d167eb89301a02e0d6299f49d35494ba5096bc10`；保留作者与许可证。
