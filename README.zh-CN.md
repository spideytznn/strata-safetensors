# strata-safetensors

已合入上游 v0.1.43.1，`STRATA_ROUTE_TAIL_SKIP` **默认关闭**，保留所有路由专家的贡献。
合并取舍与验证范围见[集成说明](docs/UPSTREAM_V0143.md)。

**在单张 GPU 和系统内存上，直接运行原始 NVIDIA NVFP4 safetensors 格式的 Qwen3.8-Flash-Next。**

[English](README.md) · [实现与验收](docs/SAFETENSORS_NATIVE.md) · [测速记录](bench/results/2026-10-10-safetensors-runtime/p16-speed-tuning/README.md) · [上游 Strata](https://github.com/Niko1221/Strata)

本项目把 Strata 已有的前向计算、专家调度与会话缓存接到原始 Hugging Face
模型目录。保留专家的 NVFP4 编码和缩放参数，将专家放入锁定内存，热专家缓存在显存，
ngram 表按需从 SSD 读取。主模型不需要先转换成 GGUF，也不生成必须持久保存的专家包。

当前实际验证的平台是 **Windows + RTX 5090 32 GB + Ryzen 9950X3D + 96 GB RAM**。
维护稳定线为 `main`，引擎程序和命令仍叫 `strata`。

`main` 维护稳定版本：已验证单路或双路文本请求、热专家晋升、INT8 KV 和视觉，
稳定回退配置不加载 MTP；本地单路后来已重新启用，双路也已根据同日对照启用共享原生 MTP，见下方快照。
后续类 vLLM 调度实验已回退，不宣称它们带来热态解码提速。
见[稳定版验收记录](bench/results/2026-10-11-two-slot-rollback/README.md)、
[主分支验收要求](docs/STABLE_NATIVE.md)与
[自动 CPU 回归检查](https://github.com/spideytznn/strata-safetensors/actions/workflows/native-stability.yml)。

相对官方的差异、实验状态与本机入口统一见[功能清单](docs/FORK_FEATURES.md)。
本次清理位于本地 `cleanup/desktop-native-20261012`，不表示已发布至远端 `main`。

## 当前能力

| 能力 | 实现范围 |
|---|---|
| 原始权重 | 直接读取 NVIDIA NVFP4 safetensors、BF16 dense 权重、FP8 ngram 数据和原生 MTP 权重。专家仅做布局适配，保留编码与缩放参数。 |
| 专家常驻 | 全部主模型专家保留锁定 RAM 副本，包括已进入 GPU 缓存的专家。加载结束后封闭专家与已加载 MTP 权重的源文件读取。 |
| CPU/GPU 协同 | 热专家在 GPU 计算；冷专家可用 CPU AVX-512 计算，或传到 GPU。自适应交换更新热专家集合。 |
| Prefill | 专用工作区、W4A8 专家批量计算、BF16 dense 投影及激活余项补偿；优化后的桌面配置先分配工作区，再确定专家缓存大小。 |
| 上下文 | 支持 FP16 / INT8 KV；桌面使用 INT8 和按需增长。已验证 262,000-token 输入及 262,144-token 总窗口的容量与状态复用。 |
| MTP | 直接使用模型原生草稿权重，由主模型验证；可调整草稿长度和置信度门槛，也可完整卸载 MTP。 |
| 多会话缓存 | 前缀复用、会话驻留与切回、A/B/A 切换，以及 KV 和线性注意力状态的磁盘保存/恢复。 |
| 视觉 | 通过已有外部 BF16 编码器接入图片；视觉侧车仍使用 mmproj / 词表 GGUF，主模型保持 safetensors。 |
| 服务 | 网页聊天、OpenAI / Anthropic 兼容 API、流式输出、工具调用。可选两路并发，支持带图历史和共享原生 MTP；超额请求排队。 |

这是针对该模型结构的后端，尚不是通用 safetensors 推理框架。

## 为什么速度提升了

- **减少搬运。** 整个专家内存区通过 CUDA 锁页并映射给 GPU，省去额外的主机 staging 复制。
  推理阶段按需读取 ngram 行；主动保存/恢复会话时另有会话文件读写。
- **使用 CPU 和 GPU。** 双路桌面配置约 75% 冷专家在 CPU 计算，25% 在 GPU 计算；单路启动时自动选择比例。
  比例只针对缓存未命中的专家。CPU 行循环展开、GPU 同专家多 token 权重复用减少重复工作。
- **保证 prefill 空间。** 先实际分配并写入工作区，再分配专家缓存，同时计算 MTP 等后续分配的预算。
  启动容量不足会报错，不再静默缩小工作区。
- **减少运行时等待。** 专家常驻后关闭启动阶段的缓冲权重句柄；Windows 上的配对测量显示明显收益，
  但具体缓存或驱动机制尚未确定。
- **复用会话。** 同时保存 KV 和线性注意力状态，续聊时处理新增后缀，减少完整历史重算。

当前保留路径使用 NVFP4 权重和 FP32 decode 激活、W4A8 专家 prefill、原始 BF16 dense 权重。
桌面没有使用的 FP4 MMA、Q8 MTP 投影等实验实现已移除，见[桌面维护范围](docs/DESKTOP_SCOPE.md)。
保留权重编码不代表整条前向逐位等同未量化 BF16 模型。

## 有条件的实测结果

以下均为 **2026-10-10，RTX 5090 32 GB / 9950X3D / 96 GB RAM，Windows，
原始 NVIDIA 权重**的历史对照。各行是不同阶段的实验，提升比例不能相乘。

| 实验 | 之前 → 优化后 | 条件 |
|---|---|---|
| 关闭启动权重句柄，prefill | 822 → 2465 tok/s | 同一程序，约 8K 输入，三组交替配对，128-token 完整输出一致。 |
| CPU/GPU 与内核组合，decode | 45.28 → 81.59 tok/s | 三组交替配对，两边都用两枚草稿，128-token 完整输出一致；候选范围 64.57–85.16。 |
| 后期 4096 行配置，文本 prefill | 2968 → 3331 tok/s | 约 24K 文本，视觉编码器已加载，三次启动中位数；首个约 8K 文本请求反而从 2688 降到 2558。 |
| 后期配置，图片请求 prefill | 2741 → 3089 tok/s | 约 8K 输入含 1024 个图片 token，三次启动中位数；总耗时从 4.51 降到 3.94 秒。 |

另一次英文、中文、代码长回答筛查测得 warm aggregate decode **91.93 → 100.82 tok/s**。
每套配置只启动一台引擎，不能当成多轮稳定吞吐估计；该候选使用四枚草稿、0.7 草稿门槛及两枚强制草稿。

**2026-10-12 本地配置文件：单路启用 native MTP / spec 3 / min-p 0.5，温度 1.0；
双路启用共享 native MTP / spec 3 / min-p 0.5，温度 0.7，并发每路最多一枚草稿。网页共享设置和请求参数可以覆盖采样默认值。**
上述成绩对应历史 MTP 配置，不是当前配置的重新测速，也不是所有任务的速度承诺。
回退后的最新验收检查正确性和实际热专家晋升，不构成新的吞吐提升证据。

完整命令、全部样本和验收范围见
[P10](bench/results/2026-10-10-safetensors-runtime/p10-native-performance/README.md)、
[P16](bench/results/2026-10-10-safetensors-runtime/p16-speed-tuning/README.md)。

## 构建与运行

新机器请使用 [英文 README 的源码构建步骤](README.md#build-and-run-on-windows)：
需要 Git、Python 3.12+、Visual Studio 2022 C++ Build Tools、CUDA 13、
CMake 3.24+ 和 Ninja。原始模型下载到仓库外，使用配置生成器填入自己的路径。

生成器选择 W4A8 专家 prefill，不能直接复现全部桌面优化参数；采样参数需在生成配置的
`sampling` 字段中明确设置。视觉还需要另行配置编码器资源。
专家内存区本身占 **63.282 GiB**，其他权重、会话、运行缓冲和操作系统还需要额外内存。
96 GB 是已验证配置；尚未验证 64 GB 下的完整原生常驻方案。

现有桌面部署使用：

```powershell
.\START-NATIVE-262K.bat
```

该脚本读取 [桌面配置](config/native/rtx5090-262k-fast-v0142.json)。
配置和本地构建辅助脚本含部署机器的绝对路径，其他机器必须适配。
服务在前台运行，Ctrl+C 停止。网页地址为 **http://127.0.0.1:8880**，
OpenAI 客户端地址为 **http://127.0.0.1:8880/v1**。
可用 `/health`、`/v1/models` 检查就绪状态和模型 ID。
对外监听前需配置 `--api-key`。

| 参数 | 2026-10-12 本地桌面配置文件 |
|---|---|
| 总上下文 / 专用 prefill | 262144 token / 4096 行 |
| KV / 视觉 | INT8 按需增长 / 外部 GPU 编码器，每张图最多 1024 token |
| 显存预留 | 单路 700 MiB / 双路 1536 MiB；启动后仍检查最终显存余量 |
| MTP | 单路 native / spec 3 / min-p 0.5；双路共享 native MTP / spec 3 / min-p 0.5，并发每路最多一枚草稿；suffix drafting 均关闭 |
| 采样 | 文件温度：单路 **1.0**、双路 **0.7**；top_p 0.95、top_k 20、min_p 0、presence_penalty 0、repetition_penalty 1 |
| 模板 | Froggeric v22.5，默认开启思考；请求未指定强度时默认 medium |
| 服务 | 127.0.0.1:8880，HTTP Host 头不设白名单 |
| 输出预算 | `fit_max_tokens: true`，把请求输出上限缩至剩余窗口，输入不截断 |

温度 0.7 是用户选择的试用设置。[Qwen 官方思考模式](https://huggingface.co/Qwen/Qwen3.8-Flash-Next#best-practices)推荐温度 **1.0**，
其模板默认思考强度 **xhigh**。客户端显式参数及网页持久化设置可以覆盖服务默认值。
输出预算适配不改变 ZCode 等客户端自己的压缩策略，客户端上下文也应设为 262144。

只维护当前单路与双路两份桌面配置；旧对照配置和额外启动入口已移除。

`START-NATIVE-262K-PARALLEL.bat` 选择可选的**两路并发**，仍为 262K 上下文、
4096 行专用 prefill。每路保留独立图片位置与带图缓存，共享一份原始 MTP 权重、各自保留草稿 KV。
每路最多一枚草稿，低置信度时该路退回单行目标验证，主模型仍验证输出。
本机并发配置还启用了按会话长度独立伸缩的 KV 和显存内槽位迁移。
内存开销、验收及首字/吞吐测量见[原生并发说明](docs/NATIVE_CONCURRENCY.md)。原桌面入口不变。
单路本地配置指向 `build-native-engine/strata-desktop-cleanup.exe`，双路指向
`build-native-engine/strata-native-batch-mtp-final.exe`。
构建此程序名时，在英文 CMake 配置命令中加上
`-DSTRATA_EXECUTABLE_NAME=strata-desktop-cleanup`；通用配置示例则使用 `strata.exe`。
原生双路 MTP 复用官方 slot verifier；异步草稿是显式选项，默认不开。官方 GGUF batch MTP 能力保留。
Git clone 不包含本机预编译程序或视觉资源，其他机器需要自行构建并适配路径。

2026-10-12 同日对照（5090 / 9950X3D / 96GB、温度 0.7、思考模式、每路 256 token）：
预热后的双路合计吞吐由约 81–82 提高到 90–93 tok/s，三组提升约 11%–14%，
各模式输出逐 token 一致。冷文档首次请求包含 prefill 的总吞吐基本持平。
共享 MTP 使热专家显存缓存减少约 2.7 GiB，专家仍全部常驻锁定 RAM。
完整条件与验收见[本日记录](bench/results/2026-10-12-native-batch-mtp/README.md)。

## 验收与边界

- 稳定回退版在本机通过 54 项服务测试和 7 类实机检查，记录到 98 轮双路热专家晋升、
  9370 次晋升，专家与 MTP 常驻后的源读取为零。GitHub 自动检查在 Windows/Linux
  执行服务和独立读取接口测试；不运行 CUDA 推理，也不代表全面质量认证。
  详见[稳定版记录](bench/results/2026-10-11-two-slot-rollback/README.md)。
- 已记录权重布局往返、真实专家 FP64 对照、完整 logits 比较，以及采样、EOS、
  输出上限、取消与恢复检查。
- efficiency17 最终验收覆盖 80 个完成请求和 4 次取消，跨零/两/四/六枚草稿，
  检查对应输出和主状态、切换会话、磁盘恢复。这些是有限回归测试，不是全面质量基准。
- 262K 文档测试证明容量及测试范围内的缓存正确性，不证明通用长上下文推理、
  检索质量或 262K 吞吐。
- 原生路径可通过 `"parallel": 2` 同时解码两路文本请求，prompt 读取仍共用一个入口。
  暂不支持多 GPU、并发槽位 MTP、流水线并行及外部草稿模型；图片请求需要独占生成。
- 本地验证了 CUDA SM120；Linux、HIP、SYCL 和其他硬件尚未验证原生后端。
- 主模型无需 GGUF；视觉仍需 GGUF 侧车。尚未实现原生视觉塔或通用原生一键安装器。

## 文档、来源与许可证

[原生实现详情](docs/SAFETENSORS_NATIVE.md) ·
[历史 NVFP4 / GGUF 文档](README.legacy-nvfp4.md) ·
[原始 Strata README](README.upstream.md)

基于 [Niko1221/Strata](https://github.com/Niko1221/Strata) 和
[sergqwer/strata-nvfp4](https://github.com/sergqwer/strata-nvfp4)，
起点为 NVFP4 fidelity 提交 `d167eb89301a02e0d6299f49d35494ba5096bc10`，
保留原作者署名与许可证。原个人 GGUF 主分支保存于 `archive/gguf-main-20261011`。

截至 2026-10-11，[上游项目提案](https://github.com/Niko1221/Strata/issues/1858)
与 [独立 safetensors 读取接口 PR](https://github.com/Niko1221/Strata/pull/1859)
均为开放状态，完整原生推理尚未合并进上游。

引擎源码使用 [MIT 许可证](LICENSE)；模型与视觉资源遵循各自许可证，仓库不分发权重。
