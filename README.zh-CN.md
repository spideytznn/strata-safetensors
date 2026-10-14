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

当前桌面使用 W4A4/CUTLASS 新版 prefill、8192 batch、BF16 KV Tensor Core，
保留原专家 decode 与调度。单路和双路都启用 native MTP，双路另启用 batch MTP。
只维护这两条入口，见[桌面维护范围](docs/DESKTOP_SCOPE.md)、
[prefill](docs/NVFP4_PREFILL_ONLY.md)及[BF16 KV](docs/CLASSIC_BF16_KV.md)。
历史测试记录对应各自当时的程序和配置。

## 当前能力

| 能力 | 实现范围 |
|---|---|
| 原始权重 | 直接读取 NVIDIA NVFP4 safetensors、BF16 dense 权重、FP8 ngram 数据和原生 MTP 权重。专家仅做布局适配，保留编码与缩放参数。 |
| 专家常驻 | 全部主模型专家保留锁定 RAM 副本，包括已进入 GPU 缓存的专家。加载结束后封闭专家与已加载 MTP 权重的源文件读取。 |
| CPU/GPU 协同 | 热专家在 GPU 计算；冷专家可用 CPU AVX-512 计算，或传到 GPU。自适应交换更新热专家集合。 |
| Prefill | 8192 行专用工作区，W4A4 FP4 Tensor Core / CUTLASS auto，BF16 舍入的 dense prefill 投影。 |
| 上下文 | 桌面使用 BF16 KV Tensor Core 和按需增长，配置总窗口 262144。历史 262K 容量测试使用 INT8；通用 FP16/INT8 路径保留。 |
| MTP | 直接使用模型原生草稿权重，由主模型验证；可调整草稿长度和置信度门槛，也可完整卸载 MTP。 |
| 多会话缓存 | 前缀复用、会话驻留与切回、A/B/A 切换，以及 KV 和线性注意力状态的磁盘保存/恢复。 |
| 视觉 | 通过已有外部 BF16 编码器接入图片；视觉侧车仍使用 mmproj / 词表 GGUF，主模型保持 safetensors。 |
| 服务 | 网页聊天、OpenAI / Anthropic 兼容 API、流式输出、工具调用。可选两路并发，支持带图历史和共享原生 MTP；超额请求排队。 |

这是针对该模型结构的后端，尚不是通用 safetensors 推理框架。

## 为什么速度提升了

- **减少搬运。** 整个专家内存区通过 CUDA 锁页并映射给 GPU，省去额外的主机 staging 复制。
  推理阶段按需读取 ngram 行；主动保存/恢复会话时另有会话文件读写。
- **使用 CPU 和 GPU。** 两份桌面配置均选择约 75% 冷专家在 CPU 计算，25% 在 GPU 计算。
  比例只针对缓存未命中的专家。CPU 行循环展开、GPU 同专家多 token 权重复用减少重复工作。
- **保证 prefill 空间。** 先实际分配并写入工作区，再分配专家缓存，同时计算 MTP 等后续分配的预算。
  启动容量不足会报错，不再静默缩小工作区。
- **减少运行时等待。** 专家常驻后关闭启动阶段的缓冲权重句柄；Windows 上的配对测量显示明显收益，
  但具体缓存或驱动机制尚未确定。
- **复用会话。** 同时保存 KV 和线性注意力状态，续聊时处理新增后缀，减少完整历史重算。

当前保留原专家 decode 和 MTP 投影，新 prefill 与 BF16 KV 的舍入方式不同于旧版。
保留权重编码不代表整体输出或任务质量完全一致。

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

上述数字对应历史程序和参数，不是当前 BF16/W4A4 部署的重新测速。
完整记录见 [P10](bench/results/2026-10-10-safetensors-runtime/p10-native-performance/README.md)
和 [P16](bench/results/2026-10-10-safetensors-runtime/p16-speed-tuning/README.md)。

## 构建与运行

先按[英文构建说明](README.md#build-and-run-on-windows)安装依赖，再执行：

```powershell
.\tools\build_safetensors_engine.ps1 -Jobs 2
```

只保留两个原生启动入口，均使用 `build-native-engine/strata-classic-bf16-kv-v0144.exe`：

- 单路：`START-NATIVE-262K-PREFILL-TC-BF16.bat`，配置 [single](config/native/rtx5090-262k-prefill-tc-bf16.json)。
- 两路：`START-NATIVE-262K-PREFILL-TC-BF16-PARALLEL.bat`，配置 [parallel](config/native/rtx5090-262k-prefill-tc-bf16-parallel.json)。

两者均为 262144 上下文、8192 prefill、BF16 KV、native MTP / spec 3 / min-p 0.5，
1536 MiB 显存预留、视觉及原专家调度。双路额外启用 `--batch-mtp`，每路最多一枚草稿。
两份文件的温度均为 0.7，top_p 0.95、top_k 20；客户端参数和网页设置可以覆盖默认值。
桌面单路/多路脚本分别调用上述入口。旧试验入口已删除。

配置含本机路径，其他机器需要修改模型、程序、tokenizer、模板及视觉资源路径。
构建脚本自动获取固定版本 CUTLASS 到 `build-deps/cutlass`，不依赖试验仓库。
构建不会覆盖正在运行的同名程序，可用 `-OutputName` 编译候选。

一次启动一个服务。网页地址 **http://127.0.0.1:8880**，API 地址
**http://127.0.0.1:8880/v1**，模型名 `Qwen3.8-Flash-Next-NVFP4-Native`。
Ctrl+C 停止前台服务，`/health`、`/v1/models` 可检查就绪状态。
保持 localhost 监听；对外监听前设置 API key。输入模型目录只读。

专家内存区实测占 **63.282 GiB**，还需其他权重、会话和系统内存；本机验证配置为 96 GB RAM。

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
  单 GPU 支持共享 native batch MTP 和带图会话；多 GPU、流水线并行及外部草稿模型不在当前原生部署范围。
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
