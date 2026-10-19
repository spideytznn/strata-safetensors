# 原始 NVFP4 部署：2026-10-20

按用户决定，保留上游 v0.2.0-beta 融合和本地引擎改造，现役权重切回
`G:/moemodels/Qwen3.8-flash-next-nvfp4`。没有证据证明 FP4-Local 质量更好。
两条 FP4-Local 桌面入口退役，配置移入 `config/native/`；实验代码、测试和模型文件保留。

## 当前入口

- 桌面 `Start-Strata-NVFP4-Parallel.bat` → 仓库 `START-NVFP4-PARALLEL.bat`。
- 唯一现役配置：[`strata-nvfp4-native.json`](../strata-nvfp4-native.json)。
- 原网页/API：localhost:8880，模型名 `Qwen3.8-Flash-Next-NVFP4-Native`。
- 新界面：`START-APP-SAFETENSORS.bat`，localhost:8891，模型页选择 NVFP4 Native。
  它加载模型时另分配受密钥保护的 API 端口；一次只启动一个模型服务。

引擎为 `build-native-engine/strata-safetensors-v020beta.exe`，SHA256：
`8ddc64bd35b38499f99235ac82434a2a8b2b2cf0c6170f86116ef2ebf51f97de`。
旧版 NVFP4 指原 checkpoint，不回退已升级的引擎。

配置由保留的 `config/native/rtx5090-262k-prefill-tc-bf16-parallel.json` 恢复：
双路、262144 配置窗口、8192 prefill、BF16 KV、FP32 激活专家 decode、
W4A4/CUTLASS prefill、native MTP / spec 3 / min-p 0.5、`--batch-mtp`、
PCIe fraction 0.25、1536 MiB 显存预留、GPU BF16 视觉、
12288 MiB 会话预算和不限固定停放槽数。并发 MTP 每路最多一枚草稿。
尾部专家跳过继续关闭。保留原采样参数与 Froggeric v22.5 模板。

本机退役入口及切换前配置备份：
`G:/Strata/Strata-data/cleanup-records/nvfp4-return-20261020/`。
仓库历史配置保留原二进制路径，供开发回溯；现役配置单独指向新版程序。

## 验证范围

平台：Windows、RTX 5090 32 GB、Ryzen 9950X3D、96 GB RAM。
本次结果记录于忽略目录 `work/nvfp4-return-20261020/`。
可随仓库检查的摘要见 [summary.json](../bench/results/2026-10-20-nvfp4-v020beta/summary.json)。

- 应用相关 28 项检查通过；推荐选择修改后另重跑对应 4 项，通过。
- Python checkpoint/doctor/MCP 检查 43 项通过，现役配置 doctor 检查通过。
- 实际启动新版应用，NVFP4 Native 就绪并被推荐；FP4-Local 不再列出，
  推荐项不继承 GGUF 安装命令或下载量。
- 原始 NVFP4 的 10 类双路检查通过：贪心、固定种子采样、惩罚、混合/严格 MTP 置信度、
  取消恢复、续聊、输出上限、EOS，以及 9043-token 长输入旁的短请求。
  该组使用 16384 上下文、8192 prefill、GPU 视觉、native batch MTP、独立 KV 和设备槽复制；
  记录 51 轮多槽自适应晋升、4018 次晋升，逐请求输出与单路参考一致。
- 图片检查沿用桌面 262144 配置窗口和 8192 prefill、GPU BF16 编码器、双路 native MTP，
  22 个请求、6 组并发组合通过：图片/图片、图片/文本、固定种子采样、
  带图历史续聊、换图隔离、取消恢复及输出一致性。
- 两组真实模型检查均以退出码 0 关闭私有引擎；专家与 MTP 常驻后的源文件读取均为 0。
  测试模型、视觉进程和源应用服务已关闭。

此前上游融合的构建与通用检查见 [升级记录](UPSTREAM_020_BETA.md)，
其中 FP4-Local 实测结果只属于当时模型，不作为 NVFP4 的验证证据。

有限的生成、会话和并发回归不能证明全面任务质量，也不证明 MTP 必然更快。
262144 是部署配置窗口，本次功能检查不构成 262K 检索或吞吐评测。
