# 桌面入口范围整理：2026-10-12

本地分支：`cleanup/desktop-native-20261012`。清理前代码：
`c6f52f10a1107e28aa223cee33a16cddb17bb750`。
目标是保留当前桌面确实使用的原生路径，删除闲置分支扩展，缩小与官方 v0.1.43 的差异。
当前功能和配置见 [FORK_FEATURES](FORK_FEATURES.md)。清理完成后，按用户要求整理并同步到远端主分支。

## 保留

- 桌面单路 → `START-NATIVE-262K.bat` → `rtx5090-262k-fast-v0142.json`，原生 MTP / spec 3 / min-p 0.5。
- 桌面双路 → `START-NATIVE-262K-PARALLEL.bat` → `rtx5090-262k-parallel.json`，不加载 MTP。
- 原始权重读取与布局、锁定 RAM/热专家、CPU/GPU 计算、W4A8 prefill、BF16 dense、INT8 KV、视觉、会话缓存和双路晋升。
- 两份配置的采样、Host、上下文、显存预留、交换重叠等现有用户设置，以及网页持久化 sidecar。
- 上游已有能力和必要的构建、正确性与回归工具；不是把所有未显式写在启动参数中的函数都删除。

其他仓库的 Q4KL/Q6KL/官方 GGUF 桌面入口、原始模型与外部视觉文件不在清理范围。

## 删除与对齐

共删除 68 个受 Git 跟踪文件；连同本地旧备份/设置文件共 71 个，均有本地归档。12 个文件恢复官方原始字节。

- 闲置 FP4 MMA/W4A4 专家路径及测试，Q8 MTP 投影、原生精简 draft 词表与 BF16 batch draft 实验。
- 原生双路 MTP 专用调度与验收入口；上游 GGUF batch MTP 保留。
- 旧配置、稳定/实验启动入口，仅保留桌面实际引用的两个原生配置。
- 旧 NVFP4 GGUF 转换、重数量化、独立便携包发布/更新代码及专用测试/数据。
- `setup.py`、`iq_pack.py`、相关安装器测试与默认 draft 词表恢复为官方 v0.1.43 原始字节。
  原生桌面使用直接构建/运行脚本，不调用官方安装器更新自身。
- 历史大文档改为当前维护说明与证据索引；历史实测记录保留，不能当作新版本测速。

## 验收

验收完成，两个桌面入口已仅把 `exe` 改为
`build-native-engine/strata-desktop-cleanup.exe`，下次启动使用精简程序。
两份配置的其余字段与网页 sidecar 保持原样；旧 `strata-upstream143.exe` 保留。
清理验收阶段没有启动桌面 HTTP 服务；此后按用户要求整理并推送代码。

- CUDA 构建通过；服务测试 719 项（跳过 12），工具测试 686 项（跳过 11）。
- 读取器 5 组、CPU 10 项、GPU/路由策略 8 项回归通过。
- 清理前后 50 个对应已完成请求的完整 token 与主状态指纹一致，覆盖 MTP 关闭/两步草稿、8K/24K prefill、冷暖缓存、取消续接及磁盘恢复。
- 双路长短会话/KV 增长通过，实际记录 137 轮双路热专家晋升、13107 次晋升。
- 8 个图片/文本请求通过；所有完成验收的运行中专家/MTP 驻留后源读取均为零。

额外严格驻留表审计在清理前程序上遇到重叠交换尚未提交的中间状态：
3 次请求结束审计各有 96 个临时主机/设备表差异，23 次审计的权重内容差异均为零。
该次尝试按失败保存，未计入通过结果；重跑保留交换重叠，检查完整输出、主状态和草稿 KV 恢复。
未改动交换实现，也不把这些检查当作新的吞吐或全面质量证明。
完整结果、程序哈希与限制见 [验收摘要](../bench/results/2026-10-12-desktop-scope/README.md)。

候选使用 Windows、RTX 5090 32GB、Ryzen 9950X3D、96GB RAM、CUDA 13、SM120。
本次没有 HIP/SYCL 构建环境，不宣称这些原生后端已验证。

## 回退与本地资料

安全分支 `backup/pre-desktop-cleanup-20261012` 保留清理前提交。
未提交配置与文档修改、每批删除/修改前的文件保存在忽略目录
`build-native-engine/desktop-scope-backup-20261012/`。历史模型测试日志保留在本机
`bench/results/local-desktop-scope-20261012-*`。这些不是远端发布资料。

## 随后的本机路径与 Q6 部署更新

两份原生桌面配置的模型/tokenizer 路径改为
`G:\moemodels\Qwen3.8-flash-next-nvfp4`，原索引引用的 11 个文件均存在。
共享视觉词表从已移除的 Q4KL pack 复制到
`G:\Strata\Strata-data\vision\qwen3.8-flash-next-vision-vocab.gguf`；
SHA256 为 `b07908c004632d72aff88e5eb6fa5462b846426c7ee6ccf7199c2712956a3bd8`，
复制前后相同。以上仅更新路径，未重新解释清理阶段的输出一致性证据。

Q4KL 桌面 Fusion/Official 两份 BAT 和对应两份 PowerShell 入口已按用户要求删除。
外部 Q6 GGUF 桌面部署仍保留，模型改为 `G:\moemodels\Qwen3.8-flash-next-gguf`。
其锁页分配故障处理与部署恢复快照见
[本机 Q6 部署](../deployment/local-q6/README.md)，不属于原生引擎的零专家源读取承诺。
