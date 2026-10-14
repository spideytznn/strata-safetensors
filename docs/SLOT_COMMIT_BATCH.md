# 多路 MTP 状态提交

`STRATA_SLOT_COMMIT_BATCH=1` 是 CUDA 的显式选择，默认关闭。
本次只改变提交阶段的调用安排：每个 slot 的多个层合并计算，slot 之间仍使用独立的
接受长度、raw 行偏移、状态地址、tail 快照、容量和图片 M-RoPE 表。
没有修改 prefill、专家计算、注意力精度、草稿采样或会话缓存策略。

实现复用上游 `gdn_commit_layers` 和 `native_qsa_indexer_append_steps_layers`，
没有新增 CUDA 算术内核。GDN 一行验证窗口保留原内核；两行及以上在符合上游
state-only 条件时合并层。调试/split 模式等不符合条件时回到原来的 GDN 调用。
QSA 层数或容量不满足层合并条件、或关闭 batch KV 时，保留原 QSA 提交。
原有 canonical spare 清理和 PLE 历史提交仍独立执行。

桌面程序：`build-native-engine/strata-classic-bf16-kv-slotcommit.exe`。
双路桌面配置启用该开关；单路不启用，沿用上游单路提交路径。
通用引擎默认仍关闭。设置为 `0` 可恢复双路原来的逐层提交。

## 验证入口

`slot_commit_layers_test` 使用生产代码中的 slot 合并 helper，与原来的逐层调用录制的
CUDA Graph 对照。覆盖不同接受长度、零接受、不同窗口行数、非连续 slot 编号、
输入行偏移、空闲 slot、带保护区的层间 stride、每路不同容量、图重放控制值变更、
文本/图片位置、RoPE 三种缩放、旋转表开关、canonical 清理和 PLE 历史。
另检查验证抛出异常时恢复 M-RoPE 注册状态。

```powershell
./tools/build_safetensors_engine.ps1 -OutputName strata-classic-bf16-kv-slotcommit -Jobs 2
./build-native-engine/slot_commit_layers_test.exe
./build-native-engine/slot_commit_layers_test.exe --bench
```

`--bench` 使用 36 个 GDN 层、12 个 QSA 层和两路各两行的验证窗口，交替录制和
测量原路径与合并路径，输出图节点数和配对耗时。该数字只代表状态提交阶段，
不能外推为整机 decode 提升。

2026-10-14，RTX 5090 / Ryzen 9950X3D / 96 GB / Windows / CUDA 13.0：
672 组 CUDA Graph 重放逐位对照通过，Compute Sanitizer memcheck 为 0 错误。
两路各两行、36 GDN + 12 QSA 层、开启 canonical 清理时，提交图从 244 节点
降到 34 节点，配对中位耗时从 595.254 µs 降到 346.320 µs（减少 41.8%）。

同一程序开关 ABBA、BF16 KV / 8192 prefill / 原生 MTP 的真实双路测试通过。
文本、代码、约 8K 文档的稳态合计吞吐分别为 98.04→99.47、104.88→105.65、
100.13→100.67 token/s。专家缓存自动定容有 3 槽差异，加上计时波动，不能把
这 0.5–1.5% 差异全部归于提交优化，也不能视为普遍整机提升。
所有开关对照生成 token 一致；另检查 seeded 采样、混合草稿置信度、取消恢复及 slot 复用。
完整测量、图片并发结果与复现命令见
[验证记录](../bench/results/slot-commit-batch-20261014/README.md)。
HIP/SYCL/Metal/Linux 未构建验证；这些后端不启用本次 CUDA 分支。
