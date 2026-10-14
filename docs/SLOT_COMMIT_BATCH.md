# 多路 MTP 状态提交：候选实现

`STRATA_SLOT_COMMIT_BATCH=1` 是 CUDA 的显式选择，默认关闭。
本次只改变提交阶段的调用安排：每个 slot 的多个层合并计算，slot 之间仍使用独立的
接受长度、raw 行偏移、状态地址、tail 快照、容量和图片 M-RoPE 表。
没有修改 prefill、专家计算、注意力精度、草稿采样或会话缓存策略。

实现复用上游 `gdn_commit_layers` 和 `native_qsa_indexer_append_steps_layers`，
没有新增 CUDA 算术内核。GDN 一行验证窗口保留原内核；两行及以上在符合上游
state-only 条件时合并层。调试/split 模式等不符合条件时回到原来的 GDN 调用。
QSA 层数或容量不满足层合并条件、或关闭 batch KV 时，保留原 QSA 提交。
原有 canonical spare 清理和 PLE 历史提交仍独立执行。

独立候选程序：`build-native-engine/strata-classic-bf16-kv-slotcommit.exe`。
当前桌面配置没有启用该开关，也没有切换到候选程序。

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

2026-10-14：候选 CUDA 构建已完成；按用户要求，GPU 正确性、memcheck 和
配对测速等待当前测试结束后进行。当前尚无 GPU 验证结果或速度结论。
HIP/SYCL/Metal/Linux 未构建验证；这些后端不启用本次 CUDA 分支。
