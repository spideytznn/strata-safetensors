# 双路 MTP 层合并提交：2026-10-14

RTX 5090 32 GiB、Ryzen 9950X3D、96 GB、Windows、CUDA 13.0、MSVC 14.44。
CPU 保持物理核布局，未启用 SMT。全部测试通过后，双路桌面配置启用
`STRATA_SLOT_COMMIT_BATCH=1`；单路不启用，通用默认关闭。
两份桌面配置均指向已测试的 `strata-classic-bf16-kv-slotcommit.exe`。
没有启动 HTTP 服务；测试私有引擎与视觉进程均已退出，下次手动启动生效。

## 正确性

- 672 组 CUDA Graph 重放，GDN、QSA、PLE、保护区与原逐层调用逐位一致。
  覆盖独立接受长度、不同窗口、空闲槽位、每路不同容量、图片位置、RoPE 缩放、
  canonical 清理和异常后的 M-RoPE 状态恢复。
- Compute Sanitizer memcheck：672 组通过，0 错误。
- 同一候选程序切换开关，ABBA 四次独立模型加载。36 个双路 batch、72 条请求、
  共 13824 个生成 token；6 条不同参考输出在全部重复及开关间 token 一致。
- 优化开启时另通过固定种子采样、混合草稿置信度、取消/恢复和 slot 复用检查。
- 图片集通过 22 条请求、6 组真实双路重叠：贪心/固定种子单路对照、历史图片
  续写复用、图片与文本混跑、替换图片隔离、取消后的复用。
- 所有模型测试结束时，expert 与 MTP 的 post-residency source bytes 均为 0。
  缓存与快照检查分别通过 4311 / 4237 项。

这些结果只证明所测状态和请求的一致性；未运行新的长图片上下文扩展集或通用质量评测。
本次没有更改权重精度、KV、prefill、采样、专家调度或缓存策略。
HIP、SYCL、Metal、Linux、多 GPU 未验证；非 CUDA 后端不启用此分支。

## 提交阶段

36 个 GDN 层、12 个 QSA 层，两路各两行，9 组交替配对、每组 10 次图重放。
耗时使用 CUDA events，中位数如下。只测状态提交，不代表整个 decode。

| 图片位置 | Canonical 清理 | 图节点 | 原路径 µs | 合并路径 µs | 耗时减少 |
|---|---|---|---:|---:|---:|
| 无 | 关 | 220→10 | 574.838 | 325.334 | 43.4% |
| 无 | 开 | 244→34 | 595.254 | 346.320 | 41.8% |
| 有 | 关 | 220→10 | 577.882 | 325.971 | 43.6% |
| 有 | 开 | 244→34 | 598.973 | 347.373 | 42.0% |

当前真实桌面启用 canonical 清理，因此完整两行窗口录制为 34 节点。
一行 GDN 仍沿用原内核；混合一/两行窗口不会获得同样的节点缩减。

## 真实双路吞吐

同一可执行文件，顺序 off/on/on/off；各臂从同一专家配置文件的副本开始。
262144 上下文、8192 W4A4/CUTLASS prefill、BF16 KV Tensor Core、native MTP
`spec=3`、`--batch-mtp`、全部路由专家、原调度、视觉常驻。
每种提示词先一组热身，再测两组；每路固定生成 192 token。
表中为两路首 token 都到达后的合计稳态吞吐，4 个测量样本的中位数。

| 提示词 | 关闭 token/s | 开启 token/s | 差异 |
|---|---:|---:|---:|
| 中英文说明文本 | 98.04 | 99.47 | +1.45% |
| 中英文代码 | 104.88 | 105.65 | +0.74% |
| 约 8K 文档 | 100.13 | 100.67 | +0.54% |

自动定容后的热专家槽位依次为 3249、3252、3252、3252；初始排序配置相同，
实际槽数有 3 槽差异。小幅吞吐差异混有计时和缓存波动，不能全部归因于提交优化。
提交占整机窗口的比例较小，当前只能确认提交阶段明显减少开销，整机收益有限。

## 复现与证据

先停止桌面服务，独占模型资源。构建与测试入口：

```powershell
./tools/build_safetensors_engine.ps1 -Jobs 2
./build-native-engine/slot_commit_layers_test.exe
./build-native-engine/slot_commit_layers_test.exe --bench
& 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/compute-sanitizer/compute-sanitizer.exe' --tool memcheck --error-exitcode 99 ./build-native-engine/slot_commit_layers_test.exe
./.venv-native/Scripts/python.exe tools/native_slot_commit_bench.py --config config/native/rtx5090-262k-prefill-tc-bf16-parallel.json --exe build-native-engine/strata-classic-bf16-kv-slotcommit.exe --output build-native-engine/slot-commit-runtime-recheck --rounds 2 --repeats 2 --new 192
./.venv-native/Scripts/python.exe tools/native_vision_concurrency_acceptance.py --config config/native/rtx5090-262k-prefill-tc-bf16-parallel.json --exe build-native-engine/strata-classic-bf16-kv-slotcommit.exe --output build-native-engine/slot-commit-vision-recheck --limit 64 --desktop-context
```

[manifest.json](manifest.json) 保存程序/源码/本地原始记录 SHA256、每组吞吐样本、
生成 token 摘要、实际专家槽数和图片复用计数。
[parity.log](parity.log)、[memcheck.log](memcheck.log)、
[microbenchmark.log](microbenchmark.log) 保存小型 GPU 测试输出。
完整私有模型日志在本机忽略目录 `build-native-engine/slot-commit-runtime-abba/`
和 `build-native-engine/slot-commit-vision-checks/`。
