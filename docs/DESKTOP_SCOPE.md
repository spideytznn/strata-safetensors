# 当前桌面维护范围：2026-10-14

桌面只保留两条 safetensors 启动链：

| 桌面入口 | 仓库脚本 | 配置 |
|---|---|---|
| `Start-Strata-Safetensors.bat` | `START-NATIVE-262K-PREFILL-TC-BF16.bat` | `config/native/rtx5090-262k-prefill-tc-bf16.json` |
| `Start-Strata-Safetensors-Parallel.bat` | `START-NATIVE-262K-PREFILL-TC-BF16-PARALLEL.bat` | `config/native/rtx5090-262k-prefill-tc-bf16-parallel.json` |

两者使用 W4A4/CUTLASS prefill、
8192 行工作区、BF16 KV Tensor Core、262144 上下文、原专家 decode 和调度、视觉，
监听 `127.0.0.1:8880`，模型名 `Qwen3.8-Flash-Next-NVFP4-Native`。
单路使用 `strata-classic-bf16-kv-slotcommit.exe`，`parallel=1`，
显式设置 `STRATA_COMMIT_BATCH=0`，关闭上游单路 GDN/QSA 层合并提交。
双路使用恢复后的 `strata-classic-bf16-kv-v0144.exe`，`parallel=2`，
额外开启 `--batch-mtp`；未设置 `STRATA_SLOT_COMMIT_BATCH`，不启用本地层合并提交。
提交优化的历史测量见 [双路提交验证](../bench/results/slot-commit-batch-20261014/README.md)，
不代表当前启用设置。
两者均为 `--mtp native --spec 3 --spec-min-p 0.5`，suffix drafting 关闭。
并发 verifier 每路最多一枚 MTP 草稿；`spec=3` 不等于并发每路三枚草稿。

已合并上游 0.1.44.1；两份配置均关闭 `STRATA_ROUTE_TAIL_SKIP`，计算全部路由专家。
保留原始 MTP 权重；arena 与 MTP 差别见 [升级说明](UPSTREAM_0144.md)。
单路 `STRATA_NATIVE_PREFILL_OVERLAP=0`，因为该实现依赖并发独立 KV。
双路保持 `STRATA_NATIVE_PREFILL_OVERLAP=1`，重叠 prefill 与已有请求的 decode。
两者均保留 `STRATA_ADAPT_OVERLAP=1` 专家拷贝重叠，未选择独立的 `--adapt-async 1` 后台调度。

## 会话缓存：只按内存淘汰

两个桌面配置均选择 `--conversation-cache-slots unlimited`，取消原先 4 个停放会话的上限。
`--conversation-cache-mib 12288`，停放快照和保留的 KV 共用 12 GiB 系统内存预算。
缓存满时淘汰最久未使用的普通会话（LRU），恢复后重新停放的会话排到最新位置。
超过预算的单个快照不入缓存；交换中仍存活的快照也计入预算。
显式 pin 的共享前缀继续受保护，无法腾出足够空间时拒绝新快照，不超预算。

系统空闲内存底线仍为 2560 MiB，必要时会在缓存达到 12 GiB 之前回收冷会话。
网页显示会话数量为 `当前数量 / ∞`，内存条继续显示预算使用率。
通用默认仍为 4 槽；`--conversation-cache-slots 0` 仍表示禁用。
桌面程序包含上述缓存策略，CPU 保持每物理核一个线程，SMT 未启用。
MTP、8192 prefill、BF16 KV、Host 不限制等其他配置不变；下次启动生效。

预算随后从 8 GiB 调至 12 GiB；历史验证记录保留当时的 8 GiB 参数。
2026-10-14 的一次运行采样：95.61 GiB 可见物理内存、11.53 GiB 系统可用、
最近日志停放缓存 3.72 GiB，扣除 2.5 GiB 底线后的缓存总空间粗估 12.75 GiB。
这些读数非原子采样，也不是长上下文峰值容量保证；运行时仍重新采样可用内存。

验证见 [memory-only cache](../bench/results/conversation-cache-memory-only-20261014/README.md)。

## 清理范围

- 移除本次 W4A16 试验的 BF16 专家展开/GEMM 分支、模式选择、配置、启动脚本和专用测试。
  专家 prefill 恢复为试验前的 W4A4 实现；试验源码和测量可从 Git 历史恢复。
- 移除 6 个闲置原生启动脚本、6 个旧 JSON 配置和 2 个旧网页设置 sidecar。
- 合并构建入口为 `tools/build_safetensors_engine.ps1`，默认构建当前桌面程序。
  CUTLASS 固定为 `56ca28adb32ffadd097566545b8f7d83b652e998`，自动获取到忽略目录
  `build-deps/cutlass`，不依赖旁边的试验仓库。
- 将未启用的共享 FP8/BF16 KV 实验缩为独立 `kv_bf16_tc` 后端，删除 FP8 KV
  编码、gather、FP8 attention 分支及未使用的 pool 标志。BF16 运算顺序保留。
  模型自身的 FP8 MTP 专家和 PLE 数据仍是当前路径的依赖，保留。
- 小组专家仍需自写 W4A4 回退路径；CUTLASS auto 会使用它，不能删除。
- 保留原始专家 decode、CPU AVX-512、热专家晋升、共享原生 MTP、会话分支保护、
  连续 prefill、视觉、缓存诊断和对应测试。
- 上游通用安装器、其他后端、必要测试和历史测量记录保留；旧桌面配置不再是维护入口。
  本地试验、dump 与网页设置加入忽略规则，不推送。

删除前副本在本机忽略目录 `build-native-engine/cleanup-backup-20261014/`。
其他仓库、模型、桌面 Q6 启动入口和正在运行的程序不在本次删除范围。

## 构建与验证

安装 CMake/Ninja 到 `.venv-native` 或通过 `-ToolDir` 指定，执行：

```powershell
./tools/build_safetensors_engine.ps1 -Jobs 2
```

正在运行同名引擎时构建脚本会拒绝覆盖。可用 `-OutputName strata-cleanup-check`
编译独立候选；不会修改桌面配置或启动模型。

W4A16 清理与当前入口核对见 [W4A4 cleanup](../bench/results/w4a4-final-cleanup-20261014/README.md)。
此前桌面范围整理见 [cleanup validation](../bench/results/desktop-prefill-tc-cleanup-20261014/README.md)。
此前 BF16 算子、快照与 sanitizer 记录见
[BF16 KV](CLASSIC_BF16_KV.md)。全模型质量、清理前后长上下文输出一致性、
HIP/SYCL/Linux 仍不能用 CPU 或编译检查代替。
