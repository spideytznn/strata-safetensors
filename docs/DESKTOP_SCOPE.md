# 当前桌面维护范围：2026-10-14

桌面只保留两条 safetensors 启动链：

| 桌面入口 | 仓库脚本 | 配置 |
|---|---|---|
| `Start-Strata-Safetensors.bat` | `START-NATIVE-262K-PREFILL-TC-BF16.bat` | `config/native/rtx5090-262k-prefill-tc-bf16.json` |
| `Start-Strata-Safetensors-Parallel.bat` | `START-NATIVE-262K-PREFILL-TC-BF16-PARALLEL.bat` | `config/native/rtx5090-262k-prefill-tc-bf16-parallel.json` |

两者使用 `build-native-engine/strata-classic-bf16-kv.exe`，W4A4/CUTLASS prefill、
8192 行工作区、BF16 KV Tensor Core、262144 上下文、原专家 decode 和调度、视觉，
监听 `127.0.0.1:8880`，模型名 `Qwen3.8-Flash-Next-NVFP4-Native`。
单路 `parallel=1`；双路 `parallel=2`，额外开启 `--batch-mtp`。
两者均为 `--mtp native --spec 3 --spec-min-p 0.5`，suffix drafting 关闭。
并发 verifier 每路最多一枚 MTP 草稿；`spec=3` 不等于并发每路三枚草稿。

## 清理范围

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

本次清理验证记录见 [cleanup validation](../bench/results/desktop-prefill-tc-cleanup-20261014/README.md)。
此前 BF16 算子、快照与 sanitizer 记录见
[BF16 KV](CLASSIC_BF16_KV.md)。全模型质量、清理前后长上下文输出一致性、
HIP/SYCL/Linux 仍不能用 CPU 或编译检查代替。
