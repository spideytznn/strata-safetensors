# 本机 Q6 GGUF 桌面部署恢复快照

记录日期：2026-10-13。Windows / RTX 5090 32GB / Ryzen 9950X3D / 96GB RAM。
这是外部 Fusion v0.1.43.1 GGUF 部署的三个文件快照，不是新的 safetensors 桌面入口，
也不包含引擎、模型、pack、ngram 或视觉资源。

已升级引擎及服务端，保留原专家预算、锁页参数、模型路径与关闭 MTP 的配置。
显式关闭路由尾部跳过并启用驻留交换重叠。下文故障修复与 `verification.json`
来自 2026-10-12；本次升级验收、输出差异和回退位置见
[v0.1.43.1 融合记录](../../docs/UPSTREAM_V01431.md)。

## 恢复位置

- 将 `Start-Strata-Fusion-Q6KL-NoMTP.ps1` 和 `strata-fusion-q6kl-no-mtp.json` 复制到 `G:\Strata\Strata`。
- 将 BAT 复制到自己的桌面。
- PS1 依赖所在目录作为引擎项目根目录，不能直接在本快照目录运行。
- 模型目录为 `G:\moemodels\Qwen3.8-flash-next-gguf`。其余依赖按 JSON 中的本机绝对路径保留。

复制后只检查文件，不启动模型：

```powershell
& G:\Strata\Strata\Start-Strata-Fusion-Q6KL-NoMTP.ps1 -CheckOnly
```

## 锁页故障与最终设置

原日志在长请求后报 `prefill: cannot allocate the CPU experts' rows`。
最终保留 `--resident-budget-gib 81`、`HeadroomGiB=1`、CPU prefill 与 CPU decode，
仅将启动器默认 `MappedGiB` 从 81 改为 64。该值传给 `STRATA_PARTIAL_PIN_GIB`。

此旧引擎只在整块 CUDA 锁页申请失败后的部分注册分支使用这一上限；它不是全局硬上限。
本次整块 78.00 GiB 注册失败后，64.00 GiB 注册成功，另约 14 GiB 通过 VirtualLock
保留在内存。显存专家缓存约 20.55 GiB。没有通过减少专家驻留量来解决分配错误。

连续 510-token 短请求 → 28449-token 长请求 → 979-token 短请求均完成，
每次生成上限 32 token，HTTP 健康检查通过。详见 [verification.json](verification.json)。
这是分配与连续请求检查，不是质量基准或稳定吞吐测量；测试输出可能在思考中达到上限。

已撤回的尝试：64 GiB 专家驻留预算虽然完成五次请求，但因增加 SSD 读取被用户否决；
增大普通 RAM 余量和关闭 CPU 短 prefill 的尝试也未解决锁页/cuBLAS 问题。

Q6 模型较大，仍有专家无法同时留在 GPU/RAM，会从文件读取。
原生 safetensors 的驻留后零专家源读取保证不适用于这个外部 GGUF 部署。
不同可用内存、驱动状态、长上下文和应用负载未穷尽验证。
