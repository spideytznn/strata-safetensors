# Native safetensors automatic PCIe selection

2026-10-11; Windows, RTX 5090 32 GB, Ryzen 9950X3D, 96 GB RAM; CUDA 13 / SM120.
Base commit: `1d309416307f21d295a9ef506825076e4f237750` plus the recorded working-tree fix.
Executable SHA256: `313bf50e8de4040acbd7ceea33be6f466c7a04f1a527d7f23099f70adbc72d13`.

## Change

The native adapter no longer resolves an unspecified PCIe share to 1.0 before the
common probe. That premature assignment also marked the share as explicit, disabling
both the per-layer cost model and the upstream early-decode adjustment.
Explicit `--pcie-frac N` still bypasses both; fixed 0.25 was exercised on the real model.
New INFO fields expose `pcie_auto` and `pcie_balance`; `pcie_frac` is the startup/global
share, not the observed per-layer percentage when the cost model is active.

## Checks

- CUDA engine built with `tools/build_safetensors_engine.ps1 -Jobs 4 -OutputName strata-pcie-auto`.
- 30 host calibration, PCIe setup and run-config tests passed; see `checks/host.log`.
- The completed screen has 24 greedy requests of 256 tokens: 18 complete vectors
  compared against the first case, all equal. All cold/warm pairs match and reuse cache.
- All completed cases report zero post-residency expert and MTP source bytes.
- Automatic cases report `pcie_auto=1`; the full auto case also reports
  `pcie_balance=1`. Fixed cases report both as 0. All start at 0.25.
- The automatic target-only runtime gate passed 20 completed requests plus cancellation:
  greedy/seeded sampling, thinking with config sampling, caps, EOS, A/B/A, disk restore
  with equal restored main-state fingerprints, and cancel/resume. MTP is unloaded.
- HIP, SYCL, other GPUs and broad model-quality benchmarks were not run.

## Timing screen

Original safetensors, original BF16 projections, INT8 262144 KV, 4096 dedicated prefill,
vision loaded before cache sizing, pinned expert RAM, adaptive exchange, MTP unloaded
and tail skipping off. All four cases use the same 1024 MiB explicit reserve.
Order: fixed-before, full auto, auto with per-layer balancing off, fixed-after.
Three short English/Chinese/code tasks, each cold and warm; one sequential run per case.

| Case | Median decode tok/s | Change against pooled fixed samples | Global share changes |
|---|---:|---:|---|
| Fixed 0.25 before | 62.35 | -0.75% | none |
| Full automatic | 60.70 | -3.38% | 0.25 -> 0.45 |
| Automatic share only (`STRATA_PCIE_BALANCE=0`) | 62.04 | -1.24% | 0.25 -> 0.40; 0.40 -> 0.45 |
| Fixed 0.25 after | 63.33 | +0.81% | none |

The pooled fixed-sample median is 62.82 tok/s.
This screen does not establish a speed benefit for automatic selection. GPU clocks
were uncontrolled and WDDM/cache placement left 5728-5762 expert slots despite equal
configuration. Startup is excluded. These short prompts do not establish long-context
or general throughput. Exact output comparisons are finite regression evidence, not
proof of equivalence to unquantized BF16.

## Initial 256 MiB attempt

`operator/` retains the original 512-output-token experiment at the desktop reserve.
The fixed-before and full-auto cases each passed six requests with equal corresponding
outputs and exact cached repeats. Their decode medians were 60.66 and 49.47 tok/s.
The final fixed case refused startup because only 488 MiB was free against the 512 MiB
prefill-first guard. That attempt is incomplete and is not the completed timing screen.
It remains as evidence; no memory guard or output assertion was weakened. The subsequent
four-case screen uniformly raised only the test reserve to 1024 MiB.

## Desktop activation

The serial desktop config and its original local fallback now select `strata-pcie-auto.exe`,
omit `--pcie-frac`, and keep MTP unloaded. This is the user-requested automatic trial,
not a claim that the automatic policy is faster. The existing 256 MiB desktop reserve,
sampling sidecars, prefill allocation and model paths are unchanged. The former executable
and byte-preserving config backups remain available.
The parallel profile retains explicit 0.25 and its existing executable. Both Q4XL
configurations and their restored MTP settings are byte-identical to before this change.
Activation and backup hashes are in `activation.json`; generated session binaries are ignored.

## Reproduction

```powershell
.venv-native/Scripts/python.exe tools/native_operator_bench.py `
  --configs build-native-engine/pcie-auto-validation/fixed-before-headroom.json `
    build-native-engine/pcie-auto-validation/auto-headroom.json `
    build-native-engine/pcie-auto-validation/auto-share-only-headroom.json `
    build-native-engine/pcie-auto-validation/fixed-after-headroom.json `
  --output <new-output-directory> --rounds 1 --new 256 --desktop-vision

.venv-native/Scripts/python.exe tools/native_runtime_acceptance.py `
  --config build-native-engine/pcie-auto-validation/auto-headroom.json `
  --output <new-runtime-directory> --variants 0 --operator-cache --official-new 64
```

The exact per-case configs, hashes, outputs, info, telemetry and startup times are
in `operator-headroom/results.json` and its numbered subdirectories. The separate
runtime commands and state hashes are in `runtime/`. The local `build-native-engine`
config files are generated artifacts; the complete config objects are retained in results.
