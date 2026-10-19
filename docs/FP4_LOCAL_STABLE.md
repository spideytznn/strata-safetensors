Archived experiment as of 2026-10-20. The active deployment uses original NVIDIA NVFP4 with the upgraded engine; see [deployment](NVFP4_DEPLOYMENT_020.md). No quality advantage for FP4-Local has been established. Profiles are retained under `config/native/`; desktop launchers are retired. The records below describe historical builds and settings.

# FP4-Local stable checkpoint: 2026-10-19

Tag: `fp4local-stable-20261019`. This freezes the user's accepted Windows
RTX 5090 implementation before experimental NVIDIA-style decode work.

Desktop entry: `START-FP4LOCAL-PARALLEL.bat`, localhost port 8880.
Profile: `strata-fp4local-native.json`; two slots, native
batch MTP, spec/max-t 3, confidence 0.5, prefill 8192, BF16 KV, automatic
expert cache, adaptive exchange every 4 windows with up to 96 promotions,
and PCIe fraction 0. The profile contains this machine's absolute paths.

The current local profile additionally enables the existing GPU BF16 vision
encoder (`--vision`, 1024 tokens per picture). This deployment change is newer
than the frozen tag; the stable executable is unchanged. The FP4-Local
checkpoint's quantized vision tensors are not executed.

Validated engine SHA256:
`7e955d7d7c67fee0a5b922b99d87bf29c419e38a304f4e9babdfe870efa4bc62`.
The executable and model weights are local build/deployment artifacts, not
part of the source tag. See [FP4_LOCAL.md](FP4_LOCAL.md) for build/runtime
evidence, arithmetic choices, and limitations. Raw outputs and session
snapshots remain local under the ignored `outputs/` directory.

Seven standalone tests pass. Earlier validation includes real-checkpoint
packing, CPU/GPU expert parity, MTP, PLE, full-model state acceptance, a
5221-token prompt and two concurrent HTTP responses. Adaptive promotion
was subsequently observed in real requests; it is not a controlled speed
comparison. Only Windows CUDA was built and exercised.

Known issue: one earlier run produced different live state before SAVE.
The preserved state can alter later seeded output. Its root cause remains
unresolved; PCIe causality is unproven. This tag is a reproducible operational
baseline, not a claim that the issue has been fixed. The configured 262144
context ceiling has not been fully validated on this checkpoint.
