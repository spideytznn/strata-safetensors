# Tensor Core prefill with the original expert decode backend

The maintained desktop entries are `START-NATIVE-262K-PREFILL-TC-BF16.bat`
and `START-NATIVE-262K-PREFILL-TC-BF16-PARALLEL.bat`. Both select W4A4/CUTLASS
prefill at 8192 tokens, BF16 KV Tensor Core attention, native MTP and the original
expert decode backend. The parallel entry adds two slots and `--batch-mtp`.
See [desktop scope](DESKTOP_SCOPE.md) and [BF16 KV](CLASSIC_BF16_KV.md).
Old INT8, 4096 and original-prefill trial launchers/configurations were removed.

## Arithmetic and dispatch

`STRATA_PREFILL_NVFP4=w4a4` selects the optional prefill port;
`STRATA_NVFP4_PREFILL_BACKEND=auto` uses NVIDIA CUTLASS grouped GEMM where
measured beneficial and the custom W4A4 implementation for small expert groups.
Packed FP4 weights and checkpoint activation scales are retained. Activations
use one FP4 term with group-16 E4M3 scales; accumulation is FP32. Dense prefill
projections round activations to BF16 for one product, replacing the original
FP32 activation remainder products. Expert decode and MTP projections retain
the original implementation; `STRATA_NATIVE_PRECISION=0` is explicit.

Expert groups retain staging-ring slots until both matrix products finish.
CUTLASS preparation permutes packed codes/scales into its layouts. Scratch is
included in the prefill workspace, with no permanent expanded weight copy.
The supported shape is embedding 2560, intermediate 640, on CUDA SM120.

The CMake options remain OFF by default, preserving the inherited generic
backend. The desktop build helper explicitly enables NVFP4 prefill, CUTLASS
and BF16 KV. Peer GPUs, pipeline windows, HIP and SYCL are outside validation.
Session identity records the changed prefill arithmetic and KV encoding.

## Quality and memory

W4A4 changes activation rounding relative to the old W4A8 path. BF16 projection
rounding and chunk size also affect initial KV/GDN/PLE state. Keeping original
decode does not prove identical answers. No broad quality-equivalence claim
is made for these hybrid profiles.

An 8192 workspace can displace more automatic hot-expert cache than 4096.
Compare cache counts as well as timings when measuring decode speed.
Historical operator/port checks are retained under
`bench/results/prefill-only-port-20261014/`; they do not measure full-model quality.

## Build

```powershell
./tools/build_safetensors_engine.ps1 -Jobs 2
```

The helper fetches pinned CUTLASS into `build-deps/cutlass` if absent and rejects
modified or mismatched dependencies. The license is retained in
[licenses/CUTLASS.txt](licenses/CUTLASS.txt). Tests include CUTLASS/custom
operator parity, graph replay and prefill selector guards.
