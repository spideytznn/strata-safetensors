# W4A16 BF16 expert prefill trial, 2026-10-14

RTX 5090, Ryzen 9950X3D, 96 GB RAM, Windows, CUDA 13.0, MSVC 14.44.
Only synthetic operator tests were run. The user's existing engine and vision
processes were left running; no candidate model or HTTP listener was started.
The two trial launchers use port 8880 and require closing the existing server.

The opt-in implementation expands one expert's original NVFP4 codes and
micro-scales into BF16, gathers BF16 activations and calls the existing cuBLAS
BF16 GEMMs with FP32 output. Global weight scales stay outside the expanded
matrices and are applied before SwiGLU / after down respectively. SwiGLU is
evaluated in FP32 and rounded to BF16 for the down product. There is no A4
activation quantization and no permanent full BF16 expert cache.

Validation:

* CUDA candidate and GPU test compiled. Nine selector tests passed, including
  unchanged defaults and refusal without the optional build.
* CPU-independent packed-weight decoding exactly matched BF16 preparation.
  The fixture covers all finite nonnegative E4M3 scale codes and all FP4 codes.
* Three reordered expert groups with 1, 3 and 37 rows passed the complete FP64
  dot-product oracle with BF16 activations/intermediates: relative L2
  `9.50811117e-6`, maximum absolute error `0.0484161377`. The broad synthetic
  scale range produces large outputs; these are arithmetic errors, not model
  quality metrics.
* Inactive output rows, workspace guards, warmed CUDA Graph replay (bit equal)
  and invalid-row rejection passed. An 8192-row zero-input case passed its exact
  zero oracle and boundary guards.
* Compute Sanitizer memcheck reported zero errors for the above suite.
* Nsight Systems recorded `cutlass_80_tensorop_s16816gemm_bf16_64x256_32x4`
  and `cutlass_80_tensorop_s16816gemm_bf16_256x128_32x3` for the 8192-row products,
  plus WMMA/Tensor Core kernels on smaller groups. One-row products used GEMV.
* W4A4 custom/CUTLASS regression passed 31 shapes, relative L2 zero in each.

The 8192-row expert workspace is 99.375 MiB, including 9.375 MiB of temporary
expanded weights. Compare actual startup hot-cache sizes when timing this
against W4A4, because changed scratch requirements affect auto-sizing.

The single and parallel trial JSON files are exact copies of their respective
current profiles except for the candidate executable, expert-prefill selector
and log file. Existing desktop launchers were not redirected. The single
profile retains the user's commit-batching/overlap opt-outs; the parallel
profile retains the rollback to the previous scheduling/commit choices.

No full-model startup, task quality, long-context parity, throughput or
HIP/SYCL/Metal/Linux/multi-GPU claim is made. Synthetic zero/nonzero fixtures
do not prove equivalence to NVIDIA's complete BF16 model.

[Manifest](manifest.json), [oracle output](parity.txt), [memcheck](memcheck.txt),
[W4A4 regression](w4a4-regression.csv), [kernel summary](kernel-summary.txt).
The complete Nsight report stays in ignored local build storage; its digest is
recorded in the manifest. Reproduction and launch instructions are in
[W4A16_PREFILL](../../../docs/W4A16_PREFILL.md).
