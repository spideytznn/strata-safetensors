Archived experiment as of 2026-10-20. The active deployment uses original NVIDIA NVFP4 with the upgraded engine; see [deployment](NVFP4_DEPLOYMENT_020.md). No quality advantage for FP4-Local has been established. Profiles are retained under `config/native/`; desktop launchers are retired. The records below describe historical builds and settings.

# Experimental NVIDIA-style NVFP4 expert decode

Stable baseline: tag `fp4local-stable-20261019` (`6fe2aca4`).
Experimental branch: `experiment/fp4local-nvidia-w4a4`.

**Status: CPU/GPU W4A4 is implemented, but cross-device repeatability is not
established. Adaptive expert placement changed greedy output in the local
256-token test. Keep this opt-in experiment separate from the stable entry.**

`STRATA_NVFP4_DECODE=w4a4` opts into SM120 block-scaled Tensor Core
computation for the main model's routed experts, during decode and MTP
verification as well as the existing W4A4 prefill. Unset or `fp32` keeps
the stable implementation. Other values are rejected.

The kernel uses NVIDIA's
`mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64`
instruction: E2M1 operands, E4M3 scales per 16 values, and FP32 accumulation.
The original packed weights and FP32 global weight scales are retained.
Activations are divided by the checkpoint's expert-specific global scale,
block maxima are rounded to E4M3 scales, and values are rounded to E2M1.
Down-projection activations are quantized after SwiGLU. There is one FP4
activation term, with no residual compensation or router changes.

Reference: [NVIDIA NVFP4 quantization](https://developer.nvidia.com/blog/creating-the-nvidia-nemotron-3-ultra-nvfp4-checkpoint-with-nvidia-model-optimizer/)
and [cuBLAS block-scaled FP4](https://docs.nvidia.com/cuda/cublas/index.html#narrow-precision-data-types-usage).
The independent GEMM reference is the repository's pinned NVIDIA CUTLASS
SM120 grouped GEMM adapter (`tools/build_safetensors_engine.ps1`).

This is not a complete TensorRT-LLM/vLLM numerical reproduction. Incoming
activations and SwiGLU are FP32 before FP4 quantization; BF16 activation
rounding at every model boundary is not imposed. Dense MXFP8 weights still
expand to BF16. The MTP drafter retains the checkpoint's weight-only NVFP4
path with FP32 arithmetic; missing MTP activation scales are not invented.
BF16 KV Tensor Core attention and the existing prefill backend remain enabled.

## Placement and session safety

The current experiment runs W4A4 on both CPU and GPU. Hot experts use SM120
Tensor Cores; missed experts use CPU AVX512 unless the configured PCIe share
selects GPU execution. The profile restores `--pcie-frac 0`, so cold experts
are computed on the CPU. Adaptive promotion still copies selected experts
into VRAM for later requests. The ordinary placement and fetch-admission
planner is retained; staging no longer needs the GPU-only experiment's 128
slots.

CPU activation quantization runs in 16-wide AVX512 registers, using the same
per-expert calibration, E4M3 block scales and E2M1 nearest-even rounding as
CUDA. Quantization is done once per expert/token. Quantized activation values
are expanded to FP32 for the existing AVX512 FMA kernel, and global scales
are applied after each dot product. This CPU has no native FP4 Tensor Core;
W4A4 describes the operand precision, not a CPU FP4 matrix instruction or
packed 4-bit CPU activation storage. CPU accumulation and SwiGLU can round
differently from GPU operations, so device placement is not promised to be
bit-identical.

Only the single-GPU safetensors configuration with kernel staging and an
AVX512 CPU is currently supported. Expert scale lookup follows routed expert
IDs, including PCIe groups whose entries start after resident groups.
GPU calibration and scratch buffers are allocated before CUDA graph capture.
The experimental arithmetic uses session fingerprint version 2, separate
from both the stable version and the earlier GPU-only prototype.

## Build and local entry

Build with the existing script and a separate output name:

```powershell
.\tools\build_safetensors_engine.ps1 -Targets strata,nvfp4_cutlass_test -OutputName strata-fp4local-w4a4-cpu
```

Profile: `strata-fp4local-w4a4-native.json`.
Entry: `START-FP4LOCAL-NVIDIA-W4A4.bat`, localhost port 8880. Stop the
stable service before using the experimental entry.
The stable launcher and executable remain available separately. The profile
uses two slots, MTP spec/max-t 3, prefill 8192 and adaptive exchange 4/96.
The profile also enables the existing GPU BF16 vision encoder, with at most
1024 tokens per picture. It starts before automatic expert-cache sizing;
the main-model W4A4 experiment does not change the image encoder arithmetic.

## CPU/GPU kernel validation

### Long-context runtime incident (2026-10-19)

With vision enabled and MTP temporarily disabled, a local ZCode request with
182,472 prompt tokens and eight historical images failed after 1,016 output
tokens. The language-model process became unavailable while the HTTP server
remained alive. The client received only an `exit code None` error; the
original native failure reason was not retained, so the cause is unresolved.

The server now persists native stdout fatal verifier errors in the engine log
and includes their retained reason in API stream errors. This improves
diagnosis; it does not change native arithmetic or fix a proven kernel cause.
Thirty-eight fatal-recovery, lifecycle, restart-waiter and concurrency checks
pass. After restart, replaying the original messages/tools and restoring all
eight image URLs completed 2,260 tokens with a `tool_calls` finish and no
stream error. The replay capped output at 4,096 rather than 78,736 tokens and
did not execute returned tools. One replay does not establish that the
intermittent runtime failure is fixed. See
[the incident and checks](../bench/results/2026-10-19-native-exit-diagnostics/summary.json).

### Quantization checks

The scalar encoding reference and AVX512 quantizer were compared against
CUDA conversion intrinsics: 131,504 inputs cover signs, ties, neighboring
floats, E4M3 subnormals and saturation. Nine full-row calibrated quantization
cases cover three global scales and three input magnitudes. All quantized
values agree bit for bit. The production CPU pool was tested with shuffled
expert IDs, independently varying scales, five tokens per expert and then
single-token calls with a different worker/task partition; grouping did not
change CPU output.

For randomized packed weights with a broad scale range, the CPU pool versus
CUTLASS relative L2 error was `2.15730486e-8`, maximum absolute difference
`3.05175781e-5`. These are synthetic-kernel results, not a model-quality
benchmark. Default and canonical CPU configurations both passed. Local
logs: `outputs/fp4local-cpu-w4a4-kernel-test-{simd,canonical}.log`.

## Default-path regression

With `STRATA_NVFP4_DECODE=fp32`, the final CPU/GPU experimental executable
produced the same response choices as the stable executable on two 256-token
solo requests and two overlapping 256-token requests. Its own repeated solo
outputs also matched. An earlier five-prompt default-path check matched all
stable token IDs. The stable executable's SHA256 remains
`7e955d7d7c67fee0a5b922b99d87bf29c419e38a304f4e9babdfe870efa4bc62`.

## Final hybrid runtime validation and limitation

The final executable passed MTP-off, one-draft and two-draft variants, with
17 completed requests per variant: greedy and seeded sampling, output caps,
EOS, A/B/A, SAVE/RESTORE and cancel/resume. Completed main-model state hashes
matched the MTP-off run. These tests used context 4096, prefill 256, a fixed
1024-expert cache and no adaptive promotions; expert/MTP file reads remained
zero after residency. See
[cpu-runtime-acceptance.json](../bench/results/2026-10-19-fp4local-w4a4/cpu-runtime-acceptance.json).

The desktop-profile test uses automatic caching, two slots, MTP 3, prefill
8192 and adaptive exchange 4/96. Both requests ran concurrently, but the
**repeated greedy-output gate failed** for W4A4. A separate control disabled
adaptive exchange and cache reuse, then ran PCIe fractions 0, 1, 0. Both
fraction-0 requests had identical tokens and main-state hashes. Changing only
where missed experts execute changed the output at zero-based token 22
(the 23rd token) and changed state hashes. The control confirms a difference
between CPU and GPU execution paths without expert exchange; it does not
identify the first differing operation. Do not describe the adaptive test as
passing, or attribute it conclusively to a particular floating-point operation.

Kernel tests show exact operand quantization and small synthetic expert error,
but these checks do not imply identical autoregressive trajectories. NVIDIA
also does not specify a portable accumulation order for MMA in its
[PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma).
This is separate from the stable baseline's older intermittent state issue,
which remains unresolved.

See [cpu-placement-parity.json](../bench/results/2026-10-19-fp4local-w4a4/cpu-placement-parity.json)
and [cpu-kernel-parity.json](../bench/results/2026-10-19-fp4local-w4a4/cpu-kernel-parity.json).
Six exported real experts at amplitudes 0.125, 1, 8 and 64 also matched
CUTLASS and the CPU at both GU and down-quantization boundaries in the sampled
cases. Their calibration values were varied by the test, not taken from a
recorded runtime activation trace.

## Hybrid profile performance on this RTX 5090

| Measurement | Final executable, stable FP32 path | CPU/GPU W4A4 |
| --- | ---: | ---: |
| First solo request, tokens/s | 65.0 | 66.7 |
| Second solo request, tokens/s | 75.6 | 80.8 |
| Two overlapping requests, combined tokens/s | 75.4 | 73.9 |
| Sampled whole-GPU peak, GiB | 31.02 | 31.06 |

Both profiles cached 4239 experts. The W4A4 solo cache-hit rates were 70.8%
and 78.1%; compute offload over PCIe was zero. Adaptive promotion still moves
expert weights to VRAM. Minimum sampled free VRAM was 798 MiB in the W4A4 run.
The hybrid result recovers the earlier GPU-only prototype's throughput loss,
but does not establish an overall speed improvement. This is one chronological
run per profile, includes prompt/admission time, and the generated texts differ.
The failed repeatability check must accompany these numbers. Evidence:
[cpu-profile-comparison.json](../bench/results/2026-10-19-fp4local-w4a4/cpu-profile-comparison.json).

## Earlier GPU-only validation scope

The decode-plan test compares shuffled experts, shuffled output destinations,
nonuniform per-expert calibration, split resident/PCIe groups, and an empty
group against NVIDIA CUTLASS. Relative L2 error was zero. The existing
prefill/CUTLASS cases also pass. Full-model five-prompt smoke completed;
arithmetic returned 372, EOS returned Blue, and the generated first-index
binary-search function passed 540 inputs. These are smoke checks, not a
quality benchmark or proof of state correctness for every prompt.

Local evidence: `outputs/fp4local-w4a4-kernel-test.log`,
`outputs/fp4local-w4a4-smoke-20261019/`. Only Windows CUDA/SM120 has been built
and tested. The stable baseline's unresolved intermittent state divergence
has not been established as fixed by this change.

## Earlier GPU-only prototype performance on this RTX 5090

These measurements describe the earlier GPU-only prototype, before CPU W4A4.
Both profiles used spec/max-t 3, two slots, prefill 8192 and adaptive exchange
4/96. Each started a fresh engine and generated two sequential 256-token
responses, then two overlapping 256-token responses. Greedy repeated solo
outputs were identical within each profile; both observed two running requests.
Rates below include prompt processing and request admission.

| Measurement | Stable FP32 experts | Experimental W4A4 experts |
| --- | ---: | ---: |
| First solo request, tokens/s | 63.8 | 55.4 |
| Second solo request, tokens/s | 75.0 | 65.1 |
| Two overlapping requests, combined tokens/s | 74.5 | 60.2 |
| Sampled whole-GPU peak, GiB | 30.96 | 30.94 |

The GPU-only prototype was slower in this small test. During the two concurrent
requests, 41.0% and 50.4% of expert routes were offloaded over PCIe. The
reported `hit_rate=1` excludes those offloads from its denominator; it does
not mean every expert was cached in VRAM. Do not promote this experiment as
a speed improvement. Different arithmetic may change generated text, and
one chronological run per profile is insufficient for a general performance
claim. Machine-readable evidence is in
[profile-comparison.json](../bench/results/2026-10-19-fp4local-w4a4/profile-comparison.json).
