# W4A16 expert prefill trial

This opt-in trial keeps NVFP4 weights in RAM and the hot-expert cache. During
prefill, one expert is expanded into temporary BF16 matrices, activations are
rounded to BF16, and the existing cuBLAS BF16 GEMMs accumulate into FP32.
Gate/up global weight scales are applied before SwiGLU; the down global scale
is applied after its product. They are not folded into FP16 weights.
Checkpoint input scales for A4 quantization do not apply to A16 activations.

The temporary weight matrices include the original FP4 codes and E4M3
micro-scales. BF16 can represent these finite products exactly. This does not
restore information lost when the checkpoint weights were originally quantized.
BF16 activation rounding, floating-point reduction and SwiGLU still affect
results; W4A16 is not a promise of full-model BF16 equivalence or better answers.
cuBLAS selects its algorithm by group size. Larger tested groups use BF16
Tensor Core kernels; a one-row group may use GEMV instead. The trial does not
force padded Tensor Core work for every tiny expert group.

Only routed-expert prefill changes. Dense prefill retains the current rounded
BF16 path, with `STRATA_PREFILL_BF16X2=0`. Decode, native MTP weights, BF16 KV,
8192-row allocation, physical-core CPU layout, cache budgets, port 8880 and
unrestricted Host headers come from the corresponding current profile.
The single profile retains `STRATA_COMMIT_BATCH=0` and prefill overlap off;
the parallel profile retains the user's rollback settings.

Launch one of these instead of the usual launcher, after closing the current
server. Both use the independently built `strata-classic-bf16-kv-w4a16.exe`:

* `START-NATIVE-262K-PREFILL-W4A16-BF16.bat`
* `START-NATIVE-262K-PREFILL-W4A16-BF16-PARALLEL.bat`

The existing desktop launchers are not redirected. The environment selector is
`STRATA_PREFILL_NVFP4=w4a16`; unselected paths keep their existing arithmetic.
Its session fingerprint differs from W4A4, preventing cross-mode state reuse.

Temporary workspace is about 99.375 MiB at 8192 rows, including 9.375 MiB of
expanded weights. Only one expert's expanded matrices are retained at a time.
No permanent full BF16 expert copy is allocated. Different workspace sizes can
change automatic hot-expert cache sizing, so compare startup cache counts along
with throughput. No end-to-end speed or quality improvement is claimed.

Build the candidate without overwriting a running engine:

```powershell
./tools/build_safetensors_engine.ps1 -OutputName strata-classic-bf16-kv-w4a16 -Targets strata,nvfp4_w4a16_test,native_prefill_policy_test,native_prefill_policy_unbuilt_test -Jobs 2
./build-native-engine/nvfp4_w4a16_test.exe
```

The GPU test checks production preparation against independent CPU decoding,
complete output against an FP64 dot-product reference with BF16 inputs,
reordered expert groups, short and uneven row counts, inactive outputs,
workspace guards and CUDA Graph replay. On 2026-10-14, CUDA compilation, nine
selector checks, the operator oracle/8192-row checks and memcheck passed.
Nsight Systems recorded BF16 Tensor Core kernels, including the 8192-row case;
one-row products also selected GEMV. The original W4A4 custom/CUTLASS comparison
still passed all 31 shapes with zero relative L2 between those two paths.
See [validation evidence](../bench/results/w4a16-prefill-20261014/README.md).
Full-model startup and quality tests are separate; HIP/SYCL/Metal
and Linux are outside this CUDA SM120 trial.
