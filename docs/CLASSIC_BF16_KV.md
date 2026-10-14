# BF16 KV with the original Safetensors expert backend

The two maintained entries use `build-native-engine/strata-classic-bf16-kv-v0144.exe`:

* `START-NATIVE-262K-PREFILL-TC-BF16.bat`: W4A4/CUTLASS/8192 prefill,
  original expert decode, BF16 KV and native MTP; one request at a time.
* `START-NATIVE-262K-PREFILL-TC-BF16-PARALLEL.bat`: the same arithmetic,
  two request slots and `--batch-mtp` enabled as well as native MTP.

Both use `--spec 3 --spec-min-p 0.5 --suffix-draft 0`, 262144 context,
vision, PCIe share 0.25, tail skip 7, original adaptation and conversation reuse.
They are selected by the single/parallel desktop launchers, respectively.
The endpoint is `http://127.0.0.1:8880`; run one server at a time.
Old trial launchers/configurations were removed; see [desktop scope](DESKTOP_SCOPE.md).

## Scope and arithmetic

`STRATA_KV_BF16_TC=1` plus `--kv bf16` selects cache storage and attention
independently of the prefill expert mode. `STRATA_NATIVE_PRECISION=0` is explicit
in the test profiles; experimental W4A4 expert decode and altered MTP projections
are not enabled. Both target sessions and the MTP draft state use BF16 KV.

K/V are written directly as BF16 without scales or Hadamard rotation. QK and
PV use BF16 Tensor Core MMA with FP32 accumulation, softmax and split-K merge.
Queries and softmax probabilities are rounded to BF16 for those products.
Prefill with at least 32 query rows uses the fused online attention kernel;
short tails use split-K. No complete KV expansion is allocated in global memory.

The independent `kv_bf16_tc.cu/hpp` implementation retains only BF16 storage
and attention. The unused FP8 KV branches were removed during desktop cleanup.
Existing INT8 and FP16 dispatch keep their inherited generic branches.

BF16 has an explicit state/pool encoding flag. Streaming and ring movers copy
the existing opaque 16-bit layout; snapshot format bit 64 and the session
fingerprint distinguish BF16 from FP16. Per-slot attention receives its own
pool pointers through the original verifier. The BF16 mode rejects legacy
attention overrides, peer GPUs and pipeline windows rather than interpreting
BF16 bits as FP16. Build support is CUDA SM120 only.

BF16 KV occupies more memory than INT8. Across the 12 target attention layers,
one 64K session adds 744 MiB; one fully mapped 262144-cell session adds 2976 MiB.
These figures exclude MTP and other buffers. Independent elastic mappings
remain enabled, so short/idle sessions need not map their full capacity. At long
contexts extra KV may reduce the GPU expert cache and affect decode speed.
BF16 storage precision alone does not prove better task quality; attention
rounding, rotation and draft acceptance differ from the INT8 path.

## Build and validation

```powershell
./tools/build_safetensors_engine.ps1 -Jobs 2
```

`STRATA_BF16_KV_TC` defaults OFF. The desktop build helper explicitly enables it, along with NVFP4 prefill and CUTLASS. The old expert,
CPU expert, dense decode and MTP projection sources are not ported from the
experimental native-precision backend.

On 2026-10-14, RTX 5090 / Ryzen 9950X3D / 96 GB RAM, Windows CUDA 13.0 and
MSVC 14.44:

* The engine, BF16 operator test and snapshot test compiled.
* BF16 operator checks passed for 1/3/8/32 queries, 5/65/2051 selected cells,
  step/batch/fused attention, independent CPU rounding/attention references,
  pagetable permutations/missing pages, host writes and changed-width graph replay.
* Compute Sanitizer memcheck and initcheck each reported zero errors.
* Snapshot checks passed: 4237 total, including BF16 resident/streamed/ring
  save/restore, cross-format rejection and direct slot copies.
* Disassembly contains `HMMA.16816.F32.BF16`; the BF16 attention uses hardware
  matrix instructions.
* All seven prefill-policy CPU regression checks passed. Nine original
  expert/decode/MTP source files matched repository HEAD. Frozen desktop and
  INT8 prefill-only executable hashes were unchanged.

Only small synthetic GPU tests were run after checking the GPU was idle. No
model service was started. Whole-model startup, two-client parity, long-context
capacity, quality and performance remain unvalidated for these hybrid profiles.
HIP and SYCL were not built or validated. Raw logs, source/executable hashes
and prepared private-pipe configurations are in
`bench/results/classic-bf16-kv-20261014/`.
