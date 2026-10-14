# Upstream 0.1.44.1 merge validation, 2026-10-14

Merged upstream 296b59d2 into the native safetensors desktop branch. Windows,
RTX 5090 32 GB, Ryzen 9950X3D, 96 GB RAM, CUDA 13.0, MSVC 14.44.

Both launch chains select the separately built strata-classic-bf16-kv-v0144.exe.
The desktop changes are the executable, explicit STRATA_ROUTE_TAIL_SKIP=0,
and STRATA_ADAPT_SIGNAL=copy to retain the previous signal choice while including
the upstream WDDM flush fix. 8192 prefill, BF16 KV, native MTP/spec 3, original
FP32-activation expert decode, physical-core CPU layout, 12 GiB memory-only
conversation cache, wildcard Host and port 8880 are preserved.

Validation:

- CUDA engine and the selected test targets built successfully.
- Server: 810 unittest cases, 15 skipped, no failures (298.446 s), mock engines.
- Conversation cache: 4311 checks; memory admission: 25; snapshots: 4237.
- Seven prefill selector/guard tests passed.
- BF16 attention: 27 cases; CUTLASS/custom: 31 shapes, relative L2 zero.
- Native FP32 expert GPU reuse: 24 cases (1/16/64 groups, 1..8 tokens),
  bit-identical including empty, uneven and strided groups and output canaries.
- Two real checkpoint expert CPU checks: FP64 relative L2 below 2.64e-7;
  T=1 and T=4 outputs bit-identical. Six test expert blobs round-tripped exactly.
- Upstream W4A8 GPU path: real expert 0 output relative error 1.123% against
  the double-precision reference (test tolerance 3%). This is a separate path,
  not the desktop's FP32-activation decode.
- AVX-512 rows passed the existing ggml tolerance; AVX2 rows bit-identical.
- GDN batched commit: 8 bitwise cases over 5 layers. QSA batched indexer append:
  3 layers, no/linear/YaRN scaling, bit-identical. Both full parity suites passed.
- CPU doorbell service during graph launch and cleanup test passed.
- JavaScript syntax, both desktop launcher chains and all retained config
  choices checked. CLI accepts unlimited cache slots without loading a model.

The existing user engine and vision processes were retained. No full model
service was started, stopped or restarted. Operator timing columns are not
end-to-end performance measurements. Full-model throughput, MTP acceptance,
quality, long-context replay and HIP/SYCL/Metal/Linux were not validated.
Logs and temporary expert exports remain in the ignored build-native-engine
folder; manifest.json records source, executable and config hashes.
