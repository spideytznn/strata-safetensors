# Desktop prefill/BF16 cleanup validation, 2026-10-14

Windows, RTX 5090 32 GB, Ryzen 9950X3D, 96 GB RAM, CUDA 13.0, MSVC 14.44.

The two desktop launch chains retain W4A4/CUTLASS prefill at 8192, BF16 KV,
original expert decode and native MTP (`spec=3`). The parallel profile adds
`parallel=2` and `--batch-mtp`; the single profile uses `parallel=1`.

Validation:

- Server unittest discovery: 750 tests completed, 12 skipped, no failures
  (274.284 seconds). Mock engines only; no model loaded by this suite.
- CUDA build passed using the unified helper and a freshly fetched pinned CUTLASS
  checkout inside this repo. The candidate was staged as `strata-cleanup-check.exe`.
- Seven prefill policy checks passed, including default/off, custom/auto/CUTLASS,
  invalid backend and missing-build rejection.
- Conversation snapshot test: 4237 checks passed.
- BF16 attention: 27 cases passed for step/batch/fused APIs, pool permutations,
  missing pages, host writes, CPU rounding/attention oracles and graph replay.
  Maximum tested absolute attention oracle error: 0.00082113 (limit 0.003).
- BF16 Compute Sanitizer memcheck: zero errors.
- CUTLASS/custom W4A4 operator comparison: 31 shapes, relative L2 zero.
- Tokenizer: 43 HF parity cases and 30 incremental cases passed.
- Eight cache-diagnostic tests passed, including failure isolation, request
  attribution, concurrency and avoiding plaintext prompt/token retention.
- Both desktop chains and config paths checked; nine original expert/decode/MTP
  source files still match the parent commit. The active desktop engine SHA256
  matches the previously deployed binary.

See `manifest.json` for source/binary hashes and profile settings. The old BF16
port evidence remains in `../classic-bf16-kv-20261014`; it describes its original
source filenames/configuration, before the BF16-only cleanup.

The existing user service was retained. No model/server was started or stopped;
only small synthetic GPU operator checks were run. Check-only timings are not
performance measurements. Full-model output parity, quality, long-context
capacity and throughput were not rerun. HIP/SYCL/Linux were not built or tested.
