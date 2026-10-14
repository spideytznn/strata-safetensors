# Memory-only conversation cache, 2026-10-14

Windows / RTX 5090 / Ryzen 9950X3D / CUDA 13.0 / MSVC 14.44.

Desktop single/parallel profiles select `--conversation-cache-slots unlimited`.
Snapshots and retained KV share the existing 8192 MiB byte budget. There is no
count ceiling. At pressure, the least recently used unpinned entry leaves;
restoring and reparking refreshes recency. Explicit shared-prefix pins retain
their existing protection. Oversized incoming snapshots are refused, including
in-flight images in the budget. The 2560 MiB physical-RAM floor remains active.
Generic defaults remain four slots, and numeric zero still disables caching.

Validation:

- CUDA engine compiled as a separate `strata-classic-bf16-kv-cache.exe`.
- 4311 conversation-cache checks passed, including retaining 64 independent
  conversations, byte-triggered LRU eviction, refreshed recency, in-flight
  accounting, oversized rejection and prior numeric/disabled/pinned behavior.
- 25 physical-memory admission checks and 4237 snapshot checks passed.
- 18 monitor/cache-diagnostic tests passed; mock HTTP metrics retain the
  `unlimited` value and 8192 MiB budget.
- JavaScript syntax and monitor rendering passed for unlimited and numeric
  limits. Memory percentage remains displayed; count reads `64 / ∞` in the
  synthetic unlimited fixture.
- CLI parsing accepts `unlimited` without a model load. Both desktop launch
  chains and retained settings checked; original CPU pool source and deployed
  binary remain unchanged.

No model service was started, stopped or restarted. Only the executable path
and conversation slot setting changed in the two desktop profiles. MTP, BF16 KV,
8192 prefill, unrestricted Host and physical-core CPU scheduling are retained.
These are cache correctness checks, not a full-model performance/quality or
long-running agent replay. HIP/SYCL/Linux were not built or validated.
See `manifest.json` for hashes and profile settings.
