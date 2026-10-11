# Strata v0.1.42 integration

This integrates the complete upstream tag `v0.1.42`
(`61b3fb5dd3f1e8ec09cf7e4e05208bc6d3c46406`) into the native concurrency
baseline `c8402280adda6ee26ee5952c5370b70609348f5e`. It includes upstream source,
tests, documentation and benchmark evidence. Upstream measurements retain their
original hardware and scope; they are not measurements of this fork.

## Quality default

`STRATA_ROUTE_TAIL_SKIP` is **off by default**, including CUDA serve. An absent
variable and `=0` compute all routed experts. An explicit value from 1 through 9
opts into upstream's approximate tail skipping; it can change answers. The
engine's startup protocol reports `INFO route_tail_skip=0` when disabled.

The native profiles retain original NVFP4 codes/scales, BF16 dense weights,
FP32 decode activations, the existing compensated prefill arithmetic, INT8 KV,
dedicated prefill allocation and RAM-resident experts. Other experimental
routing/fusion options remain opt-in. Concurrency still unloads MTP completely;
native batch MTP and native multi-GPU remain unsupported.

## Integration decisions

- Upstream batch adaptation replaces the fork's batch adaptation tick. It runs
  once per due window, waits for copies before publishing residency, and pauses
  during prompt chunks or cache-slot loans. Native promotion diagnostics remain
  available, including the actual number of active rows.
- Retain the fork's independent elastic KV and direct device slot migration.
  These copies now use upstream's private non-blocking conversation stream;
  callers synchronize before reading or reusing the destination.
- Retain owned residency snapshots for router lookahead, alongside upstream's
  cancellation generation and statistics changes.
- Keep native BF16/NVFP4 CPU/GPU dispatch while adding upstream's older-CPU
  kernels and generic Q8 dequantization. Explicit PCIe shares and the fork's
  measured cost model remain available.
- Merge server ownership/queue fixes, cancellation/cache fixes, metrics,
  logit bias, reasoning controls, replicas, memory guards, setup updates and
  backend-specific improvements. Hardware-specific features retain their
  upstream opt-in requirements and the native backend's existing constraints.
- Upstream's slow-direct-I/O probe and automatic mmap fallback apply to GGUF.
  Native FP8 ngram tables currently require segmented direct reads; automatically
  selecting mmap for them is refused by the reader. The native path therefore
  retains its established direct-I/O default. Supporting native mmap is separate
  work, not a claimed capability of this integration.
- Setup uses upstream's current/initial page-file size accounting, rather than
  counting possible future growth as already available commit.

## Validation scope

The local validation machine is Windows, RTX 5090 32 GB, Ryzen 9950X3D and
96 GB RAM, using CUDA 13.0, MSVC 14.44 and SM120. The staged executable is
`build-native-engine/strata-upstream142.exe`. Validation evidence and exact
commands are stored under `bench/results/2026-10-11-upstream-v0142/`.

The original model and the Q4XL deployment are read-only inputs. The previous
accepted executable is retained for comparisons and rollback. HIP, SYCL,
Linux, multi-GPU and other CPU generations are not built or hardware-validated
on this machine; merging their upstream changes does not establish support for
native safetensors on those configurations.
