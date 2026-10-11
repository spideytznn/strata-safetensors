# Native two-slot MTP experiment — 2026-10-11

Hardware: RTX 5090 32GB, Ryzen 9950X3D, 96GB RAM; Windows, CUDA 13, SM120.
Checkpoint: original Qwen3.8-Flash-Next NVIDIA NVFP4 safetensors.
Engine/server base: stable `77e9886a`, with the engine sources inherited from
`34541c37`. This is an opt-in experiment, not a change to stable `main`.

## What changed

The native path now accepts `--batch-mtp`. Each request has a private drafter
state and shares the original draft weights/head. The verifier combines both
slots' independent windows, applies the request's confidence floor and commits
only each slot's matching prefix. `--spec 3 --spec-min-p 0.5` means up to three
rows per slot, including its target row. No minimum draft count is forced.
Native penalty history, residual staging, caps and proposal counters follow the
actual width. No forward kernels or quantization formats were replaced.

## End-to-end comparison

All runs: two active text requests, 262144 configured context, 4096 dedicated
prefill rows, INT8 KV, vision loaded, fixed PCIe fraction 0.25, hot-expert
adaptation every four windows, original BF16 projections, route-tail skip off.
The script uses two complete warmups, then three measured rounds; each round
generates 256 tokens per request. Prompts are the TCP/QUIC and Chinese algorithm
tutorials in `tools/native_batch_mtp_benchmark.py`. Sampling is greedy for this
speed comparison. Reported output throughput includes admission/cache restore
and excludes model startup. The original target-only engine is run before and
after MTP. All compared token sequences match exactly.

| Mode | Three rounds, total output tok/s | Median |
| --- | --- | --- |
| Original target-only, before | 67.42, 69.56, 69.24 | 69.24 |
| Native two-slot MTP, spec 3 / confidence 0.5 | 75.07, 74.09, 72.44 | 74.09 |
| Original target-only, after | 65.36, 67.87, 67.96 | 67.87 |

This workload improves by about 7–9%; it does not establish a large or universal
gain. The MTP run starts with 3825 expert-cache slots (10086 MiB), versus 4883
slots (12876 MiB) in the first target-only run. The solo MTP allocation reports
2620 MiB and each shared slot adds about 47 MiB of private state/buffers. These
are startup measurements, not a general peak-memory estimate. The present MTP
mode also disables the native independent-target-KV/device-copy optimizations.

## Correctness and limits

- 54 existing service/concurrency/settings tests passed.
- Real-model acceptance checks greedy and seeded sampling, penalties,
  cancel/resume, followups, caps/EOS and actual overlapping generation.
- Both slots offer and accept proposals while the other is active, including
  three-row windows. Adaptive promotion continues with both slots active.
- With request-local confidence 1.0, both slots report zero offered proposals,
  31 confidence skips and one-row windows, while reproducing target output.
- Two 8913-token inputs pass serial/batch parity, reuse and comparison against
  the original no-MTP engine. Post-residency expert and MTP source bytes are zero.
- A final run on the benchmark executable repeats all acceptance cases and
  verifies overlapping 40000/40007-token requests across the 32768-cell draft
  window, generating 256 tokens each with exact serial/batch parity. In that
  long batch the slots accept 135/195 and 129/196 proposals, including proposals
  offered while both are active. This is a window-boundary check, not a full
  262144-token capacity test.
- This is finite functional coverage, not a broad model-quality evaluation.
  Full 262144-token operation, HIP, SYCL and coupled draft sampling are not
  validated by these measurements. Ngram SSD reads remain allowed.

The machine-readable verification includes executable hashes, exact output
hashes, case coverage and raw diagnostic hashes. Raw local files are in
`bench/results/local-native-batch-mtp-20261011-*`; they are not staged as public
performance claims. See `docs/NATIVE_BATCH_MTP.md` for the optional launcher and
reproduction tools.
