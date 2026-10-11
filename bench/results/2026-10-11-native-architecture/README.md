# Native scheduler architecture evidence

Hardware: RTX 5090 32 GB, Ryzen 9950X3D, 96 GB RAM, Windows, CUDA 13, SM120.
Model: original Qwen3.8-Flash-Next NVIDIA NVFP4 safetensors. Dedicated 4096-row
prefill, original BF16 projection weights, INT8 KV, no MTP, route-tail skip off.

Implementation and reproduction: [scheduler documentation](../../../docs/NATIVE_BATCH_SCHEDULER.md).

## Curated files

- `cpu-dag-screen.json`: six completed two-/four-client warmed decode runs from
  binary architecture20, off/on/off, two rounds of 128 output tokens. Replies
  match across variants. The later protocol-mismatched cold test from that
  orchestrator is excluded; the original local `screen20/summary.json` records
  that failure.
- `prefill-screen21/summary.json`: completed four-client cold-request off/on/off
  screen, two rounds, natural arrivals and actual owner eviction. This measures
  request wall time, including prefill and 128-token generation. Ragged replies
  match their individual canonical references, but differ from baseline replies.
- `acceptance.json`: combined final acceptance and source-read audits, using the
  six passed stages from `final21/summary.json` plus the corrected visual stage
  in `final21-vision-retry/summary.json`. Exact commands and results cover released
  arithmetic parity, packed owners, full-context scheduler regression,
  same-binary physical-prefix sharing off/on, eight-slot budget/root checkpoint
  admission and real image exclusion. The first visual attempt correctly received
  `PREFDONE 0 0 0` but its harness compared an unstripped newline. The original
  failed summary is retained; only that stage was rerun after fixing the harness.
- `dag-final-low.txt`: 144 byte comparisons; timings are a microbenchmark, not
  an end-to-end speed claim. In this repetition the single-expert case regressed.
- `conversation-final-low.txt`: 4236 conversation-snapshot checks.
- `server-final-tests.txt`: 55 service tests, including restart, cancellation,
  token-budget admission, collector coalescing and checkpoint boundaries.
- `canonical-vs-original-shapes.json`: diagnostic comparison isolating dense
  projection shape changes while both sides already use complete-K W4A8. It is
  **not** a released-binary comparison. Worst measured first-step KL is 0.01740;
  this supports keeping ragged prefill experimental, not claiming quality parity.
- `packed-root-turn21/results.json`: the additional turn-checkpoint run with a
  2945-token shared system root; both owners' first-step logits and 32 generated
  tokens match their individual canonical references.

Architecture21 executable SHA256:
`d4b0b2e8c067169616268ee45ca854efd6e553790e57cdc67b7b79dd89b5fb44`.
Released reference `strata-continuous12.exe` SHA256:
`87b1fc740dbc67c9b031c946b3ab2cfa884f006fccc6afc0ead6830af1153a2b`.

Raw logits, layer dumps, sessions, images and other development attempts remain
local. Do not sum `physical_alias_bytes` across operations and call that peak
VRAM savings. The paired prefix test records unique physical KV peaks instead.
All expert SSD assertions refer to post-residency execution, not initial loading;
ngram SSD reads and requested conversation restoration are separate.

## Interpretation

The CPU DAG keeps each dot product's arithmetic and passed byte comparisons.
Ragged prefill changes reduction and projection shapes; within its canonical
path the tested long owners' logits and greedy replies agree with individual
execution. These are finite regression tests, not a broad model-quality study.
The live-token budget limits admissions rather than total allocated VRAM.
Neither four slots nor eight slots promises a particular throughput multiplier.
HIP, SYCL, multi-GPU, pipeline windows and batched MTP were not validated here.
