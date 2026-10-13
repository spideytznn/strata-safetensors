# Native safetensors backend

This model-specific backend loads the original NVIDIA Qwen3.8-Flash-Next
checkpoint directory with `strata --model <directory>` (`--safetensors` is an
alias). It adapts Strata's existing forward pass, scheduler and conversation
state. The original checkpoint is read-only. The current maintenance scope is
[the desktop inventory](FORK_FEATURES.md) and [cleanup record](DESKTOP_SCOPE.md).
Upstream v0.1.43.1 integration is recorded in [UPSTREAM_V01431](UPSTREAM_V01431.md).
Earlier v0.1.43 validation remains in [UPSTREAM_V0143](UPSTREAM_V0143.md).

## Weight and memory contract

The loader validates fixed model geometry, tensor types, ranges and layouts.
Expert repacking preserves NVFP4 codes and scales. Dense weights remain BF16;
ngram data remain FP8. No main-model GGUF, persistent expert pack or converted
MTP artifact is required. Vision still uses the existing external BF16 mmproj
and vocabulary GGUF sidecars.

All main experts have owned, physically locked RAM copies, including experts
cached in VRAM. This checkpoint's expert arena is about 63.282 GiB. Failure to
lock it aborts startup. The desktop uses `STRATA_NATIVE_ALLOC_PINNED=1` for a
CUDA-mapped arena, avoiding the separate host staging copy. Batched parallel
loading and startup-handle closure remain enabled.

Main expert and loaded MTP source reads are sealed after loading. Runtime
counters measure requested source bytes, not physical SSD traffic. The owned
expert arena cannot be paged out. Ngram rows remain demand-read; explicit
conversation save/restore reads and writes session files.

## Retained computation

- Decode uses NVFP4 weights with FP32 activations/intermediates, CPU AVX-512
  specialization and GPU weight reuse. Existing hot-expert and PCIe scheduling
  remain in use.
- Desktop expert prefill uses W4A4 Tensor Core / CUTLASS auto and BF16-rounded
  dense projections, with an 8192-row workspace allocated before the expert cache.
  The inherited W4A8 and activation-remainder path remain the generic default.
- Desktop KV uses BF16 storage and BF16 QK/PV Tensor Core products with FP32
  accumulation. On-demand growth remains enabled. Generic INT8/FP16 paths remain.
- Unused FP8 KV code was removed; the retained W4A4 kernels serve prefill only.
  Q8 MTP projection and reduced draft-head experiments are not enabled.
  The old `STRATA_NVFP4_TC=0` field is retained for fingerprint compatibility.

Weight preservation is a storage/layout guarantee. It does not make the complete
quantized forward pass byte-identical to an unquantized BF16 model.

## MTP and conversation state

`--mtp native` loads 29 BF16/F32 draft tensors and 512 original FP8 experts.
It uses the full original draft output head. The main model verifies proposals
through the existing rollback and commit path. MTP indexer weights are validated
and retained; draft attention currently follows Strata's dense/window path rather
than using those weights for sparse selection.

The desktop single-request entry uses `--spec 3 --spec-min-p 0.5`: one target row
and up to two draft rows, shortened at low confidence. The parallel entry selects
`--batch-mtp`, which supports one native proposal
per slot, shared draft weights and private KV/graphs, including image histories.
See [native concurrency](NATIVE_CONCURRENCY.md). Native reduced draft-head subsets
remain refused; upstream GGUF batch MTP and vocabulary options remain unchanged.

Native requests reuse KV, GDN and PLE state, parked conversations and disk
snapshots. Two text slots support adaptive hot-expert promotion and independent
KV growth/device slot copying with target-only or native batch MTP. A native single-GPU batch
engine with vision and either of those modes advertises `native_vision_slots=1`: each slot keeps
an independent image position table and matches image identities when reusing
its live state or checkpoints. The server retains exclusive image generation
for older engines and configurations without that capability. Batch slots do
not provide disk slot save/restore.
See [native concurrency](NATIVE_CONCURRENCY.md).

## Serving and launch settings

The two retained launchers are `START-NATIVE-262K-PREFILL-TC-BF16.bat` and
`START-NATIVE-262K-PREFILL-TC-BF16-PARALLEL.bat`; the desktop shortcuts call them.
Both select 262144 context, 8192 W4A4/CUTLASS prefill, BF16 KV Tensor Core,
native MTP, GPU vision and port 8880. The parallel entry adds batch MTP.
See [current desktop scope](DESKTOP_SCOPE.md).
Host headers are unrestricted; the run script binds to `127.0.0.1`.

File sampling defaults are temperature 0.7 for both,
with top_p 0.95, top_k 20, min_p 0, presence_penalty 0 and repetition_penalty 1.
Explicit requests and the existing shared web-settings sidecars can override
file defaults. The template is `config/native/froggeric-v22.5.jinja`; its default
thinking effort is medium. These user settings are preserved by cleanup.

`tools/native_config.py` creates a new local profile without overwriting an
existing file or writing inside the model directory. Upstream `setup.py` is
restored to the official installer; it is not the native desktop updater.
Use [the README source-build route](../README.md#build-and-run-on-windows).

## Evidence and limits

Cleanup is checked against the pre-cleanup executable, using complete token
sequences and main-state fingerprints, cache switching/restoration, cancellation,
zero expert/MTP source reads, two-slot promotion and KV growth. The dated result
is linked from [the cleanup record](DESKTOP_SCOPE.md). No cleanup speedup is claimed.
CUDA is the validated native backend on this Windows/SM120 machine; HIP, SYCL,
Linux and multi-GPU native execution have not been validated here.

Historical results remain archived with their exact profiles and limitations:

- [P10](../bench/results/2026-10-10-safetensors-runtime/p10-native-performance/README.md): startup-handle closure and CPU/GPU/kernel comparisons.
- [P16](../bench/results/2026-10-10-safetensors-runtime/p16-speed-tuning/README.md): repeated prefill/visual profiles, sampled output/state checks and long-context capacity.
- [Stable two-slot rollback](../bench/results/2026-10-11-two-slot-rollback/README.md): retained concurrency and adaptive promotion after scheduler experiments were reverted.
- [v0.1.43](../bench/results/2026-10-11-upstream-v0143/README.md): integration-time validation; removed experiments in that record are historical.

Those measurements are not new measurements of today's combined desktop settings.
Historical source/config commands may refer to deleted experiments; recover them
from their recorded Git revision, rather than treating them as current launch instructions.
Finite output/state checks do not establish broad model quality, retrieval
performance, universal MTP gains or vLLM-level throughput.
