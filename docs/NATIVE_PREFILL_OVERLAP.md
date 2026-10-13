# Windows native prefill/decode overlap

`STRATA_NATIVE_PREFILL_OVERLAP=1` lets an existing batch slot decode while the
native prompt path reads the next request on its own thread and CUDA stream.
It is off by default. The existing desktop configurations and deployed
`strata-v01431.exe` are unchanged.

The opt-in entry is `START-NATIVE-262K-OVERLAP.bat`, using
`config/native/rtx5090-262k-prefill-overlap.json` and
`build-native-engine/strata-continuous-prefill.exe` on `127.0.0.1:8880`.
The desktop profile also enables [continuous prefill while slots are idle](NATIVE_CONCURRENCY.md#continuous-prefill-with-idle-batch-slots-2026-10-13).
The current opt-in profile enables parked checkpoint branch preservation too
(described below). The preceding `strata-prefill-overlap-final.exe` and its
profile backup remain available.
Stop the other model service before starting it. The original parallel entry
remains available for rollback.

At the user's request, the desktop `Start-Strata-Safetensors-Parallel.bat`
now calls the overlap entry. Its previous contents are backed up under
`build-native-engine/prefill-overlap-backup-20261013/`. The separate single-request
desktop entry is unchanged. This desktop routing change does not start a service.

## Execution

The main session owns prefill state; live batch slots keep their independent
target and draft states. Prefill and its main-session MTP prompt KV preparation
run on the worker. Decode uses a high-priority CUDA stream. The main thread
still performs request progress and conversation checkpoints after joining the
chunk, preserving the existing checkpoint and parked-conversation paths.

Prefill has an independent PLE reader. Its row cache is capped at the smaller
of the configured budget and 65536 rows; it does not duplicate the full default
1048576-row decode cache budget. This changes cache capacity only, not PLE rows
or model weights.

Expert placement stays fixed while the worker runs. Before each overlapped
chunk, the main thread reserves live slots' KV headroom on a quiet device.
If a slot runs out of mapped headroom, it waits for the worker before any VMM
mapping or expert residency changes. Every error path joins the worker before
state or buffers are reused.

This path currently requires native safetensors, one GPU, independent session
KV, dedicated fixed-size prefill, `STRATA_PREFILL_CPU_SHARE=0`, and synchronous
expert adaptation. Layer splits, pipelined windows, borrowed prompt scratch,
and asynchronous expert adaptation are refused when the opt-in path is used.
It does not introduce shared-prefix pages or change the existing two-slot limit.

The desktop opt-in profile retains 262144 context, 4096 prefill, INT8 KV,
native batch MTP, GPU vision, 8 GiB conversation cache and the current desktop
sampling and tail-skip settings. No default is changed based on these tests.

## Build and checks

```powershell
tools/build_safetensors_engine.ps1 -OutputName strata-continuous-prefill
```

The full-model acceptance uses all routed experts (tail skip 0), frozen expert
placement, PCIe fraction 0 and no prefix reuse for cold numerical checks.
The performance profile retains the current desktop settings, including tail
skip 7. Do not mix these two sets of numbers.

Finite validation and the before/after mixed-load measurements are recorded
under `bench/results/local-windows-overlap-final-20261013/`. Earlier iterations
remain under `local-windows-overlap-opt-20261013/` and
`local-windows-overlap-opt-v2-20261013/`; their binary hashes and scope differ.
The pre-change reference is `local-windows-prefill-overlap-20261013/`.

CUDA SM120 is the available build/test backend on this machine. HIP, SYCL,
multi-GPU, and the full 262K context under overlap have not been validated.
An SSE text chunk can contain a different number of tokens on different paths;
visible chunk intervals are not GPU single-token latency.

## Preserving a parked checkpoint's original branch

`STRATA_CONVERSATION_BRANCH_PRESERVE=1` is a separate opt-in switch for
single-stage parked conversation restores. When another request matches only
a checkpoint of a parked long conversation, the engine copies the matching
prefix's running checkpoints and returns the original full image to the RAM
cache. Without the switch, taking that checkpoint consumes the original
image; a later return to its long branch can re-read most of the history.

The new branch gets independent prefix state. The original KV payload moves
back with its original image; it is neither copied nor lent to the new
branch's mutable retained-buffer pool. Full-live continuation keeps its
existing path. The normal parked-byte and slot limits still apply, including
superseded-copy pruning and capacity eviction.

Prefix-state copying uses the physical-RAM floor already configured by
`--conversation-cache-min-free-mib`, checked before and after allocation.
Unknown/insufficient headroom or allocation failure falls back to the old
restore path and logs `branch preservation skipped`. Preserving an original
image costs a parked slot and keeps its bytes charged to the existing budget;
it does not make all branches immune to later capacity eviction. Layer-split
restores keep their existing path.

The investigation and CPU ownership replay are under
`bench/results/local-kv-reread-investigation-20261013/`. Full-model validation
of the branch-preservation build is recorded separately under
`bench/results/local-kv-branch-preserve-20261013/`; the earlier overlap speed
table applies to the preceding binary and is not a new measurement of this
build. This change does not alter model weights, precision, sampling or MTP
verification.
