# Native continuous batching

Further opt-in CPU scheduling, token-budget admission, ragged prefill and physical
prefix sharing are described in [scheduler extensions](NATIVE_BATCH_SCHEDULER.md).

The native CUDA path has three opt-in architecture changes for target-only
serving. They use the original safetensors weights and existing forward kernels.
MTP and route-tail skipping are not required.

## Request lifetime

`STRATA_NATIVE_CONTINUOUS=1` makes the engine report `continuous_batch=1`.
The Python server admits each text request with BGEN, even when it is alone.
It retains that slot until completion or cancellation. A later arrival joins
the batch; an earlier completion removes its row from the next window. The
server no longer stops and reissues a request to move between GEN and BGEN as
the number of clients changes. Image requests still take the exclusive GENI
path because the vision position table is shared.

A repeated text prompt prefers the idle slot that served that prompt. The full
held conversation includes the reply, so the older prefix-only slot heuristic
could miss this case and move the turn checkpoint to another slot. The engine
still validates the checkpoint; this preference does not cache generated output.

The default remains the existing solo-promotion/demotion policy. Continuous
mode is reported only for native, single-GPU, target-only batch serving.

## Sequence state ownership

`STRATA_NATIVE_KV_HANDOFF=1`, with `STRATA_NATIVE_KV_INDEPENDENT=1`, changes
admission and idle-slot restoration from copies into ownership transfers.
The prompt reader and request slots each hold a view of a physical sequence:

- KV codes/scales and their elastic allocation IDs;
- GDN recurrent state and convolution history;
- PLE history and its preceding-token pair;
- indexer tails, pooled keys and spare-row state.

At a synchronized boundary, these pointers change owners together. Forward
scratch and RoPE tables remain attached to their executors. Admission does
not copy sequence buffers or remap physical KV pages. A checkpoint rewind
still restores its saved recurrent/indexer state, and a parked RAM conversation
still needs restoration. This is exclusive ownership, not shared writable KV.

CUDA Graphs contain addresses. Main-reader graph bundles therefore follow the
physical backing ID; batch graph keys include the backing IDs of their rows.
Independent KV growth follows the new owner. The idle reader can return excess
pages to the expert cache instead of retaining a duplicate long KV sequence.

For an unrelated admission or a prefix cloned from an active slot, the reader
borrows the chosen idle destination's backing before filling it. Admission hands
it back to that slot. This avoids growing an extra long reader pool while an
idle destination already owns storage, and keeps ordinary slot graph bindings
stable. A valid reader prefix and an idle source selected for continuation are
preserved instead of being overwritten by this borrowing step.

A replaced idle slot is parked through the existing bounded RAM conversation
cache before its storage is reused. The same protection applies when a request
continues a different idle source into the chosen destination. Without this,
cycling more conversations than slots can discard a long prompt and force a
full reread. Parking still obeys the configured byte/slot limits and physical
RAM floor; under memory pressure it can be skipped, as on the serial path.

The old owner's token/checkpoint identity is invalidated. Retained RAM snapshot
chunks also lose their reusable-prefix extent on transfer: their prefix cannot
be assumed to describe the new backing. A regression initially caught this
case as a seeded-output mismatch; the failed record is retained in the evidence.

A request can also reuse a prefix held by an active slot. At a quiescent boundary,
the reader copies that prefix into its own storage, then restores the matching
recurrent/indexer checkpoint and reads the new suffix. The source stays active
with independent writable state. This avoids rereading a common long prompt
without introducing shared KV pages or copy-on-write. An idle source still uses
the ownership transfer above.

The bounded checkpoint chain follows a transferred sequence in both directions, rather than keeping
only its last turn checkpoint. A sibling request may diverge earlier in the
prompt. These are CPU recurrent/indexer snapshots; retaining more than one per
slot can use more host RAM than the old one-checkpoint slot policy.

## Ordinary batch decode

`STRATA_NATIVE_BATCH_INPLACE=1` records a separate target-only execution path.
When every row belongs to a distinct slot, its input token is unconditionally
accepted. The forward pass updates GDN state and history directly. It leaves
the accepted PLE and indexer state in place, avoiding snapshots and the second
GDN replay/commit graph and host synchronization.

Rows proposing multiple tokens for one slot retain the existing speculative
snapshot/commit implementation. The standalone single-request verifier's
`STRATA_ONE_TOKEN_COMMIT` setting is unchanged. No projection weights, expert
selection, activation formats or sampling parameters change with these flags.

## Scope and validation

This is continuous batch decode with transferable sequence storage. Admissions
still share one prompt reader and active decode slots run between prompt
chunks. It does not implement ragged multi-request prefill GEMMs, shared-prefix
KV pages or the complete vLLM PagedAttention scheduler.

Use `tools/native_concurrency_acceptance.py` for real GEN/BGEN token comparisons,
unequal output lengths, cancellation, cache reuse, near-capacity KV and BYIELD.
`tools/native_continuous_screen.py` runs same-binary HTTP comparisons with all
three flags disabled/enabled, two/four slots, excess-client queuing, vision,
Anthropic and disconnected-client recovery. Raw evidence is saved under
`bench/results/2026-10-11-continuous-batching`.

Only CUDA SM120 on the RTX 5090 is being validated here. HIP, SYCL and multi-GPU
are not validated by these runs. Finite token and state checks establish those
tested paths, not general equivalence to unquantized weights.

## Local build and measured profile

Build the CUDA executable with:

```powershell
tools/build_safetensors_engine.ps1 -Jobs 4 -OutputName strata-continuous-batching
```

The local `rtx5090-262k-parallel.json` enables all three flags with two slots,
independent KV and device slot copying. The existing desktop parallel launcher
uses that config and port 8880. Its context, prefill, sampling, vision, PCIe and
MTP settings are retained. Generic configs remain opt-in; an absent flag does
not enable these changes. The native single-request and Q4XL entries are
unchanged by this deployment.

The accepted binary matches `strata-continuous12.exe` byte for byte; source
commit, SHA-256, pre-change config backup and deployment checks are recorded
in [the evidence](../bench/results/2026-10-11-continuous-batching/README.md).
The performance matrix compares the same binary with the three flags off/on.
Short two-slot throughput improves about 4-5%, with single-client throughput
essentially unchanged; mixed-long two-request throughput improves about 19%.
Three mixed-long clients queue behind two slots and have essentially unchanged
throughput. Four-slot short-request results are separate from this local default.
