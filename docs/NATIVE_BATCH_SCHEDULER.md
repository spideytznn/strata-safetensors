# Native scheduler extensions

These extensions build on the native continuous-batching implementation. They
are opt-in. The existing desktop executable and launch profiles are unchanged.
They do not convert weights, prune routed experts,
enable MTP, or change the INT8 KV format.

## Controls

Add these values to a **copy** of the native server config's `env` object. Keep
`STRATA_NATIVE_BATCH_INPLACE`, `STRATA_NATIVE_KV_HANDOFF`,
`STRATA_NATIVE_CONTINUOUS`, `STRATA_NATIVE_KV_INDEPENDENT` and
`STRATA_NATIVE_SLOT_COPY` enabled. `parallel` sets the maximum number of slots,
from 2 to 8. Native continuous decode remains target-only.

| Setting | Default | Effect |
|---|---|---|
| `STRATA_NATIVE_EXPERT_DAG` | `0` | Combine native NVFP4 CPU gate/up and down jobs into one dependency queue. |
| `STRATA_NATIVE_KV_TOKEN_BUDGET` | `0` | Limit the sum of live prompt plus requested output tokens; zero disables this admission limit. |
| `STRATA_NATIVE_RAGGED_PREFILL` | `0` | Collect pending text requests into bounded multi-owner prefill chunks. |
| `STRATA_NATIVE_PREFIX_PAGES` | `0` | Alias complete immutable KV chunks when cloning a cached prefix; copy on write. |

Check `INFO` for `expert_dag`, `kv_token_budget`, `ragged_prefill` and
`prefix_pages`. A configured value alone does not prove that a supported
execution path was selected. The tested combination is Windows, one RTX 5090,
native Qwen3.8-Flash-Next NVFP4, dedicated 4096-row prefill, INT8 KV and no MTP.
HIP, SYCL, multi-GPU and pipeline windows have not been validated for these
extensions. Prefix sharing is disabled for multiple pipeline windows.

Two additional launchers leave the existing single/Q4XL/parallel entries alone:

- `START-NATIVE-SCHEDULER.bat`: four slots, CPU DAG, physical prefix sharing,
  and a 524288-cell live-token budget. Ragged prefill is off.
- `START-NATIVE-SCHEDULER-RAGGED.bat`: the same limits with experimental ragged
  prefill enabled. Use this entry only when accepting the arithmetic difference
  described below.

Both use context 262144, prefill 4096, INT8 KV, vision, no MTP, host checking `*`,
loopback binding and port 8880. Stop the other server before using either entry.
The budget admits up to four short requests or two full-context requests; it
does not promise four simultaneous full-context requests. The inherited
`--spec 2` does not enable MTP without an MTP model; native continuous decode is
target-only.

The launchers expect `build-native-engine/strata-scheduler.exe`. To reproduce
that artifact from source on the configured Windows development machine:

```powershell
./tools/build_safetensors_engine.ps1 -Jobs 4 -OutputName strata-scheduler -Targets strata,native_prefix_pages_test,native_expert_dag_test,conversation_snapshot_test
```

## CPU expert dependencies

Decode already groups rows that route to the same expert. The new CPU queue
keeps the existing AVX-512 dot products and their reduction order, publishes each
expert's completed gate/up rows with an atomic counter, and lets its down rows
consume the FP32 intermediate directly. It removes the second pool-wide phase
barrier and the intermediate copy. All gate/up producers are claimed before any
down consumer, so waiting consumers cannot starve unclaimed producers.

Task weights account for four-row activation reuse rather than treating an
eight-token expert as eight independent weight reads. Both the original pool
and the queue remain available for same-binary comparison. Pool diagnostics name
overlapping DAG time instead of reporting a misleading split-stage bandwidth.

## Admission by token budget

Before reserving a slot, the server charges `prompt_tokens + max_tokens`, capped
by the context limit and rounded up to 8192 cells. An unlimited output request
is charged the whole context. Admission checks and reservations run under the
same slot lock. A request larger than the configured budget fails rather than
waiting forever. Queued requests release the engine control lock while waiting.

This is a **logical live-token budget**, not a hard VRAM limit. Idle cached
conversations, fixed GDN state, indexer buffers, workspace, graph captures and
expert residency have separate costs. Creating eight slots still allocates
their fixed state even if the live-token budget permits only four active
requests. `/v1/status` live slots expose `reserved_kv_tokens` for diagnosis.

## Ragged prefill

The server collects arrivals briefly, and includes arrivals that finished
tokenizing while it waited for the current reader. It reserves slots using the
same token-budget rules, divides one prefill workspace between the owners, and
releases control between chunks. An unfinished owner keeps its reservation;
new requests can join later chunks. Completed owners continue through normal
`BGEN` admission and target decode. Rejected packs fall back to ordinary
admission. Cancellation, generator abandonment and engine restart release the
associated reservations.

The internal protocol is:

```text
BPREF slot@position:token_ids;slot@position!:token_ids
PREFDONE owners rows milliseconds
```

The optional `!` retains a root, turn or periodic checkpoint; an unmarked
transient chunk snapshot replaces the preceding transient snapshot. The server
checks `INFO ragged_boundaries=1` before using the marker, so an older engine's
protocol remains usable. Root and turn boundaries divide each owner's work;
the existing root-pinned/LRU retention policy applies. Explicit pinned prefixes,
tail-role and message-boundary policies fall back to ordinary admission, as do
short segments that require the FP32 reader.

The engine validates the complete command before changing state. Each owner
has private GDN recurrence, convolution history, PLE history, absolute positions,
KV and QSA indexer state. PLE ngram lookup resets at owner boundaries. Expert
routing and products combine rows and reuse the staged expert weights.

Numerical stability required two additional measures. Opt-in ordinary and
packed prefill both use 256-row dense projection blocks, including the PLE
projections; the coordinator uses aligned quotas. W4A8 expert MMQ performs each
complete K reduction in one CTA, avoiding a shape-dependent stream-k fixup
order. Neither changes the weight or activation format. They **do change
floating-point arithmetic relative to the released prefill path**. Do not
describe this path as byte-identical to that release or infer general model
quality from a small greedy-output test.

`STRATA_NATIVE_RAGGED_DENSE=1` rejoins dense GEMMs for investigation. It bypasses
the canonical projection shape and has not passed the numerical gate; leave it
unset. Prompts at or below `--short-read` retain the original FP32 verifier
path. Image requests remain exclusive; packed text is refused while the shared
M-RoPE table contains image positions, until ordinary text admission resets it.

## Physical prefix pages

The CUDA VMM allocator retains fixed virtual addresses for captured graphs, but
several owners can map the same immutable physical handle. It shares only
complete logical KV pages and complete physical chunks. Partial chunks, GDN,
PLE and pooled-indexer state remain private. INT8 codes and their FP16 scale
arrays are tracked independently.

Before a verifier, prefill or RAM restoration writes, the allocator detaches
every overlapping shared chunk and copies its bytes. Resets detach the retained
working set, zero it, and shrink excess mappings. Replaced unique backing goes
back to the expert/KV recycler. Removing an alias never recycles a handle that
another owner still reads; only its last owner can return that handle.

Prefix cloning installs aliases **before** growing the destination. Otherwise
temporarily allocating a full private duplicate would evict experts even when
sharing could have avoided that allocation. Pool accounting counts unique
physical chunks. The log's `physical_alias_bytes` counts an individual clone
operation; sums across operations are not simultaneous VRAM savings.

## Validation and reproduction

Evidence is kept under
`bench/results/2026-10-11-native-architecture/`. The hardware is RTX 5090 32 GB,
Ryzen 9950X3D, 96 GB RAM, Windows, CUDA 13, SM120. Original safetensors were used.
All completed real-engine runs report zero expert and MTP source bytes after
residency; ngram reads and explicit conversation restoration are separate.
The final combined record is `acceptance.json`: seven passed stages and eight
engine source-read audits, including both executables in the release comparison.
The original visual harness's newline-comparison failure remains recorded in
`final21/summary.json`; `final21-vision-retry` contains the corrected stage. No
engine code changed for that retry.

Current checks include:

- With all four new switches off, nine solo/batch outputs and their complete
  first-step logit arrays are byte-identical to `strata-continuous12.exe`.
- Sixteen full-context scheduler cases pass with CPU DAG and prefix pages on:
  seeded sampling, penalties, unequal output limits, cancellation/followup,
  owner replacement, RAM restoration, actual GEN/BGEN yield/resume, 261000 /
  261007-token simultaneous owners, and growth beside a short admission.
- 144 CPU comparisons against the original pool, byte-identical across task
  counts, worker counts, host participation and experts with unequal token counts.
- VMM source/branch isolation, stable addresses, last-owner recycling, source
  eviction, reset shrinking, and INT8 code **and scale** COW at 131100 cells.
- Four real-model 131100-token prefix forks, identical 64-token outputs, then
  replacement of every owner, RAM restore and another fork.
- The final same-binary sharing off/on comparison also passes four 261000-token
  forks and replacement of every owner followed by RAM restore. Both variants'
  reference outputs agree. Unique physical KV peaks are 12912 / 4128 MiB;
  minimum expert-cache capacities are 570 / 3858 slots, respectively.
- Ragged prompts of 346/484 tokens, 2049/3073/785 tokens and 8193/2305 tokens,
  including multiple nonzero-position appends. Within the canonical prefill
  path the long owners' first-step logits are byte-identical and their greedy
  generations match their individual runs. Very short prompts exercise a
  different FP32 reader and are compared with an explicit KL gate.
- A shared 2945-token system root with 4763/4161-token requests is split at the
  root/turn boundaries. Both packed owners have byte-identical first-step logits
  and matching 32-token generations against their individual canonical runs.
- A real HTTP eight-slot run with a 32768-token live budget: eight clients
  queued, at most four owners active, peak reserved tokens 32768. Outputs
  matched each individual reference; disconnect recovery and Anthropic serving
  passed. This proves admission behavior, not eight-way throughput superiority.
- 43 parallel/native-slot/control tests and 12 restart tests, including packed
  collectors waiting across engine restart.

Run private acceptance with the new binary, leaving the desktop service stopped:

```powershell
python -m unittest serve.test_parallel serve.test_native_concurrency serve.test_slot_alloc serve.test_control_cancel serve.test_restart_waiters
build-native-engine\native_expert_dag_test.exe
build-native-engine\native_prefix_pages_test.exe
python tools/native_packed_acceptance.py --config local-native.json --exe build-native-engine\strata.exe --output bench\packed-check
python tools/native_prefix_acceptance.py --config local-native.json --exe build-native-engine\strata.exe --output bench\prefix-check --tokens 131100 --slots 4
python tools/native_architecture_screen.py --config local-native.json --exe build-native-engine\strata.exe --output bench\scheduler-screen
```

The screen compares the same executable with the new flags off/on/off. The
existing continuous lifetime, ownership handoff and in-place commit stay on.
CPU-DAG output must match across variants. Cold-prefill rounds first replace
every actual slot owner, so cached admissions cannot masquerade as prefill.
No existing desktop default is changed on the strength of a kernel microbenchmark.

## Measured request performance

Same-binary off/on/off comparisons used the hardware above, target-only decode,
INT8 KV, original BF16 projection weights and dedicated 4096-row prefill. Vision
was disabled for these throughput screens. These are aggregate output tokens
divided by request wall time, not isolated prompt-processing rates.

| Workload | Before / after baseline | New path | Change against baseline mean |
|---|---|---|---|
| Two warmed clients, 128 output tokens each, two rounds | 85.28 / 83.81 tok/s | CPU DAG: 86.36 tok/s | +2.1% |
| Four warmed clients, 128 output tokens each, two rounds | 108.82 / 108.94 tok/s | CPU DAG: 112.62 tok/s | +3.4% |
| Four cold clients, about 1.7–2K prompt tokens each, 128 output tokens each, two rounds | 55.80 / 55.43 tok/s | Ragged prefill: 65.16 tok/s | +17.2% |
| Last client's first-token wait in the cold workload | 4.751 / 4.755 s | Ragged prefill: 3.484 s | −26.7% |

The warmed CPU-DAG screen left adaptive expert promotion enabled. Its logs show
different VRAM/CPU routing shares between arms (about 72–76% VRAM hits in sampled
two-client windows). The small +2.1% and +3.4% differences therefore do not isolate
the queue change from cache variation. They do not demonstrate a noticeable
improvement over the previous two-client warm throughput in the eighties.
Future CPU comparisons must reuse one private learned profile, fix expert-cache
capacity, disable promotion and check identical output and routed expert counts.
`tools/native_hot_decode_screen.py` performs that off/on/off comparison (CPU DAG
by default). A subsequent paired-row CPU experiment with vision allocated gave
81.00 / 86.72 / 87.13 aggregate tok/s over the overlapping two-client decode
interval. Since the following original baseline matched the enabled path, that
experiment was rejected and its source changes removed. Its byte comparisons,
frozen routing audit and full result are archived under
`bench/results/2026-10-11-hot-decode-pair2/`. It does not establish a warm-decode
speed improvement.

CPU-DAG replies were identical across all three variants. Ragged replies matched
their individual canonical references but differed from the released arithmetic
path. This is not evidence of unchanged model quality; the path remains opt-in.
Every completed screen reported zero expert SSD source bytes. Cold rounds
replaced all actual slot owners and used natural arrivals, without holding the
control lock to manufacture a burst.

The CPU screen used `strata-architecture20.exe`; its six decode runs are valid,
but its later cold screen failed because that binary predates the checkpoint
protocol marker. Those cold results are excluded. The completed cold screen
used `strata-architecture21.exe`, SHA256
`d4b0b2e8c067169616268ee45ca854efd6e553790e57cdc67b7b79dd89b5fb44`.
Evidence: `screen20/summary.json` and `prefill-screen21/summary.json` under the
evidence directory. These small samples describe these workloads; they do not
establish a general throughput multiplier or a single-request speed increase.

The four-owner 261000-token shared-prefix comparison uses architecture21, vision
off, admission budget disabled, 64 output tokens and the same 5105-slot initial
expert cache. Sharing reduces
peak **unique physical KV** from 12912 to 4128 MiB (8784 MiB, about 8.58 GiB).
This excludes private GDN/indexer state, workspace and model/expert allocations;
it is not a reduction in total GPU usage by that amount in every workload.
Unrelated prefixes cannot obtain this saving. The minimum available expert-cache
capacity rises from 570 to 3858 slots. Four-fork request wall time is 3.205 /
2.364 s and eviction/restore wall time is 2.581 / 2.242 s, but this is one off/on
pair, not a repeated speed estimate. Both variants preserve the tested outputs
and report zero expert SSD source bytes. See `final21/prefix261k-before` and
`final21/prefix261k-shared` result files.
