# Native safetensors concurrency

The CUDA native backend can decode two text requests in one target-model window
with `"parallel": 2` in the server config (`--batch 2` / `--slots 2` in the
engine). This is opt-in. The desktop parallel entry enables native batch MTP;
the local single-request entry uses solo native MTP. See the dated
[feature/profile inventory](FORK_FEATURES.md) for current settings.
The measurements below retain their historical executables and configurations.

The Windows [prefill overlap path](NATIVE_PREFILL_OVERLAP.md) can run native
prompt processing alongside live-slot decode. The current parallel desktop
profile enables it; the generic engine default is off.

## Image histories and progress

Native single-GPU batches can register a stable M-RoPE table for each
slot. Admission copies the main request's table into that slot. Query/key norms
and rotations and pooled indexer forward/commit use the slot's table, including
when two requests have the same token position. Projections and expert matrix
operations retain the existing multi-row path. Tables for two 262144-cell slots
add approximately 6 MiB of VRAM (the exact allocation includes the engine's cell
padding).

Image identity (grid and embedding hash) is retained with cached slot tokens and
checkpoints. A later turn containing the same historical images can restore that
state; equal image-pad token IDs alone cannot match a different picture. Image
prefill can yield at existing chunk boundaries and resume from the slot. The
existing in-memory parked-conversation budget and eviction rules still apply.

The engine reports `native_vision_slots=1` only for this supported configuration.
The Python server uses the existing exclusive image gate for older binaries,
solo-MTP batch configurations and layer splits. Merely loading the vision encoder does not
mean every request contains images.

Parallel monitor progress now comes from the request that consumed the engine's
control lines. A waiting request reports that it is queued and does not display
another request's completed prefill. Request-scoped progress also applies to
requests outside batch slots.

`vision_slot_positions_test` checks captured image/image and text/image rotations
and pooled indexer state against separate calls, bit for bit. The server gate
and monitor tests run without a model in `serve.test_native_concurrency`. Full
model acceptance uses generated images rather than user conversations:

```powershell
.venv-native/Scripts/python.exe tools/native_vision_concurrency_acceptance.py --config config/native/rtx5090-262k-parallel.json --exe build-native-engine/strata-vision-cache.exe --output bench/results/local-vision-cache-acceptance
```

That command requires the desktop service to be stopped to free model memory.
Compilation and small synthetic GPU tests alone do not establish full-model
parity or a speedup. On 2026-10-12, the full model also passed private-stdin tests
with both 16K/512 and desktop 262144/4096 context/prefill allocations on the
RTX 5090, Ryzen 9950X3D, 96 GB Windows CUDA machine. Two different image grids
in 141098- and 126147-token prompts matched their single-request token outputs
in parallel. Their continuations reused 141193 and 126242 tokens. A cold image
request yielded at 8192 of 141106 tokens and resumed there; its warm repeat
reused 141099 tokens. Expert file reads remained zero. These are correctness
checks with generated pictures and repetitive text, not a quality or speed
benchmark. This change has not been validated on HIP or SYCL.

The restored desktop HTTP service also passed image-history and thinking-history
continuation with two decoding slots. A 141092-token request with its image before
the long text yielded at 12288, resumed there, and reused 141085 tokens on repeat
with the same output. See the [dated repair record](../bench/results/2026-10-12-vision-slot-cache/README.md)
for counts, exact settings and limits.

## Local launch

Use `START-NATIVE-262K-PREFILL-TC-BF16-PARALLEL.bat`, or the desktop
`Start-Strata-Safetensors-Parallel.bat`. The current configuration is
`config/native/rtx5090-262k-prefill-tc-bf16-parallel.json`: BF16 KV Tensor Core,
8192 W4A4/CUTLASS prefill, 262144 context, native/batch MTP, original expert
scheduling, vision, continuous prefill and independent elastic KV. It uses
`strata-classic-bf16-kv.exe` and localhost port 8880. See
[current desktop scope](DESKTOP_SCOPE.md). Stop another server before starting it.

The dated sections and commands below record earlier engines and configurations.
Old INT8/4096 trial launchers and configurations have since been removed;
those performance measurements do not describe the current BF16 hybrid.

Excess requests wait for a free slot. Admissions share one prompt reader, with
decode opportunities between prefill chunks. In-memory checkpoint and parked
conversation reuse remain available; disk `/slots/0` save/restore is unavailable
while batch mode is enabled.

## Optional native batch MTP

`--mtp native --batch 2 --batch-mtp --spec 3 --spec-min-p 0.5` enables the
existing slot verifier with native draft weights. The main drafter owns one
shared copy of the original BF16/F32 and FP8 weights and full output head.
Each slot owns its draft KV, activation scratch, residual rows and graph inputs.
Each slot proposes at most **one** token per batch window, regardless of the
solo `--spec` setting. Two proposing slots therefore verify four rows. A slot
below its request's confidence threshold verifies one row; the other slot can
continue proposing. Length/EOS/cancellation still apply per request.

Draft graphs capture each slot's stable image position table. Independent
elastic KV includes resident draft pools, while streamed draft KV retains its
existing snapshot fallback. Target slot migration can still use device copies.
The default submits and waits for each slot's draft using the upstream interface.
`STRATA_NATIVE_BATCH_MTP_ASYNC=1` opts into submitting both private streams
before joining completion, for a same-build A/B. It does not change target
verification or confidence gating. Draft work still
runs in low-confidence windows to obtain the next confidence estimate.

The current desktop parallel entry explicitly selects this option. MTP consumes
VRAM that could otherwise hold hot experts; acceptance rate alone does not prove
a throughput improvement. The private benchmark retains actual output tokens,
loads vision before expert-cache sizing, clones the same initial expert profile,
and measures warm end-to-end and decode throughput with both slots active:

```powershell
.venv-native/Scripts/python.exe tools/native_batch_mtp_bench.py --config config/native/rtx5090-262k-parallel.json --exe build-native-engine/strata-native-batch-mtp-final.exe --output bench/results/local-native-batch-mtp-bench
```

On RTX 5090 32 GB, Ryzen 9950X3D and 96 GB RAM, the same-day private-stdin
comparison used 262144 context, a dedicated 4096-row prefill workspace, GPU
vision loaded before cache sizing, INT8 KV, the same cloned initial expert
profile, adaptive promotion, temperature 0.7, thinking enabled and seed 9950.
Each result below is the median of two warm repeats after one retained warmup;
each pair emits 256 tokens per slot. Total throughput counts the 512 actual
emitted tokens over wall time including both admissions. It is not the engine's
verify-row counter.

| Pair | MTP unloaded | Shared native MTP | Change |
|---|---:|---:|---:|
| Short text | 81.24 tok/s | 91.05 tok/s | +12.1% |
| Code | 81.69 tok/s | 93.43 tok/s | +14.4% |
| 8K document, cached repeat | 81.59 tok/s | 90.39 tok/s | +10.8% |

The first uncached document pair was 42.00 versus 42.04 tok/s including prefill;
this is no demonstrated cold-prefill improvement. A separate greedy screen
found no clear extra benefit from overlapping draft streams, so it stays opt-in.
All retained outputs matched token for token across each screen's modes. These
finite checks are not a broad quality evaluation or a guarantee for other prompts
and context lengths. The MTP allocation takes about 2.7 GiB from the hot-expert
cache on this configuration; experts remain resident in locked RAM.

See [the validation record](../bench/results/2026-10-12-native-batch-mtp/README.md)
for configurations, raw-run references and limits. Generic engine/server defaults
remain target-only unless the user supplies the MTP flags. Remove `--mtp native`
and `--batch-mtp`, and set `--spec 2`, to unload MTP in the desktop profile.

## State, sampling and memory

### Adaptive exchange correction

Inspection after the initial runs found that `strata-concurrency2.exe` accumulates
routing heat in batch windows but does not periodically schedule new adaptive
rounds there. Solo generation still exchanges experts. Initial logs' cumulative
promotion counts therefore do not prove promotions during two-row decode.

The correction was built separately as `strata-concurrency3.exe` and accepted
on this PC on 2026-10-11 with MTP entirely unloaded and vision loaded. The initial
acceptance data below use `concurrency2`; the new records are separate.
The native batch boundary now reuses the existing hot/cold swap policy every
`--adapt-every` windows, with at most `--adapt-swaps` promotions per round (4 and
96 in this profile). Heat includes routed entries from all active rows. Copies
complete before the next window; exchanges are deferred while prefill borrows
expert-cache slots. The dedicated workspace profile does not take that loan.
The correction uses a synchronized boundary.

CUDA SM120 compilation and GPU parity passed, including 128 tokens per sampling
pair, cancellation/resume, continuation, limits and EOS. The run logged 98 actual
two-row adaptive rounds and 9367 promotions in those rounds. Post-residency expert
and MTP source bytes were zero. The acceptance harness's `--adaptive` option
retains configured adaptation and requires nonzero promotions from logged
`active_rows=2` rounds:

```powershell
.venv-native/Scripts/python.exe tools/native_concurrency_acceptance.py --config config/native/rtx5090-262k-parallel.json --exe build-native-engine/strata-concurrency3.exe --output bench/results/local-batch-adaptive --context 262144 --vision --limit 128 --adaptive
```

### Per-request state

Each slot has its own GDN recurrence/conv history, PLE history and previous token
pair, QSA KV/indexer and committed token list. The target verifier's batch commit
now applies the same opt-in canonical QSA spare-row cleanup as its solo commit.
Native windows stage a separate penalty tail for each row, including greedy
repetition/frequency/presence penalties. Temperature, top_p, top_k, min_p, seed
and Philox position are taken from that row's request. Legacy batch callers that
do not supply row histories retain their existing sampler path.

The native elastic KV registry can include the main session and both slots.
The original path initially maps 16384 cells, grows all those pools to their common required
length, and recycles physical pages from the expert cache. A short admission
must retain the longest active, yielded or cached slot's cells. Growth is also
checked before each batch window. Virtual addresses and graph pointers remain
stable; the original resident RAM expert mirrors remain available when a GPU
expert slot is surrendered. This first implementation maps all slot pools to
the same length, rather than growing each slot independently.

On the recorded 262K two-slot run, the combined elastic KV pools initially used
0.72 GiB against 9.54 GiB at full capacity. The two sessions' non-KV/initial
allocations were reported as 1.12 GiB total. These are overlapping accounting
views, not amounts to add together. There is still less room for GPU experts
than in the serial profile; concurrency can reduce single-request speed.
The dedicated prefill workspace stays allocated before expert-cache sizing.

### Independent KV and direct slot copies

Two optional native, target-only paths are implemented in `strata-concurrency5.exe`:

- `STRATA_NATIVE_KV_INDEPENDENT=1` grows and shrinks only the selected session's
  QSA pools. The main prompt reader and each slot keep their own mapped extent.
  A destination grows before cached KV is copied into it; a replaced slot can
  release pages beyond its new conversation. Other live/cached slots retain
  their pages. Growth is checked before every batch window. Physical pages still
  come from the existing expert-cache VMM allocator; graph addresses remain stable.
- `STRATA_NATIVE_SLOT_COPY=1` transfers the fully resident KV, GDN and PLE state
  directly between device allocations. It avoids temporary host snapshots when
  migrating a request into/out of a slot. An earlier checkpoint still restores
  its saved running state, then copies the authoritative KV from the live slot.
  RAM conversation snapshots and their file format are unchanged.

Both flags default off and require native target-only or native batch-MTP batching. Direct
copies require fully resident device KV (mode 0); they do not implement streaming
or draft-ring migration. No projection, expert arithmetic, quantization or
attention kernel changes are involved. Source and destination remain idle until
copies finish. Copy failure uses the existing admission refusal/slot invalidation
or full-prompt recomputation path; partial destination state is not committed.

The combined paths passed raw-protocol GPU acceptance with the vision encoder
loaded: greedy/seeded/penalized sampling, cancellation/resume, followups, limits,
EOS, two overlapping 261000/261007-token slots with 256 output tokens each, and
a 124135-token conversation beside a short request. Live-slot continuation
matched solo; after both slots were replaced by short conversations, the long
request was admitted again and matched solo. That last admission reread its
prompt rather than hitting the RAM cache, so it does not prove a RAM-cache hit.
175 actual two-row adaptive rounds promoted 15279 experts; post-residency
expert and MTP source bytes stayed zero.

At full capacity, the three target KV groups mapped 9504 MiB (9.28 GiB) with MTP
unloaded. With just the main group full and two initial 16384-cell slots, they
mapped 3648 MiB. With one 131072-cell slot and the main/other slot shrunk to
8192, they mapped 1872 MiB (1.83 GiB); replacing that slot with a short one
returned the total to 432 MiB. These are mapped KV bytes, not total GPU memory.
Two full-length slots still need their full KV; the savings apply to unequal
lengths and unused state. The historical 9.54 GiB figure above included draft KV.

4219 CUDA snapshot checks passed, including byte equality of direct copies
against the snapshot path at page/block boundaries, different KV formats,
rotation mismatch refusal, and another session's size/bytes surviving selected
growth and shrink. Server/gate/config tests also passed. CUDA SM120 was built;
HIP/SYCL were not built or validated on this PC.

Before the image-slot fix described above, vision kernels used a shared mutable
M-RoPE position table. The server therefore allowed text requests to share the
engine but gave image requests exclusive
generation, after existing text slots and cancellation drains finished. Waiting
images took precedence over newly arriving text. That guard remains for engines
without `native_vision_slots=1`, including MTP and layer-split configurations.
The external encoder can prepare the image before generation gets its permit.

## Validation on this PC

RTX 5090 32 GB, Ryzen 9950X3D, 96 GB RAM, native Windows CUDA SM120,
2026-10-11. These initial runs loaded MTP for solo requests, before it was
disabled in the desktop parallel profile. Raw data and commands are in
[`bench/results/2026-10-11-native-concurrency`](../bench/results/2026-10-11-native-concurrency/README.md).

- Raw protocol checks compare solo and two-slot outputs exactly for English and
  Chinese/code prompts, greedy decoding, different seeded temperatures/top-k/
  top-p, and repetition/frequency/presence penalties. Short output limits,
  cancellation/resume and subsequent turns are exercised.
- The 262K configuration passes those checks with the vision encoder loaded.
  A 22777-token fixture grows the common KV mapping to 24576 cells; its output
  and later continuation match solo after a short request is admitted.
- Two near-capacity prompts of 261000 and 261007 tokens each generate 256
  tokens matching solo, with overlapping output intervals (6.488–24.715 s and
  16.532–27.110 s from admission). The common KV mapping reaches 262144 cells
  and 9.54 GiB; the GPU expert cache falls to 128 slots. These repeated-text
  fixtures establish tested capacity/state coexistence, not retrieval quality.
- Real HTTP SSE checks cover one, two and three simultaneous clients over the
  OpenAI route, an Anthropic request, client disconnection while another request
  runs, recovery afterwards, and a red image between text requests. Text matches
  the corresponding serial responses; the image returns red. HTTP usage retains
  cached input tokens. Each round's answer cap is 96 target tokens.
- Runtime diagnostics report zero expert file bytes; engine shutdown reports
  zero post-residency expert and MTP source bytes. These are logical source
  counters, not physical-disk telemetry. Ngram SSD reads remain allowed.

These are finite regression tests against the same NVFP4 target, not a broad
quality benchmark or proof of equivalence to unquantized BF16. Native HIP/SYCL,
other GPUs, multi-GPU and more than two native slots have not been validated.

## Target-only latency and throughput screen

After the adaptive correction, `strata-concurrency3.exe` was tested with MTP
entirely unloaded, INT8 262K KV, 4096 dedicated prefill, vision loaded, reserve
1536 MiB and adapt 4/96. Only the server's parallel setting differed. Each client
requested 96 greedy target tokens; three rounds followed reference warmups.
These medians include admission and HTTP time.

| Simultaneous clients | Serial aggregate tok/s | Two-slot aggregate tok/s | Serial last first-content latency | Two-slot last first-content latency |
|---|---:|---:|---:|---:|
| 1 | 64.67 | 61.43 | 0.136 s | 0.141 s |
| 2 | 58.10 | 73.27 | 1.640 s | 0.420 s |
| 3 | 55.19 | 62.24 | 3.511 s | 2.866 s |

Two-client aggregate throughput rose 26.1% and three-client throughput 12.8%;
the one-client rate fell 5.0%. The measured two-client first-content wait fell
74.4%. All 24 shared completed HTTP responses matched across configurations.
Both runs passed disconnect/recovery and Anthropic checks; the parallel run also
passed text/image exclusion. Expert source bytes stayed zero. Raw records are
`http3-no-mtp-serial`, `http3-no-mtp-parallel` and
`no-mtp3-http-summary.json` in the evidence directory.

These sequential, warm-cache screens do not control GPU clocks or establish a
peak rate. Startup varied from 70.878 to 482.566 seconds as source-read time
changed; startup is excluded from the request measurements.

## Independent-KV mixed-length HTTP screen

The same `strata-concurrency5.exe` was run with both new flags off, then both
on, with two slots, MTP unloaded and the same profile above. The first fixture
has 124135 formatted tokens of repeated text; the other two are short code/tree
questions. Each request asks for 96 greedy tokens. Three measured rounds follow
reference warmups; repeated long prompts reuse 124128 input tokens. These are
warm conversation measurements, not fresh 124K prefill measurements.

| Clients | Flags off aggregate tok/s | Flags on aggregate tok/s | Flags off last first-content latency | Flags on last first-content latency |
|---|---:|---:|---:|---:|
| 1 | 42.47 | 47.11 | 0.587 s | 0.367 s |
| 2 | 44.33 | 51.97 | 1.260 s | 1.273 s |
| 3 | 42.21 | 49.84 | 4.776 s | 3.865 s |

Measured aggregate gains were 10.9%, 17.2% and 18.1%. The two-client last
first-content wait was essentially unchanged (+1.0%); the three-client wait
fell 19.1%. All 27 shared completed responses matched, including disconnect
recovery, image exclusion and Anthropic. Both shutdown logs report zero expert
and MTP source bytes. The same binary SHA256 is recorded in both results.

Common KV growth mapped 4.64 GiB at 131072 cells in the flags-off run.
With flags on, one 131072-cell group plus two 8192-cell groups mapped 1872 MiB;
two long groups plus a short one mapped 3312 MiB. Main/slot copies can temporarily
require both long groups. Freed pages return to the GPU expert cache.

Direct copies do not eliminate allocation work. The median logged slot
admission rose from 56 to 107 ms because admissions also grow/shrink their own
pools and adjust the expert cache. Median slot restoration fell from 41.9 to
22.05 ms. End-to-end gains above include those costs.

Raw records are `http5-long-baseline`, `http5-long-optimized` and
`optimized5-long-http-summary.json`. Runs were sequential; clocks were not
controlled, cache placement adapted and startup (104.458/61.728 s) is excluded.
These figures do not establish general speed or retrieval quality.

The same off/on comparison also passed with all three prompts short. For
1/2/3 clients, median aggregate rates were 65.26/75.00/66.02 tok/s off and
66.18/78.43/69.75 on (+1.4%/+4.6%/+5.6%). Last first-content latency was
0.134/0.429/2.679 s off and 0.124/0.376/2.574 s on. Median slot admission was
63.4 versus 22.8 ms; restoration was 35.1 versus 27.9 ms. All 27 completed
responses matched, and the same HTTP/residency checks passed. Raw records are
`http5-short-baseline`, `http5-short-optimized` and
`optimized5-short-http-summary.json`. The same measurement limits apply; the
small short-request differences are not a guaranteed gain.

The separate local parallel desktop profile now opts into both paths. Their
engine defaults remain off; the serial entry is a separate target-only profile.

## Historical latency screen with solo MTP

Before disabling MTP in the desktop parallel profile, the same executable was
run serially and with two slots, using 262K INT8 KV,
4096 dedicated prefill, vision loaded, native MTP for solo requests, 1536 MiB
reserve, greedy sampling and adaptive expert exchange. Three rounds per client
count followed reference warmups. Each client requested 96 tokens. Values below
are medians; throughput includes prompt/admission and HTTP time.

| Simultaneous clients | Serial aggregate tok/s | Two-slot aggregate tok/s | Serial last visible first-content latency | Two-slot last visible first-content latency |
|---|---:|---:|---:|---:|
| 1 | 79.19 | 61.25 | 0.131 s | 0.144 s |
| 2 | 71.34 | 48.05 | 1.438 s | 0.455 s |
| 3 | 76.94 | 71.65 | 2.519 s | 2.744 s |

The two-client last first-content wait fell in this screen, while aggregate
throughput and the one-client rate fell. Three clients did not improve their
last first-content wait. These were sequential profile runs, not interleaved
paired trials; cache state adapted and GPU clocks were not recorded. They are
functional/latency screening evidence, not a stable peak-throughput claim.
The serial desktop entry remains the choice for one user's maximum speed.

## Continuous prefill with idle batch slots (2026-10-13)

`STRATA_NATIVE_PREFILL_CONTINUOUS=1` opts into a single-GPU safetensors prompt
path for a batch-enabled engine with no slot decoding. The default remains off.
It reads a whole prompt segment in one `Prefill::run`, retaining the existing
chunk size, MTP callbacks and checkpoints. This keeps next-chunk PLE host reads
overlapped with the current chunk's GPU work instead of restarting that prefetch
at every outer scheduler piece. It does not change weight or KV precision.

The engine still checks for `BYIELD` at every completed chunk. A cooperative
finish runs the normal prefill epilogue before the scheduler copies the partial
session into a slot: PLE token history is committed and outstanding copies are
drained. A speculative next-chunk PLE read is joined before returning. STOP
continues to use the existing cancellation checks. With a decoding slot, peer
GPU, layer split or pipelined groups, the existing scheduling path is retained.

The executable is `build-native-engine/strata-continuous-prefill.exe`. Following
the user's activation request, the desktop `Start-Strata-Safetensors-Parallel.bat`
entry selects it through `START-NATIVE-262K-OVERLAP.bat` and
`config/native/rtx5090-262k-prefill-overlap.json`, with the continuous switch on.
The existing overlap and branch-preservation switches remain enabled, as do the
other desktop settings. Port 8880 is unchanged; activation did not start a server.
The previous configuration is saved in
`work/codecheck/continuous-prefill/desktop-overlap-before.json`.
The same-source pre-change executable is `strata-continuous-baseline.exe`.

`tools/native_continuous_prefill_test.py` compares that baseline with the
candidate disabled/enabled, using private engine pipes and copied expert
profiles. It checks first-token logits, generated tokens, long-prompt run
lengths, a mid-prefill BYIELD followed by an intervening request and slot
restoration, and STOP followed by a retry. It opens no HTTP listener.

On the RTX 5090 / Ryzen 9950X3D / 96 GB Windows machine, the same candidate
binary with the switch off/on measured the following prompt times. Both used
two batch slots but one request, 64K INT8 KV, 4096-token chunks, native MTP,
fixed 4000 requested expert-cache slots, all missed experts on the GPU, no
adaptive swaps or vision, and a copied identical initial expert profile.
Tracing and prefill timing were enabled in both runs. These are single samples
after a short warmup, not medians or a general throughput guarantee.

| Prompt | Switch off | Switch on | Prompt time change |
| --- | ---: | ---: | ---: |
| 8192 tokens, repeated report text | 2861.9 ms | 2874.5 ms | +0.4% |
| 32768 tokens, mixed DETAILS.md text | 13943.2 ms | 12007.7 ms | -13.9% |

The independently compiled pre-change source measured 2831.9 / 13687.9 ms
respectively. The candidate with the switch off matched its output tokens and
first-token logits byte-for-byte on all four 8K/32K/24K/20K requests. With the
switch on, the same comparisons passed, including a 24K prompt yielded after
8192 tokens, an intervening side request, and restoration of all 8192 tokens
from the slot. STOP interrupted a separate 20K prompt after 8192 tokens; retry
matched the uninterrupted reference. The largest continuous run processed
32761 tokens while retaining 4096-token internal chunks.

Evidence: `bench/results/local-continuous-prefill-20261013-r2/results.json`
and per-run logs. The initial run stopped during startup because the harness
combined fixed expert-cache size with `STRATA_NATIVE_PREFILL_FIRST=1`; r2
corrected this to 0 consistently for all three engines. CUDA compilation and
the above real-model checks passed; HIP, SYCL, multi-GPU and vision are not
validated for this change. No HTTP listener was started.

The existing `native_concurrency_acceptance.py` suite also passed with the
switch enabled, independent KV and device slot copying. Its ten cases covered
greedy/seeded/penalized sampling, mixed and strict draft-confidence thresholds,
cancel/resume, followups, small output caps, EOS, and a 7801-token long slot
growing/reusing KV beside a short admission. Concurrent outputs matched solo
references. This suite uses its required prefill-first/automatic-cache setup;
its initial fixed-cache attempt stopped before inference and was corrected.
Records: `bench/results/local-continuous-prefill-concurrency-20261013-r2`.

Candidate SHA256:
`100c2e506b1f4a2d929328236bfad2591b0abd258ac73f1fac49503d9f69f120`.
