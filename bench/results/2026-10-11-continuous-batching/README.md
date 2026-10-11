# Continuous batching development evidence — 2026-10-11

Hardware: RTX 5090 32 GB (driver 617.14), Ryzen 9 9950X3D, 96 GB RAM,
Windows, CUDA 13, SM120. Original NVIDIA NVFP4 safetensors are loaded directly.
Vision stays loaded, INT8 KV, context limit 262144, dedicated 4096-row prefill,
no MTP, no route-tail skipping. This is finite regression evidence, not a general
model-quality or long-context retrieval evaluation.

## Development records

- `inplace-acceptance2`: in-place target-only batch commit. Two-slot greedy,
  seeded sampling, penalties, cancellation/resume, follow-up, limits, EOS and
  124135-token growth/replacement/restoration passed against GEN token IDs.
- `handoff-acceptance`: rejected. First seeded output differed after a backing
  transfer; retained RAM snapshot chunks still assumed the previous prefix.
  Transfers now invalidate that retained prefix extent.
- `handoff-fixed`: the preceding two-slot and long-context tests passed with
  the RAM-prefix correction.
- `fullstate-4slot`: two/four-slot comparisons passed. The initial BYIELD fixture
  used different checkpoint settings for reference and resume and was corrected
  to preserve the same prefill partition. Do not treat this run as acceptance.
- `continuous6-4slot`: two/four-slot token comparisons and real BYIELD passed.
  Both 261000/261007-token outputs matched GEN, but their decode did not overlap:
  the second request reread the prompt after the first took ownership. Rejected
  as a concurrency acceptance. Subsequent candidates add independent prefix
  cloning from active slots and transfer the bounded checkpoint chain.
- `continuous8-4slot`: short two/four-slot and BYIELD checks passed. Stopped
  intentionally before long-context acceptance when review identified that the
  latest-checkpoint-only policy also had to change. See `SUPERSEDED.txt`.
- `continuous9-4slot`: all checks passed: two/four-slot greedy and seeded
  sampling, penalties, cancellation/resume, follow-up, output caps, EOS, a real
  11441-token BYIELD/resume, 261000/261007-token overlapping slots, and
  124135-token growth/reuse/replacement/restoration. The second near-capacity
  request cloned 245760 prefix tokens from an active slot in 212.3 ms, then read
  its suffix. Both near-capacity outputs matched GEN. Shutdown reports zero
  expert and MTP source bytes; actual two-row hot-expert promotions occurred.
- `continuous10-4slot`: all correctness checks passed. This adds
  idle-destination borrowing, reverse checkpoint-chain transfer and exact-prompt
  slot affinity. Both GEN and BGEN yielded and resumed a real 11441-token prompt
  with identical output. Two 261000/261007-token requests overlapped in a
  four-slot engine; all outputs matched GEN. The 124135-token growth, reuse,
  follow-up, replacement and restoration checks also passed, as did unequal
  four-slot output caps and seeded sampling with penalties. Expert and MTP
  source bytes remained zero, and hot-expert promotion occurred in two-row
  decode. This does **not** establish capacity for four simultaneous 261K
  contexts. See `acceptance10-final.txt` and `continuous10-4slot/results.json`.

`screen9/short-summary.json` separately verifies all four completed short HTTP
runs (old two-slot policy, continuous two/four slots, old two-slot rerun). All
59 corresponding completed responses matched across processes. Both two-slot
policies had essentially the same throughput; the four-slot gain in this screen
also changes slot count and must not be attributed entirely to the new engine
architecture. Baseline before/after medians stayed close. Candidate 9 captured
12 batch graphs with two slots and 24 with four; the old two-slot policy captured
3. This prompted the subsequent idle-destination borrowing and prompt-affinity
changes. The remaining candidate-9 runs were stopped intentionally, after the
short rerun passed and shut down, to test the revised implementation instead.
Any partially initialized `screen9/long-baseline` record is not an acceptance.

Raw JSON contains binary hashes, actual configurations, output IDs or HTTP text,
and timestamps. Candidate names identify separate local builds, not releases.
Do not compare correctness-run timings as controlled throughput measurements:
some builds and CPU checks ran concurrently with those tests.

## Supporting checks

- `conversation_snapshot_test`: 4236 checks, including actual CUDA elastic-pool
  handoffs, incompatible-state rejection, reverse transfer, unchanged unrelated
  pool, full GDN/PLE/indexer bytes and aliases.
- 39 Python parallel, native concurrency, slot allocation and quiet-cancel tests,
  including continuous request lifetime and exact-prompt slot affinity.
- 30 calibration, setup PCIe and run-configuration tests.

Supporting test transcripts are saved beside this file; build and harness logs
are also under the local `build-native-engine/continuous-validation` directory.
CUDA SM120 is built here. HIP and SYCL builds are not available on
this machine; these changes are not presented as validated on those backends.

## Throughput protocol

`tools/native_continuous_screen.py` compares all three new architecture flags
off/on with the same binary, model, sampling and hardware. It includes a
baseline rerun, 1/2/4/8 HTTP clients, two/four slots, and separate mixed-long
prompts. Warm serial reference outputs are checked for each completed response;
separate-process outputs are also compared. Timings include HTTP, admission,
queueing and generation; startup is excluded. Three-round medians are used.
Clocks are not locked and the hot-expert cache adapts.

## Candidate 10 short-request comparison

`screen10-final/short-summary.json` uses candidate 10 for all three runs:
four slots with flags off, four slots with flags on, then flags off again.
All 59 corresponding completed responses are byte-identical across processes.
The baseline below is the pooled median of six before/after rounds; the new
path uses three rounds. Each request generates up to 256 tokens. These are
warm repeated prompts, not cold prefill measurements. PCIe fraction is fixed
at 0.25 in both variants. Startup and reference warmup are excluded.

| HTTP clients | Baseline aggregate tok/s | Continuous aggregate tok/s | Change | Baseline last first-content latency | Continuous last first-content latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 62.34 | 60.66 | -2.7% | 0.157 s | 0.110 s |
| 2 | 75.21 | 78.74 | +4.7% | 0.437 s | 0.233 s |
| 4 | 96.81 | 101.88 | +5.2% | 0.766 s | 0.457 s |
| 8 | 92.40 | 93.77 | +1.5% | 12.338 s | 12.367 s |

The four-client first-content latency falls about 40%; the single-client
throughput is slightly lower. The eight-client throughput difference is small
and its queue wait is unchanged. Do not describe these results as doubled
throughput or as an improvement to the separate single-request desktop profile.

The continuous run also tests cold admission with checkpoint reuse disabled:
a 30263-token body is read while a short request arrives later. The engine
actually yields prompt reading; both responses match their serial references.
The late short request starts returning content after 2.463 s. The long request
takes 12.559 s to first content with this competing work, versus 9.518 s alone.
This demonstrates admission fairness, not faster isolated prefill. The cold
case is separate from the warm-prompt table above.

Candidate 10 was **not deployed**. The later mixed-long screen exposed missing
RAM parking when a cached idle batch slot was replaced. In its two-slot run,
the first three-client round took 47.911 s, with the long request's first content
at 43.302 s; it reread the 124135-token prompt. The pipeline was stopped after
recording this regression. `screen10-final` is the original directory name,
not a claim of final acceptance; its summary intentionally has no passed flag.
Candidate 11 parks replaced idle slots using the existing bounded RAM cache.
The acceptance harness now explicitly requires RAM-prefix restoration after
both ordinary eviction and admission from a different idle source.

The first candidate-11 fixture restored 30058 prefix tokens from RAM with
identical output after replacing all slots. Its second assertion was too
strict for the generated continuation: that continuation ended at admission,
so it never overwrote the destination, and the following request correctly
reused the still-live slot instead of the parked RAM copy. The fixture now
uses a complete new user turn and requires a continued batch admission before
testing destination eviction. The initial record is retained in
`continuous11-4slot` and `acceptance11-initial.txt`; it is not final acceptance.

## Accepted candidate 11 state and output checks

Source commit: `2e2bd372281a4da1326c856822888bade9f0521e` on
`codex/continuous-batching`. Binary and source hashes are in
`build11-manifest.json`. `continuous11-4slot-fixed-fixture/results.json` and
`acceptance11-final.txt` record the complete successful run:

- two/four-slot output equality, including seeded sampling and penalties;
- cancellation, continuation, unequal output caps and EOS at admission;
- ordinary and different-source destination eviction, both restoring 30058
  prompt tokens from RAM with identical output;
- real GEN and BGEN yield/resume at 11441 tokens;
- overlapping 261000/261007-token contexts with outputs identical to GEN;
- 124135-token growth, reuse, follow-up, replacement and restoration;
- actual two-row hot-expert promotions and zero post-residency expert/MTP
  source bytes.

This is a four-slot engine but the near-capacity case contains two long active
contexts, not four. Snapshot budgets can prevent parking a very large context;
the new path follows the existing configured RAM limits. A refused cache park
is a reuse miss, not loss of the active request's state.

Candidate 11's performance screen was stopped after the completed short
two-slot baseline. Code review found that the temporary mount/unmount calls
must pass the owner's token history when reconstructing PLE previous-token
metadata. Candidate 12 corrects those two arguments. No output mismatch was
observed for candidate 11; the correction preserves the state-transfer contract
even when the reader retains a valid prefix. A separate acceptance case now
combines that live reader prefix with eviction of an unrelated destination.
`screen11/SUPERSEDED.txt` records the stop; `screen11` is not final performance
evidence.

## Final candidate 12 correctness acceptance

Source commit: `5e71556c91612c09a9165d69584440b9dcd05279`.
`continuous12-4slot/results.json` passed all 16 cases, including the full
candidate-11 suite and the new live-reader-prefix/destination-eviction case.
`acceptance-final.txt` contains this final transcript; `build12-manifest.json`
contains the exact binary and source hashes. Expert and MTP source reads after
residency are zero. The performance matrix uses this frozen binary.

## Final candidate 12 short-request performance

`screen12/short-summary.json` verifies 59 corresponding completed responses
across all five processes. The two-slot baseline pools six rounds from the
before/after runs; the continuous path uses three rounds. The four-slot
comparison uses three rounds each. All use the same candidate-12 binary,
256-token output cap, original weights, vision loaded, INT8 KV, dedicated
4096 prefill, fixed PCIe fraction 0.25, MTP off and tail skipping off.

| Slots | HTTP clients | Baseline aggregate tok/s | Continuous aggregate tok/s | Change | Baseline last first-content latency | Continuous last first-content latency |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 1 | 64.07 | 63.86 | -0.3% | 0.173 s | 0.107 s |
| 2 | 2 | 79.40 | 83.26 | +4.9% | 0.418 s | 0.224 s |
| 2 | 4 | 79.82 | 83.09 | +4.1% | 6.857 s | 6.454 s |
| 2 | 8 | 77.59 | 81.23 | +4.7% | 20.467 s | 19.620 s |
| 4 | 1 | 62.72 | 61.45 | -2.0% | 0.166 s | 0.110 s |
| 4 | 2 | 74.71 | 79.10 | +5.9% | 0.468 s | 0.234 s |
| 4 | 4 | 95.30 | 101.65 | +6.7% | 0.878 s | 0.458 s |
| 4 | 8 | 90.07 | 101.72 | +12.9% | 12.975 s | 10.812 s |

These are warm repeated synthetic prompts and end-to-end HTTP rates, including
admission and queueing. The raw summary includes ranges. The isolated
single-request desktop entry is a different profile and is not represented by
the one-client rows above. Short-request throughput is not doubled. Keeping
the same slot count shows modest throughput gains and a larger first-content
latency reduction. Comparing new four-slot throughput against old two-slot
throughput also changes memory allocation and scheduling capacity.

## Final candidate 12 mixed-long performance

`screen12/long-summary.json` compares two-slot serving, three rounds per case.
The first request has a 124135-token formatted prompt; additional clients use
short prompts. These are warmed repeated conversations with a 256-token output
cap. Startup and initial serial reference prefill are excluded. All 27
corresponding completed responses match across processes.

| HTTP clients | Baseline aggregate tok/s | Continuous aggregate tok/s | Change | Baseline last first-content latency | Continuous last first-content latency |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 55.23 | 60.00 | +8.6% | 0.414 s | 0.142 s |
| 2 | 63.33 | 75.56 | +19.3% | 1.338 s | 0.280 s |
| 3 | 64.37 | 64.46 | +0.1% | 7.563 s | 7.300 s |

Three clients still queue behind two slots; aggregate throughput there is
essentially unchanged. These results do not show faster isolated prefill or
universal decode gains. The new path changes scheduling, state movement and
memory use together; this matrix is not an ablation of the three flags.

In this mixed-long workload the maximum **logged KV-pool mapping** falls from
4752 MiB to 2064 MiB (2688 MiB less). These are KV pool mappings, not total
process VRAM or a guarantee for arbitrary contexts. Each run performs exactly
one full read of the 124135-token prompt, during its initial reference; later
requests reuse it. The final implementation therefore avoids candidate 10's
eviction/reread regression.

The separate cold-admission cases use a 30263-token body with checkpoint reuse
disabled. A late short request begins returning content in 2.572 s (two slots)
or 2.594 s (four slots). The long request's first content takes 12.356/12.700 s
with competition versus 9.266/9.557 s alone. Both requests match their serial
references and logs confirm a real prefill yield. This is fairness evidence,
not an isolated-prefill speedup.

## Final compatibility and deployment

`screen12/summary.json` reports `passed: true` for the complete seven-run matrix.
The previous deployed `strata-upstream142.exe` also matches all 29 corresponding
completed responses in its separate compatibility run. That one-round run is
an output-compatibility check, not the baseline used for the tables above.

At 2026-10-11 14:23 Hong Kong time, the accepted executable was copied byte for
byte to `build-native-engine/strata-continuous-batching.exe`, SHA-256
`87b1fc740dbc67c9b031c946b3ab2cfa884f006fccc6afc0ead6830af1153a2b`.
The desktop parallel entry's existing JSON now selects this executable and
sets the three new flags to `1`. It retains two slots, 262144 context, dedicated
4096 prefill, INT8 KV, vision, MTP off, route-tail skipping off, the PCIe settings,
port 8880 and unrestricted Host headers. The separate single-request and Q4XL
profiles and saved web settings were verified unchanged.

`deployment.json` records before/after hashes and the exact backup:
`build-native-engine/continuous-release-20261011-142327/rtx5090-262k-parallel.json`.
Restoring that backup to `config/native/rtx5090-262k-parallel.json` restores the
prior local executable and flags. The new desktop service was not started;
the benchmark engines use private ports and close on completion.

Reproduce the final correctness and HTTP matrix with the installed alias:

```powershell
.venv-native/Scripts/python.exe tools/native_concurrency_acceptance.py --config config/native/rtx5090-262k-parallel.json --exe build-native-engine/strata-continuous-batching.exe --output bench/results/local-continuous-acceptance --slots 4 --context 262144 --vision --adaptive --independent-kv --device-slot-copy --limit 128 --growth-tokens 131000 --yield-tokens 20000 --capacity-tokens 261000 --eviction-tokens 32768
.venv-native/Scripts/python.exe tools/native_continuous_screen.py --config config/native/rtx5090-262k-parallel.json --exe build-native-engine/strata-continuous-batching.exe --reference-exe build-native-engine/strata-upstream142.exe --output bench/results/local-continuous-screen --mode full --rounds 3 --limit 256 --long-tokens 131000 --cold-admission-tokens 32768
```

Output directories must be new. These commands consume the GPU and RAM; they
were run without a production server or another timed GPU workload active.
