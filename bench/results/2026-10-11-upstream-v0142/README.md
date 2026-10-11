# Upstream v0.1.42 integration evidence

Hardware: Windows, RTX 5090 32 GB, Ryzen 9950X3D, 96 GB RAM; CUDA 13.0,
MSVC 14.44, SM120. Original NVIDIA NVFP4 safetensors, BF16 projections,
INT8 KV, dedicated 4096-row prefill, vision encoder loaded first, resident
experts and explicit 0.25 PCIe share. Tail skipping is off.

Upstream tag: `61b3fb5dd3f1e8ec09cf7e4e05208bc6d3c46406`.
Pre-merge baseline: `c8402280adda6ee26ee5952c5370b70609348f5e`.
Candidate executable SHA256:
`7ccbe8580bc78b95d3747f72ef203b19b612a7ce444e08fc774306d55ae13307`.

## Build and unit checks

- CUDA engine compiled with `tools/build_safetensors_engine.ps1 -Jobs 8
  -OutputName strata-upstream142`.
- Server suite: `python -m unittest discover -s serve -p 'test_*.py'`:
  703 tests, OK, 11 skipped, 252.719 s.
- Selected setup/tokenizer suite: 142 tests, OK, 3 skipped, 27.480 s.
- Twelve host CTest targets pass: logit bias, batch read, split/share rules,
  host limits, draft policy, split ring, native draft vocabulary, draft KV
  plan, batch rows/groups and auxiliary CPUs.
- Eleven conversation/tail-skip CTest targets pass, including GPU snapshot
  copies, session reclamation and separate processes testing unset, explicit
  zero, explicit seven and invalid tail-skip settings.

The first staged startup exposed the GGUF-only ngram fallback reopening an
empty native GGUF path. The second attempt established that native segmented
FP8 reads do not implement mmap. The final candidate keeps native direct I/O
and applies the new probe/fallback to GGUF only. Failed startup records remain
in `native-acceptance/` and `native-acceptance-r2/`; they are not passing evidence.

## Real-model commands

```powershell
python tools/native_concurrency_acceptance.py `
  --config config/native/rtx5090-262k-parallel.json `
  --exe build-native-engine/strata-upstream142.exe `
  --output bench/results/2026-10-11-upstream-v0142/native-acceptance-r3 `
  --context 262144 --vision --limit 128 --adaptive `
  --independent-kv --device-slot-copy `
  --growth-tokens 131000 --capacity-tokens 261000

python tools/native_runtime_acceptance.py `
  --config build-native-engine/upstream142-serial-regression.json `
  --output bench/results/2026-10-11-upstream-v0142/serial-mtp `
  --variants 0,2 --operator-cache --spec-min-p 0.5 --official-new 64
```

The serial test config is copied from the user's current fast config with the
candidate executable and the existing `--mtp native` removed; the harness adds
MTP separately for each variant. Original local configuration files are retained
unchanged and checked against SHA256 backups.

HTTP before/after commands use `tools/native_concurrency_http.py`, the parallel
profile, respectively `strata-concurrency5.exe` and `strata-upstream142.exe`,
`--parallel 2 --rounds 3 --long-tokens 131000 --independent-kv
--device-slot-copy`. Requests use greedy sampling. These are sequential
same-day screens with warm caches and uncontrolled GPU clocks. Repeated-text
long prompts test capacity and state reuse, not retrieval quality. The raw
results and logs, not an extrapolated peak number, are the measurement record.

HIP, SYCL, Linux, multi-GPU and non-Zen-5 CPUs were not built or tested here.

## Native regression results

The full raw acceptance passed: greedy, seeded and penalty sampling, overlapping
slots, cancellation/resume, followups, caps, EOS, two 261000/261007-token slots,
and 124135-token growth/reuse beside a short request. All 35 compared output
vectors match the saved pre-merge baseline. There were 181 actual two-row
adaptive rounds and 15472 promotions. The startup protocol reports
`route_tail_skip=0`, dedicated 4096 prefill and 67948118016 bytes of CUDA-pinned
expert RAM. Post-residency expert and MTP source reads remain zero.

Serial target-only and `--spec 3 --spec-min-p 0.5` each passed 20 completed
requests plus cancellation. Greedy, seeded and configured thinking sampling
outputs match; completed main-model state fingerprints match. Disk save/restore,
A/B/A reuse, EOS and output caps pass. The test used the local fast profile's
256 MiB reserve without changing the original config; startup reported 582/590
MiB free after allocations. Session binaries are local generated artifacts and
are excluded from Git.

## Same-day HTTP comparison

Both builds passed arbitrary Host headers, Prometheus text metrics, 1/2/3
clients, excess-client queueing, disconnect recovery, image exclusion and the
Anthropic API. All 27 compared complete responses match exactly. Three rounds
per client count after reference warmup, with vision loaded and MTP unloaded:

| Clients | Before aggregate tok/s | After aggregate tok/s | Change |
|---|---:|---:|---:|
| 1 | 43.71 | 47.79 | +9.34% |
| 2 | 56.09 | 58.75 | +4.74% |
| 3 | 50.65 | 52.30 | +3.24% |

The first fresh 124135-token repeated-text prompt took 36722/36376 ms
(3380.4/3412.5 tok/s) before/after; its 96-token decode took 1748/1646 ms
(54.9/58.3 tok/s). These are single fresh-prompt observations, not medians.
Aggregate rates include prompt admission and HTTP. Cache state and clocks were
not controlled; these results do not establish a universal speedup.

## Activation and preservation

Both desktop entry names and port 8880 remain. The parallel config now selects
`strata-upstream142.exe` and explicit `STRATA_ROUTE_TAIL_SKIP=0`. It retains
262144 context, 4096 dedicated prefill, INT8 KV, vision, two text slots and no
loaded MTP. The serial entry selects `rtx5090-262k-fast-v0142.json`, copied from
the user's local fast config with the new executable/log path and tail skipping
off. Its `--spec 3 --spec-min-p 0.5` and other settings are retained. The web's
saved sampling sidecar was copied byte for byte to the new profile's sidecar.
All four original local config/backup/settings file SHA256 values are unchanged.
The previous executable and original fast profile remain available for rollback.
