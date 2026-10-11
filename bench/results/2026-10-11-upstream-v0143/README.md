# Upstream v0.1.43 native regression record

Machine: Windows, RTX 5090 32 GB, Ryzen 9950X3D, 96 GB RAM; CUDA 13.0,
MSVC 14.44, SM120. Model: original NVIDIA NVFP4 safetensors at
`D:\迅雷下载\Qwen3.8-flash-next-nvfp4`, with the existing GPU vision encoder.

Upstream: `7695859aa13afb248bdb4d36f9b2e95707db079d` (`v0.1.43`).
Local base: `36902c911f21adf1fc18c4364772108f907ae8f0`, after reverting the
unified-expert experiment. The candidate is `build-native-engine/strata-upstream143.exe`.
Hashes and compact results are in [verification.json](verification.json).
Raw logs, model outputs and session snapshots remain local generated artifacts.

## Build and tests

```powershell
tools/build_safetensors_engine.ps1 -OutputName strata-upstream143 -Targets @('strata')
.venv-native/Scripts/python.exe -m unittest discover -s serve -p 'test_*.py'
.venv-native/Scripts/python.exe -m unittest tools.test_strata_bench tools.test_make_profile_mix tools.test_setup_hotfix_tag tools.test_setup_sycl tools.test_setup_amd tools.test_setup_choices tools.test_setup_nvfp4 tools.test_setup_fork_engine tools.test_safetensors_tokenizer
```

- CUDA engine build passed.
- Service suite: 719 tests, OK, 12 skipped.
- Tools suite: 173 tests, 171 passed, two pre-existing installer failures.
  Both failures reproduce when the harness loads `setup.py` from local base
  `36902c91`: `Keep.test_a_card_the_release_has_no_code_for` and
  `PageFile.test_check_and_install_say_it_framed_and_again_at_the_end`.
- 20 rebuilt CTest checks passed: conversation cache/split failure/memory/file,
  save reclaim/validation/snapshot, batch rows/groups, draft policy/vocabulary,
  native NVFP4 CPU parity, file expert source, mapped copies, interleaved GEMV,
  resident routing and four tail-skip policy checks.
- Mapped-copy coverage includes 375 NaN/positive-infinity/negative-infinity
  graph replays at 1/2/3/6/8 rows and widths 1/255/257/4096/248320, plus 25
  finite replays after poisoned rows. Logits and guard bytes are unchanged;
  every written flag is checked. The test covers the last row of three-row
  native MTP windows and clearing flags between graph replays.

CTest logs are `build-native-engine/upgrade-v0143-backup/ctest-cpu.log` and
`ctest-gpu-r2.log`. Targets were rebuilt after enabling `STRATA_BUILD_TESTS`
and `STRATA_BUILD_CONVERSATION_TESTS`; stale test binaries were not used.

## Original-model checks

Private engines are loaded sequentially. They use context 262144, dedicated
prefill 4096, INT8 KV, vision and full expert routing. The desktop service on
port 8880 is not started by validation.

`tools/native_runtime_acceptance.py` compares the previous `strata-pcie-auto.exe`
against the candidate, then compares candidate MTP off against spec 3 with
`--spec-min-p 0.5`. Each variant covers English, Chinese, code, arithmetic,
seeded sampling, official thinking parameters, output caps, EOS, A/B/A cache
reuse, disk save/restore and cancellation/resume. All 20 complete token vectors,
the cancelled prefix and 20 completed main-model state fingerprints match the
previous executable. Deliberately interrupted state is excluded from the state
comparison because STOP can arrive after different amounts of in-flight work.
Expert and MTP source reads after residency are zero.

```powershell
.venv-native/Scripts/python.exe tools/native_runtime_acceptance.py --config build-native-engine/upgrade-v0143-backup/single-before.json --output bench/results/local-upstream143-single-before --variants 0 --operator-cache --spec-min-p 0.5 --official-new 64
.venv-native/Scripts/python.exe tools/native_runtime_acceptance.py --config build-native-engine/upgrade-v0143-backup/single-after.json --output bench/results/local-upstream143-single-after --variants 0,2 --operator-cache --spec-min-p 0.5 --official-new 64
.venv-native/Scripts/python.exe tools/native_concurrency_acceptance.py --config config/native/rtx5090-262k-parallel-mtp.json --exe build-native-engine/strata-upstream143.exe --output bench/results/local-upstream143-dual-mtp --context 262144 --vision --adaptive --batch-mtp --limit 128 --growth-tokens 40000 --capacity-tokens 40000
.venv-native/Scripts/python.exe tools/native_concurrency_acceptance.py --config config/native/rtx5090-262k-parallel.json --exe build-native-engine/strata-upstream143.exe --output bench/results/local-upstream143-desktop-parallel --context 262144 --vision --adaptive --independent-kv --device-slot-copy --limit 128
```

The dual-MTP run passed nine case groups: greedy, seeded, penalties,
cancel/resume, followup, caps, EOS, two long slots and long-slot replacement/
restore. Both slots accepted proposals while overlapping; confidence gating
reduced windows and three-row windows ran. Long slot prompts were 40000 and
40007 tokens; the growth/reuse case used 37933 tokens. There were 83 actual
two-slot adaptive rounds with 7496 expert promotions. Post-residency expert
and MTP source bytes were zero, and the engine exited with code 0.

The actual desktop parallel profile is checked separately, retaining its local
MTP arguments without adding `--batch-mtp`. This covers serial MTP followed by
ordinary overlapping batch windows, with the configured KV/copy options.
The seven short case groups, adaptive promotion and post-residency source-read
checks pass. The dedicated parallel-MTP profile remains the explicit way to
request per-slot drafting.

These finite checks do not establish general model quality, near-capacity
long-context retrieval quality or a throughput gain.

## HTTP and fault recovery

The private HTTP suite uses two slots, independent KV and direct device slot
copying, with MTP unloaded. It checks arbitrary Host headers, metrics, serial/
parallel response equality, excess-client queueing, disconnect recovery,
vision exclusion and the Anthropic API. The ephemeral HTTP listener binds to
127.0.0.1. Test-only `STRATA_NAN_INJECT=1` reports the first verifier window as
non-finite; the server must retry before emitting a token and the engine must
invalidate its poisoned cache. The injection is not added to desktop profiles.
The real-model suite passed and the logs confirm the one retry and cache flush.

```powershell
.venv-native/Scripts/python.exe tools/native_concurrency_http.py --config build-native-engine/upgrade-v0143-backup/http-after.json --exe build-native-engine/strata-upstream143.exe --output bench/results/local-upstream143-http --parallel 2 --rounds 1 --independent-kv --device-slot-copy
```

## Activation and limits

The two desktop entry scripts continue to call the same repository launchers.
The fast, historical fast, parallel and opt-in parallel-MTP profiles select
the new executable. Only `exe` changes: sampling, spec/minimum confidence,
context, prefill, vision, reserve, Host and port settings remain as found.
Pre-existing local profile edits are kept outside the merge commit. All old
profile executable hashes remain unchanged, and backups stay under
`build-native-engine/upgrade-v0143-backup/`.

`STRATA_ROUTE_TAIL_SKIP` stays off. Windows adaptive overlap is not enabled
without a separate measurement. HIP/SYCL builds, Linux, multi-GPU and other
GPU generations were not validated here. No new speedup or general quality
claim is made. The original model, Q4XL deployment and Git remotes are not
modified by this validation; the upgrade is committed locally without pushing.
