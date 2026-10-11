# Two-slot rollback, 2026-10-11

Restored engine, server, tests and build sources to `34541c37`, before the native
continuous-batching experiment. The source comparison against that commit is
empty for `src/`, `include/`, `serve/` and `CMakeLists.txt`. Later continuous
sequence lifetime/handoff, parked-slot extension, CPU DAG, prefix-page sharing,
token-budget admission and ragged prefill are removed from the active source.
The experiment history and evidence remain available on
`archive/vllm-experiments-20261011` (head `9b27f3f3`).

The earlier double-slot hot-expert promotion correction, independent KV mapping,
device slot copies, v0.1.42 integration and route-tail-skip-off policy remain.
The single-request automatic PCIe path and its existing config were preserved.

Rebuilt with MSVC and CUDA 13 for RTX 5090 SM120 as
`build-native-engine/strata-two-slot-rollback.exe`, then copied byte-for-byte to
`strata-two-slot.exe` after validation. The parallel desktop profile points to
that alias. SHA256:
`3e1d34b276a00190adff0f32cd99b5ed2c611f5fddf5e741ec866f8362a69389`.

Deployment keeps two slots, 262144 context, dedicated 4096-row prefill, INT8 KV,
vision, no MTP weights, adaptation every 4 windows with at most 96 promotions,
temperature 0.7, unrestricted HTTP Host headers and loopback port 8880. The
existing `Start-Strata-Safetensors-Parallel.bat` desktop entry is unchanged.
The separate four-slot scheduler desktop entry was moved to the local rollback
backup directory; its project launchers and configs were removed by restoration.
Original model files and Q4XL deployment were not changed. Hash checks preserve
the single-request configs, saved web settings, learned expert profile and the
other desktop entries, including existing uncommitted local changes.

Validation on RTX 5090 32GB, Ryzen 9950X3D, 96GB RAM, Windows:

- 54 server tests passed, including concurrency, slot allocation, cancellation,
  restart and saved run settings.
- Seven real-model cases passed: greedy, seeded sampling, penalties,
  cancellation/resume, follow-up cache reuse, output caps and EOS. The paired
  greedy/seeded/penalty outputs match their individual references and overlap.
- The runtime logged **98 actual two-row adaptive rounds and 9370 promotions**.
- Final post-residency expert and MTP source bytes are both zero. Ngram reads
  remain separate and allowed.
- Private test engine and vision processes exited. The desktop service was not
  started. This rollback validation is not a new throughput benchmark.

`verification.json` records the deployment and protected-file hashes;
`runtime/results.json`, `runtime-checks.txt` and `server-tests.txt` retain the
checks. Raw logs remain local. HIP and SYCL were not tested.

```powershell
./tools/build_safetensors_engine.ps1 -Jobs 6 -OutputName strata-two-slot-rollback
.venv-native/Scripts/python.exe -m unittest serve.test_parallel serve.test_native_concurrency serve.test_slot_alloc serve.test_control_cancel serve.test_restart_waiters serve.test_runconfig
.venv-native/Scripts/python.exe tools/native_concurrency_acceptance.py --config config/native/rtx5090-262k-parallel.json --exe build-native-engine/strata-two-slot-rollback.exe --output bench/results/local-rollback-check --context 262144 --vision --limit 128 --adaptive --independent-kv --device-slot-copy
```
