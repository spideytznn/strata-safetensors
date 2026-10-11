# Experimental native two-slot MTP

This branch adds an opt-in `--batch-mtp` path for original NVIDIA NVFP4
safetensors. It reuses Strata's slot drafters, batch verifier and prefix commit;
it does not change the model's weights or replace the forward kernels.
The stable `main` engine and existing desktop launchers are separate.

The trial profile is `config/native/rtx5090-262k-parallel-mtp.json`, launched by
`START-NATIVE-262K-PARALLEL-MTP.bat`. Its relevant options are:

```text
--batch 2 --mtp native --spec 3 --spec-min-p 0.5
--batch-mtp --kv int8 --prefill 4096 --max-context 262144
```

`parallel: 2` adds `--batch 2` through the server. `--spec 3` permits a target row
and up to two draft rows per slot. Each slot has its own draft KV, residuals,
probabilities and request sampling state. Draft weights and the head are shared.
Both slots can propose and verify drafts in the same target batch, rather than
using MTP only while a request is alone. The confidence floor can shorten each
slot independently; no minimum draft count is forced. All emitted tokens are
target picks, and only the matching prefix commits into recurrent/PLE state.
Caps, cancellation and EOS stop that prefix. Request-local `spec_min_p` is
also honored.

The native path now supplies the correct number of penalty-history and residual
rows to each slot. Per-slot diagnostics report offered/accepted proposals,
proposals offered while both slots are active, confidence skips and the largest
window. `INFO native_batch_mtp=1 mtp_enabled=1 mtp_slot_drafters=2` confirms the
mode. The older `mtp_max=0` field is an optional window-cap setting, **not** an
indicator that MTP is unloaded; zero means use `--spec` when a drafter is loaded.

The profile preserves original BF16 projections, original FP8 MTP experts, the
full draft head, all routed expert contributions, dedicated prefill space and
resident expert RAM. It explicitly leaves coupled draft sampling off; that
optional sampling algorithm has not been validated by this experiment.
The existing independent-target-KV and device slot-copy optimizations are
currently guarded off when MTP is loaded. Thus long contexts can consume more
physical KV memory and admission can cost more than the target-only profile.

## Build and verify

On the supported Windows / CUDA 13 / SM120 machine:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File tools/build_safetensors_engine.ps1 -OutputName strata-native-batch-mtp
.venv-native/Scripts/python.exe tools/native_concurrency_acceptance.py --config config/native/rtx5090-262k-parallel-mtp.json --exe build-native-engine/strata-native-batch-mtp.exe --output bench/results/local-batch-mtp-check --context 262144 --vision --limit 128 --adaptive --batch-mtp
```

The acceptance run checks overlapping greedy and seeded outputs, penalties,
cancellation/continuation, conversation reuse, caps/EOS, real adaptive
promotions, confidence reduction, three-row windows and zero post-residency
expert/MTP source reads. These are finite regression cases, not a general
quality evaluation. HIP and SYCL builds/runtime are not validated here.

`tools/native_batch_mtp_benchmark.py` compares private engines without starting
HTTP or editing launch configs. It performs two warmups, three measured rounds
and long-slot parity/reuse checks; `--reference` also compares the exact outputs
against a target-only run. Rates include admission and reuse, exclude startup,
and use greedy sampling. Startup cache sizes and executable hashes are recorded.
Run the original target-only engine again after the MTP run to check drift.
An optional benchmark profile is not evidence that MTP improves every workload.

Measured results and their scope are recorded under
`bench/results/2026-10-11-native-batch-mtp/`.
