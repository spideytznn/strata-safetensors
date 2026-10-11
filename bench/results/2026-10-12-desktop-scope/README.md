# Desktop-scope cleanup — 2026-10-12

Local branch `cleanup/desktop-native-20261012`, before commit `c6f52f10`.
Official baseline: Strata v0.1.43, `7695859aa13afb248bdb4d36f9b2e95707db079d`.
Machine: Windows, RTX 5090 32GB, Ryzen 9950X3D, 96GB RAM, CUDA 13 / SM120.
This is a correctness/maintenance record with no new throughput claim. The verified cleanup was subsequently prepared for the project main branch at the user's request.

The cleanup retains the two actual native desktop entries, their runtime dependencies
and regression tools. It removes unused fork-only FP4/W4A4 experiments, Q8 MTP
projections, native batch MTP/reduced draft head, old conversion/requantization and
portable publishing/update code. There are 68 tracked deletions, 71 removed files
including local backup/settings artifacts, and 12 files restored byte-for-byte to
upstream (installer, packer, related tests and default draft vocabulary).
See [maintenance scope](../../../docs/DESKTOP_SCOPE.md).

## Verified

| Check | Result |
|---|---|
| CUDA engine and retained MMQ diagnostics | Build passed. |
| Server tests | 719 run, 12 skipped, passed. |
| Tool tests | 686 run, 11 skipped, passed with existing gguf-py dependency. |
| Standalone reader/layout/source tests | 5 CTest groups passed, including malformed-input Python fixtures with explicit reader executable. |
| CPU engine regressions | 10 passed. |
| GPU/snapshot/route policy regressions | 8 passed. |
| Full native request comparison | MTP off and two draft steps: 40 corresponding completed requests, exact tokens and main-state fingerprints; cancelled prefixes and resumed continuations match. |
| Long prefill comparison | 8199 / 24583 inputs, actual 4096 chunks, cold/warm reuse, A/B/A, short and long disk restoration: 10 corresponding requests with exact tokens/main states. Draft-KV snapshot verification enabled. |
| Two-slot native target-only | Greedy/seeded/penalty cases, cancellation/resume, followups, caps/EOS, 37933-token long slot beside short admission and KV growth/reuse passed. 137 two-row adaptive rounds, 13107 promotions. |
| Resident source reads | Expert/MTP requested source bytes zero on completed real-engine runs. Ngram and explicit session files are separate. |
| Generator / retained quality tools | Single/parallel generation, imports, argument parsing and removed-module reference checks passed. |

Image checks passed: 8 requests covering red/blue/red switching, warm image reuse,
return to text and seeded thinking sampling. Both desktop configs now select the
validated `strata-desktop-cleanup.exe`; only the executable field changed, shared
web settings and all runtime parameters are preserved. The old executable remains
for rollback. During this cleanup acceptance stage, no desktop HTTP service was started. Later model/vision path edits and the external Q6 deployment are recorded in the maintenance scope; those are separate from these native parity measurements.
The executable hashes and individual scope fields are in [verification.json](verification.json).
Raw model logs remain local under `bench/results/local-desktop-scope-20261012-*`;
build/test logs and original files are under ignored
`build-native-engine/desktop-scope-backup-20261012/`.

## Failed attempts and limits

The first baseline runtime was interrupted when the app closed; it is not counted.
Initial generic tool discovery lacked gguf-py and the reader's required `--exe`,
and caught the fork-modified default draft vocabulary. The corrected invocation,
separate reader CTest and exact upstream vocabulary rerun passed.

An additional strict `STRATA_KVG_CHECK=2` attempt failed on the **pre-cleanup**
executable with overlap enabled: three request-end audits each saw 96 device/host
table differences, while all 23 audits had zero weight-content errors. Source
review shows evictions marked on the host before `apply_pending` publishes
completed swaps. Waiting for CUDA copies inside the audit does not publish that
host transition. Settled prompt/grow/trim audits were clean. The failed attempt is
retained as `local-desktop-scope-20261012-long-pair`; it is not counted as passing.
The successful rerun retains overlap, verifies outputs/main states and restored
draft KV, and omits that incompatible residency-table audit. The scheduling code
and desktop overlap settings were not changed to make the audit pass.

This is finite regression evidence, not a broad reasoning/retrieval quality suite.
Auto-sized expert caches varied between starts and diagnostics add overhead, so
timing samples here must not be used to claim speed changes. Native HIP/SYCL,
Linux and multi-GPU execution were not validated on this machine.

## Real-model commands

Temporary baseline/candidate JSON copies differed only in the executable for the
request comparison. Long-pair copies also enabled `STRATA_SNAPSHOT_VERIFY=1`.
They are local acceptance inputs, not additional supported desktop profiles.

```powershell
python tools/native_runtime_acceptance.py --config <baseline-or-candidate.json> --output <new-directory> --variants 0,2 --operator-cache --spec-min-p 0.5 --official-new 32
python tools/native_config_pair.py --configs <baseline.json> <candidate.json> --fixture bench/results/2026-10-10-safetensors-runtime/p9-framework-efficiency/details-requests.json --cases details-8199,details-24583 --desktop-vision --long-restore --output <new-directory>
python tools/native_concurrency_acceptance.py --config config/native/rtx5090-262k-parallel.json --exe build-native-engine/strata-desktop-cleanup.exe --output <new-directory> --context 262144 --vision --limit 128 --adaptive --independent-kv --device-slot-copy --growth-tokens 40000
python tools/native_vision_smoke.py --config <candidate.json> --output <new-directory>
```
