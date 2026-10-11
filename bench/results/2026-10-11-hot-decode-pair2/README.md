# Frozen expert decode experiment, 2026-10-11

Decision: **do not deploy the paired-row CPU experiment**. Its apparent gain
against the first baseline disappeared when the original path was run again.
The source changes are archived in `rejected-cpu-row-pair.patch`, and have been
removed from the active source tree. No desktop profile or deployed executable
was changed.

RTX 5090 32GB, Ryzen 9950X3D, 96GB RAM, Windows, CUDA 13 / SM120.
Original NVFP4 safetensors, original BF16 projections, INT8 KV, 262144 context,
dedicated 4096-row prefill, vision allocated, target-only two-client decode.
One learned private expert ranking was reused in all three arms; actual cache
capacity was 4999 slots and promotion was disabled. CPU DAG, ragged prefill,
prefix sharing and token-budget admission were off. Fixed cache allocation
requires prefill-first allocation off in this benchmark; daily configs were not
changed. The PCIe fraction was 0.25.

Four short English/Chinese/code prompts were run as two pairs, each generating
256 tokens per client. Each arm had two warm rounds followed by three measured
rounds (six measured batches). Each arm restarted the same candidate executable.
The numbers below are medians across those six batches, in aggregate output
tokens/second, not per-client or prompt-processing rates.

| Measure | Original before | Paired + VBMI | Original after |
|---|---:|---:|---:|
| Whole-request wall rate | 76.87 | 82.23 | 82.54 |
| Overlapping decode interval rate | 81.00 | 86.72 | 87.13 |

The enabled path was 7.1% ahead of the first decode baseline, but 0.5% below the
following baseline. The baseline itself moved 7.6%. This run does not establish
a repeatable end-to-end gain. Do not average the two baselines and advertise the
result as a causal CPU speedup.

All ten batches per arm, including warmups, produced identical token sequences.
The GPU/PCIe/CPU routing signatures match for every corresponding batch in all
three arms. The two prompt pairs route 50.8% / 75.3% of entries to resident VRAM;
"warm" does not mean every expert is resident. All three final expert and MTP
source-read audits are zero. Ngram SSD reads are separate.

`controlled-results.json` retains all measured case rates, diagnostics, binary
and ranking hashes, startup information and source audits. Raw logs, timestamps
and the private ranking remain local. The low-level helper passed 120 byte
comparisons, and the combined CPU-DAG helper passed 144 byte comparisons with
paired rows enabled. Their timing results are microbenchmarks. HIP and SYCL
were not tested.

The rejected change pairs gate/up or adjacent down rows to reuse activation
loads, expands FP4 nibbles with AVX512VBMI, and preserves each dot product's FMA
and reduction order. It applies only to canonical/GPU-order F32 with one or two
tokens; other cases fall back. The experiment is opt-in and did not prune routes
or requantize weights.

To rebuild the archived experiment from the recorded base commit:

```powershell
git apply bench/results/2026-10-11-hot-decode-pair2/rejected-cpu-row-pair.patch
./tools/build_safetensors_engine.ps1 -Jobs 4 -OutputName strata-cpu-pair2 -Targets strata,native_cpu_row_pair_test,native_expert_dag_test
.venv-native/Scripts/python.exe tools/native_hot_decode_screen.py --config config/native/rtx5090-262k-parallel.json --exe build-native-engine/strata-cpu-pair2.exe --output bench/results/frozen-repeat --toggle STRATA_NVFP4_CPU_ROW_PAIR --vision
```

Without `--profile`, the tool trains a new private ranking first. For the exact
local ranking, pass the profile path recorded in the JSON. The current tool also
supports a frozen-cache CPU-DAG comparison without applying the patch; its
default toggle is `STRATA_NATIVE_EXPERT_DAG`. It now checks route equality and
records effective engine settings in addition to the original harness's output
and source-read assertions.

Two earlier harness attempts remain local: pair1 rejected the incompatible
fixed-cache/prefill-first configuration before measurement; pair1b trained the
ranking but then incorrectly required exactly 5000 slots rather than the engine's
rounded 4999. Neither supplied performance results.
