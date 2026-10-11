# Native CPU/GPU expert batch scheduling — local experiment

Windows RTX 5090 32GB, Ryzen 9950X3D, 96GB RAM; CUDA 13 / SM120.

All features are opt-in. Existing desktop profiles and binaries were left in place. No remote push was made.

Weights remain original NVFP4 with original BF16 projections. These comparisons retain canonical FP32 decode arithmetic and existing W4A8 prefill; they do not enable FP4 activation requantization.

The same executable is used within each off/feature/off family. The final direct comparison brackets the new binary with the unchanged deployed executable, recording each row's binary hash. Four text prompts form two paired workloads; each arm runs two complete warmups (four paired batches) and three measured rounds (six paired batches). Each request generates 256 greedy tokens. Rates below sum actual output tokens and divide by total measured batch wall time, including admission/cache restore and excluding startup. The JSON also records every case, median rates, overlapping decode rates, exact output hashes, binary hashes and cache capacities. GPU clocks were unlocked; these modest effects remain sensitive to clock and background variation. All cases load the same vision helper.

| Comparison | Arm | Aggregate output tok/s | Expert slots |
|---|---|---:|---:|
| Target, frozen cache | off | 72.68 | 4999 |
| Target, frozen cache | warm | 75.08 | 4999 |
| Target, frozen cache | queue_warm | 75.67 | 4999 |
| Target, frozen cache | all_warm | 75.74 | 4999 |
| Target, frozen cache | off | 74.14 | 4999 |
| MTP 3 / p=0.5, frozen cache | off | 80.54 | 3799 |
| MTP 3 / p=0.5, frozen cache | queue_warm | 85.08 | 3799 |
| MTP 3 / p=0.5, frozen cache | all_warm | 85.75 | 3799 |
| MTP 3 / p=0.5, frozen cache | off | 81.25 | 3799 |
| Target, adaptive tier | off | 76.95 | 4799 |
| Target, adaptive tier | queue_warm | 79.12 | 4799 |
| Target, adaptive tier | off | 78.54 | 4799 |
| Direct deployed/new comparison | reference | 76.96 | 4999 |
| Direct deployed/new comparison | off | 76.75 | 4999 |
| Direct deployed/new comparison | queue_warm | 79.45 | 4999 |
| Direct deployed/new comparison | reference | 76.74 | 4999 |
| Restored final source, frozen cache | off | 71.83 | 4999 |
| Restored final source, frozen cache | queue_warm | 67.28 | 4999 |
| Restored final source, frozen cache | off | 68.01 | 4999 |
| Ready queue only versus deployed | reference | 67.45 | 4999 |
| Ready queue only versus deployed | queue | 71.49 | 4999 |
| Ready queue only versus deployed | reference | 72.17 | 4999 |

The frozen target comparison uses one trained ranking with promotion disabled and 4999 slots. The MTP comparison uses the same ranking with 3799 slots in every arm, accounting for draft KV/workspace. Do not infer MTP benefit from absolute rates across these different cache capacities. The adaptive comparison starts each arm from the same existing profile and retains real hot-expert promotion, with identical requested cache capacity (4800, rounded to 4799). Automatic capacities differed in an earlier trial, which the harness rejected before feature timing. Explicit capacity requires prefill-first=0 in the existing engine. Controlled tests use this ordinary allocation order, retaining dedicated prefill workspace. Actual launch-profile acceptance retains prefill-first=1 and automatic capacity.

The separate launch profile enables ready scheduling and CPU cache warming. Planner and fetch overlap remain optional experimental switches; these results do not justify a production default change. Earlier positive differences did not reproduce consistently: the restored final-source queue/warming repeat regressed against the mean of its two controls. The profiles remain comparison entries, not recommended replacements for the deployed launchers. These results do not demonstrate stable acceleration or a near-doubling of throughput.

Functional acceptance passed nine cases with all features on for target-only execution and nine with dual MTP: greedy/seeded sampling, penalties, cancellation/continuation, conversation reuse, caps/EOS, two long contexts (40000 / 40007 tokens), and long-slot grow/reuse (37933 tokens). Both modes confirmed overlapping generation and real hot-expert promotion. Each separate launch profile repeats the same nine checks. All normal exits report zero post-residency expert/MTP source bytes. Ngram reads remain allowed.

The unchanged deployed target executable produces the same greedy output hash as all retained scheduler arms, including the MTP arms. The CPU pool passed 216 byte comparisons including mixed widths, sleeping workers, cancellation/replacement/destruction of cache warming and partial cache lines. The planner passed 600 exhaustive subset comparisons and capacity bounds. Service tests: 69 passed, 1 skipped.

An early planner trial was discarded because its guard never enabled the planner. A later final off control was discarded because compilation overlapped it. The complete later off/feature/off families determine the conclusions. Raw local trials remain available under the folders named in verification.json.

A separate AVX512 output-row tiling prototype passed 80 direct CPU byte checks and improved hot-cache kernel microbenchmarks, but its combination with ready scheduling/cache warming changed greedy output IDs in the complete model. It was removed from the final sources and launch profiles. The restored sources were rebuilt, passed the three C++ tests, and repeated the complete off/queue_warm/off model comparison. The rejected prototype is recorded separately in the JSON and supplies no claimed performance gain.

Finite token parity is not a broad model-quality evaluation. Vision was loaded, not evaluated on image content. The configured context is 262144; this run did not fill it. HIP, SYCL, Linux, other GPUs and more than two requests were not tested. No precision reduction, route-tail skipping, extra weight copy or expert SSD path was introduced. CPU cache-warming byte counters are logical touched spans.

Implementation and switches: [docs/NATIVE_UNIFIED_BATCH.md](../../../docs/NATIVE_UNIFIED_BATCH.md).
