# Stable native line

`main` carries the maintained native deployment. Its validated scope is Windows,
CUDA 13 / SM120, RTX 5090 32GB, Ryzen 9950X3D and 96GB RAM, using the original
Qwen3.8-Flash-Next NVIDIA NVFP4 checkpoint. The supported desktop profiles are
single-request or two-request text decoding. Images run exclusively.

The current desktop profiles keep MTP unloaded, INT8 KV, original BF16 dense
weights and all routed expert contributions (`STRATA_ROUTE_TAIL_SKIP=0`). The
two-request profile keeps independent KV/device copies and adaptive hot-expert
promotion. Expert RAM stays resident; post-residency expert source reads must
stay zero. Ngram reads and explicitly requested session files are separate.

The later continuous-batching, CPU-DAG, shared-page and ragged-prefill experiments
were rolled back. Their archived records do not establish a warm-decode speed
gain. Historical MTP timings are not current target-only performance figures.
Local build executables are not shipped by Git clone; follow the source-build
instructions and adapt the checked-in machine-specific paths.

## Checks before updating main

- GitHub's `Native CPU regression` workflow runs the existing service tests and
  standalone safetensors tests on Windows and Linux. It covers slot allocation,
  cancellation/restart, saved settings, weight layout, source-read guards,
  segmented ngram access and malformed inputs. No model download or GPU is used.
- Native engine changes require a CUDA build and real-model checks on the
  supported hardware. Check greedy and seeded outputs, penalties, caps/EOS,
  cancellation/resume, conversation reuse, actual overlapping two-slot decode,
  actual two-row hot-expert promotion and final expert/MTP source-read counters.
- Changes to KV, ownership, caching or vision also require the affected long
  context, growth/restore and image/text state checks. A CPU CI pass cannot
  substitute for these GPU checks.
- Performance claims need repeated end-to-end comparisons that separate prompt
  work from output tokens. Control the learned expert set, cache capacity and
  routing when comparing CPU execution paths. An enabled run that fails to beat
  the following original baseline is not evidence of improvement.
- Keep future scheduler experiments on separate branches until correctness and
  measured end-to-end benefit justify promoting them. Do not silently switch
  stable launchers, quantization, routing or arithmetic to a new experiment.

The [2026-10-11 rollback record](../bench/results/2026-10-11-two-slot-rollback/README.md)
contains the latest real-model checks: 54 server tests, seven runtime cases,
98 two-row adaptive rounds and 9370 promotions, with zero post-residency expert
and MTP source bytes. These are finite regression checks, not a broad reasoning
quality benchmark or a guarantee for untested hardware and workloads. The
inherited HIP/SYCL support does not establish native runtime validation there.
