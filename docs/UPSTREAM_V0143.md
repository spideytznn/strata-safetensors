# Strata v0.1.43 integration


Current feature status is in [the desktop cleanup record](DESKTOP_SCOPE.md). Native batch-MTP and other unused experiments described below were subsequently removed. This is an integration-time record.
This merges upstream tag `v0.1.43`, commit
`7695859aa13afb248bdb4d36f9b2e95707db079d`, into the safetensors fork after
`36902c911f21adf1fc18c4364772108f907ae8f0`. The latter reverted the local
unified-expert scheduling experiment; this upgrade does not restore it.
Upstream source, tests, tools and backend documentation are included.

## Integration decisions

- Keep all routed experts. `STRATA_ROUTE_TAIL_SKIP` remains off when absent or
  `0`, including AMD. Upstream's RDNA4 default does not replace this quality
  default.
- Keep original NVFP4 expert codes/scales, BF16 projections, FP32 decode,
  compensated W4A8 prefill, INT8 KV, dedicated prefill allocation, RAM-resident
  experts and the existing native expert source seals.
- Retain the native two-slot scheduler, adaptive hot-expert promotion,
  independent KV/device slot copying where configured, and opt-in native
  per-slot MTP. The NaN guard checks each slot's actual row interval: native
  confidence gating can produce one, two or three rows per slot. A fixed
  two-row interval would miss the last draft or examine a neighbour's rows.
- Include upstream non-finite logits detection, poisoned-cache invalidation
  and the server's single retry before emitting output. A partial response
  ends instead of retrying already emitted tokens.
- Include adaptive-copy overlap with upstream's platform defaults. Windows
  leaves `STRATA_ADAPT_OVERLAP` off; enabling it requires a new measurement.
  Canonical Q2 automatic PCIe selection also remains opt-in on Windows.
  The fork's measured native PCIe selection and timing stamps are retained.
- Include request accounting across solo/batch transitions, optional cache
  dropping, Windows GGUF mapping cleanup, NUMA allocation checks and the new
  benchmark commands. The benchmark version reader explicitly uses UTF-8
  for Windows systems whose default text encoding is GBK.
- Include upstream NVIDIA, AMD and SYCL changes. Merging source does not
  establish native safetensors support on additional backends.

## Validation and deployment scope

Local validation uses Windows, RTX 5090 32 GB, Ryzen 9950X3D, 96 GB RAM,
CUDA 13.0, MSVC 14.44 and SM120. The candidate is a separate executable,
`build-native-engine/strata-upstream143.exe`; previous binaries are retained.
See [the validation record](../bench/results/2026-10-11-upstream-v0143/README.md)
for commands, hashes and results. Finite token/state comparisons are regression
screens, not a general model quality benchmark.

Desktop profile updates change the executable only. Existing sampling, MTP,
context, prefill, vision, unrestricted Host handling and port settings remain.
The original model and Q4XL deployment are untouched. No remote push is part
of this upgrade.

HIP, SYCL, Linux, multi-GPU and other GPU generations cannot be built or
hardware-validated on this machine. Upstream benchmark numbers retain their
original hardware and scope; no RTX 5090 safetensors speedup is inferred.

Two installer tests also fail against pre-upgrade `setup.py` under the same
local harness: `Keep.test_a_card_the_release_has_no_code_for` and
`PageFile.test_check_and_install_say_it_framed_and_again_at_the_end`.
They are recorded as pre-existing failures, not passing upgrade checks.
