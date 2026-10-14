# W4A4 desktop cleanup, 2026-10-14

Windows, RTX 5090, Ryzen 9950X3D, 96 GB RAM, CUDA 13.0, MSVC 14.44.

Removed the opt-in W4A16 expert-prefill implementation, selectors, trial
launchers/configs, dedicated tests and trial documentation. Those files and
their validation remain recoverable from Git history (`72385781`, `a7e2ea84`).
All ten modified engine/build files match the pre-trial commit `9d9f56e9` in Git.
Current desktop documentation now describes the actual rolled-back commit
settings and the integrated upstream v0.1.44.1.

Validation:

* CUDA engine and operator test compiled as an independent candidate,
  `build-native-engine/strata-w4a4-cleanup.exe`.
* Seven prefill policy tests passed: default, original, custom, auto, CUTLASS,
  invalid backend and missing optional build.
* CUTLASS/custom W4A4 comparison passed 31 shapes, each with relative L2 zero.
  Check-only mode was used; these are not performance measurements.
* Desktop wrappers, both root launchers and both profile files were checked.
  Root launchers and profiles are unchanged from the parent commit. Active
  engine hashes are recorded in [manifest.json](manifest.json).
* W4A4/CUTLASS auto, prefill 8192, BF16 KV, native MTP, two-slot concurrency,
  parallel prefill/decode overlap, route-tail skip off and unrestricted Host
  headers remain selected. The single profile still disables commit batching.

No model or HTTP server was started or stopped. The active desktop binaries
were not overwritten. Full-model quality, performance and long-context checks
were not repeated. HIP/SYCL/Linux were not built or tested.
