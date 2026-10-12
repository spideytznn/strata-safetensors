# Upstream v0.1.43.1 integration

2026-10-13. Release: [Niko1221/Strata v0.1.43.1](https://github.com/Niko1221/Strata/releases/tag/v0.1.43.1).
The annotated tag resolves to `a0116b2782a5b82622358c04a06496485d12b2d6`.

## Integration

The native fork merges the complete release into the existing main branch. The previously validated
image-slot/cache fixes and shared native batch MTP were first saved in local commit `ba011090`.
No remote push or upstream pull request is part of this update.

Conflict resolutions retain the native safetensors library, dedicated prefill changes, native M-RoPE
tables, request-local progress, image-aware yield/resume and asynchronous device slot migration.
The upstream BGEN refusal retry also permits image prefill yielding when the engine advertises
`native_vision_slots`. Conversation copies retain their asynchronous option and report watchdog progress
after a synchronous copy lands. The pinned upload ring covers upstream's pageable prompt sources.

Both native desktop launchers use `build-native-engine/strata-v01431.exe`. Context 262144,
prefill 4096, INT8 KV, GPU vision, 8192 MiB / four-slot conversation cache, sampling,
unrestricted Host headers and local port 8880 remain configured as before. Single and parallel
MTP remain native / spec 3 / min-p 0.5; parallel retains shared weights and private per-slot draft state.
`STRATA_ROUTE_TAIL_SKIP=0` remains explicit. Explicit adaptive overlap remains enabled;
the upstream change to opt-in defaults does not override the desktop configuration.

The external GGUF Fusion repository also merges the release, retaining its local memory planner,
RAM promotion and optional session storage. The renamed upstream memory probe is used by the local
session storage code. Its staged executable enables `STRATA_Q6K_EXPERTS` and `STRATA_MMQ_KQUANTS`.
The Q6 desktop configuration retains its 81 GiB resident budget, 64 GiB partial-pin setting,
1 GiB RAM headroom, INT8 streamed KV, GPU vision and disabled MTP. Tail skipping is explicitly disabled
when moving from 0.1.41, where that upstream default did not exist.

## Validation

Windows, RTX 5090 32 GB, Ryzen 9950X3D, 96 GB RAM, CUDA 13.0, MSVC 14.44, SM120.

- Both CUDA engine builds pass. Native SHA256:
  `02781fb369e5aeba90b0a595e40360258eb7688ffa38040c770d1e79dbcf6a17`.
  GGUF SHA256: `e5a806002eb06a29912ac5044d8051ed617da7993f9fb6307a3122e961eabd76`.
- Native server/frontend/parallel tests: 368 passed. GGUF counterpart plus slots/monitor:
  396 run, one skipped, all others passed. Upstream setup/version/benchmark tests: 77 passed.
- Six native CTests pass: batch rows/groups, QSA spare commit, image positions,
  conversation snapshots and conversation cache. Four GGUF CTests pass: local memory planner,
  expert source, conversation snapshots and session files.
- Native desktop-context image acceptance: 22 requests, six concurrent pairs; image/text
  isolation, greedy/seeded parity, history reuse, replacement images and cancellation pass.
- Native desktop-context MTP acceptance: ten cases, including different confidence thresholds,
  cancellation/resume, follow-up, EOS, caps and a 18979-token slot growing beside a short admission.
  Serial and concurrent token sequences match within every checked pair. Actual row layouts include
  target-only and proposing slots. Hot expert promotion occurs with both two and one active slots.
- Both native runs exit with code zero; post-residency expert and MTP source bytes are zero.
- Native single desktop settings also pass short / 27026-token long / short / cached continuation,
  with its unchanged 700 MiB reserve, native MTP, dedicated workspace and vision loaded.
  It reuses 72 of 73 continuation input tokens, exits with code zero and seals both weight sources at zero reads.
- PLE selftests pass for both readers against every format's dequantizer and the 90/110-byte row tests.
- GGUF old/new desktop settings each pass short / 27026-token long / short / cached continuation,
  and exit with code zero. The continuation reuses 72 of 73 input tokens. The first two responses
  match all 48 emitted tokens across versions. The third matches 27 tokens, then chooses different
  wording; its continuation therefore also differs. This is not a byte-identical cross-version result.
  Actual auto-sized GPU expert counts, available-RAM residency and automatic PCIe share differ between runs;
  the checks do not isolate which of these or the upstream changes causes the divergence.
  The resident budget remains 81 GiB, but the engine clamps the allocation to available memory.
  This larger Q6 model still uses a file tier; the native zero-expert-read guarantee does not apply to Q6.

Raw local results are under `bench/results/local-upstream1431-*`; build and focused test logs
are under `build-native-engine/upstream1431-*`. Earlier 0.1.43 and batch MTP measurements
remain historical measurements, not speed claims for this release. Finite regression checks are not
a general quality evaluation. HIP, SYCL, Linux and multi-GPU builds were not available on this host.

## Local directories and recovery

- `G:\Strata\Strata`: GGUF desktop runtime, Python/build tools and shared llama.cpp checkout.
  Its actual Fusion Git source is `work\fusion-20261007-v0401`; that name is historical,
  and its release ancestry is now 0.1.43.1. The runtime source/server files are synchronized from it.
- `G:\Strata\Strata-Safetensors`: maintained native source, runtime and two desktop configurations.
- `G:\Strata\Strata-data`: shared Q6 pack, FP8 ngram and vision resources. Model weights remain
  in `G:\moemodels`.
- `G:\Strata\Strata-Official`: upstream reference checkout updated to the release; its existing
  `engine\q6\strata-vision.exe` remains referenced by all three desktop configurations.
  That encoder is reused unchanged; this release changes its callers, not its source.
- `G:\Strata\Strata-Safetensors-Upstream`: Git worktree for the earlier upstream contribution,
  not a desktop runtime. Keep only for that development work; remove through Git worktree tools when requested.
- `G:\Strata\NVFP4-Converter`: historical conversion tools, not referenced by current desktop paths.

No directory or model is deleted. Several older clones/builds inside `Strata\work` are historical;
the active Fusion source, build dependency paths and rollback material must be excluded from any future cleanup.
Rollback material: native `build-native-engine/upstream1431-backup-20261013`, existing native binaries,
GGUF `backups/before-v01431-20261013.zip`, old `engine/fusion-v041/strata.exe`,
and each repository's pre-upgrade commit/backup branch. Web settings sidecars are preserved.
