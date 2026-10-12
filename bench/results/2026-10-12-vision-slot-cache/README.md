# Image-history slot and progress repair

2026-10-12: RTX 5090 32 GB, Ryzen 9950X3D, 96 GB RAM, Windows, CUDA 13, SM120.
Executable SHA256: `037f8c4896eeafac28c750dda009456a41d3ff097635ad21cbaad6cfb448347a`.
The local parallel profile selects `strata-vision-cache.exe`; the single-request
MTP profile continues to select `strata-desktop-cleanup.exe`.

## Problem and change

The native server held an exclusive image permit throughout generation, even
when only an old picture remained in the request's history. Native batch slots
could not safely share the mutable image position table and did not retain image
conversations for slot reuse. The monitor could also display one request's final
prefill progress against another request's prompt length.

Single-GPU target-only native slots now have stable independent M-RoPE tables,
including pooled-indexer forward and commit. Slot prefixes and checkpoints match
image grids and embedding hashes. Image prefill can use the existing yield/resume
path. Monitor progress belongs to the request consuming the control lines.
The older exclusive guard remains for engines without this capability, MTP and
layer splits. Expert GEMMs, quantization and the existing adaptive exchange path
are unchanged. Both local desktop cache budgets are restored to 8192 MiB.

## Validation

- CUDA build and five selected GPU/state CTest targets pass, including captured
  slot-position parity, conversation snapshots and INT8 KV restoration.
- Server regressions: 69 tests pass, with one skipped. The new cases cover the
  image capability gate and the reported 141600/154172 progress mismatch.
- Private-stdin acceptance: 22 requests at 16384/512 context/prefill and 31
  requests at the desktop 262144/4096 allocation pass. Generated image/image and
  image/text pairs run in two slots. Greedy and seeded output comparisons match
  complete token sequences. Changed pictures miss the old image state;
  cancellation and warm continuation recover. All 53 recorded requests report
  zero runtime expert file blobs and bytes.
- Real HTTP acceptance on port 8880 passes 13 requests: two simultaneous decoding
  slots, color recognition, historical images, thinking history, repeated output
  parity, unrestricted Host headers and a long image-first prompt. That 141092-token
  prompt yields at 12288 tokens and resumes there; its warm repeat reuses 141085
  tokens and produces the same response. The final cumulative expert file byte
  counter remains zero. The desktop service is left loaded and idle.

| Desktop long case | Input tokens | Reused tokens |
|---|---:|---:|
| Red image, parallel repeat | 141098 | 141091 |
| Different blue image grid, parallel repeat | 126147 | 126140 |
| Red history continuation | 141216 | 141193 |
| Blue history continuation | 126265 | 126242 |
| Cold prompt resumed after yielding | 141106 | 8192 |
| Same prompt, warm repeat | 141106 | 141099 |

The cold read yielded after 8192 tokens and restored that partial slot, rather
than restarting from zero. The two long parallel repeats match their solo output
token sequences. Two-row adaptive promotion is present in the logs.

To repeat the private desktop check with exclusive GPU/RAM resources:

```powershell
.venv-native/Scripts/python.exe tools/native_vision_concurrency_acceptance.py --config config/native/rtx5090-262k-parallel.json --exe build-native-engine/strata-vision-cache.exe --output bench/results/local-vision-cache-acceptance-new --desktop-context --extended
```

See [the machine-readable record](summary.json) for exact arguments, executable
and result hashes, counters and HTTP acceptance results. The raw local runs are
`bench/results/local-vision-cache-acceptance-20261012-r2` and
`bench/results/local-vision-cache-acceptance-20261012-desktop`.

An initial harness launch failed because its private expert-profile file did not
exist. The harness was corrected to copy the existing profile before startup;
that attempt is excluded from the passing checks above.

These are finite correctness checks with generated solid-color pictures and
repetitive text, not a speed, broad quality, retrieval or BF16-equivalence
benchmark. A changed rendered prefix, changed image or cache eviction can still
require prefill. HIP/SYCL, multi-GPU and more than two slots were not validated.
