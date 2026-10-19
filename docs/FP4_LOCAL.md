# FP4-Local loader

See [the stable checkpoint](FP4_LOCAL_STABLE.md) and the separate
[experimental NVIDIA-style W4A4 decode](FP4_LOCAL_NVIDIA_W4A4.md).

Implementation and local validation: 2026-10-19, based on `5d9a4db2`.

The loader accepts the NVIDIA FP4-Local formats alongside the previous
NVFP4/BF16/FP8 checkpoint contract. Select the checkpoint directory through
the existing safetensors configuration. No conversion or intermediate model
files are required.

## Implemented

- Routed experts: `.weight_scale.nvfp4_csf_fixed` plus U32 exceptions decode
  to the original E4M3 bytes before the existing `pack_nvfp4`. Single-expert
  and bounded parallel startup loading use the same decoder. Plain scales
  use the previous path. Ambiguous or incomplete components are rejected.
- FP4-Local has different activation calibration across experts in every
  layer (177--286 distinct gate/up/down tuples per layer in this checkpoint).
  Custom and CUTLASS W4A4 prefill receive each expert's original scales.
  Gate/up equality is checked per expert; layer-wide equality is not assumed.
  The packed weight bytes and the existing W4A16 decode path are unchanged.
- CSF uses low-nibble-first offsets and 24-bit row-major exception positions.
  Positions must be ordered, unique and in range. Partial 16-row slabs are
  supported. `0x38` is an upstream padding base, not a decoding bias.
  `csf_32x2_v1` is explicitly unsupported.
- Attention and shared experts: E4M3 weights with UE8M0 U8 scales per 32
  columns stream into BF16, preserving row/column permutations and indexer
  slices. Normal-range values are exact; BF16 subnormals round to nearest,
  ties to even. NaNs and values exceeding BF16 range are rejected.
- F32 norm source weights stay F32 through the existing norm arithmetic.
- PLE: separate NVFP4 weights and E4M3 scales are read through bounded
  direct-I/O readers, including multiple source files and segment boundaries.
  Gather applies E2M1 x E4M3 x the original F32 global scale. All three I64
  hash constants still come directly from the checkpoint.
- MTP: W4A16 NVFP4 experts retain packed codes, micro scales and F32 global
  scales, with a dedicated CUDA dispatch. An absent activation scale is
  accepted only when explicitly binding W4A16. The existing FP8 dispatch stays
  available. No FP4-to-FP8 requantization is used.
- Indexes can refer to `tensors/*.safetensors`. Relative paths are checked
  for traversal and symlink/junction escapes. The embedded ModelOpt recipe
  is accepted without requiring `hf_quant_config.json`; CSF tensor names are
  authoritative and recipe disagreement is a warning. Missing
  `norm_topk_prob` uses the upstream Qwen3Next default, true.

## Validation

Run `tools/build_safetensors.ps1` for the standalone loader and seven tests:
CSF golden bytes, malformed inputs, plain/CSF batch equality, W4A16 binding,
MXFP8 bit conversion and permutations, F32 norm precision, segmented PLE,
source guards, adversarial headers/paths and actual dense header descriptors.

The CUDA build uses `tools/build_safetensors_engine.ps1`. The local output
name for this implementation is `strata-fp4local.exe`. Additional targets:
`mtp_fp8_test`, `native_nvfp4_ple_test`, `nvfp4_cpu_f32_test`,
`nvfp4_f32_device_test` (the last two take an expert test blob).

On RTX 5090, the synthetic MTP test compares 10,240 outputs against
independent FP64 dot products with FP32 hidden storage:

| Format | Relative L2 error | Max absolute error |
|---|---:|---:|
| Existing FP8 | 2.7933e-7 | 5.9117e-12 |
| NVFP4 | 2.5538e-7 | 2.9802e-8 |

The dense header test binds 1,079 tensors: 583 BF16, 100 F32, 396 MXFP8.
It also binds the previous all-BF16 descriptor contract.

The completed FP4-Local download at
`G:/moemodels/Qwen3.8-Flash-Next-FP4-Local` passed header/index validation:
375,487 tensors in 21 files. Six real expert samples passed lossless adapter
checks and FP64 SwiGLU comparison. Real NVFP4 MTP GPU comparison had relative
L2 error 2.6729e-7 and maximum absolute error 1.5646e-7.
`tools/validate_fp4local_dense.py` compared 12 real MXFP8 linear projections
(Q/K/V/O, GDN row/column reorder, indexer slices, shared experts) with two
independent FP64 input vectors each: maximum absolute difference was zero.
These projection checks do not compare the complete attention/softmax block
against another framework. Reports are under
`outputs/fp4local-real-validation-20261019/`.

`nvfp4_cutlass_test --check-only` also checks nonuniform per-expert calibration:
16 shuffled experts with unequal row counts match separate calibrated launches
bit for bit; custom/CUTLASS relative L2 difference was zero on RTX 5090.
The six real expert blobs also passed `nvfp4_f32_device_test`: all 61,440
output values were bit-identical between canonical CPU and GPU execution,
including shuffled token/output indices and single/grouped GPU execution.

The previous real checkpoint at `G:/moemodels/Qwen3.8-flash-next-nvfp4`
also passed validation: 299,545 tensors in 11 files, six expert adapters,
FP64 SwiGLU equality, eight PLE boundary rows, and 1,079 BF16 dense bindings.
All six packed expert SHA256 hashes equal the historical 2026-10-09 report
in `bench/results/2026-10-09-safetensors-p1/checkpoint-validation.json`.
The new report is in `outputs/fp4local-legacy-validation-20261019/`.
The real FP8 MTP GPU comparison had relative L2 error 2.8416e-7; the real
expert CPU test had relative L2 error 3.1470e-7 and identical T=1/T=4 results.
The runtime NVFP4 PLE test also covers split issue/collect, batch gather,
single-row reads and closing with an outstanding read.

For a completed download, run:

```powershell
.\build-safetensors\strata-safetensors.exe dense-plan 'G:\path\to\FP4-Local'
python .\tools\validate_safetensors_checkpoint.py --model 'G:\path\to\FP4-Local' --exe .\build-safetensors\strata-safetensors.exe --output 'G:\path\to\new-validation-directory'
.\build-native-engine\mtp_fp8_test.exe 'G:\path\to\FP4-Local'
```

The model directory is the parent of `config.json`, the index, and `tensors/`.
The validation output directory must be new and outside the model directory.

The local validation profile is
`config/native/rtx5090-fp4local-validation.json`: 4,096 context, 256-token
prefill capacity, 1,024 cached GPU experts, BF16 KV, full-head native MTP and
no PCIe miss sharing. It points to the parent directory of `tensors/`.
It is a small-cache correctness profile. The separate
`config/native/rtx5090-fp4local.json` profile uses automatic expert-cache sizing,
an allocated 8,192-token prefill workspace and a 262,144-token context ceiling.
It retains a 1,536 MiB reserve, BF16 KV, full-head native MTP, and disables
PCIe miss sharing and adaptive cache swaps. The context ceiling is a configured
limit, not evidence that a 262K input has been tested. To run the local API
with automatic cache sizing from the repository:

```powershell
.\.venv-native\Scripts\python.exe -m serve.server --config config/native/rtx5090-fp4local.json --host 127.0.0.1 --port 8096
```

## Runtime acceptance

Full loading and text generation now run on the completed checkpoint.
Expert residency is 63.282 GiB, physically page-locked and GPU-mapped.
In the initial 4,096-context test, all 17 completed requests had identical
tokens with MTP off and one draft enabled, including seeded sampling, EOS,
output caps, A/B/A continuation, disk restore and cancellation/resume.
Both runs recorded zero expert/MTP source reads after startup.
The initial MTP run with `--pcie-frac 0.25` failed the stricter restored-main-state
hash check for A+ despite identical output tokens. The failure did not recur
in a reduced A/B/A test or a second complete run. A diagnostic run with
`--pcie-frac 0` passed every token and main-state comparison against the original
MTP-off baseline. The tested validation profile disables PCIe miss sharing;
this does not disable the resident GPU expert cache. The root cause of the
one intermittent difference is not established, so the initial report
`outputs/fp4local-runtime-20261019-run3/` is retained as a failed strict test.

Follow-up snapshot analysis on 2026-10-19 narrowed this occurrence. The first
A+ continuation already had different state **before SAVE**; restoring and
recomputing from the saved 43-token checkpoint produced the reference state.
That checkpoint's main-state arrays are byte-identical to the MTP-off baseline.
The first differing persisted GDN layer is zero-based layer 18; earlier GDN
layers match. The first differing valid K/V cells are at QSA layer 19, position
46 (`<think>` in the 50-token prompt), and the difference continues through
position 56. Earlier QSA layers' valid K/V cells match. Unused cells beyond
the 57-token live extent were excluded from this comparison.

The final GDN arrays have relative L2 difference 0.00177995 (0.178%) and
maximum absolute difference 0.0734303, with no NaN/Inf. This describes internal
state, not a measured accuracy loss. PLE state and all 57 live token IDs match.
A healthy MTP-on snapshot compared against the same baseline has zero
differences in every valid main-state array, providing a control for the
snapshot diagnostic. `tools/inspect_session_difference.py` parses trusted v1
snapshots for this Qwen geometry and BF16 KV; it is not a replacement for the
engine's hash-validating session reader.

Three further independent full-history MTP-one-draft runs with the original
PCIe fraction 0.25 all passed (51 completed requests), using the same executable
as the failure. Evidence points to computation/state update in the prompt-tail
region, but does not yet identify a faulty instruction or establish PCIe as the
cause. Disabling PCIe sharing remains a conservative configuration choice,
not a proven root-cause fix. Reports and raw-snapshot differences are under
`outputs/fp4local-state-investigation-20261019/`.

The impact is reproducible from the retained snapshots. A controlled follow-up
loaded the healthy and anomalous 57-token live states into the same executable
with `--pcie-frac 0`, then asked the same next question about the remembered
identifier. Every branch reused exactly 57 tokens and recorded zero offloaded
experts. Each branch was repeated twice:

- Greedy: both snapshots returned the same 178-token answer, bit for bit.
- Temperature 0.8, top-p 0.9, seed 42: the healthy snapshot returned 181 tokens,
  the anomalous snapshot 190; the first differing output was token 74
  (zero-based index 73). Each snapshot reproduced its own output exactly on
  repetition. Both answers recalled `ZEBRA-417` correctly, but their subsequent
  wording and examples differed.

This demonstrates changed future sampling from the saved state; it does not
quantify a model-quality loss. Turning off PCIe sharing after the state was
already affected did not repair it. Model weight files are read-only during
these requests; the affected objects are the live state and snapshots derived
from it. The controlled report is
`outputs/fp4local-state-investigation-20261019/impact-controlled/results.json`.
The earlier `impact/` run retained PCIe fraction 0.25 (its attempted request
override was not forwarded); it is retained with a diagnostic note and is
not the basis for the zero-offload claim.

Remaining root-cause work needs an occurrence captured around the prompt-tail
steps: compare the inputs/outputs of layer 17 experts and layer 18 GDN, plus
their commit buffers, before and after the first differing QSA cells. Current
snapshots locate the persisted divergence but cannot distinguish an arithmetic
path difference from a buffer/synchronization fault. No runtime fix was made
on that unproven distinction.

The previous unchanged executable and old checkpoint passed the same complete
MTP-off/one-draft acceptance suite, including state hashes, in
`outputs/fp4local-old-baseline-runtime-20261019/`.

`tools/fp4local_generation_smoke.py` then ran the same five greedy, no-thinking
prompts on the patched executable with each checkpoint. The old checkpoint's
five output prefixes matched the unchanged executable's saved baseline.
Both checkpoints returned `372` for the arithmetic prompt and `Blue` followed
by EOS for the stop prompt. Both Chinese explanations hit the 192-token cap.
The old code answer also hit the cap; the new answer completed in 185 tokens
and its first-index binary-search function passed 106 concrete input cases.
English answers completed in 54 and 58 tokens respectively. Raw prompts,
token IDs, text and metrics are retained in
`outputs/fp4local-generation-comparison-20261019/`.
These five short prompts do not establish a model-quality or speed advantage.
The automatic-cache profile passed all three complete acceptance runs with
MTP off, one draft and two drafts, including every completed output and main
state hash. Each run completed 17 requests and recorded zero expert/MTP source
reads after startup. The command was:

```powershell
.\.venv-native\Scripts\python.exe tools/native_runtime_acceptance.py --config config/native/rtx5090-fp4local.json --output outputs/fp4local-auto-runtime-20261019 --operator-cache --variants 0,1,2
```

On the RTX 5090, automatic cache sizing selected 5,639 expert slots without
MTP (14.52 GiB of expert cache) and 5,029/5,028 with one/two draft tokens
(12.95 GiB). Small slot-count differences reflect available VRAM at startup.
The allocated prefill workspace was 4,694,433,024 bytes for 8,192 tokens.
The ready engines reported 1,675--1,706 MiB free. Whole-GPU `nvidia-smi`
sampling every second observed a peak of 30,842 MiB used and a minimum of
1,346 MiB free across startup, generation and teardown. This includes the
desktop and other processes and may miss subsecond peaks. Reports are in
`outputs/fp4local-auto-runtime-20261019/` and
`outputs/fp4local-auto-cache-gpu-20261019/`.

The same automatic-cache profile, with its configured MTP threshold 0.5,
also passed a 5,221-token archive/code-retrieval prompt. It used an actual
5,214-token batched prefill plus the prompt tail, streamed 19,548 expert
instances and returned `RIVER-827` followed by EOS. The engine reported
1,640.8 ms prompt processing and 123.1 ms decoding for seven output tokens,
with zero expert file reads. This is one warm-system smoke request, not a
throughput benchmark. Whole-GPU one-second samples peaked at 31,059 MiB used
(30.33 GiB), with at least 1,129 MiB free. No allocation failure occurred.
Reports: `outputs/fp4local-long-prefill-20261019/` and
`outputs/fp4local-long-prefill-gpu-20261019/`.

The existing Infernix executable and its converted old NVFP4 model also ran
the same five prompts through a temporary localhost API on port 8891.
Its deployment files were not edited; the test disabled expert-state saves
and stopped its own process afterward. It returned correct arithmetic and
the requested `Blue` stop response, while Chinese and code answers reached
the same 192-token limit. The recorded completion counts are:

| Prompt | Strata, old NVFP4 | Strata, FP4-Local | Infernix, old NVFP4 |
|---|---:|---:|---:|
| English | 54, EOS | 58, EOS | 57, EOS |
| Chinese | 192, limit | 192, limit | 192, limit |
| Code | 192, limit | 185, EOS | 192, limit |
| Arithmetic | 4, EOS | 4, EOS | 4, EOS |
| Single-word stop | 2, EOS | 2, EOS | 2, EOS |

The Strata comparison used MTP off and a fixed 1,024-expert cache for both
checkpoints. Infernix used its original W4A16 engine, automatic cache and
four draft tokens, so timings across engines do not isolate checkpoint
effects. This is a deployment smoke comparison, not a controlled performance
or quality benchmark. Complete commands, health response, logs and HTTP
results: `outputs/fp4local-infernix-comparison-20261019/`.

Only Windows/MSVC/CUDA was built here. HIP, SYCL and Metal toolchains were
unavailable; shared PLE source lists include the added host reader.

Vision continues to use the existing external BF16 encoder. This work does
not execute the FP4-Local quantized vision tensors. NVFP4 PLE supports the
split issue/collect and batch gather APIs; speculative row prefetch, SSD
keepalive and the old per-file batch-reader tuning are not yet implemented
for the new multi-file path. Its latency and memory use need measurement on
the real table.

## FP4-Local desktop parallel launcher (2026-10-19)

`START-FP4LOCAL-PARALLEL.bat` and the desktop
`Start-Strata-FP4Local-Parallel.bat` start the local UI/API on
`http://127.0.0.1:8880` using
`config/native/rtx5090-fp4local-parallel.json`. The profile copies the
FP4-Local auto-cache settings and adds `parallel: 2` and `--batch-mtp`;
PCIe sharing remains zero. Extra requests wait for a slot.

A real HTTP smoke check observed two simultaneously processing slots and
two successful 128-token responses (about 5.3 seconds each). The two slot
sessions consumed 1.12 GiB; the engine reported 1644 MiB free at startup.
Whole-GPU memory immediately after the requests was 31015 MiB used and
1172 MiB free (not a sampled peak). The test service was stopped afterward.
Evidence: `outputs/fp4local-desktop-parallel-20261019/result.json`.
This short smoke does not validate two simultaneous maximum-length contexts.

The desktop parallel profile subsequently restores hot-expert promotion with
`--adapt-every 4 --adapt-swaps 96`, at the user's request, and retains
`--prefill 8192`. Restart the desktop service to apply these startup settings.
The preceding smoke measurements were taken with promotion disabled.
