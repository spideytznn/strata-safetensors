# strata-safetensors

**Run Qwen3.8-Flash-Next directly from the original NVIDIA NVFP4 safetensors on one GPU and system RAM.**

[中文说明](README.zh-CN.md) · [Runtime details](docs/SAFETENSORS_NATIVE.md) · [Measurements](bench/results/2026-10-10-safetensors-runtime/p16-speed-tuning/README.md) · [Upstream Strata](https://github.com/Niko1221/Strata)

Upstream v0.1.44.1 is integrated with `STRATA_ROUTE_TAIL_SKIP` **off by default**.
See [integration and validation scope](docs/UPSTREAM_0144.md) and the
[fork feature inventory / local profile snapshot](docs/FORK_FEATURES.md).

strata-safetensors adapts Strata's forward pass, expert scheduler and session cache
to the original Hugging Face checkpoint. It keeps the checkpoint's expert codes
and scales, loads experts into locked RAM, caches hot experts in VRAM, and reads
ngram rows from SSD on demand. No main-model GGUF conversion or persistent
converted expert pack is required.

The validated deployment is **Windows + RTX 5090 32 GB + Ryzen 9950X3D + 96 GB RAM**.
The maintained stable line is `main`; the executable and CLI retain the name `strata`.
The current desktop deployment uses W4A4/CUTLASS prefill at 8192 tokens,
BF16 KV Tensor Core attention and the original expert decode/scheduling backend.
Both single-request and two-request entries enable native MTP; the latter also
enables batch MTP. See [desktop scope](docs/DESKTOP_SCOPE.md),
[prefill](docs/NVFP4_PREFILL_ONLY.md) and [BF16 KV](docs/CLASSIC_BF16_KV.md).
Earlier validation records describe their own builds and settings.

## What it does

| Capability | Current native implementation |
|---|---|
| Original weights | Loads NVIDIA NVFP4 safetensors, BF16 dense weights, FP8 ngram data and native MTP weights directly. Expert repacking changes layout, preserving codes and scales. |
| Expert residency | All main experts have locked RAM copies, including experts cached on the GPU. Expert and loaded MTP source reads are sealed after loading. |
| CPU/GPU scheduling | Hot experts execute on GPU. Cold experts can execute on CPU with AVX-512 or be transferred to GPU; adaptive swaps update the hot set. |
| Prefill | 8192-token dedicated workspace, W4A4 FP4 Tensor Core experts with CUTLASS auto, and BF16-rounded dense prefill projections. |
| Context | Desktop BF16 KV Tensor Core attention with elastic growth. Inherited FP16/INT8 paths remain generic options. The configured total window is 262,144 tokens; older capacity records used INT8. |
| MTP | Uses the checkpoint's native draft weights. The main model verifies proposals; draft length and confidence gating are configurable. MTP can be fully unloaded. |
| Session reuse | Prefix reuse, multiple parked conversations, A/B/A switching, and disk save/restore of KV and linear-attention state. |
| Vision | Image input through Strata's existing external BF16 encoder. The visual sidecar uses mmproj/vocabulary GGUF files; the main model stays in safetensors. |
| Serving | Web chat, OpenAI-compatible and Anthropic-compatible APIs, streaming and tool calling. Opt-in two-slot concurrency, including image histories and shared native batch MTP on supported single-GPU engines. Excess requests queue. |

This is a model-specific backend, not a general safetensors runner. See
[scope and validation](docs/SAFETENSORS_NATIVE.md) for the supported geometry,
numerical comparisons and remaining limits.

## How the optimizations work

- **Keep experts resident.** The full expert arena is CUDA-pinned and GPU-mapped,
  avoiding an extra host staging copy. Runtime SSD reads are for ngram rows,
  plus session files when explicitly saving or restoring.
- **Use both processors.** Both desktop profiles select about 75% of cold experts on
  CPU and 25% on GPU.
  These fractions apply to cache misses, not all model work.
  AVX-512 row-loop specialization and GPU weight reuse reduce repeated work.
- **Allocate prefill space first.** Allocate and write the requested workspace
  before sizing the expert cache. Budget late allocations, including MTP, and
  refuse an insufficient startup allocation instead of silently reducing capacity.
- **Reduce runtime waits.** Release buffered startup weight handles once resident
  reads are sealed. This produced a measured prefill improvement on Windows;
  the specific Windows cache/driver mechanism has not been established.
- **Reuse conversation state.** Keep both KV and linear-attention state so a
  resumed session processes its new suffix rather than its entire history.

The desktop path keeps original expert decode and MTP projections. Prefill and
KV arithmetic differ from the historical INT8/W4A8 desktop path. Preserving
weight codes does not establish equal task quality or identical output.

## Measured performance

All numbers below were measured on **RTX 5090 32 GB / Ryzen 9950X3D / 96 GB RAM,
Windows, 2026-10-10**, using the original NVIDIA checkpoint. Rows describe
different historical comparisons; their gains must not be multiplied.

| Comparison | Before → after | Conditions and evidence |
|---|---|---|
| Startup-handle release, prefill | 822 → 2,465 tok/s | Same binary, about 8K input, three alternating pairs; complete 128-token outputs equal. [P10](bench/results/2026-10-10-safetensors-runtime/p10-native-performance/README.md) |
| CPU/GPU and kernel profile, decode | 45.28 → 81.59 tok/s | Three alternating pairs; both profiles use two drafts; complete 128-token outputs equal. Candidate range 64.57–85.16. [P10](bench/results/2026-10-10-safetensors-runtime/p10-native-performance/README.md) |
| Later 4096-row profile, text prefill | 2,968 → 3,331 tok/s | About 24K text, visual encoder loaded, three-start medians. First about-8K text prefill instead falls 2,688 → 2,558. [P16](bench/results/2026-10-10-safetensors-runtime/p16-speed-tuning/README.md) |
| Later profile, image-request prefill | 2,741 → 3,089 tok/s | About 8K including 1,024 image tokens, three-start medians; total wall 4.51 → 3.94 s. [P16](bench/results/2026-10-10-safetensors-runtime/p16-speed-tuning/README.md) |

A separate P16 screen measured warm aggregate decode of **91.93 → 100.82 tok/s**
over English, Chinese and code. It used one fresh engine per profile and is
not a repeated throughput estimate. That candidate used four drafts, a 0.7
confidence gate and a two-draft floor.

The figures above describe archived configurations, not fresh measurements of
the current BF16/W4A4 desktop deployment.

## Build and run on Windows

Install Git, Python 3.12+, Visual Studio 2022 C++ Build Tools, CUDA 13,
CMake 3.24+ and Ninja. Model and vision assets are not distributed in this repo.

```powershell
git clone https://github.com/spideytznn/strata-safetensors.git
cd strata-safetensors
python -m venv .venv-native
.\.venv-native\Scripts\python.exe -m pip install -r requirements-native.txt
.\.venv-native\Scripts\python.exe -m pip install cmake ninja
.\tools\build_safetensors_engine.ps1 -Jobs 2
```

The helper builds `build-native-engine/strata-classic-bf16-kv-slotcommit.exe` with W4A4
prefill, pinned CUTLASS and BF16 KV. It fetches CUTLASS into `build-deps/cutlass`;
CMake fetches its pinned ggml unless `-GgmlDir` is supplied. It refuses to overwrite
a running engine; use `-OutputName` to build a separate candidate.

### Current desktop entries

| Entry | Config | Slots |
|---|---|---:|
| `START-NATIVE-262K-PREFILL-TC-BF16.bat` | [single](config/native/rtx5090-262k-prefill-tc-bf16.json) | 1 |
| `START-NATIVE-262K-PREFILL-TC-BF16-PARALLEL.bat` | [parallel](config/native/rtx5090-262k-prefill-tc-bf16-parallel.json) | 2 |

These deployment-specific configs contain Windows paths; adapt model, executable,
tokenizer, template and vision asset paths on another PC. Both select 262144
context, 8192 prefill, BF16 KV, native MTP (`spec=3`, confidence 0.5), 1536 MiB
VRAM reserve, vision and original expert scheduling. The two-slot config also
sets `--batch-mtp`; each concurrent slot can propose one MTP draft.
The single and parallel desktop BAT files call these entries. Old trial entries
were removed. Upstream generic setup and the config generator remain available.

Start one entry at a time. Open **http://127.0.0.1:8880** for chat; API clients use
**http://127.0.0.1:8880/v1**, model `Qwen3.8-Flash-Next-NVFP4-Native`.
Ctrl+C stops the foreground server. Check `/health` and `/v1/models` for readiness.
The runner binds localhost; configure an API key before exposing it externally.

Both files default to temperature 0.7, top_p 0.95, top_k 20. Requests and persisted
web settings can override sampling. The template is Froggeric v22.5; output
budget fitting is enabled. Model inputs are read-only. The measured expert RAM
arena alone occupies **63.282 GiB**, excluding other weights, sessions and OS;
96 GB is the validated machine configuration.

## Validation and limits

- The stable rollback passes 54 server tests and seven real-model cases on the
  validated Windows/CUDA machine. It records 98 two-row adaptive rounds and
  9370 promotions, with zero post-residency expert/MTP source bytes. GitHub's CPU
  workflow runs server and standalone reader tests on Windows and Linux; it does
  not run CUDA inference or certify model quality. See [stable checks](bench/results/2026-10-11-two-slot-rollback/README.md).
- Weight-layout roundtrips, real-expert FP64 checks, full-logit comparisons,
  seeded sampling, output caps/EOS and cancellation/resume have recorded evidence.
- Final efficiency17 acceptance covers 80 completed requests and four
  cancellations across zero/two/four/six drafts, including output/main-state,
  session switching and disk-restore checks. These are finite regression tests.
- The 262K fixture proves capacity and tested state reuse. It does not establish
  general long-context reasoning quality, retrieval accuracy or 262K throughput.
- Native text requests can decode together with `"parallel": 2`; admissions
  share the prompt reader. Shared native batch MTP and image histories are supported on the desktop
  single-GPU path. Multi-GPU, pipeline parallelism and external drafts are outside
  this native deployment scope.
- Native CUDA SM120 is validated locally. Linux, HIP, SYCL and other hardware
  have not been validated for this backend. Inherited backend support is not
  evidence for the new native path.
- The main model requires no GGUF. Vision still uses an external GGUF sidecar.
  No native vision-tower implementation or automatic native installer is claimed.

## Documentation and provenance

- [Differences from official Strata, feature status and profile catalog](docs/FORK_FEATURES.md)
- [Native implementation, diagnostics and acceptance](docs/SAFETENSORS_NATIVE.md)
- [Earlier CPU/GPU performance comparisons](bench/results/2026-10-10-safetensors-runtime/p10-native-performance/README.md)
- [Workspace and speculation measurements](bench/results/2026-10-10-safetensors-runtime/p16-speed-tuning/README.md)
- [Inherited NVFP4/GGUF documentation](README.legacy-nvfp4.md) and [original Strata README](README.upstream.md)

Built on [Niko1221/Strata](https://github.com/Niko1221/Strata) and
[sergqwer/strata-nvfp4](https://github.com/sergqwer/strata-nvfp4), starting from the
NVFP4 fidelity commit `d167eb89301a02e0d6299f49d35494ba5096bc10`.
Their authorship and licenses are retained. The previous personal GGUF main
branch is preserved as `archive/gguf-main-20261011`.

The [upstream proposal](https://github.com/Niko1221/Strata/issues/1858) and
[standalone safetensors-source PR](https://github.com/Niko1221/Strata/pull/1859)
are open as of 2026-10-11. The complete native runtime has not been merged
into upstream Strata.

Engine source is [MIT-licensed](LICENSE). Model weights and visual assets retain
their respective licenses; weights are not distributed in this repository.
