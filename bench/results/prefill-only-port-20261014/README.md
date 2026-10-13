# Prefill-only port, 2026-10-14

The CUDA SM120/CUTLASS engine and copied operator test compiled. Seven CPU policy checks passed. Twelve decode/state/CPU expert source files matched desktop repository HEAD. See manifest and build log.

The user was running strata-nvfp4-concurrent.exe on the RTX 5090 (30958 MiB used / 97% utilization observed before model testing), so no GPU operator/model test was run. Correctness, quality and throughput remain unvalidated for this hybrid build. Prepared folders contain isolated profiles only; no engine was started.

Desktop launcher, active config and old executable remain in place. New launchers use 8192 and 4096 prefill chunks with original decode, INT8 KV, native/batch MTP, 262144 context, two slots and vision. The expert prefill is W4A4/CUTLASS; BF16 prefill projections use rounded activations. Keeping decode code does not imply identical output from the changed prefill state.
