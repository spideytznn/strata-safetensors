# Original expert decode with BF16 KV, 2026-10-14

Engine compiled. BF16 attention operator/oracle checks passed. Snapshot suite: 4237 checks passed, including BF16 copy and format isolation. Memcheck/initcheck: zero errors each. BF16 HMMA instructions recorded. Seven prefill-policy CPU regressions passed. See raw logs and manifest.

Two launchers retain original expert decode, native/batch MTP, two slots, vision, 262144 context and port 8880. One keeps original 4096/W4A8 prefill; the other selects 8192/W4A4/CUTLASS prefill. Frozen desktop and INT8 prefill-only executables were not overwritten. Nine expert/decode/MTP source files matched HEAD.

GPU tests were small synthetic checks after verifying no Strata model was running. No full model or HTTP model service was started; whole-model performance, parity, quality and long-context capacity remain unvalidated. Prepared folders contain isolated test configurations only.
