# Single-request startup headroom fix

The desktop single-request entry loads
`config/native/rtx5090-262k-fast-v0142.json`. Its old 256 MiB explicit reserve
could leave just 379 MiB free after the dedicated 4096-row workspace, expert
cache, KV and verification buffers were allocated. The unchanged final readiness
guard requires 512 MiB, so the engine correctly refused startup.

Only `--vram-reserve-mib` changes, from 256 to 1024 MiB. This matches the reserve
previously used in the completed single-request PCIe-auto acceptance. No engine
source or binary changes are needed. More free space leaves fewer slots for the
automatic expert cache; this is a startup reliability correction, not a speed
claim or an arithmetic change. The 4096-row workspace and guard remain intact.

Private validation used the actual single-request desktop config and GPU vision
sidecar on RTX 5090 / Ryzen 9950X3D / 96GB RAM, Windows. No HTTP listener was
started and both private processes exited:

- Reached READY in 64182 ms with **1249 MiB free**, above the 512 MiB guard.
- Kept **4096 prefill rows**, workspace 2906123008 bytes, INT8 262144 context,
  original BF16 projections, all routed experts and unloaded MTP.
- Generated 32 greedy tokens for an **11026-token** input, then repeated it.
  Complete outputs match. The repeat reuses 11019 tokens and reads only 7.
- Final post-residency expert and MTP source bytes are both zero.

`verification.json` records startup INFO, complete DONE lines, counters and the
unchanged executable hash. Raw engine logs and the initial diagnostic helper
record remain local. That helper mistakenly parsed prompt milliseconds (DONE
field 3) as an integer cache-reuse count after both requests had completed. The
corrected independent audit checks reuse at field 8 and read count at field 14;
this harness error is preserved in the verification note and is not an engine
failure. No throughput, broad quality or other-hardware claim is made here.
