# v0.1.43.1 local integration checks

Windows / RTX 5090 32 GB / Ryzen 9950X3D / 96 GB RAM / CUDA 13 / SM120.
Upstream release commit: `a0116b2782a5b82622358c04a06496485d12b2d6`.

See [the integration record](../../../docs/UPSTREAM_V01431.md) for deployment paths,
conflict resolutions, validation limits and recovery. [summary.json](summary.json)
records the completed local checks. Raw logs remain in `local-upstream1431-*` on the test machine.

These are functional regression checks, not throughput or general quality measurements.
The native image and batch MTP runs preserve zero post-residency expert/MTP source reads.
GGUF before/after outputs are not wholly byte-identical; the record explicitly describes the
checked responses, changed auto-sized residency and automatic PCIe division. This larger Q6
deployment still has a file tier. HIP, SYCL, Linux and multi-GPU were not tested.
