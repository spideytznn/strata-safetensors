# FP4-Local format fixtures

`csf/` contains a 32 x 8 scale plane (logical weights 32 x 128), its CSF
slabs, and five little-endian U32 exceptions. Regenerate with
`python tools/make_csf_fixture.py`. The construction is adapted from
`tests/quantization/csf_fixtures.py::matrix` in local-inference-lab/vllm,
Apache-2.0, Copyright contributors to the vLLM project. It needs NumPy only.

`fp4local/config.json` is the configuration captured for
`nvidia/Qwen3.8-Flash-Next-FP4-Local` on 2026-10-19.
`fp4local/dense-descriptors.json` retains names, dtypes and shapes from the
five captured shard headers (1, 2, 14, 15, 16); offsets and payloads are absent.
The input files are recorded in the local research directory
`G:/Strata/fp4local-research/checkpoint-probe/`.

`test_fp4local_dense_plan.py` creates temporary sparse files from those
descriptors. It reads no weight payloads and does not download a checkpoint.
It validates all 1,079 dense bindings, including sliced indexer weights,
396 MXFP8 bindings and 100 F32 norm bindings. Its second pass synthesizes
the previous BF16 descriptor contract; that pass is not a real-checkpoint
regression or an inference quality test.
