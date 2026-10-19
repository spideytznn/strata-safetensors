# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reproduce the NVFP4 finite-scales fixture without importing torch/vLLM.

Adapted from vLLM tests/quantization/csf_fixtures.py::matrix (Apache-2.0,
Copyright contributors to the vLLM project). Run from any directory.
"""
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parents[1] / "tests/fixtures/csf"
root.mkdir(parents=True, exist_ok=True)
rows, columns, seed = 32, 8, 11
bases = ((np.arange(rows) + seed) % 16 + 43).astype(np.uint8)
offsets = ((np.arange(rows * columns) + seed) % 16).reshape(rows, columns).astype(np.uint8)
logical = bases[:, None] + offsets
positions = np.unique([0, columns-1, rows//2*columns-1, rows//2*columns, rows*columns-1]).astype(np.uint32)
values = (np.arange(len(positions), dtype=np.uint32)*7+seed) % 127
logical.flat[positions] = values.astype(np.uint8)
packed = offsets[:, ::2] | (offsets[:, 1::2] << 4)
fixed = np.concatenate((bases.reshape(rows//16, 16), packed.reshape(rows//16, -1)), axis=1)
fixed.tofile(root / "fixed.bin")
(positions | (values << 24)).astype("<u4").tofile(root / "exceptions.bin")
logical.tofile(root / "decoded.bin")
