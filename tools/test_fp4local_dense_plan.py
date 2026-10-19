"""Bind actual FP4-Local dense header descriptors without downloading payloads.

The test file is sparse: only its header is allocated/read. Also exercises the
previous BF16 contract by converting the descriptor inventory, not weights.
"""
import argparse
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import tempfile


def sparse(f):
    if os.name == "nt":
        import ctypes
        import msvcrt
        from ctypes import wintypes
        fn = ctypes.windll.kernel32.DeviceIoControl
        fn.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD,
                       ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
        returned = wintypes.DWORD()
        if not fn(msvcrt.get_osfhandle(f.fileno()), 0x900C4, None, 0, None, 0, ctypes.byref(returned), None):
            raise ctypes.WinError()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", type=Path, required=True)
    args = parser.parse_args()
    fixture = Path(__file__).resolve().parents[1] / "tests/fixtures/fp4local"
    counts = []
    for legacy in (False, True):
        descriptors = json.loads((fixture / "dense-descriptors.json").read_text())
        config = json.loads((fixture / "config.json").read_text())
        if legacy:
            descriptors = {k: v for k, v in descriptors.items() if not k.endswith(".weight_scale")}
            for v in descriptors.values():
                if v["dtype"] in ("F8_E4M3", "F32"):
                    v["dtype"] = "BF16"
            config.pop("quantization_config")
        header, extent = {}, 0
        sizes = {"BF16": 2, "F32": 4, "I64": 8, "F8_E4M3": 1, "U8": 1}
        for name, desc in descriptors.items():
            end = extent + math.prod(desc["shape"]) * sizes[desc["dtype"]]
            header[name] = {**desc, "data_offsets": [extent, end]}
            extent = end
        with tempfile.TemporaryDirectory(prefix="strata-dense-plan-") as tmp:
            root = Path(tmp)
            (root / "tensors").mkdir()
            raw = json.dumps(header).encode()
            with (root / "tensors/dense.safetensors").open("wb") as f:
                sparse(f)
                f.write(struct.pack("<Q", len(raw)) + raw)
                f.truncate(8 + len(raw) + extent)
            (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
                name: "tensors/dense.safetensors" for name in header}}))
            (root / "config.json").write_text(json.dumps(config))
            if legacy:
                (root / "hf_quant_config.json").write_text(json.dumps({"producer": {"name": "modelopt"},
                    "quantization": {"quant_algo": "MIXED_PRECISION", "group_size": 16}}))
            result = subprocess.run([str(args.exe.resolve()), "dense-plan", str(root)],
                                    capture_output=True, text=True, encoding="utf-8")
            assert result.returncode == 0, result.stdout + result.stderr
            report = json.loads(result.stdout)
            assert sum(report["io"]["source_read_bytes"].values()) == 0
            counts.append(report["dense_bindings"])
            if not legacy:
                assert report["dense_source_dtypes"]["F8_E4M3"] > 300
                assert report["dense_source_dtypes"]["F32"] > 90
            print("legacy" if legacy else "FP4-Local", report["dense_source_dtypes"])
    assert counts[0] == counts[1]


if __name__ == "__main__":
    main()
