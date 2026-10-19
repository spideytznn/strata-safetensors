"""Native checkpoints remain usable through the 0.2 control layer."""
import json
from pathlib import Path
import tempfile
import unittest

from strata_control import discovery, doctor
from strata_control.native_checkpoint import checkpoint_inventory


class NativeCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.index = self.root / "model.safetensors.index.json"
        self.index.write_text(json.dumps({"weight_map": {"x": "tensors/one.safetensors"}}))

    def test_lists_missing_index_shards(self):
        files = dict(checkpoint_inventory(self.root))
        self.assertEqual(files["checkpoint/tensors/one.safetensors"], self.root / "tensors/one.safetensors")
        self.assertFalse(files["checkpoint/tensors/one.safetensors"].exists())

    def test_changed_index_invalidates_cache(self):
        checkpoint_inventory(self.root)
        self.index.write_text(json.dumps({"weight_map": {"x": "two.safetensors"}}))
        self.assertIn("checkpoint/two.safetensors", dict(checkpoint_inventory(self.root)))

    def test_rejects_escape(self):
        self.index.write_text(json.dumps({"weight_map": {"x": "../outside.safetensors"}}))
        with self.assertRaisesRegex(ValueError, "escapes"):
            checkpoint_inventory(self.root)

    def test_native_mtp_and_hf_tokenizer_do_not_require_converted_files(self):
        (self.root / "tensors").mkdir()
        for file in ["engine.exe", "config.json", "tokenizer.json", "tokenizer_config.json", "tensors/one.safetensors"]:
            (self.root / file).write_bytes(b"{}")
        cfg = {"exe": str(self.root / "engine.exe"), "cwd": str(self.root), "tokenizer": str(self.root),
               "tokenizer_format": "safetensors", "args": ["--model", str(self.root), "--mtp", "native"]}
        config = self.root / "run.json"
        config.write_text(json.dumps(cfg))
        report = doctor.inspect(config)
        self.assertTrue(report["ok"], report["issues"])
        fp = discovery.footprint(cfg, config)
        self.assertIsNone(fp.mtp)
        self.assertFalse(fp.missing())
        (self.root / "tensors/one.safetensors").unlink()
        self.assertEqual(discovery.footprint(cfg, config).missing()[0].kind, "weights")
        self.assertFalse(doctor.inspect(config)["ok"])


if __name__ == "__main__":
    unittest.main()
