"""Real MXFP8 attention/shared projection parity against an independent FP64 oracle.

Exercises production stream_dense, including GDN head order and indexer slices.
This checks linear projections, not complete attention/softmax or model logits.
"""
import argparse
import json
from pathlib import Path
import struct
import subprocess

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--exe', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    root = a.model.resolve()
    if a.output.resolve().is_relative_to(root):
        raise ValueError('report must be outside the checkpoint')
    descriptors = {}
    for path in sorted(root.rglob('*.safetensors')):
        with path.open('rb') as f:
            size = struct.unpack('<Q', f.read(8))[0]
            header = json.loads(f.read(size))
        for name, d in header.items():
            if name != '__metadata__':
                descriptors[name] = (path, size + 8, d)

    def read(name):
        path, base, d = descriptors[name]
        assert d['dtype'] in ('F8_E4M3', 'U8')
        with path.open('rb') as f:
            lo, hi = d['data_offsets']
            f.seek(base + lo)
            raw = f.read(hi - lo)
        assert len(raw) == hi - lo
        return np.frombuffer(raw, np.uint8).reshape(d['shape'])

    # Independent named mappings, not the adapter's reported geometry.
    cases = [
        ('blk.0.attn_qkv.weight', 0, 'linear_attn.in_proj_qkv', 'qkv'),
        ('blk.0.attn_gate.weight', 0, 'linear_attn.in_proj_z', 'rows'),
        ('blk.0.ssm_alpha.weight', 0, 'linear_attn.in_proj_a', 'heads'),
        ('blk.0.ssm_out.weight', 0, 'linear_attn.out_proj', 'cols'),
        ('blk.3.attn_q.weight', 3, 'self_attn.q_proj', None),
        ('blk.3.attn_k.weight', 3, 'self_attn.k_proj', None),
        ('blk.3.attn_v.weight', 3, 'self_attn.v_proj', None),
        ('blk.3.attn_output.weight', 3, 'self_attn.o_proj', None),
        ('blk.3.indexer.q_proj.weight', 3, 'self_attn.indexer.index_qk_proj', 'iq'),
        ('blk.3.indexer.k_proj.weight', 3, 'self_attn.indexer.index_qk_proj', 'ik'),
        ('blk.47.ffn_gate_shexp.weight', 47, 'mlp.shared_expert.gate_proj', None),
        ('blk.47.ffn_down_shexp.weight', 47, 'mlp.shared_expert.down_proj', None),
    ]
    native = json.loads(subprocess.check_output(
        [str(a.exe.resolve()), 'dense-probe', str(root), *[c[0] for c in cases]], text=True))
    lut = np.zeros(256, np.float64)
    for i in range(256):
        e, m = (i >> 3) & 15, i & 7
        lut[i] = (-1 if i & 128 else 1) * (np.ldexp(1 + m / 8, e - 7) if e else m / 512)
    results = []
    for case, got in zip(cases, native['projections'], strict=True):
        binding, layer, suffix, order = case
        name = f'model.language_model.layers.{layer}.{suffix}'
        codes, scales = read(name + '.weight'), read(name + '.weight_scale')
        assert np.all((codes & 127) != 127) and np.all(scales != 255)
        w = np.ldexp(lut[codes], np.repeat(scales.astype(np.int32) - 127, 32, axis=1))
        # All sampled real values must be exactly representable by BF16.
        f = w.astype(np.float32)
        assert np.array_equal(f.astype(np.float64), w)
        assert np.all((f.view(np.uint32) & 65535) == 0)
        if order in ('rows', 'heads', 'qkv'):
            prefix = 4096 if order == 'qkv' else 0
            head = 1 if order == 'heads' else 128
            tail = w[prefix:].reshape(16, 3, head, w.shape[1]).transpose(1, 0, 2, 3)
            w = np.concatenate((w[:prefix], tail.reshape(-1, w.shape[1])))
        elif order == 'cols':
            w = w.reshape(w.shape[0], 16, 3, 128).transpose(0, 2, 1, 3).reshape(w.shape)
        elif order == 'iq':
            w = w[:512]
        elif order == 'ik':
            w = w[512:]
        assert got['name'] == binding and got['source'] == name + '.weight'
        assert w.shape == (got['rows'], got['columns'])
        j = np.arange(w.shape[1])
        errors = []
        for key, x in [('y0', (j % 31 - 15) / 16), ('y1', ((j * 17) % 97 - 48) / 64)]:
            expected = np.sum(w * x, axis=1)
            actual = np.array(got[key])
            error = float(np.max(np.abs(actual - expected)))
            assert np.allclose(actual, expected, rtol=1e-12, atol=1e-12), (binding, key, error)
            errors.append(error)
        results.append({'name': binding, 'elements': int(w.size), 'max_abs': max(errors)})
        print(f'{binding}: PASS, max_abs={max(errors):.3g}', flush=True)
    a.output.write_text(json.dumps({'status': 'pass', 'scope': __doc__, 'source': str(root),
                                   'projections': results}, indent=2), encoding='utf8')


if __name__ == '__main__':
    main()
