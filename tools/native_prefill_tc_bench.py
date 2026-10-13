"""Private cold-prompt comparison for old decode with optional Tensor Core prefill."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import subprocess
import threading
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
from serve.server import StrataEngine, Vision, child_env, vision_env, parallel_args
from safetensors_tokenizer import SafetensorsTokenizer


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--exe', type=Path)
    p.add_argument('--lengths', default='1024,8192,8192')
    p.add_argument('--new-tokens', type=int, default=32)
    p.add_argument('--timing', action='store_true')
    p.add_argument('--reference', type=Path)
    p.add_argument('--expert-cache', type=int, help='fixed common expert slots for a paired comparison')
    p.add_argument('--prepare-only', action='store_true', help='write an isolated config without loading a model')
    a = p.parse_args()
    a.output = a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=False)
    c = json.loads(a.config.read_text(encoding='utf-8-sig'))
    if a.exe:
        c['exe'] = str(a.exe.resolve())
    flags = c['args']
    flags += parallel_args(c, flags)
    if a.expert_cache is not None:
        flags[flags.index('--expert-cache')+1] = str(a.expert_cache)
    for key, value in [('--conversation-cache-mib', 0), ('--prompt-cache', 0)]:
        flags[flags.index(key)+1] = str(value)
    i = flags.index('--expert-profile')+1
    shutil.copyfile(flags[i], a.output/'profile.bin')
    flags[i] = str(a.output/'profile.bin')
    c['env']['STRATA_DUMP_FIRST_LOGITS'] = str(a.output/'logits.f32')
    if a.timing:
        c['env']['STRATA_PREFILL_TIMING'] = '1'
    env = child_env(c)
    if not a.timing:
        env.pop('STRATA_PREFILL_TIMING', None)
    (a.output/'config.json').write_text(json.dumps(c, indent=2), encoding='utf8')
    if a.prepare_only:
        print('Prepared', a.output/'config.json', '; no model started')
        return
    active = subprocess.check_output(['powershell.exe','-NoProfile','-Command', "@(Get-Process -Name 'strata*' -ErrorAction SilentlyContinue).Count"], text=True)
    assert int(active.strip() or '0') == 0, 'Another Strata process is running; leave it alone'
    tok = SafetensorsTokenizer.from_directory(c['tokenizer'])
    prefix = tok.encode('<|im_start|>user\n', parse_special=True)
    tail = tok.encode('\nSummarize this report in detail.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n', parse_special=True)
    texts = [tok.encode('The report describes a schedule, a budget, and a plan for checking progress. '),
             tok.encode((ROOT/'docs/DETAILS.md').read_text(encoding='utf8'))]
    reference = json.loads((a.reference/'results.json').read_text(encoding='utf8')) if a.reference else None
    result = dict(config=c, exe_sha256=hashlib.sha256(Path(c['exe']).read_bytes()).hexdigest(),
                  diagnostic_timing=a.timing, status='running', requests=[])
    def save():
        (a.output/'results.json').write_text(json.dumps(result, indent=2), encoding='utf8')
    engine = vision = None
    vision_log = None
    save()
    try:
        if c.get('vision'):
            vision_log = (a.output/'vision.log').open('w', encoding='utf8')
            vision = Vision(c['vision'], log=vision_log, env=vision_env(c, env))
        engine = StrataEngine(c['exe'], flags, cwd=c['cwd'], log=str(a.output/'engine.log'), env=env)
        result['info'] = dict(engine.info)
        for j, length in enumerate(map(int, a.lengths.split(','))):
            unit = texts[1 if j >= 2 else 0]
            body = (unit*((length+len(unit)-1)//len(unit)))[:length-len(prefix)-len(tail)]
            ids = prefix+body+tail
            out = [v for v in engine.generate(ids, a.new_tokens, {'temperature': 0}, threading.Event()) if v is not None]
            path = a.output/f'logits.f32.{j}'
            logits = np.fromfile(path, dtype='<f4')
            assert logits.shape == (248320,) and np.isfinite(logits).all()
            row = dict(input_tokens=len(ids), input_sha256=hashlib.sha256(np.asarray(ids, dtype='<i4').tobytes()).hexdigest(),
                       ids=out, logits_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), **engine.last)
            assert row['reused'] == 0 and row['file_blobs'] == row['file_mb'] == 0
            if reference:
                ref = reference['requests'][j]
                assert row['input_sha256'] == ref['input_sha256']
                assert row['ids'] == ref['ids'] and row['logits_sha256'] == ref['logits_sha256'], 'changed arithmetic'
            result['requests'].append(row)
            save()
            print(j, length, row['prompt_ms'], 'decode_tps', round(len(out)*1000/row['decode_ms'], 2), flush=True)
        result['status'] = 'completed'
    finally:
        if engine:
            proc = engine.proc
            engine.close()
            result['exit_code'] = proc.returncode
        if vision:
            vision.shutdown()
        if vision_log:
            vision_log.close()
        save()
    assert result['exit_code'] == 0


if __name__ == '__main__':
    main()
