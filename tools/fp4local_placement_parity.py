"""Isolate CPU/GPU placement from adaptive cache exchange on one private engine."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
from serve.server import StrataEngine, child_env
from serve.frontend import ChatTemplate
from safetensors_tokenizer import SafetensorsTokenizer
from conversation_cache_parity import state_hashes


def summarize(report):
    runs = report['runs']
    assert all(r['metrics']['reused'] == 0 for r in runs), 'unexpected prompt cache reuse'
    assert runs[0]['metrics']['offloaded'] == 0 and runs[1]['metrics']['offloaded'] > runs[1]['metrics']['lookups'] and runs[2]['metrics']['offloaded'] == 0, 'placement override was not applied'
    report['repeat_cpu_tokens_equal'] = runs[0]['ids'] == runs[2]['ids']
    report['cpu_gpu_tokens_equal'] = runs[0]['ids'] == runs[1]['ids']
    states = report['state_hashes']
    report['repeat_cpu_state_equal'] = len(states) == 3 and states[0] == states[2]
    report['cpu_gpu_state_equal'] = len(states) == 3 and states[0] == states[1]
    report['first_cpu_gpu_token_difference'] = next((i for i, (x, y) in enumerate(zip(runs[0]['ids'], runs[1]['ids'])) if x != y), None)
    return report


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    cfg = json.loads(a.config.read_text(encoding='utf8'))
    args = list(cfg['args'])
    for flag in ('--mtp',):
        if flag in args:
            at = args.index(flag)
            del args[at:at+2]
    for flag in ('--batch-mtp',):
        if flag in args:
            args.remove(flag)
    for flag, value in (('--max-context', 4096), ('--prefill', 256),
                        ('--expert-cache', 1024), ('--adapt-swaps', 0),
                        ('--prompt-cache', 6), ('--conversation-cache-mib', 0),
                        ('--spec', 2), ('--suffix-draft', 0)):
        if flag in args:
            args[args.index(flag)+1] = str(value)
        else:
            args.extend([flag, str(value)])
    env = child_env(cfg)
    env['STRATA_STATE_HASH'] = '1'
    env['STRATA_NATIVE_PREFILL_FIRST'] = '0'
    tok = SafetensorsTokenizer.from_directory(cfg['tokenizer'])
    template = ChatTemplate(Path(cfg['chat_template']))
    prompt = 'Explain how a CPU and a GPU cooperate to run a large language model on a desktop PC. Cover memory bandwidth, expert caching, and batch scheduling in detail.'
    ids = tok.encode(template.render([{'role': 'user', 'content': prompt}],
                                    enable_thinking=False), parse_special=True)
    log = a.output / 'engine.log'
    report = {'args': args, 'exe_sha256': hashlib.sha256(Path(cfg['exe']).read_bytes()).hexdigest(),
              'prompt': prompt, 'runs': []}
    engine = StrataEngine(cfg['exe'], args, cwd=cfg['cwd'], log=str(log.resolve()), env=env)
    try:
        report['engine'] = dict(engine.info)
        for frac in (0, 1, 0):
            output = [v for v in engine.generate(ids, 256, {'temperature': 0, 'strata_tune': {'pcie_frac': frac}, 'strata_checkpoint': False},
                                                threading.Event()) if v is not None]
            report['runs'].append({'pcie_frac': frac, 'ids': output, 'text': tok.decode(output),
                                   'metrics': dict(engine.last)})
            (a.output / 'results.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf8')
            print('pcie_frac', frac, 'tokens', len(output), flush=True)
    finally:
        engine.close()
        report['state_hashes'] = state_hashes(log.read_text(encoding='utf8'))
        (a.output / 'results.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf8')
    summarize(report)
    (a.output / 'results.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf8')
    print(json.dumps({k: v for k, v in report.items() if k.endswith('equal') or k.startswith('first_')}), flush=True)
    assert report['repeat_cpu_tokens_equal'] and report['repeat_cpu_state_equal'], 'fixed CPU placement is not repeatable'


if __name__ == '__main__':
    main()
