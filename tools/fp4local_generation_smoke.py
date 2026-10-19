"""Small deterministic text smoke comparison, not a model-quality benchmark.

Pass --config repeatedly for sequential private Strata engines, or an existing
--infernix-url. Results retain prompts, token IDs, output caps and timings.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import threading
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
from serve.server import StrataEngine, child_env
from serve.frontend import ChatTemplate
from safetensors_tokenizer import SafetensorsTokenizer

PROMPTS = {
    'english': 'Explain why the sky is blue in two sentences.',
    'chinese': '用中文解释二分查找的前提和时间复杂度，给出一个简短例子。',
    'code': 'Write a Python function returning the first index of a target in a sorted list, or -1.',
    'math': 'Calculate (17 * 23) - 19. Give only the answer.',
    'eos': 'Reply with exactly the single word Blue and stop.',
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, action='append', default=[])
    ap.add_argument('--infernix-url')
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--max-new', type=int, default=192)
    a = ap.parse_args()
    assert a.config or a.infernix_url
    a.output.mkdir(parents=True, exist_ok=False)
    report = {'scope': __doc__, 'max_new': a.max_new, 'temperature': 0, 'thinking': False, 'runs': {}}

    def save():
        (a.output / 'results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')

    for path in a.config:
        cfg = json.loads(path.read_text(encoding='utf8'))
        label = path.stem
        tok = SafetensorsTokenizer.from_directory(cfg['tokenizer'])
        template = ChatTemplate(Path(cfg['chat_template']))
        run = {'config': cfg, 'exe_sha256': hashlib.sha256(Path(cfg['exe']).read_bytes()).hexdigest(), 'requests': []}
        report['runs'][label] = run
        save()
        start = time.monotonic()
        engine = StrataEngine(cfg['exe'], cfg['args'], cwd=cfg['cwd'],
                              log=str((a.output / (label + '.log')).resolve()), env=child_env(cfg))
        run.update(startup_s=time.monotonic() - start, info=engine.info)
        try:
            for name, text in PROMPTS.items():
                rendered = template.render([{'role': 'user', 'content': text}], enable_thinking=False)
                ids = tok.encode(rendered, parse_special=True)
                start = time.monotonic()
                out = [v for v in engine.generate(ids, a.max_new, {'temperature': 0}, threading.Event()) if v is not None]
                run['requests'].append({'name': name, 'prompt': text, 'rendered': rendered,
                    'input_ids': ids, 'ids': out, 'text': tok.decode(out),
                    'wall_s': time.monotonic() - start, 'metrics': dict(engine.last)})
                save()
                assert out and len(out) <= a.max_new
                print(f'{label} {name}: {len(out)} tokens', flush=True)
        finally:
            engine.close()
        run['status'] = 'pass'
        save()
    if a.infernix_url:
        run = {'url': a.infernix_url, 'requests': []}
        report['runs']['infernix'] = run
        for name, text in PROMPTS.items():
            payload = {'model': 'Qwen3.8-Flash-Next-NVFP4-Infernix',
                'messages': [{'role': 'user', 'content': text}], 'temperature': 0,
                'reasoning_effort': 'none', 'max_tokens': a.max_new, 'stream': False}
            req = urllib.request.Request(a.infernix_url.rstrip('/') + '/v1/chat/completions',
                json.dumps(payload).encode('utf8'), {'Content-Type': 'application/json'})
            start = time.monotonic()
            with urllib.request.urlopen(req, timeout=300) as response:
                result = json.load(response)
            run['requests'].append({'name': name, 'prompt': text,
                'text': result['choices'][0]['message']['content'],
                'wall_s': time.monotonic() - start, 'response': result})
            save()
            print(f'infernix {name}: {result.get("usage", {})}', flush=True)
        run['status'] = 'pass'
        save()
    report['status'] = 'pass'
    save()


if __name__ == '__main__':
    main()
