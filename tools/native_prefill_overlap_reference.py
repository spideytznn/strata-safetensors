"""Finite full-model check that the disabled overlap path matches the deployed engine."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

from native_concurrency_acceptance import Protocol
from safetensors_tokenizer import SafetensorsTokenizer
from serve.frontend import ChatTemplate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    original = json.loads(args.config.read_text(encoding='utf-8-sig'))
    tokenizer = SafetensorsTokenizer.from_directory(original['tokenizer'])
    template = ChatTemplate(Path(original['chat_template']))
    prompts = [tokenizer.encode(template.render([{'role': 'user', 'content': text}],
                    enable_thinking=False), parse_special=True) for text in (
        'Explain TCP and QUIC in detail, including congestion control and multiplexing.',
        '写一个 Python 二分查找函数，解释边界条件和时间复杂度。')]
    profile = Path(original['args'][original['args'].index('--expert-profile')+1]).read_bytes()
    result = {'passed': False, 'runs': [], 'scope':
        'Two cold greedy prompts, 96 output-token cap, same fixed placement, no prefix reuse or PCIe split; overlap disabled.'}
    for name, executable in [('deployed', args.baseline), ('candidate-disabled', args.candidate)]:
        folder = args.output / name
        folder.mkdir(parents=True, exist_ok=True)
        cfg = copy.deepcopy(original)
        cfg['exe'] = str(executable.resolve())
        cfg['env']['STRATA_NATIVE_PREFILL_OVERLAP'] = '0'
        flags = cfg['args']
        if '--batch' not in flags:
            flags.extend(['--batch', '2'])
        for flag, value in [('--adapt-swaps', '0'), ('--adapt-every', '1000000')]:
            flags[flags.index(flag)+1] = value
        own_profile = folder / 'expert-profile.bin'
        own_profile.write_bytes(profile)
        flags[flags.index('--expert-profile')+1] = str(own_profile.resolve())
        engine = Protocol(cfg, folder, vision=True)
        try:
            requests = [engine.solo(prompt, 96, 'temperature=0') for prompt in prompts]
            result['runs'].append({'name': name, 'exe_sha256': hashlib.sha256(executable.read_bytes()).hexdigest(),
                                   'config': cfg, 'info': engine.info, 'requests': requests})
        finally:
            engine.close()
        (args.output / 'results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        assert engine.proc.returncode == 0
    first, second = result['runs']
    assert [r['ids'] for r in first['requests']] == [r['ids'] for r in second['requests']], 'Disabled path output mismatch'
    result['passed'] = True
    (args.output / 'results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print('PASS: disabled overlap matches the deployed engine on both cold greedy prompts', flush=True)


if __name__ == '__main__':
    main()
