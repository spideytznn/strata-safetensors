"""Full-model A/side/B/A regression for parked shared-checkpoint borrowers.

Private sequential engines; never alters the desktop profile or original
expert placement file. Numerical checks use all experts and fixed placement.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import time

from native_concurrency_acceptance import Protocol
from safetensors_tokenizer import SafetensorsTokenizer


def request(engine, ids, limit):
    started = time.perf_counter()
    engine.send(f"GEN {limit} temperature=0 {','.join(map(str, ids))}")
    out, resume, progress = [], None, []
    while True:
        line = engine.read()
        if line.startswith('T '):
            out.append(int(line.split()[1]))
        elif line.startswith('RESUME '):
            resume = int(line.split()[1])
        elif line.startswith('PROGRESS '):
            progress.append(line)
        elif line.startswith('DONE '):
            fields = line.split()
            assert int(fields[1]) == len(out) <= limit, line
            assert int(fields[12]) == 0 and float(fields[13]) == 0, line
            return {'ids': out, 'resume': resume, 'done': line,
                    'wall_s': time.perf_counter() - started, 'progress': progress}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cfg0 = json.loads(args.config.read_text(encoding='utf-8-sig'))
    cfg0 = cfg0.get('config', cfg0)
    tok = SafetensorsTokenizer.from_directory(cfg0['tokenizer'])
    header = tok.encode('<|im_start|>user\n', parse_special=True)
    tail = tok.encode('\nWrite a detailed, 2000-word explanation of why schedules are useful, '
                      'with many examples.<|im_end|>\n<|im_start|>assistant\n'
                      '<think>\n\n</think>\n\n', parse_special=True)

    def body(text, count):
        piece = tok.encode(text)
        return (piece * ((count + len(piece) - 1) // len(piece)))[:count]

    parent = header + body('A report covers the schedule and budget. ', 59697 - len(header) - len(tail)) + tail
    child = parent[:19968] + body('A separate research team studies rainfall and river flow. ',
                                 36470 - 19968 - len(tail)) + tail
    side = header + tok.encode('Reply with a short definition of a queue.') + tail
    follow = tok.encode('<|im_end|>\n<|im_start|>user\nGive one concrete example in detail.'
                        '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n', parse_special=True)
    profile = Path(cfg0['args'][cfg0['args'].index('--expert-profile') + 1]).read_bytes()
    result = {'passed': False, 'scope': 'Finite full-model cache branch regression; greedy, tail skip 0, '
              'PCIe fraction 0, frozen placement. Not production-speed or general-quality evidence.',
              'prompt_tokens': {'parent': len(parent), 'child': len(child)}, 'runs': []}

    def save():
        (args.output / 'results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')

    save()
    for name, exe, enabled in [('original', args.baseline, False),
                               ('candidate-disabled', args.candidate, False),
                               ('candidate-enabled', args.candidate, True)]:
        folder = args.output / name
        folder.mkdir()
        cfg = copy.deepcopy(cfg0)
        cfg['exe'] = str(exe.resolve())
        cfg['env']['STRATA_CONVERSATION_BRANCH_PRESERVE'] = '1' if enabled else '0'
        cfg['env']['STRATA_ROUTE_TAIL_SKIP'] = '0'
        cfg['env']['STRATA_NVFP4_F32_CANONICAL'] = '1'
        flags = cfg['args']
        for flag, value in [('--batch', 2), ('--max-context', 262144), ('--prompt-cache', 6),
                            ('--conversation-cache-mib', 8192), ('--conversation-cache-slots', 4),
                            ('--adapt-swaps', 0), ('--adapt-every', 1000000), ('--pcie-frac', 0)]:
            if flag in flags:
                flags[flags.index(flag) + 1] = str(value)
            else:
                flags.extend([flag, str(value)])
        own_profile = folder / 'expert-profile.bin'
        own_profile.write_bytes(profile)
        flags[flags.index('--expert-profile') + 1] = str(own_profile.resolve())
        run = {'name': name, 'config': cfg, 'exe_sha256': hashlib.sha256(exe.read_bytes()).hexdigest(),
               'requests': {}}
        result['runs'].append(run)
        save()
        print(f'{name}: loading full model', flush=True)
        engine = Protocol(cfg, folder)
        try:
            run['info'] = engine.info
            assert any('conversation_cache_mib=8192' in line for line in engine.info), engine.info
            for label, ids, limit in [('parent', parent, 64)]:
                run['requests'][label] = request(engine, ids, limit)
                save()
            first = run['requests']['parent']
            assert len(first['ids']) == 64, 'parent ended before the intended cap'
            parent_next = parent + first['ids'] + follow
            for label, ids, limit in [('reference-followup', parent_next, 48), ('side', side, 16),
                                      ('child', child, 32), ('returned-parent', parent_next, 48)]:
                item = request(engine, ids, limit)
                run['requests'][label] = item
                save()
                print(f'{name}: {label}, prompt={len(ids)}, reused={item["resume"]}, '
                      f'wall={item["wall_s"]:.3f}s', flush=True)
            reference = run['requests']['reference-followup']
            returned = run['requests']['returned-parent']
            assert reference['ids'] == returned['ids'], f'{name}: restored output differs from undisturbed reference'
            assert run['requests']['child']['resume'] == 16384, 'test did not borrow 16K parked checkpoint'
            if enabled:
                assert returned['resume'] >= len(parent), 'enabled path lost the deep parent checkpoint'
            else:
                assert returned['resume'] == 16384, 'original failure did not reproduce'
            run['passed'] = True
        finally:
            engine.close()
            run['exit_code'] = engine.proc.returncode
            save()
        assert run['exit_code'] == 0
        log = engine.log_path.read_text(encoding='utf8')
        run['preserved_branches'] = re.findall(r'branch source preserved (\d+) tokens; borrowed=(\d+)', log)
        run['post_residency_expert_file_bytes'] = list(map(int, re.findall(r'expert_file_bytes=(\d+)', log)))
        assert not any(run['post_residency_expert_file_bytes'])
        if enabled:
            assert any(int(n) > len(parent) and int(prefix) == 16384 for n, prefix in run['preserved_branches'])
        save()
    original, disabled, enabled = result['runs']
    for label in original['requests']:
        assert original['requests'][label]['ids'] == disabled['requests'][label]['ids'], f'disabled differs: {label}'
        assert original['requests'][label]['ids'] == enabled['requests'][label]['ids'], f'enabled differs: {label}'
    result['passed'] = True
    save()
    print('PASS: original reread reproduced, disabled matches, enabled preserves deep parent and exact tokens', flush=True)


if __name__ == '__main__':
    main()
