"""Private-stdin A/B of slot-layer commits; same executable, cloned profiles, no HTTP listener.

Greedy output identity, seeded/cancellation checks, ABBA engine ordering and paired steady throughput.
Requires exclusive model resources. All child processes are closed, including on failure.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import shutil
import socket
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
from native_concurrency_acceptance import Protocol
from safetensors_tokenizer import SafetensorsTokenizer
from serve.frontend import ChatTemplate


def functional(engine, prompts):
    results = []
    keys = ['temperature=0.7 top_p=0.95 top_k=20 seed=6101',
            'temperature=0.7 top_p=0.95 top_k=20 seed=2718']
    refs = [engine.solo(p, 64, key) for p, key in zip(prompts, keys)]
    batch = engine.batch([(p, 64, key) for p, key in zip(prompts, keys)])
    assert all(a['ids'] == b['ids'] for a, b in zip(refs, batch['jobs'])), 'seeded solo/batch mismatch'
    results.append(dict(case='seeded', batch=batch))
    keys = ['temperature=0 spec_min_p=0', 'temperature=0 spec_min_p=1']
    refs = [engine.solo(p, 64, key) for p, key in zip(prompts, keys)]
    batch = engine.batch([(p, 64, key) for p, key in zip(prompts, keys)])
    assert all(a['ids'] == b['ids'] for a, b in zip(refs, batch['jobs'])), 'mixed row counts mismatch'
    results.append(dict(case='mixed_confidence', batch=batch))
    keys = 'temperature=0'
    refs = [engine.solo(p, 64, keys) for p in prompts]
    batch = engine.batch([(p, 64, keys) for p in prompts], cancel_after=12)
    prefix = batch['jobs'][0]['ids']
    assert prefix == refs[0]['ids'][:len(prefix)] and batch['jobs'][1]['ids'] == refs[1]['ids'], 'cancel corrupted a slot'
    remaining = 64 - len(prefix)
    resumed = engine.solo(prompts[0] + prefix, remaining, keys) if remaining else {'ids': []}
    assert prefix + resumed['ids'] == refs[0]['ids'], 'cancel/resume mismatch'
    results.append(dict(case='cancel_resume', batch=batch, resumed=resumed))
    followups = [p + r['ids'] for p, r in zip(prompts, refs)]
    refs = [engine.solo(p, 32, keys) for p in followups]
    batch = engine.batch([(p, 32, keys) for p in followups])
    assert all(a['ids'] == b['ids'] for a, b in zip(refs, batch['jobs'])), 'slot reuse mismatch'
    results.append(dict(case='slot_reuse', batch=batch))
    return results


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--exe', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rounds', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--new', type=int, default=192)
    args = parser.parse_args()
    assert args.rounds > 0 and args.repeats > 0 and args.new >= 64
    args.output.mkdir(parents=True, exist_ok=False)
    base = json.loads(args.config.read_text(encoding='utf-8-sig'))
    tok = SafetensorsTokenizer.from_directory(base['tokenizer'])
    template = ChatTemplate(Path(base['chat_template']))
    document = tok.encode((ROOT / 'docs/DETAILS.md').read_text(encoding='utf-8'))
    document = tok.decode((document * 3)[:8000])
    texts = [
        ('text', ['Explain TCP and QUIC in depth, including congestion control and multiplexing. Write at least 1000 words.',
                  '详细解释数据库事务隔离级别，每个级别给出并发读写例子。至少写两千字。']),
        ('code', ['Write a complete Python LRU cache with detailed tests and a long explanation of all edge cases.',
                  '写一个完整 Python 红黑树实现，并给出详细的插入、删除和边界测试。']),
        ('doc8k', [document + '\nAnalyze this document in detail. Write a comprehensive technical review.',
                   document + '\n请详细分析本文的性能取舍，给出具体实现例子。']),
    ]
    prompts = [(name, [tok.encode(template.render([{'role': 'user', 'content': text}], enable_thinking=False),
                                 parse_special=True) for text in pair]) for name, pair in texts]
    initial = args.output / 'initial-expert-profile.bin'
    shutil.copyfile(base['args'][base['args'].index('--expert-profile') + 1], initial)
    result = dict(status='running', exe=str(args.exe.resolve()),
                  exe_sha256=hashlib.sha256(args.exe.read_bytes()).hexdigest(),
                  profile_sha256=hashlib.sha256(initial.read_bytes()).hexdigest(), runs=[],
                  scope='Desktop BF16 KV / 8192 prefill / native MTP / tail skip off. Only slot commit flag differs.',
                  warmups_excluded=True, greedy_identity_required=True)
    report = args.output / 'results.json'
    def save():
        report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    references = {}
    save()
    for rnd in range(args.rounds):
        for mode in (['off', 'on'] if rnd % 2 == 0 else ['on', 'off']):
            with socket.socket() as probe:
                assert probe.connect_ex(('127.0.0.1', 8880)) != 0, 'User service started; stop private tests'
            folder = args.output / f'r{rnd+1}-{mode}'
            folder.mkdir()
            cfg = copy.deepcopy(base)
            cfg['exe'] = str(args.exe.resolve())
            profile = folder / 'expert-profile.bin'
            shutil.copyfile(initial, profile)
            flags = cfg['args']
            flags[flags.index('--expert-profile') + 1] = str(profile.resolve())
            if '--batch' in flags:
                flags[flags.index('--batch') + 1] = '2'
            else:
                flags += ['--batch', '2']
            assert '--batch-mtp' in flags and flags[flags.index('--kv') + 1] == 'bf16'
            cfg['env']['STRATA_SLOT_COMMIT_BATCH'] = '1' if mode == 'on' else '0'
            run = dict(round=rnd+1, mode=mode, config=cfg, requests=[])
            result['runs'].append(run)
            engine = None
            print(f'START round={rnd+1} mode={mode}', flush=True)
            try:
                born = time.perf_counter()
                engine = Protocol(cfg, folder, vision=True)
                run.update(startup_s=time.perf_counter()-born, info=engine.info)
                for name, pair in prompts:
                    for repeat in range(args.repeats+1):
                        row = engine.batch([(p, args.new, 'temperature=0 seed=9950') for p in pair])
                        row.update(case=name, repeat=repeat, warmup=repeat == 0)
                        start = max(job['times'][0] for job in row['jobs'])
                        end = max(job['times'][-1] for job in row['jobs'])
                        row['steady_tok_s'] = sum(sum(t > start for t in job['times']) for job in row['jobs']) / (end-start)
                        row['total_tok_s'] = sum(len(job['ids']) for job in row['jobs']) / row['wall_s']
                        for slot, job in enumerate(row['jobs']):
                            assert len(job['ids']) == args.new, 'early EOS invalidates fixed-work timing'
                            reference = references.setdefault((name, slot), job['ids'])
                            assert job['ids'] == reference, f'output changed: {mode} {name} slot={slot}'
                        run['requests'].append(row)
                        save()
                        print(f'{mode} {name} repeat={repeat}: steady={row["steady_tok_s"]:.2f} tok/s', flush=True)
                if mode == 'on' and rnd == 0:
                    run['functional'] = functional(engine, prompts[0][1])
                    print('seeded/mixed-confidence/cancel/resume/slot-reuse checks passed', flush=True)
            except BaseException as error:
                run['error'] = repr(error)
                result['status'] = 'fail'
                save()
                raise
            finally:
                if engine:
                    engine.close()
            assert engine.proc.returncode == 0, 'private engine did not exit cleanly'
            log = (folder / 'engine.log').read_text(encoding='utf-8')
            run['commit_graphs'] = re.findall(r'slot-layer commit graph: groups=(\d+) gdn_batches=(\d+) qsa_batches=(\d+) nodes=(\d+)', log)
            if mode == 'on':
                assert any(g[:3] == ('2','2','2') for g in run['commit_graphs']), 'optimized two-slot path never ran'
            for family in ['expert', 'MTP']:
                reads = re.findall(r'post-residency ' + family + r' source bytes=(\d+)', log)
                assert reads and all(int(n) == 0 for n in reads), 'unexpected runtime weight reads'
            run['status'] = 'pass'
            save()
    result['medians'] = {
        mode: {name: statistics.median(row['steady_tok_s'] for run in result['runs'] if run['mode'] == mode
                                      for row in run['requests'] if row['case'] == name and not row['warmup'])
               for name, _ in prompts} for mode in ['off','on']}
    result['status'] = 'pass'
    save()
    print(json.dumps(result['medians'], indent=2), flush=True)


if __name__ == '__main__':
    main()
