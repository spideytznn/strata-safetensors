"""Private two-slot MTP A/B. Clone expert profiles, retain tokens and time both admissions and decode.

No HTTP listener or desktop setting is changed. Each profile starts from the same expert placement.
Warmups are retained but excluded from the reported median. Requires otherwise idle model resources.
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


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--exe', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--modes', nargs='+', choices=['off', 'serial', 'async'], default=['off', 'serial', 'async'])
    parser.add_argument('--rounds', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--new', type=int, default=256)
    parser.add_argument('--context', type=int, default=262144)
    parser.add_argument('--temperature', type=float, default=0)
    parser.add_argument('--thinking', action='store_true')
    args = parser.parse_args()
    assert args.new >= 64 and args.rounds > 0 and args.repeats > 0 and args.temperature >= 0
    args.output.mkdir(parents=True, exist_ok=False)
    base = json.loads(args.config.read_text(encoding='utf-8-sig'))
    tok = SafetensorsTokenizer.from_directory(base['tokenizer'])
    template = ChatTemplate(Path(base['chat_template']))
    document = tok.encode((ROOT / 'docs/DETAILS.md').read_text(encoding='utf-8'))
    document = tok.decode((document * 3)[:8000])
    fixtures = [
        ('text', ['Explain TCP and QUIC in depth, including congestion control and multiplexing. Write at least 1000 words.',
                  '详细解释数据库事务隔离级别，每个级别给出并发读写例子。至少写两千字。']),
        ('code', ['Write a complete Python LRU cache with detailed tests and a long explanation of all edge cases.',
                  '写一个完整 Python 红黑树实现，并给出详细的插入、删除和边界测试。']),
        ('doc8k', [document + '\nAnalyze this document in detail. Write a comprehensive technical review.',
                   document + '\n请详细分析本文的性能取舍，给出具体实现例子。']),
    ]
    prompts = [(name, [tok.encode(template.render([{'role': 'user', 'content': text}], enable_thinking=args.thinking),
                                 parse_special=True) for text in texts]) for name, texts in fixtures]
    profile = args.output / 'initial-expert-profile.bin'
    shutil.copyfile(base['args'][base['args'].index('--expert-profile') + 1], profile)
    result = dict(status='running', exe=str(args.exe.resolve()),
                  exe_sha256=hashlib.sha256(args.exe.read_bytes()).hexdigest(),
                  initial_profile_sha256=hashlib.sha256(profile.read_bytes()).hexdigest(), runs=[],
                  temperature=args.temperature, thinking=args.thinking,
                  fixtures={name: [dict(tokens=len(p), sha256=hashlib.sha256(','.join(map(str, p)).encode()).hexdigest())
                                   for p in ids] for name, ids in prompts},
                  note='Two active slots, vision loaded first, adaptive exchange, INT8 KV, desktop workspace. '
                       'Total throughput includes admissions; steady throughput counts tokens after both first tokens. '
                       'Warmups excluded. Seeded target output parity required across modes.')
    report = args.output / 'results.json'
    def save():
        report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    references = {}
    save()
    for round_index in range(args.rounds):
        modes = args.modes if round_index % 2 == 0 else list(reversed(args.modes))
        for mode in modes:
            with socket.socket() as probe:
                probe.settimeout(1)
                assert probe.connect_ex(('127.0.0.1', 8880)) != 0, 'A desktop service started; leave it running and stop this benchmark'
            cfg = copy.deepcopy(base)
            cfg['exe'] = str(args.exe.resolve())
            flags = cfg['args']
            for flag in ('--mtp', '--spec', '--spec-min-p', '--batch', '--max-context', '--expert-profile'):
                while flag in flags:
                    at = flags.index(flag)
                    del flags[at:at+2]
            while '--batch-mtp' in flags:
                flags.remove('--batch-mtp')
            folder = args.output / f'r{round_index+1}-{mode}'
            folder.mkdir()
            own_profile = folder / 'expert-profile.bin'
            shutil.copyfile(profile, own_profile)
            flags += ['--batch', '2', '--max-context', str(args.context), '--expert-profile', str(own_profile.resolve())]
            flags += ['--spec', '2' if mode == 'off' else '3']
            if mode != 'off':
                flags += ['--mtp', 'native', '--batch-mtp', '--spec-min-p', '0.5']
            cfg['env']['STRATA_NATIVE_BATCH_MTP_ASYNC'] = '1' if mode == 'async' else '0'
            run = dict(round=round_index+1, mode=mode, config=cfg, requests=[])
            result['runs'].append(run)
            engine = None
            print(f'START round={round_index+1} mode={mode}', flush=True)
            try:
                begin = time.perf_counter()
                engine = Protocol(cfg, folder, vision=True)
                run.update(startup_s=time.perf_counter()-begin, info=engine.info)
                for name, ids in prompts:
                    for repeat in range(args.repeats+1):
                        keys = f'temperature={args.temperature} top_p=0.95 top_k=20 seed=9950'
                        row = engine.batch([(p, args.new, keys) for p in ids])
                        row.update(case=name, repeat=repeat, warmup=repeat == 0, sampling=keys)
                        run['requests'].append(row)
                        start = max(job['times'][0] for job in row['jobs'])
                        end = max(job['times'][-1] for job in row['jobs'])
                        count = sum(sum(t > start for t in job['times']) for job in row['jobs'])
                        row['total_tok_s'] = sum(len(job['ids']) for job in row['jobs']) / row['wall_s']
                        row['steady_tok_s'] = count / (end-start)
                        for slot, job in enumerate(row['jobs']):
                            assert len(job['ids']) == args.new, 'Early EOS invalidates fixed-work timing'
                            key = (name, slot)
                            reference = references.setdefault(key, job['ids'])
                            assert job['ids'] == reference, f'Target output mismatch: {mode} {key}'
                            job['decode_tok_s'] = (len(job['ids'])-1) / (job['times'][-1]-job['times'][0])
                        save()
                        print(f'{mode} {name} repeat={repeat}: total={row["total_tok_s"]:.2f}, '
                              f'steady={row["steady_tok_s"]:.2f} tok/s', flush=True)
            except BaseException as error:
                run.update(status='fail', error=repr(error))
                result['status'] = 'fail'
                save()
                raise
            finally:
                if engine:
                    engine.close()
            log = (folder / 'engine.log').read_text(encoding='utf-8')
            for family in ('expert', 'MTP'):
                reads = re.findall(r'post-residency ' + family + r' source bytes=(\d+)', log)
                assert reads and all(int(n) == 0 for n in reads), (mode, family, reads)
            assert engine.proc.returncode == 0
            run['status'] = 'pass'
            save()
    result['medians'] = {
        mode: {name: {metric: statistics.median(row[metric] for run in result['runs'] if run['mode'] == mode
                                               for row in run['requests'] if row['case'] == name and not row['warmup'])
                     for metric in ('total_tok_s', 'steady_tok_s')}
               for name, _ in prompts}
        for mode in args.modes}
    result['status'] = 'pass'
    save()
    print(json.dumps(result['medians'], indent=2), flush=True)


if __name__ == '__main__':
    main()
