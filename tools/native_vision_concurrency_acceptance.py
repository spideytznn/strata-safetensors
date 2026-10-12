"""Private-stdin correctness checks for native image-history slots. Requires exclusive model resources.

Uses generated red/blue pictures. Checks image/image and image/text overlap, seeded and greedy parity,
historical-image continuation, replacement-image misses, cancellation and zero runtime expert file reads.
No speed claim; no HTTP service or user conversation is used.
"""
import argparse
import copy
import hashlib
import json
import re
import shutil
import struct
import sys
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
from native_vision_smoke import solid_png
from safetensors_tokenizer import SafetensorsTokenizer
from serve.frontend import ChatTemplate
from serve.server import Service, StrataEngine, Vision, child_env, vision_env


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--exe', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=96)
    parser.add_argument('--desktop-context', action='store_true', help='Keep the desktop context and prefill allocation.')
    parser.add_argument('--extended', action='store_true', help='Also check long image histories and interrupted prefill.')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = json.loads(args.config.read_text(encoding='utf-8-sig'))
    engine_args = list(cfg['args'])
    def option(key, value):
        if key in engine_args:
            engine_args[engine_args.index(key) + 1] = str(value)
        else:
            engine_args.extend([key, str(value)])
    # Full desktop context is unnecessary for these isolation checks, but use the same kernels, INT8 KV,
    # adaptive exchange, dedicated prefill workspace and source residency settings.
    if not args.desktop_context:
        option('--max-context', 16384)
        option('--prefill', 512)
    option('--batch', 2)
    profile = args.output.resolve() / 'expert-profile.bin'
    if '--expert-profile' in engine_args:
        shutil.copyfile(engine_args[engine_args.index('--expert-profile') + 1], profile)
    option('--expert-profile', profile)
    result = dict(status='running', exe=str(args.exe.resolve()),
                  exe_sha256=hashlib.sha256(args.exe.read_bytes()).hexdigest(), args=engine_args,
                  note='Functional parity only, not a throughput benchmark.', requests=[], pairs=[])
    report = args.output / 'results.json'
    def save():
        report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    save()
    engine = vision = svc = engine_proc = None
    log_path = (args.output / 'engine.log').resolve()
    log = log_path.open('w', encoding='utf-8')
    try:
        env = child_env(cfg)
        vision = Vision(cfg['vision'], log=log, env=vision_env(cfg, env))
        engine = StrataEngine(str(args.exe.resolve()), engine_args, cwd=cfg['cwd'], log=str(log_path), env=env)
        assert engine.info.get('native_vision_slots') == 1, engine.info
        result['info'] = dict(engine.info)
        tok = SafetensorsTokenizer.from_directory(cfg['tokenizer'])
        svc = Service(engine, tok, ChatTemplate(Path(cfg['chat_template'])), vision=vision)
        fixtures = {}
        for name, rgb in [('red', (255, 0, 0)), ('blue', (0, 0, 255))]:
            image = (args.output / (name + '.png')).resolve()
            solid_png(image, rgb)
            fixtures[name] = [{'role': 'user', 'content': [
                {'type': 'image', 'source': str(image)},
                {'type': 'text', 'text': 'Describe this solid color and its use in painting in a long detailed paragraph. '
                 'Begin by naming the color. Continue for at least 300 words.'}]}]
        fixtures['text'] = [{'role': 'user', 'content': 'Write a long detailed paragraph about mountain hiking. '
                            'Continue for at least 300 words.'}]
        lock = threading.Lock()
        def run(name, messages, sampling, barrier=None, cancel_after=None):
            ids, _, limit = svc.prepare(messages, [], {'enable_thinking': False}, max_new=args.limit)
            emb = getattr(svc.embeddings, 'path', None)
            state = {}
            engine.bind_request_status(state)
            cancel = threading.Event()
            out = []
            try:
                if barrier:
                    barrier.wait(timeout=30)
                for token in engine.generate(ids, limit, sampling, cancel, embeddings=emb):
                    if token is not None:
                        out.append(token)
                        if cancel_after and len(out) >= cancel_after:
                            cancel.set()
                row = dict(name=name, input_tokens=len(ids), image_tokens=ids.count(248056), ids=out,
                           text=tok.decode(out), last=copy.deepcopy(engine.last))
                assert row['last'].get('file_blobs', 0) == row['last'].get('file_mb', 0) == 0, row
                with lock:
                    result['requests'].append(row)
                    save()
                return row
            finally:
                engine.bind_request_status(None)
                svc.drop_embeddings()
        def pair(name, messages, sampling, cancel_first=False):
            barrier = threading.Barrier(2)
            observed = set()
            with ThreadPoolExecutor(max_workers=2) as pool:
                tasks = [pool.submit(run, name + '-' + str(i), m, {**sampling, 'seed': sampling.get('seed', 9950) + i},
                                     barrier, 12 if cancel_first and i == 0 else None)
                         for i, m in enumerate(messages)]
                while not all(t.done() for t in tasks):
                    with engine.slot_cv:
                        observed.add(sum(engine.slot_busy))
                    time.sleep(0.005)
                rows = [t.result() for t in tasks]
            result['pairs'].append(dict(name=name, max_busy_slots=max(observed, default=0)))
            save()
            assert 2 in observed, (name, observed)
            return rows
        for mode, sampling in [('greedy', {'temperature': 0}),
                               ('seeded', {'temperature': 0.7, 'top_p': 0.95, 'top_k': 20, 'seed': 9950,
                                           'repetition_penalty': 1.05, 'presence_penalty': 0.1})]:
            baseline = [run(mode + '-solo-' + name, fixtures[name], {**sampling, 'seed': sampling.get('seed', 9950) + i})
                        for i, name in enumerate(('red', 'blue'))]
            concurrent = pair(mode + '-image-image', [fixtures['red'], fixtures['blue']], sampling)
            for a, b in zip(baseline, concurrent):
                assert a['ids'] == b['ids'], (mode, a, b)
            # Both follow-up requests still include the old picture in their history.
            followups = [fixtures[name] + [{'role': 'assistant', 'content': baseline[i]['text']},
                          {'role': 'user', 'content': 'Continue the paragraph. Keep discussing the same color.'}]
                         for i, name in enumerate(('red', 'blue'))]
            following = pair(mode + '-historical-image-continuation', followups, sampling)
            for i, row in enumerate(following):
                assert row['last'].get('reused', 0) >= baseline[i]['input_tokens'] - 10, row
        text_ref = run('solo-text', fixtures['text'], {'temperature': 0, 'seed': 9951})
        mixed = pair('image-text', [fixtures['red'], fixtures['text']], {'temperature': 0})
        assert mixed[1]['ids'] == text_ref['ids'], (mixed[1], text_ref)
        # Same image-pad tokens, a different image: never restore state that includes the old image.
        changed = copy.deepcopy(fixtures['red'])
        changed[0]['content'].insert(0, {'type': 'text', 'text': 'Replacement image isolation case. '})
        changed_red = copy.deepcopy(changed)
        changed[0]['content'][1]['source'] = fixtures['blue'][0]['content'][0]['source']
        run('replacement-base-red', changed_red, {'temperature': 0})
        replacement = run('replacement-blue', changed, {'temperature': 0})
        prepared, _, _ = svc.prepare(changed, [], {'enable_thinking': False}, max_new=args.limit)
        svc.drop_embeddings()
        assert replacement['last'].get('reused', 0) <= prepared.index(248056), replacement
        # The repeat must now match the blue image and agree token for token.
        blue_ref = run('replacement-blue-repeat', changed, {'temperature': 0})
        assert replacement['ids'] == blue_ref['ids']
        assert replacement['image_tokens'] > 0 and blue_ref['last'].get('reused', 0) > 0
        cancelled = pair('image-cancel', [fixtures['red'], fixtures['blue']], {'temperature': 0}, cancel_first=True)
        restored = run('image-after-cancel', fixtures['red'], {'temperature': 0})
        red_ref = run('image-after-cancel-repeat', fixtures['red'], {'temperature': 0})
        assert restored['ids'] == red_ref['ids']
        assert restored['last'].get('reused', 0) > 0
        if args.extended:
            assert args.desktop_context, 'Long tests require --desktop-context'
            # Different image grids and image positions produce different M-RoPE tables.
            wide = (args.output / 'blue-wide.png').resolve()
            def chunk(kind, data):
                return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))
            wide.write_bytes(b'\x89PNG\r\n\x1a\n' +
                             chunk(b'IHDR', struct.pack('>IIBBBBB', 448, 224, 8, 2, 0, 0, 0)) +
                             chunk(b'IDAT', zlib.compress((b'\0' + bytes((0, 0, 255)) * 448) * 224)) +
                             chunk(b'IEND', b''))
            long_messages = []
            for i, name in enumerate(('red', 'blue')):
                messages = copy.deepcopy(fixtures[name])
                if i:
                    messages[0]['content'][0]['source'] = str(wide)
                filler = 'Historical reference: a quiet garden has trees and a stone path.\n' * (8400 if i else 9400)
                messages.insert(0, {'role': 'system', 'content': filler})
                long_messages.append(messages)
            long_ref = [run('long-solo-' + str(i), m, {'temperature': 0}) for i, m in enumerate(long_messages)]
            long_pair = pair('long-different-grid-images', long_messages, {'temperature': 0})
            for ref, row in zip(long_ref, long_pair):
                assert ref['input_tokens'] > 100000 and ref['ids'] == row['ids'], (ref, row)
                assert row['last'].get('reused', 0) >= ref['input_tokens'] - 10, row
            assert long_ref[0]['image_tokens'] != long_ref[1]['image_tokens'], long_ref
            histories = [m + [{'role': 'assistant', 'content': long_ref[i]['text']},
                             {'role': 'user', 'content': 'Continue discussing the picture color in detail.'}]
                         for i, m in enumerate(long_messages)]
            long_follow = pair('long-historical-images', histories, {'temperature': 0})
            for ref, row in zip(long_ref, long_follow):
                assert row['last'].get('reused', 0) >= ref['input_tokens'] - 10, row
            # A genuinely cold image prompt yields to a shorter waiting request, then resumes.
            cold = copy.deepcopy(long_messages[0])
            cold[0]['content'] = 'Independent interrupted-prefill case.\n' + cold[0]['content']
            with ThreadPoolExecutor(max_workers=2) as pool:
                pending = pool.submit(run, 'long-image-prefill-yield', cold, {'temperature': 0})
                deadline = time.monotonic() + 60
                while engine.progress is None and not pending.done() and time.monotonic() < deadline:
                    time.sleep(0.01)
                time.sleep(0.2)
                short = pool.submit(run, 'short-during-image-prefill', fixtures['blue'], {'temperature': 0})
                short.result(timeout=240)
                resumed = pending.result(timeout=240)
            repeated = run('long-image-after-yield-repeat', cold, {'temperature': 0})
            assert resumed['ids'] == repeated['ids'] and repeated['last'].get('reused', 0) > 100000
            assert 'prompt read gives way' in log_path.read_text(encoding='utf-8'), 'No image-prefill yield exercised'
    except Exception as error:
        result.update(status='fail', error=repr(error))
        save()
        raise
    finally:
        if svc: svc.drop_embeddings()
        if engine:
            engine_proc = engine.proc
            engine.close()
        if vision: vision.shutdown()
        log.close()
    # Shutdown seals source-read accounting, including the shared native MTP weights.
    source_log = log_path.read_text(encoding='utf-8')
    for family in ('expert', 'MTP'):
        reads = re.findall(r'post-residency ' + family + r' source bytes=(\d+)', source_log)
        assert reads and all(int(n) == 0 for n in reads), (family, reads)
    assert engine_proc.returncode == 0
    result.update(status='pass', post_residency_source_bytes={'expert': 0, 'MTP': 0},
                  private_engine_exit_code=engine_proc.returncode)
    save()
    print('PASS: image concurrency, token parity, history reuse, replacement images, cancellation, zero resident source reads')


if __name__ == '__main__':
    main()
