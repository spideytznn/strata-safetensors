"""Private HTTP acceptance for native parallel requests. Does not start the desktop service."""
import argparse
import base64
import hashlib
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import threading
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
from serve.server import Service, StrataEngine, Vision, child_env, engine_args, serve, vision_env
from serve.frontend import ChatTemplate
from safetensors_tokenizer import SafetensorsTokenizer
from native_vision_smoke import solid_png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--exe", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--clients", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--limit", type=int, default=96)
    ap.add_argument("--independent-kv", action="store_true", help="opt into per-session elastic KV mappings")
    ap.add_argument("--device-slot-copy", action="store_true", help="opt into direct device slot migration")
    ap.add_argument("--long-tokens", type=int, default=0, help="replace the first fixture with a repeated-text long prompt")
    ap.add_argument("--cold-admission-tokens", type=int, default=0, help="also check a cold long read yielding to a late short request")
    ap.add_argument("--fixture-tokens", type=int, default=0, help="pad every prompt to exercise real multi-request prefill")
    ap.add_argument("--cold-prefill", action="store_true", help="replace all slot owners before measured bursts and disable turn checkpoints")
    ap.add_argument("--gather-burst", action="store_true", help="hold admission for 100 ms to test a deterministic queued burst; includes that wait in TTFT")
    ap.add_argument("--no-vision", action="store_true")
    ap.add_argument("--system-tokens", type=int, default=0, help="exercise a long shared system-root checkpoint")
    ap.add_argument("--turn-checkpoints", action="store_true", help="retain normal checkpoint policy during cold-slot tests")
    ap.add_argument("--packed-vision-guard", action="store_true", help="check packed-text refusal immediately after real image positions")
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    cfg = json.loads(a.config.read_text(encoding="utf8"))
    cfg["exe"] = str(a.exe.resolve())
    cfg["parallel"] = a.parallel
    if a.no_vision:
        cfg.pop("vision", None)
        cfg["args"] = [v for v in cfg["args"] if v != "--vision"]
    cfg["env"]["STRATA_NATIVE_KV_INDEPENDENT"] = "1" if a.independent_kv else "0"
    cfg["env"]["STRATA_NATIVE_SLOT_COPY"] = "1" if a.device_slot_copy else "0"
    args = cfg["args"]
    args[args.index("--vram-reserve-mib") + 1] = "1536"
    result = {"config": cfg, "exe_sha256": hashlib.sha256(a.exe.read_bytes()).hexdigest(), "rounds": [], "requests": []}
    def save():
        (a.output / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8")
    save()
    env = child_env(cfg)
    vision_log = (a.output / "vision.log").open("w", encoding="utf8")
    vision, engine, httpd = None, None, None
    try:
        if cfg.get("vision"):
            vision = Vision(cfg["vision"], log=vision_log, env=vision_env(cfg, env))
        engine = StrataEngine(cfg["exe"], engine_args(cfg), cwd=cfg["cwd"],
                              log=str((a.output / "engine.log").resolve()), env=env)
        tok = SafetensorsTokenizer.from_directory(cfg["tokenizer"])
        svc = Service(engine, tok, ChatTemplate(Path(cfg["chat_template"])), vision=vision,
                      sampling_defaults=cfg.get("sampling"), fit_max_tokens=True)
        svc.allowed_hosts = cfg.get("allowed_hosts", [])
        httpd = serve(svc, port=0)
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        result["info"] = dict(engine.info)
        assert engine.batch == (a.parallel if a.parallel > 1 else 0)
        if "route_tail_skip" in engine.info:
            assert engine.info["route_tail_skip"] == 0, "quality screen requires tail skipping off"
        with urllib.request.urlopen(urllib.request.Request(base + "/health",
                headers={"Host": "native-regression.example"}), timeout=30) as response:
            assert response.status == 200, "the profile must allow arbitrary Host headers"
        with urllib.request.urlopen(urllib.request.Request(base + "/metrics",
                headers={"Accept": "text/plain; version=0.0.4"}), timeout=30) as response:
            metrics = response.read().decode("utf8")
            assert response.status == 200 and "# TYPE" in metrics and "vllm:" in metrics
        result["http_checks"] = {"unrestricted_host": True, "prometheus_metrics": True}
        system_unit = "You are a careful tutor. Check units, assumptions and boundary conditions. "
        system_text = system_unit * (a.system_tokens // len(tok.encode(system_unit)) + 1) if a.system_tokens else ""

        def chat(name, content, limit=a.limit, barrier=None, started=None, disconnect=False, anthropic=False, checkpoint=None):
            if barrier:
                barrier.wait(timeout=30)
            body = {"model": "qwen3.8-flash-next", "messages": [{"role": "user", "content": content}],
                    "max_tokens": limit, "temperature": 0, "stream": True,
                    "chat_template_kwargs": {"enable_thinking": False}}
            if system_text and not name.startswith("eviction-"):
                body["messages"].insert(0, {"role": "system", "content": system_text})
            if checkpoint is False or (checkpoint is None and a.cold_prefill and not a.turn_checkpoints):
                body["strata_checkpoint"] = False
            if anthropic:
                body.pop("chat_template_kwargs")
                body["thinking"] = {"type": "disabled"}
            else:
                body["stream_options"] = {"include_usage": True}
            req = urllib.request.Request(base + ("/v1/messages" if anthropic else "/v1/chat/completions"),
                data=json.dumps(body).encode("utf8"), headers={"Content-Type": "application/json"})
            t0 = time.perf_counter()
            times, text, usage, finish = [], "", None, None
            with urllib.request.urlopen(req, timeout=300) as response:
                for line in response:
                    if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                        continue
                    ev = json.loads(line[6:])
                    if "error" in ev:
                        raise RuntimeError(ev)
                    if anthropic:
                        delta = ev.get("delta", {})
                        part = delta.get("text", "")
                        usage = ev.get("usage", usage)
                        finish = delta.get("stop_reason", finish)
                    else:
                        choice = (ev.get("choices") or [{}])[0]
                        part = choice.get("delta", {}).get("content", "")
                        usage = ev.get("usage") or usage
                        finish = choice.get("finish_reason") or finish
                    if part:
                        text += part
                        times.append(time.perf_counter() - t0)
                        if started:
                            started.set()
                        if disconnect and len(times) >= 12:
                            break
            row = {"name": name, "text": text, "times": times, "wall_s": time.perf_counter() - t0,
                   "usage": usage, "finish": finish, "disconnected": disconnect}
            result["requests"].append(row)
            return row

        questions = ["Explain TCP and QUIC in detail, including congestion control and multiplexing.",
                     "写一个 Python 二分查找函数，解释边界条件和时间复杂度。",
                     "Explain how a B-tree splits a node; give a worked example."]
        if max(a.clients) > len(questions):
            questions.extend([
                "Explain database transactions and isolation levels with detailed examples.",
                "Describe merge sort with Python code, correctness and complexity.",
                "Explain how an operating system schedules threads on many CPU cores.",
                "Describe DNS resolution from a browser to an authoritative name server.",
                "Explain Unicode encoding with examples of UTF-8 and UTF-16.",
            ][:max(a.clients) - len(questions)])
        if not a.clients or min(a.clients) < 1 or max(a.clients) > len(questions):
            raise ValueError("clients must be in 1..8")
        if a.long_tokens:
            paragraph = "The committee reviewed the budget, the schedule and open questions. A report follows next quarter. "
            questions[0] = paragraph * (a.long_tokens // len(tok.encode(paragraph)) + 1) + \
                "\nExplain why a written schedule is useful, in detail."
        if a.fixture_tokens:
            for i in range(len(questions)):
                paragraph = f"Independent notebook {i} records units, dates, observations and open questions. "
                questions[i] = paragraph * ((a.fixture_tokens + i * 128) // len(tok.encode(paragraph)) + 1) + "\n" + questions[i]
        result["fixture_body_tokens"] = [len(tok.encode(q)) for q in questions]
        refs = [chat(f"reference-{i}", text) for i, text in enumerate(questions)]
        save()
        with ThreadPoolExecutor(max_workers=max(4, max(a.clients), a.parallel)) as pool:
            for count in a.clients:
                for round_id in range(a.rounds):
                    if a.cold_prefill:
                        evict_gate = threading.Barrier(a.parallel)
                        engine.ctl.acquire()
                        try:
                            evictions = [pool.submit(chat, f"eviction-{count}-{round_id}-{i}",
                                f"Replacement {count}/{round_id}/{i}. Explain merge sort in detail with Python.",
                                limit=32, barrier=evict_gate, checkpoint=False) for i in range(a.parallel)]
                            time.sleep(0.1)
                        finally:
                            engine.ctl.release()
                        for future in evictions:
                            future.result(timeout=300)
                        # Every owner must really be replaced: merely clearing the
                        # Python hints would leave native KV cached and fake a cold run.
                        rendered = [svc.template.render(([{"role": "system", "content": system_text}] if system_text else []) +
                                    [{"role": "user", "content": q}], enable_thinking=False) for q in questions]
                        prompt_ids = [tok.encode(q, parse_special=True) for q in rendered]
                        assert all(p not in prompt_ids for p in engine.slot_prompts), "eviction did not replace every native slot"
                    gate = threading.Barrier(count)
                    samples, stop_sample = [], threading.Event()
                    def sample_slots():
                        while not stop_sample.wait(0.01):
                            with engine.slot_cv:
                                samples.append({"active": sum(engine.slot_busy), "reserved_tokens": sum(
                                    n for b, n in enumerate(engine.slot_tokens) if engine.slot_busy[b])})
                    sampler = threading.Thread(target=sample_slots, daemon=True)
                    sampler.start()
                    t0 = time.perf_counter()
                    if a.gather_burst:
                        engine.ctl.acquire()
                    try:
                        futures = [pool.submit(chat, f"c{count}-r{round_id}-{i}", questions[i], barrier=gate)
                                   for i in range(count)]
                        if a.gather_burst:
                            time.sleep(0.1)
                    finally:
                        if a.gather_burst:
                            engine.ctl.release()
                    try:
                        rows = [f.result(timeout=300) for f in futures]
                    finally:
                        stop_sample.set()
                        sampler.join(timeout=2)
                    for i, row in enumerate(rows):
                        assert row["text"] == refs[i]["text"], row["name"] + " differs from serial reference"
                        assert row["usage"] and row["usage"]["completion_tokens"] <= a.limit
                    elapsed = time.perf_counter() - t0
                    result["rounds"].append({"clients": count, "round": round_id, "wall_s": elapsed,
                        "aggregate_tok_s": sum(r["usage"]["completion_tokens"] for r in rows) / elapsed,
                        "ttft_s": [r["times"][0] for r in rows],
                        "peak_active": max((s["active"] for s in samples), default=0),
                        "peak_reserved_tokens": max((s["reserved_tokens"] for s in samples), default=0)})
                    if engine.kv_token_budget:
                        assert all(s["reserved_tokens"] <= engine.kv_token_budget for s in samples), "live KV token budget exceeded"
                    save()
                    print(f"HTTP clients={count} round={round_id}: identical; {elapsed:.3f} s; "
                          f"TTFT {[round(r['times'][0],3) for r in rows]}", flush=True)
            # A client disappearing must not kill another slot or leave the engine's protocol out of step.
            barrier = threading.Barrier(2)
            cancelled = pool.submit(chat, "disconnect", questions[0], barrier=barrier, disconnect=True)
            survivor = pool.submit(chat, "survivor", questions[1], barrier=barrier)
            cancelled.result(timeout=300)
            assert survivor.result(timeout=300)["text"] == refs[1]["text"]
            assert chat("after-disconnect", questions[0])["text"] == refs[0]["text"]
            if vision and a.parallel > 1:
                image = a.output / "red.png"
                solid_png(image, (255, 0, 0))
                content = [{"type": "image_url", "image_url": {
                    "url": "data:image/png;base64," + base64.b64encode(image.read_bytes()).decode("ascii")}},
                    {"type": "text", "text": "What is the solid background color? Answer with one English color word only."}]
                started = threading.Event()
                active = pool.submit(chat, "text-before-image", questions[0], started=started)
                assert started.wait(timeout=120)
                picture = pool.submit(chat, "image-waiting", content, limit=16)
                # This allows the image to encode and enter the engine's writer queue.
                deadline = time.monotonic() + 20
                while not engine.vision_waiting and not picture.done() and time.monotonic() < deadline:
                    time.sleep(0.02)
                late = pool.submit(chat, "text-after-image", questions[1])
                assert active.result(timeout=300)["text"] == refs[0]["text"]
                assert "red" in picture.result(timeout=300)["text"].lower()
                assert late.result(timeout=300)["text"] == refs[1]["text"]
                if a.packed_vision_guard:
                    assert engine.info.get("ragged_prefill") == 1
                    assert "red" in chat("image-before-packed-guard", content, limit=16)["text"].lower()
                    guard_ids = [tok.encode(svc.template.render([{"role": "user", "content": q}],
                                 enable_thinking=False), parse_special=True) for q in questions[:2]]
                    assert sum(len(ids) - 1 for ids in guard_ids) <= int(engine.info["prefill_chunk"])
                    with engine.ctl:
                        engine._send("BPREF " + ';'.join(f"{i}:" + ','.join(map(str, ids[:-1])) for i, ids in enumerate(guard_ids)))
                        line = engine.lines.get(timeout=30)
                        assert line is not None and line.strip() == "PREFDONE 0 0 0", repr(line)
                    assert chat("text-reset-after-packed-guard", questions[0])["text"] == refs[0]["text"]
                    result["packed_vision_guard"] = {"refused_image_positions": True, "text_reset_equal": True}
                    save()
            anth = chat("anthropic", questions[0], anthropic=True)
            assert anth["text"] == refs[0]["text"]
            if a.cold_admission_tokens:
                paragraph = "An independent cold admission exercises the prompt scheduler and sequence state. "
                cold_text = paragraph * (a.cold_admission_tokens // len(tok.encode(paragraph)) + 1) + \
                    "\nExplain why a written schedule helps a project."
                cold_ref = chat("cold-reference", cold_text, limit=64, checkpoint=False)
                log_before = (a.output / "engine.log").stat().st_size
                cold = pool.submit(chat, "cold-staggered", cold_text, limit=64, checkpoint=False)
                deadline = time.monotonic() + 30
                while not any(s and s.get("state") == "reading" and s.get("prompt_tokens", 0) > 10000
                              for s in engine.slot_live) and not cold.done() and time.monotonic() < deadline:
                    time.sleep(0.02)
                assert not cold.done(), "cold admission finished before a short request could arrive"
                short = pool.submit(chat, "cold-short", questions[1])
                assert short.result(timeout=300)["text"] == refs[1]["text"]
                assert cold.result(timeout=300)["text"] == cold_ref["text"]
                with (a.output / "engine.log").open("rb") as log:
                    log.seek(log_before)
                    trace = log.read().decode("utf8")
                assert "the prompt read gives way" in trace, "no real chunk-boundary yield occurred"
                result["cold_admission"] = {"body_tokens": len(tok.encode(cold_text)), "yielded": True,
                                             "output_identical": True}
                save()
                print("cold long admission yielded to a late short request; both outputs identical", flush=True)
        assert not any(engine.slot_busy)
        assert engine.vision_readers == 0 and not engine.vision_writer
        result["passed"] = True
        save()
        print("HTTP parity, excess-client queue, disconnect, vision exclusion, Anthropic: passed", flush=True)
    finally:
        if httpd:
            httpd.shutdown()
            httpd.server_close()
        if engine:
            engine.unload()
        if vision:
            vision.shutdown()
        vision_log.close()


if __name__ == "__main__":
    main()
