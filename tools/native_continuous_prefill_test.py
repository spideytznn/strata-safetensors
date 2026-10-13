"""Private-pipe regression for continuous solo prefill in a batch-enabled engine.

Same-day source baseline, candidate off/on, identical fixed expert placement.
Checks chunk geometry, first logits, output tokens, BYIELD/slot resume and STOP.
No server listener or deployment files are changed.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re

from native_concurrency_acceptance import Protocol
from safetensors_tokenizer import SafetensorsTokenizer


def request(engine, ids, command=None, limit=64):
    engine.send(f"GEN {limit} temperature=0 {','.join(map(str, ids))}")
    result = {"ids": [], "progress": [], "yielded": None}
    sent = False
    while True:
        line = engine.read()
        if line.startswith("T "):
            result["ids"].append(int(line.split()[1]))
        elif line.startswith("PP "):
            done = int(line.split()[1])
            result["progress"].append(done)
            if command and not sent:
                engine.send(command)
                result["command_at"] = done
                sent = True
        elif line.startswith("YIELDED "):
            result["yielded"] = int(line.split()[2])
        elif line.startswith("DONE "):
            fields = line.split()
            result.update(done=line, prompt_ms=float(fields[3]), reused=int(fields[8]),
                          read=int(fields[14]), finish=fields[5])
            assert int(fields[1]) == len(result["ids"])
            assert int(fields[12]) == 0 and float(fields[13]) == 0, line
            if command:
                assert sent and result["finish"] == "cancel", result
                assert result["read"] < len(ids) - result["reused"], result
            return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    original = json.loads(a.config.read_text(encoding="utf-8-sig"))
    tok = SafetensorsTokenizer.from_directory(original["tokenizer"])
    header = tok.encode("<|im_start|>user\n", parse_special=True)
    tail = tok.encode("\nExplain the preceding material in detail.<|im_end|>\n<|im_start|>assistant\n"
                      "<think>\n\n</think>\n\n", parse_special=True)

    def prompt(n, text):
        part = tok.encode(text)
        count = n - len(header) - len(tail)
        return header + (part * (count // len(part) + 1))[:count] + tail

    prompts = [prompt(8192, "A report covers schedules and budgets. "),
               prompt(32768, Path("docs/DETAILS.md").read_text(encoding="utf-8")),
               prompt(24576, "Networks carry packets through routers and switches. "),
               prompt(20480, "Databases store durable records and indexed transactions. ")]
    side = prompt(192, "Describe the color blue. ")
    results = {"passed": False, "runs": []}
    def save():
        (a.output / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    for name, exe, enabled in [("baseline", a.baseline, False), ("disabled", a.candidate, False),
                                ("enabled", a.candidate, True)]:
        folder = a.output / name
        folder.mkdir()
        cfg = copy.deepcopy(original)
        cfg["exe"] = str(exe.resolve())
        flags = cfg["args"]
        def setting(key, val):
            if key in flags:
                flags[flags.index(key) + 1] = str(val)
            else:
                flags.extend([key, str(val)])
        if "--vision" in flags:
            flags.remove("--vision")
        for key, val in [("--batch", 2), ("--max-context", 65536), ("--prefill", 4096),
                         ("--expert-cache", 4000), ("--pcie-frac", 1), ("--adapt-swaps", 0),
                         ("--adapt-every", 1000000), ("--conversation-cache-mib", 0),
                         ("--prompt-cache", 6)]:
            setting(key, val)
        profile = folder / "profile.bin"
        profile.write_bytes(Path(flags[flags.index("--expert-profile")+1]).read_bytes())
        setting("--expert-profile", profile.resolve())
        cfg["env"].update(STRATA_NATIVE_PREFILL_CONTINUOUS=str(int(enabled)),
                          STRATA_NATIVE_PREFILL_FIRST="0",
                          STRATA_NATIVE_PREFILL_OVERLAP="0", STRATA_ROUTE_TAIL_SKIP="0",
                          STRATA_TRACE="1", STRATA_PREFILL_TIMING="1",
                          STRATA_CONVERSATION_BRANCH_PRESERVE="1",
                          STRATA_DUMP_FIRST_LOGITS=str((folder / "logits.f32").resolve()))
        (folder / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        run = {"name": name, "sha256": hashlib.sha256(exe.read_bytes()).hexdigest(), "requests": []}
        results["runs"].append(run)
        save()
        eng = Protocol(cfg, folder)
        try:
            run["info"] = eng.info
            request(eng, side, limit=8)
            for i, ids in enumerate(prompts):
                if enabled and i >= 2:
                    interrupted = request(eng, ids, "BYIELD 0" if i == 2 else "STOP")
                    run.setdefault("interruptions", []).append(interrupted)
                    if i == 2:
                        assert interrupted["yielded"] is not None, interrupted
                        assert interrupted["yielded"] <= interrupted["command_at"] + 4096, interrupted
                        request(eng, side, limit=8)
                before = set(folder.glob("logits.f32.*"))
                result = request(eng, ids)
                added = set(folder.glob("logits.f32.*")) - before
                assert len(added) == 1, added
                result["logits_sha256"] = hashlib.sha256(added.pop().read_bytes()).hexdigest()
                if enabled and i == 2:
                    assert result["reused"] >= run["interruptions"][0]["yielded"], result
                run["requests"].append(result)
                save()
                print(name, len(ids), result["prompt_ms"], result["reused"], flush=True)
        finally:
            eng.close()
        assert eng.proc.returncode == 0
        log = (folder / "engine.log").read_text(encoding="utf-8")
        run["max_run_tokens"] = max(map(int, re.findall(r"prefill executed: tokens=(\d+)", log)))
        if enabled:
            assert run["max_run_tokens"] > 4096
        else:
            assert run["max_run_tokens"] <= 4096
        save()
    base = results["runs"][0]["requests"]
    for run in results["runs"][1:]:
        for ref, got in zip(base, run["requests"]):
            assert ref["ids"] == got["ids"], (run["name"], "tokens")
            assert ref["logits_sha256"] == got["logits_sha256"], (run["name"], "logits")
    results["passed"] = True
    save()
    print("PASS: default path, continuous prefill, yield/resume and cancellation", flush=True)


if __name__ == "__main__":
    main()
