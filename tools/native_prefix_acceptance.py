"""Real-model shared-prefix forks, source eviction and rewind. Private process only."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
from native_concurrency_acceptance import Protocol
from safetensors_tokenizer import SafetensorsTokenizer
from serve.frontend import ChatTemplate


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--exe", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tokens", type=int, default=131100)
    p.add_argument("--slots", type=int, default=4, choices=range(2, 9))
    p.add_argument("--limit", type=int, default=64)
    p.add_argument("--sharing", type=int, default=1, choices=(0, 1))
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    cfg = json.loads(a.config.read_text(encoding="utf-8-sig"))
    cfg["exe"] = str(a.exe.resolve())
    cfg["args"] = [v for v in cfg["args"] if v != "--vision"]
    cfg["args"].extend(["--batch", str(a.slots)])
    cfg["env"].update(STRATA_NATIVE_PREFIX_PAGES=str(a.sharing), STRATA_NATIVE_RAGGED_PREFILL="0", STRATA_NATIVE_EXPERT_DAG="0")
    tok = SafetensorsTokenizer.from_directory(cfg["tokenizer"])
    template = ChatTemplate(Path(cfg["chat_template"]))
    def prompt(text):
        return tok.encode(template.render([{"role": "user", "content": text}], enable_thinking=False), parse_special=True)
    unit = tok.encode("A laboratory notebook records the measurements, dates, units and open questions. ")
    rendered = template.render([{"role": "user", "content": "STRATA_PREFIX_INSERT\nExplain why recording units is useful, with detailed examples."}], enable_thinking=False)
    left, right = rendered.split("STRATA_PREFIX_INSERT")
    head, tail = tok.encode(left, parse_special=True), tok.encode(right, parse_special=True)
    count = a.tokens - len(head) - len(tail)
    assert count > 0
    ids = head + (unit * ((count + len(unit) - 1) // len(unit)))[:count] + tail
    assert len(ids) == a.tokens
    result = {"status": "running", "config": cfg, "exe_sha256": hashlib.sha256(a.exe.read_bytes()).hexdigest(),
              "prompt_tokens": len(ids), "cases": []}
    def save():
        (a.output / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8")
    save()
    engine = Protocol(cfg, a.output)
    try:
        result["info"] = engine.info
        assert any(f"prefix_pages={a.sharing}" in line for line in engine.info)
        ref = engine.solo(ids, a.limit, "temperature=0")
        start = len(engine.log_path.read_text(encoding="utf8"))
        forks = engine.batch([(ids, a.limit, "temperature=0")] * a.slots)
        trace = engine.log_path.read_text(encoding="utf8")[start:]
        aliases = list(map(int, re.findall(r"physical_alias_bytes=(\d+)", trace)))
        result["cases"].append({"name": "active_prefix_forks", "reference": ref, "batch": forks,
                                "alias_operation_bytes_sum": sum(aliases), "clone_count": trace.count("active prefix cloned")})
        save()
        assert all(row["ids"] == ref["ids"] for row in forks["jobs"]), "fork output changed"
        assert bool(aliases) == bool(a.sharing), "unexpected physical sharing"
        assert "active prefix cloned" in trace, "fixture never cloned an active source"
        print(f"{a.slots} forks, {len(ids)} tokens: identical, {sum(aliases)} bytes across alias operations (not peak savings)", flush=True)
        # Replace every owner. A later RAM restore must detach shared mappings
        # before overwriting; no surviving alias may enter the expert recycler.
        replacements = [prompt(f"Independent replacement {i}: explain merge sort with Python.") for i in range(a.slots)]
        engine.batch([(v, 32, "temperature=0") for v in replacements])
        restored = engine.batch([(ids, a.limit, "temperature=0")] * 2)
        result["cases"].append({"name": "evict_all_owners_restore", "batch": restored})
        save()
        assert all(row["ids"] == ref["ids"] for row in restored["jobs"]), "restored fork output changed"
        print("all owners evicted, RAM restoration and subsequent fork: identical", flush=True)
        result["status"] = "passed"
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        raise
    finally:
        engine.close()
        trace = engine.log_path.read_text(encoding="utf8")
        result["expert_ssd_bytes_zero"] = "post-residency expert source bytes=0" in trace
        mappings = [(float(m.group(1)), int(m.group(2))) for m in re.finditer(r"mapped_mib=([\d.]+) expert_slots=(\d+)", trace)]
        result["physical_kv_peak_mib"] = max((row[0] for row in mappings), default=None)
        result["minimum_expert_slots"] = min((row[1] for row in mappings), default=None)
        result["cache_sizing"] = [line for line in trace.splitlines() if "expert cache auto:" in line]
        save()


if __name__ == "__main__":
    main()
