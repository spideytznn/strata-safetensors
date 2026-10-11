"""Same-binary CPU-DAG and cold ragged-prefill A/B/A. Private engines only."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--exe", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=("decode", "prefill", "all"), default="all")
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--limit", type=int, default=128)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    base = json.loads(a.config.read_text(encoding="utf-8-sig"))
    variants = {}
    for name, env in (("baseline", {}), ("dag", {"STRATA_NATIVE_EXPERT_DAG": "1"}),
                      ("ragged", {"STRATA_NATIVE_RAGGED_PREFILL": "1"})):
        cfg = json.loads(json.dumps(base))
        cfg["exe"] = str(a.exe.resolve())
        cfg["env"].update(STRATA_NATIVE_BATCH_INPLACE="1", STRATA_NATIVE_KV_HANDOFF="1",
                          STRATA_NATIVE_CONTINUOUS="1", STRATA_NATIVE_PREFIX_PAGES="0",
                          STRATA_NATIVE_EXPERT_DAG="0", STRATA_NATIVE_RAGGED_PREFILL="0",
                          STRATA_NATIVE_KV_TOKEN_BUDGET="0")
        cfg["env"].update(env)
        path = a.output / f"{name}.json"
        path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf8")
        variants[name] = path
    runs = []
    if a.mode in ("decode", "all"):
        for slots in (2, 4):
            runs.extend((f"decode{slots}-{label}", variant, slots, False)
                        for label, variant in (("before", "baseline"), ("dag", "dag"), ("after", "baseline")))
    if a.mode in ("prefill", "all"):
        runs.extend((f"cold4-{label}", variant, 4, True)
                    for label, variant in (("before", "baseline"), ("ragged", "ragged"), ("after", "baseline")))
    result = {"status": "running", "exe_sha256": hashlib.sha256(a.exe.read_bytes()).hexdigest(), "runs": []}
    def save():
        (a.output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8")
    save()
    try:
        references = {}
        for name, variant, slots, cold in runs:
            folder = a.output / name
            command = [sys.executable, str(ROOT / "tools/native_concurrency_http.py"),
                       "--config", str(variants[variant].resolve()), "--exe", str(a.exe.resolve()),
                       "--output", str(folder.resolve()), "--parallel", str(slots), "--rounds", str(a.rounds),
                       "--clients", str(slots), "--limit", str(a.limit), "--independent-kv", "--device-slot-copy", "--no-vision"]
            if cold:
                command.extend(["--fixture-tokens", "1800", "--cold-prefill"])
            record = {"name": name, "variant": variant, "slots": slots, "command": command, "status": "running"}
            result["runs"].append(record)
            save()
            print(f"Starting {name}", flush=True)
            with (a.output / f"{name}.log").open("w", encoding="utf8") as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            data = json.loads((folder / "results.json").read_text(encoding="utf8"))
            trace = (folder / "engine.log").read_text(encoding="utf8")
            assert data["passed"] and data["exe_sha256"] == result["exe_sha256"]
            assert data["info"]["mtp_max"] == 0 and data["info"]["continuous_batch"] == 1
            assert data["info"]["expert_dag"] == int(variant == "dag")
            assert data["info"]["ragged_prefill"] == int(variant == "ragged")
            assert "post-residency expert source bytes=0" in trace
            assert "post-residency MTP source bytes=0" in trace
            replies = {row["name"]: row["text"] for row in data["requests"] if not row["disconnected"]}
            group = (slots, cold)
            same = group not in references or replies == references[group]
            if variant == "baseline" and group not in references:
                references[group] = replies
            # The DAG promises identical arithmetic. Ragged uses canonical dense
            # shapes and complete-K reductions; its numerical change is reported,
            # never silently accepted as byte-identical to the released path.
            if not cold:
                assert same, "CPU DAG changed output"
            record.update(status="passed", output_equal_to_first_baseline=same,
                aggregate_tok_s=statistics.median(row["aggregate_tok_s"] for row in data["rounds"]),
                last_ttft_s=statistics.median(max(row["ttft_s"]) for row in data["rounds"]),
                peak_active=max(row["peak_active"] for row in data["rounds"]),
                packed_chunks=re.findall(r"strata packed prefill:.*", trace),
                cache_sizing=[line for line in trace.splitlines() if "expert cache auto:" in line],
                expert_ssd_bytes=0)
            if variant == "ragged":
                assert record["packed_chunks"], "cold benchmark never exercised packed prefill"
            save()
            print(name, record["aggregate_tok_s"], record["last_ttft_s"], flush=True)
        result["status"] = "passed"
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        if result["runs"]:
            result["runs"][-1].update(status="failed", error=repr(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
