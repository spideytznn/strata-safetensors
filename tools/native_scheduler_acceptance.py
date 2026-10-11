"""Run final scheduler correctness checks sequentially, one private GPU engine at a time."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def source_read_audit(folder):
    """Check final counters after each private engine has shut down."""
    logs = list(folder.rglob("engine.log"))
    assert logs, f"no engine diagnostics in {folder}"
    rows = []
    for path in logs:
        trace = path.read_text(encoding="utf8")
        counters = {}
        for source in ("expert", "MTP"):
            values = [int(n) for n in re.findall(rf"post-residency {source} source bytes=(\d+)", trace)]
            assert values and all(n == 0 for n in values), f"{source} source reads in {path}: {values}"
            counters[source] = 0
        rows.append({"log": str(path.relative_to(folder)), "source_bytes": counters})
    return rows


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--exe", type=Path, required=True)
    p.add_argument("--reference-exe", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--only", action="append", help="Run only this named step; repeat to select multiple steps.")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    base = json.loads(a.config.read_text(encoding="utf-8-sig"))
    variants = {}
    for name, env in (("baseline", {}), ("prefix-dag", {"STRATA_NATIVE_PREFIX_PAGES": "1", "STRATA_NATIVE_EXPERT_DAG": "1"}),
                      ("all", {"STRATA_NATIVE_PREFIX_PAGES": "1", "STRATA_NATIVE_EXPERT_DAG": "1", "STRATA_NATIVE_RAGGED_PREFILL": "1"}),
                      ("budget8", {"STRATA_NATIVE_PREFIX_PAGES": "1", "STRATA_NATIVE_EXPERT_DAG": "1", "STRATA_NATIVE_RAGGED_PREFILL": "1", "STRATA_NATIVE_KV_TOKEN_BUDGET": "32768"})):
        cfg = json.loads(json.dumps(base))
        cfg["exe"] = str(a.exe.resolve())
        cfg["env"].update(STRATA_NATIVE_PREFIX_PAGES="0", STRATA_NATIVE_EXPERT_DAG="0",
                          STRATA_NATIVE_RAGGED_PREFILL="0", STRATA_NATIVE_KV_TOKEN_BUDGET="0")
        cfg["env"].update(env)
        path = a.output / f"{name}.json"
        path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf8")
        variants[name] = path
    steps = [
        ("released-parity", "baseline", "native_default_parity.py", ["--reference-exe", str(a.reference_exe.resolve())]),
        ("packed-all", "baseline", "native_packed_acceptance.py", []),
        ("full-long", "prefix-dag", "native_concurrency_acceptance.py", ["--context", "262144", "--slots", "4", "--limit", "128", "--vision", "--adaptive", "--independent-kv", "--device-slot-copy", "--yield-tokens", "20000", "--growth-tokens", "131000", "--capacity-tokens", "261000", "--eviction-tokens", "32768"]),
        ("prefix261k-before", "baseline", "native_prefix_acceptance.py", ["--tokens", "261000", "--slots", "4", "--sharing", "0"]),
        ("prefix261k-shared", "baseline", "native_prefix_acceptance.py", ["--tokens", "261000", "--slots", "4", "--sharing", "1"]),
        ("http-budget8-root", "budget8", "native_concurrency_http.py", ["--parallel", "8", "--rounds", "1", "--clients", "8", "--limit", "64", "--independent-kv", "--device-slot-copy", "--fixture-tokens", "1000", "--system-tokens", "2400", "--turn-checkpoints", "--cold-prefill", "--gather-burst", "--no-vision"]),
        ("http-vision", "all", "native_concurrency_http.py", ["--parallel", "4", "--rounds", "1", "--clients", "4", "--limit", "64", "--independent-kv", "--device-slot-copy", "--fixture-tokens", "1200", "--turn-checkpoints", "--cold-prefill", "--gather-burst", "--packed-vision-guard"]),
    ]
    if a.only:
        unknown = set(a.only) - {step[0] for step in steps}
        if unknown:
            raise ValueError(f"unknown acceptance steps: {sorted(unknown)}")
        steps = [step for step in steps if step[0] in a.only]
    result = {"status": "running", "exe_sha256": hashlib.sha256(a.exe.read_bytes()).hexdigest(), "steps": []}
    def save():
        (a.output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8")
    save()
    try:
        for name, variant, script, flags in steps:
            command = [sys.executable, str(ROOT / "tools" / script), "--config", str(variants[variant].resolve()),
                       "--exe", str(a.exe.resolve()), "--output", str((a.output / name).resolve()), *flags]
            row = {"name": name, "status": "running", "command": command}
            result["steps"].append(row)
            save()
            print(f"Starting {name}", flush=True)
            with (a.output / f"{name}.log").open("w", encoding="utf8") as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            row["source_read_audit"] = source_read_audit(a.output / name)
            row["status"] = "passed"
            save()
            print(f"Passed {name}", flush=True)
        result["status"] = "passed"
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        if result["steps"]:
            result["steps"][-1].update(status="failed", error=repr(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
