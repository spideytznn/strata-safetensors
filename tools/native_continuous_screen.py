"""Same-binary architectural A/B over the real HTTP server. Private engines, original model, no desktop edits.

Run after correctness acceptance; stops on any failed screen. All child harnesses close their engine/vision.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--exe", type=Path, required=True)
    ap.add_argument("--reference-exe", type=Path, help="also check one short round against the deployed previous binary")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--limit", type=int, default=256)
    ap.add_argument("--long-tokens", type=int, default=131000)
    ap.add_argument("--mode", choices=("full", "final"), default="full",
                    help="final compares four-slot short A/B/A and two/four-slot mixed long requests")
    ap.add_argument("--cold-admission-tokens", type=int, default=0)
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    cfg = json.loads(a.config.read_text(encoding="utf8"))
    configs = {}
    for name, on in (("baseline", False), ("continuous", True)):
        c = json.loads(json.dumps(cfg))
        c["exe"] = str(a.exe.resolve())
        for flag in ("STRATA_NATIVE_BATCH_INPLACE", "STRATA_NATIVE_KV_HANDOFF", "STRATA_NATIVE_CONTINUOUS"):
            c.setdefault("env", {})[flag] = str(int(on))
        configs[name] = a.output / (name + ".json")
        configs[name].write_text(json.dumps(c, ensure_ascii=False, indent=2), encoding="utf8")
    runs = [("short-before", "baseline", 2, 0), ("short-baseline4", "baseline", 4, 0), ("short-continuous2", "continuous", 2, 0),
            ("short-continuous4", "continuous", 4, 0), ("short-after", "baseline", 2, 0)]
    if a.long_tokens:
        runs += [("long-baseline", "baseline", 2, a.long_tokens),
                 ("long-continuous2", "continuous", 2, a.long_tokens)]
    if a.mode == "final":
        runs = [("short-baseline4", "baseline", 4, 0), ("short-continuous4", "continuous", 4, 0),
                ("short-after4", "baseline", 4, 0)]
        if a.long_tokens:
            runs += [("long-baseline", "baseline", 2, a.long_tokens),
                     ("long-continuous2", "continuous", 2, a.long_tokens),
                     ("long-continuous4", "continuous", 4, a.long_tokens)]
    manifest = {"exe_sha256": hashlib.sha256(a.exe.read_bytes()).hexdigest(), "commands": [], "runs": []}
    def save():
        (a.output / "summary.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf8")
    for name, variant, parallel, long_tokens in runs:
        command = [sys.executable, str(ROOT / "tools/native_concurrency_http.py"),
                   "--config", str(configs[variant].resolve()), "--exe", str(a.exe.resolve()),
                   "--output", str((a.output / name).resolve()), "--parallel", str(parallel),
                   "--rounds", str(a.rounds), "--limit", str(a.limit), "--independent-kv", "--device-slot-copy",
                   "--clients", *(["1", "2", "3"] if long_tokens else ["1", "2", "4", "8"])]
        if long_tokens:
            command += ["--long-tokens", str(long_tokens)]
        elif variant == "continuous" and a.cold_admission_tokens:
            command += ["--cold-admission-tokens", str(a.cold_admission_tokens)]
        manifest["commands"].append(command)
        save()
        print(f"Starting {name}", flush=True)
        with (a.output / (name + ".log")).open("w", encoding="utf8") as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        r = json.loads((a.output / name / "results.json").read_text(encoding="utf8"))
        assert r["passed"] and r["exe_sha256"] == manifest["exe_sha256"]
        for flag in ("batch_inplace", "kv_handoff", "continuous_batch"):
            assert r["info"][flag] == int(variant == "continuous"), (name, flag, r["info"])
        assert r["info"]["weight_source"] == "safetensors" and r["info"]["mtp_max"] == 0
        runtime_log = (a.output / name / "engine.log").read_text(encoding="utf8")
        assert "post-residency expert source bytes=0" in runtime_log
        assert "post-residency MTP source bytes=0" in runtime_log
        by_count = {}
        for row in r["rounds"]:
            by_count.setdefault(row["clients"], []).append(row)
        stats = {n: {"aggregate_tok_s": statistics.median(x["aggregate_tok_s"] for x in rows),
                     "aggregate_range_tok_s": [min(x["aggregate_tok_s"] for x in rows), max(x["aggregate_tok_s"] for x in rows)],
                     "last_ttft_s": statistics.median(max(x["ttft_s"]) for x in rows)} for n, rows in by_count.items()}
        manifest["runs"].append({"name": name, "parallel": parallel, "stats": stats,
                                 "batch_graph_captures": runtime_log.count("captured the batch window"),
                                 "cache_sizing": [line for line in runtime_log.splitlines() if "expert cache auto:" in line],
                                 "kv_mappings": [line for line in runtime_log.splitlines() if "native K/V sessions:" in line]})
        save()
        print(name, stats, flush=True)
    # Output parity across separate processes, including the historical migration path.
    for names in ([x[0] for x in runs if not x[3]], [x[0] for x in runs if x[3]]):
        reference = None
        for name in names:
            r = json.loads((a.output / name / "results.json").read_text(encoding="utf8"))
            replies = {x["name"]: x["text"] for x in r["requests"]
                       if not x["disconnected"] and not x["name"].startswith("cold-")}
            if reference is None:
                reference = replies
            else:
                assert replies == reference, f"cross-run output mismatch: {name}"
    if a.reference_exe:
        command = [sys.executable, str(ROOT / "tools/native_concurrency_http.py"),
                   "--config", str(configs["baseline"].resolve()), "--exe", str(a.reference_exe.resolve()),
                   "--output", str((a.output / "released-defaults").resolve()), "--parallel", "2",
                   "--rounds", "1", "--limit", str(a.limit), "--independent-kv", "--device-slot-copy",
                   "--clients", "1", "2", "4", "8"]
        manifest["commands"].append(command)
        save()
        print("Starting released-defaults parity check", flush=True)
        with (a.output / "released-defaults.log").open("w", encoding="utf8") as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        previous = json.loads((a.output / "released-defaults/results.json").read_text(encoding="utf8"))
        current = json.loads((a.output / runs[0][0] / "results.json").read_text(encoding="utf8"))
        assert previous["passed"]
        expected = {x["name"]: x["text"] for x in current["requests"] if not x["disconnected"]}
        checked = 0
        for row in previous["requests"]:
            if not row["disconnected"]:
                assert row["text"] == expected[row["name"]], f"previous-default output differs: {row['name']}"
                checked += 1
        manifest["previous_default_parity"] = {"exe_sha256": previous["exe_sha256"], "responses": checked}
    manifest["passed"] = True
    save()


if __name__ == "__main__":
    main()
