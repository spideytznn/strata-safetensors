"""Private, same-binary CPU/GPU scheduler ablations with exact output checks.

No HTTP, launch-profile edits, model writes or remote operations. All child
engines and vision processes are closed, including on a failed assertion.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
from native_concurrency_acceptance import Protocol
from safetensors_tokenizer import SafetensorsTokenizer
from serve.frontend import ChatTemplate


def audit(engine):
    assert engine.proc.returncode == 0, f"engine exit {engine.proc.returncode}"
    trace = engine.log_path.read_text(encoding="utf8")
    for family in ("expert", "MTP"):
        values = list(map(int, re.findall(r"post-residency " + family + r" source bytes=(\d+)", trace)))
        assert values and all(x == 0 for x in values), f"{family} source reads: {values}"
    return trace


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--exe", type=Path, required=True)
    ap.add_argument("--reference-exe", type=Path, help="separate unchanged binary for reference arms")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--limit", type=int, default=256)
    ap.add_argument("--arms", default="off,queue_warm,all_warm,off")
    ap.add_argument("--mtp", action="store_true")
    ap.add_argument("--freeze", action="store_true", help="train/reuse one profile and disable promotion for controlled timing")
    ap.add_argument("--profile", type=Path)
    ap.add_argument("--expert-slots", type=int, help="override expert capacity; frozen default is 5000 target-only or 3800 with MTP")
    ap.add_argument("--solo", action="store_true", help="also measure each prompt serially after the paired cases")
    args = ap.parse_args()
    capacity_override = args.expert_slots is not None
    if args.expert_slots is None:
        args.expert_slots = 3800 if args.mtp else 5000
    assert args.expert_slots > 0
    assert args.rounds > 0 and args.limit >= 32
    arms = args.arms.split(",")
    assert all(arm in ("off", "queue", "fetch", "queue_fetch", "plan", "all", "warm", "queue_warm", "all_warm", "reference") for arm in arms)
    assert "reference" not in arms or args.reference_exe is not None
    args.output.mkdir(parents=True, exist_ok=False)
    base = json.loads(args.config.read_text(encoding="utf8"))
    base["exe"] = str(args.exe.resolve())
    flags = base["args"]
    for flag in ("--mtp", "--mtp-max-t", "--spec-min-p", "--spec-min-drafts", "--expert-profile-save", "--expert-profile-save-every"):
        if flag in flags:
            i = flags.index(flag)
            del flags[i:i + 2]
    if "--batch-mtp" in flags:
        flags.remove("--batch-mtp")
    def setting(cfg, flag, value):
        flags = cfg["args"]
        if flag in flags:
            flags[flags.index(flag) + 1] = str(value)
        else:
            flags.extend((flag, str(value)))
    setting(base, "--batch", 2)
    setting(base, "--spec", 3 if args.mtp else 2)
    if args.mtp:
        flags.extend(("--mtp", "native", "--spec-min-p", "0.5", "--batch-mtp"))
    base["env"].update(STRATA_UNIFIED_EXPERT_BATCH="0", STRATA_UNIFIED_EXPERT_PLAN="0",
                       STRATA_NATIVE_READY_QUEUE="0", STRATA_BATCH_MTP="0", STRATA_SPEC_COUPLED="0",
                       STRATA_ROUTE_TAIL_SKIP="0", STRATA_UNIFIED_FETCH_OVERLAP="0", STRATA_NATIVE_WARM_AHEAD="0")
    if args.freeze or capacity_override:
        setting(base, "--expert-cache", args.expert_slots)
        # Existing prefill-first startup requires automatic expert capacity.
        # A controlled explicit capacity retains dedicated workspace but uses
        # the supported ordinary startup allocation order in every arm.
        base["env"]["STRATA_NATIVE_PREFILL_FIRST"] = "0"
    tokenizer = SafetensorsTokenizer.from_directory(base["tokenizer"])
    template = ChatTemplate(Path(base["chat_template"]))
    texts = ["Explain TCP and QUIC in detail, including multiplexing and failure recovery. Write a long tutorial with worked examples.",
             "用中文详细解释哈希表、碰撞和扩容，并给出 Python 示例。写一份长篇教程。",
             "Write a Python LRU cache and explain correctness, complexity and tests, with a long worked example.",
             "Explain binary search invariants and boundary cases with many worked examples. Write a long tutorial."]
    prompts = [tokenizer.encode(template.render([{"role": "user", "content": text}], enable_thinking=False), parse_special=True)
               for text in texts]
    result = {"passed": False, "exe_sha256": hashlib.sha256(args.exe.read_bytes()).hexdigest(),
              "config": str(args.config.resolve()), "base_config": base, "prompt_ids": prompts,
              "mtp": args.mtp, "frozen": args.freeze, "limit": args.limit, "runs": []}
    def save():
        (args.output / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8")
    save()
    try:
        if args.freeze:
            profile = args.profile.resolve() if args.profile else args.output.resolve() / "trained-profile.bin"
            if not args.profile:
                folder = args.output / "train"
                folder.mkdir()
                cfg = copy.deepcopy(base)
                setting(cfg, "--expert-profile-save", profile)
                setting(cfg, "--expert-profile-save-every", 0)
                print("training private expert ranking", flush=True)
                engine = Protocol(cfg, folder, vision=True)
                try:
                    for _ in range(2):
                        for pair in (prompts[:2], prompts[2:]):
                            engine.batch([(ids, args.limit, "temperature=0") for ids in pair])
                finally:
                    engine.close()
                audit(engine)
            assert profile.is_file()
            setting(base, "--expert-profile", profile)
            setting(base, "--adapt-every", 0)
            setting(base, "--adapt-swaps", 0)
            result["profile"] = str(profile)
            result["profile_sha256"] = hashlib.sha256(profile.read_bytes()).hexdigest()
        elif "--expert-profile" in base["args"]:
            profile = Path(base["args"][base["args"].index("--expert-profile") + 1]).resolve()
            result["profile"] = str(profile)
            result["profile_sha256"] = hashlib.sha256(profile.read_bytes()).hexdigest()
        reference = None
        capacity = None
        for index, arm in enumerate(arms):
            if "profile" in result:
                assert hashlib.sha256(Path(result["profile"]).read_bytes()).hexdigest() == result["profile_sha256"], "ranking changed between arms"
            folder = args.output / f"{index}-{arm}"
            folder.mkdir()
            cfg = copy.deepcopy(base)
            if arm == "reference":
                cfg["exe"] = str(args.reference_exe.resolve())
            cfg["env"]["STRATA_NATIVE_READY_QUEUE"] = "1" if arm in ("queue", "queue_fetch", "all", "queue_warm", "all_warm") else "0"
            cfg["env"]["STRATA_UNIFIED_EXPERT_PLAN"] = "1" if arm in ("plan", "all", "all_warm") else "0"
            cfg["env"]["STRATA_UNIFIED_FETCH_OVERLAP"] = "1" if arm in ("fetch", "queue_fetch", "all", "all_warm") else "0"
            cfg["env"]["STRATA_NATIVE_WARM_AHEAD"] = "1" if arm in ("warm", "queue_warm", "all_warm") else "0"
            row = {"arm": arm, "config": cfg, "cases": [], "passed": False,
                   "exe_sha256": hashlib.sha256(Path(cfg["exe"]).read_bytes()).hexdigest()}
            result["runs"].append(row)
            save()
            print(f"starting {index}-{arm}", flush=True)
            engine = Protocol(cfg, folder, vision=True)
            outputs = []
            try:
                row["info"] = engine.info
                info = " ".join(engine.info)
                for feature, enabled in (("unified_expert_plan", arm in ("plan", "all", "all_warm")),
                                         ("native_ready_queue", arm in ("queue", "queue_fetch", "all", "queue_warm", "all_warm")),
                                         ("expert_fetch_overlap", arm in ("fetch", "queue_fetch", "all", "all_warm")),
                                         ("cpu_warm_ahead", arm in ("warm", "queue_warm", "all_warm"))):
                    if enabled:
                        assert feature + "=1" in info, f"requested {feature} did not activate"
                    else:
                        assert feature + "=1" not in info, f"unexpected {feature}"
                current = int(re.search(r"\bexpert_slots=(\d+)", engine.info[0])[1])
                if capacity is None:
                    capacity = current
                assert current == capacity, f"cache capacity changed: {current} vs {capacity}"
                for repeat in range(args.rounds + 2):
                    for pair_index, pair in enumerate((prompts[:2], prompts[2:])):
                        batch = engine.batch([(ids, args.limit, "temperature=0") for ids in pair])
                        outputs.append([job["ids"] for job in batch["jobs"]])
                        t0, t1 = [job["times"] for job in batch["jobs"]]
                        assert t0[1] < t1[-1] and t1[1] < t0[-1], "requests did not overlap"
                        if repeat < 2:
                            continue
                        begin = max(job["times"][0] for job in batch["jobs"])
                        end = max(job["times"][-1] for job in batch["jobs"])
                        count = sum(sum(t > begin for t in job["times"]) for job in batch["jobs"])
                        case = {"round": repeat - 2, "pair": pair_index,
                                "wall_tok_s": sum(len(job["ids"]) for job in batch["jobs"]) / batch["wall_s"],
                                "overlap_tok_s": count / (end - begin), "batch": batch}
                        row["cases"].append(case)
                        save()
                if args.solo:
                    row["solo"] = [engine.solo(ids, args.limit, "temperature=0") for ids in prompts]
            finally:
                engine.close()
            trace = audit(engine)
            row["post_residency_source_bytes"] = {"expert": 0, "MTP": 0}
            row["private_engine_exit_code"] = engine.proc.returncode
            row["diagnostics"] = [line for line in trace.splitlines() if line.startswith(("strata batch:", "strata unified", "strata native ready", "strata warm ahead"))]
            if arm in ("queue", "queue_fetch", "all", "queue_warm", "all_warm"):
                batches = re.search(r"strata native ready queue: batches=(\d+)", trace)
                assert batches and int(batches[1]) > 0, "CPU ready queue did not execute"
            if arm in ("plan", "all", "all_warm"):
                counts = re.search(r"strata unified experts: samples=(\d+) decisions=(\d+) changed=(\d+)", trace)
                assert counts and all(int(v) > 0 for v in counts.groups()), "planner did not change ownership"
            if arm in ("warm", "queue_warm", "all_warm"):
                counts = re.search(r"strata warm ahead: batches=(\d+) RAM_bytes=(\d+)", trace)
                assert counts and all(int(v) > 0 for v in counts.groups()), "RAM warming did not execute"
            row["output_sha256"] = hashlib.sha256(json.dumps(outputs).encode()).hexdigest()
            if reference is None:
                reference = outputs
            assert outputs == reference, f"{arm}: output mismatch"
            for metric in ("wall_tok_s", "overlap_tok_s"):
                row["median_" + metric] = statistics.median(case[metric] for case in row["cases"])
            row["aggregate_wall_tok_s"] = (
                sum(sum(len(job["ids"]) for job in case["batch"]["jobs"]) for case in row["cases"]) /
                sum(case["batch"]["wall_s"] for case in row["cases"]))
            row["pair_median_wall_tok_s"] = [statistics.median(case["wall_tok_s"] for case in row["cases"]
                                                              if case["pair"] == pair) for pair in (0, 1)]
            row["passed"] = True
            save()
            print(f"{index}-{arm}: wall {row['median_wall_tok_s']:.2f}, overlap {row['median_overlap_tok_s']:.2f} tok/s; exact outputs", flush=True)
        result["passed"] = True
    except BaseException as error:
        result["error"] = repr(error)
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
