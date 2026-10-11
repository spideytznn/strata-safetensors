"""Train one private expert profile, then freeze it for same-binary hot decode A/B/A."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
from native_concurrency_acceptance import Protocol
from native_scheduler_acceptance import source_read_audit
from safetensors_tokenizer import SafetensorsTokenizer
from serve.frontend import ChatTemplate


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--exe", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--limit", type=int, default=256)
    p.add_argument("--expert-slots", type=int, default=5000)
    p.add_argument("--vision", action="store_true")
    p.add_argument("--profile", type=Path, help="Reuse an already trained private profile.")
    p.add_argument("--toggle", default="STRATA_NATIVE_EXPERT_DAG",
                   choices=("STRATA_NATIVE_EXPERT_DAG", "STRATA_NVFP4_CPU_ROW_PAIR"),
                   help="Off/on/off feature; CPU_ROW_PAIR requires the archived experiment patch.")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    base = json.loads(a.config.read_text(encoding="utf-8-sig"))
    base["exe"] = str(a.exe.resolve())
    args = base["args"]
    def setting(flag, value):
        if flag in args:
            args[args.index(flag) + 1] = str(value)
        else:
            args.extend([flag, str(value)])
    setting("--batch", 2)
    setting("--expert-cache", a.expert_slots)
    if not a.vision:
        base["args"] = args = [v for v in args if v != "--vision"]
    base["env"].update(STRATA_NATIVE_CONTINUOUS="1", STRATA_NATIVE_KV_HANDOFF="1",
        STRATA_NATIVE_BATCH_INPLACE="1", STRATA_NATIVE_KV_INDEPENDENT="1", STRATA_NATIVE_SLOT_COPY="1",
        STRATA_NATIVE_EXPERT_DAG="0", STRATA_NATIVE_PREFIX_PAGES="0", STRATA_NATIVE_RAGGED_PREFILL="0",
        STRATA_NATIVE_KV_TOKEN_BUDGET="0", STRATA_NVFP4_CPU_ROW_PAIR="0")
    # Prefill-first requires auto cache sizing. The controlled test fixes cache
    # capacity and contents explicitly; all three arms use the same allocation order.
    base["env"]["STRATA_NATIVE_PREFILL_FIRST"] = "0"
    tok = SafetensorsTokenizer.from_directory(base["tokenizer"])
    template = ChatTemplate(Path(base["chat_template"]))
    prompts = [tok.encode(template.render([{"role": "user", "content": text}], enable_thinking=False), parse_special=True)
               for text in ["Explain TCP and QUIC in detail, including multiplexing and failure recovery.",
                            "用中文详细解释哈希表、碰撞和扩容，并给出 Python 示例。",
                            "Write a Python LRU cache and explain correctness, complexity and tests.",
                            "Explain binary search invariants and boundary cases with worked examples."]]
    result = {"status": "running", "exe_sha256": hashlib.sha256(a.exe.read_bytes()).hexdigest(),
        "expert_slots": a.expert_slots, "vision": a.vision, "limit": a.limit,
        "toggle": a.toggle, "rounds": a.rounds, "runs": []}
    result["config"] = str(a.config.resolve())
    result["config_sha256"] = hashlib.sha256(a.config.read_bytes()).hexdigest()
    result["prompts_sha256"] = hashlib.sha256(json.dumps(prompts).encode()).hexdigest()
    def save():
        (a.output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8")
    save()
    profile = a.profile.resolve() if a.profile else a.output.resolve() / "trained-profile.bin"
    try:
        if not a.profile:
            folder = a.output / "train"
            folder.mkdir()
            setting("--expert-profile-save", profile)
            setting("--expert-profile-save-every", 0)
            engine = Protocol(base, folder, vision=a.vision)
            try:
                for _ in range(2):
                    for i in range(0, len(prompts), 2):
                        engine.batch([(ids, a.limit, "temperature=0") for ids in prompts[i:i+2]])
            finally:
                engine.close()
            result["training_source_read_audit"] = source_read_audit(folder)
        assert profile.is_file()
        result["profile"] = str(profile)
        result["profile_sha256"] = hashlib.sha256(profile.read_bytes()).hexdigest()
        # No replacements after loading this same learned ranking in each arm.
        for flag in ("--expert-profile-save", "--expert-profile-save-every"):
            if flag in args:
                i = args.index(flag)
                del args[i:i+2]
        setting("--expert-profile", profile)
        setting("--adapt-every", 0)
        reference = None
        reference_routes = None
        expected_slots = None
        for name, paired in (("before", "0"), ("enabled", "1"), ("after", "0")):
            folder = a.output / name
            folder.mkdir()
            cfg = json.loads(json.dumps(base))
            cfg["env"][a.toggle] = paired
            row = {"name": name, "enabled": paired, "status": "running", "cases": []}
            row["engine_args"] = cfg["args"]
            row["engine_env"] = {key: value for key, value in cfg["env"].items()
                                 if key.startswith(("STRATA_", "GGML_"))}
            result["runs"].append(row)
            save()
            print("Starting frozen hot decode", name, flush=True)
            engine = Protocol(cfg, folder, vision=a.vision)
            outputs = []
            try:
                row["info"] = engine.info
                actual_slots = int(re.search(r"\bexpert_slots=(\d+)", engine.info[0])[1])
                if expected_slots is None:
                    expected_slots = actual_slots
                assert actual_slots == expected_slots, "expert cache capacity changed between arms"
                row["actual_expert_slots"] = actual_slots
                for round_id in range(a.rounds + 2):
                    for i in range(0, len(prompts), 2):
                        batch = engine.batch([(ids, a.limit, "temperature=0") for ids in prompts[i:i+2]])
                        outputs.append([job["ids"] for job in batch["jobs"]])
                        if round_id < 2:
                            continue
                        begin = max(job["times"][0] for job in batch["jobs"])
                        end = max(job["times"][-1] for job in batch["jobs"])
                        count = sum(sum(t > begin for t in job["times"]) for job in batch["jobs"])
                        row["cases"].append({"round": round_id-2, "pair_index": i//2,
                            "wall_tok_s": sum(len(job["ids"]) for job in batch["jobs"])/batch["wall_s"],
                            "overlap_tok_s": count/(end-begin), "batch": batch})
                        save()
            finally:
                engine.close()
            row["source_read_audit"] = source_read_audit(folder)
            trace = (folder / "engine.log").read_text(encoding="utf8")
            assert "batch adaptive: round=" not in trace
            row["batch_diagnostics"] = [line for line in trace.splitlines() if line.startswith("strata batch:") and "windows, avg" in line]
            routes = [line.split("routed entries per window: ", 1)[1].split(";", 1)[0]
                      for line in row["batch_diagnostics"]]
            assert len(routes) == (a.rounds + 2) * 2, "missing batch routing diagnostics"
            if reference_routes is None:
                reference_routes = routes
            else:
                assert routes == reference_routes, f"{name} changed expert tier routing"
            row["routing_equal"] = True
            row["output_sha256"] = hashlib.sha256(json.dumps(outputs).encode()).hexdigest()
            row["median_wall_tok_s"] = statistics.median(c["wall_tok_s"] for c in row["cases"])
            row["median_overlap_tok_s"] = statistics.median(c["overlap_tok_s"] for c in row["cases"])
            if reference is None:
                reference = outputs
            else:
                assert outputs == reference, f"{name} changed output"
            row["status"] = "passed"
            save()
            print(name, row["median_wall_tok_s"], row["median_overlap_tok_s"], flush=True)
        result["status"] = "passed"
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
