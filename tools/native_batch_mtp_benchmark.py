"""Finite, private two-slot MTP comparison; never starts HTTP or edits launch configs."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import statistics

from native_concurrency_acceptance import Protocol
from safetensors_tokenizer import SafetensorsTokenizer
from serve.frontend import ChatTemplate


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--exe", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--mode", choices=("target", "mtp"), required=True)
    ap.add_argument("--reference", type=Path)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--limit", type=int, default=256)
    ap.add_argument("--long-tokens", type=int, default=10000)
    args = ap.parse_args()
    assert args.rounds > 0 and args.limit >= 32
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = copy.deepcopy(json.loads(args.config.read_text(encoding="utf8")))
    cfg["exe"] = str(args.exe.resolve())
    flags = cfg["args"]
    for flag in ("--mtp", "--mtp-max-t", "--spec-min-p", "--spec-min-drafts"):
        if flag in flags:
            index = flags.index(flag)
            del flags[index:index + 2]
    if "--batch-mtp" in flags:
        flags.remove("--batch-mtp")
    def setting(flag, value):
        if flag in flags:
            flags[flags.index(flag) + 1] = str(value)
        else:
            flags.extend((flag, str(value)))
    setting("--batch", 2)
    setting("--spec", 3 if args.mode == "mtp" else 2)
    if args.mode == "mtp":
        flags.extend(("--mtp", "native", "--spec-min-p", "0.5", "--batch-mtp"))
    cfg["env"]["STRATA_BATCH_MTP"] = "0"
    cfg["env"]["STRATA_SPEC_COUPLED"] = "0"
    result = {"mode": args.mode, "config": cfg, "passed": False,
              "exe_sha256": hashlib.sha256(args.exe.read_bytes()).hexdigest(), "rounds": []}
    def save():
        (args.output / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8")
    save()
    reference = json.loads(args.reference.read_text(encoding="utf8")) if args.reference else None
    tok = SafetensorsTokenizer.from_directory(cfg["tokenizer"])
    template = ChatTemplate(Path(cfg["chat_template"]))
    def prompt(text):
        return tok.encode(template.render([{"role": "user", "content": text}], enable_thinking=False), parse_special=True)
    prompts = [prompt("Write a very detailed tutorial about TCP and QUIC: congestion control, flow control, multiplexing and packet loss. Include worked examples and at least 30 numbered points. Keep explaining for at least 2000 words."),
               prompt("写一份详细的 Python 算法教程，讲解二分查找、排序、复杂度和边界条件。每种算法都给出代码和多组推演，分至少三十个要点，篇幅不少于三千字。")]
    result["prompt_ids"] = prompts
    if reference:
        assert prompts == reference["prompt_ids"], "comparison prompts differ"
    keys = "temperature=0"
    jobs = [(ids, args.limit, keys) for ids in prompts]
    engine = None
    try:
        engine = Protocol(cfg, args.output, vision=True)
        result["info"] = engine.info
        if args.mode == "mtp":
            assert any("native_batch_mtp=1" in line for line in engine.info)
        print(args.mode + ": ready", flush=True)
        # Two complete warmups precede all reported wall-clock measurements.
        result["warmup"] = [engine.batch(jobs), engine.batch(jobs)]
        expected = [job["ids"] for job in result["warmup"][-1]["jobs"]]
        if reference:
            assert expected == [job["ids"] for job in reference["rounds"][0]["jobs"]], "MTP changes greedy output"
        for index in range(args.rounds):
            measured = engine.batch(jobs)
            result["rounds"].append(measured)
            save()
            assert [job["ids"] for job in measured["jobs"]] == expected, "repeat changes output"
            t0, t1 = [job["times"] for job in measured["jobs"]]
            assert t0[1] < t1[-1] and t1[1] < t0[-1], "requests did not overlap"
            rate = sum(len(job["ids"]) for job in measured["jobs"]) / measured["wall_s"]
            print(f"{args.mode}: round {index + 1}: {rate:.2f} output tok/s including admission/reuse", flush=True)
        rates = [sum(len(job["ids"]) for job in row["jobs"]) / row["wall_s"] for row in result["rounds"]]
        result["median_output_tok_s"] = statistics.median(rates)
        if args.mode == "mtp":
            # An explicit 1.0 floor must remove all non-certain proposals without changing target picks.
            blocked = engine.batch([(ids, 32, keys + " spec_min_p=1") for ids in prompts])
            assert all(job["ids"] == ids[:32] for job, ids in zip(blocked["jobs"], expected)), "confidence gating changes target output"
            result["confidence_one"] = blocked
        if args.long_tokens:
            piece = tok.encode("The committee reviewed the budget and schedule. ")
            text = tok.decode((piece * (args.long_tokens // len(piece) + 1))[:args.long_tokens])
            long_prompts = [prompt(text + "\nWrite a long numbered explanation of why schedules are useful."),
                            prompt(text + "\nWrite a long numbered explanation of why budgets are useful.")]
            assert all(len(ids) + 256 < int(flags[flags.index("--max-context") + 1]) for ids in long_prompts)
            long_refs = [engine.solo(ids, 96, keys) for ids in long_prompts]
            long_jobs = [(ids, 96, keys) for ids in long_prompts]
            long_batch = engine.batch(long_jobs)
            assert all(a["ids"] == b["ids"] for a, b in zip(long_refs, long_batch["jobs"])), "long-slot parity failed"
            long_repeat = engine.batch(long_jobs)
            assert all(a["ids"] == b["ids"] for a, b in zip(long_refs, long_repeat["jobs"])), "long-slot reuse failed"
            assert all(int(job["admission"].split()[8]) > 0 for job in long_repeat["jobs"]), "long prompts were not reused"
            result["long_context"] = {"prompt_tokens": list(map(len, long_prompts)), "solo": long_refs,
                                      "batch": long_batch, "repeated": long_repeat}
            if reference:
                assert [row["ids"] for row in long_refs] == [row["ids"] for row in reference["long_context"]["solo"]], "long outputs differ from target-only"
            print(args.mode + ": long-context parity and reuse passed", flush=True)
    finally:
        if engine is not None:
            engine.close()
            result["private_engine_exit_code"] = engine.proc.returncode
        save()
    assert engine.proc.returncode == 0
    log = engine.log_path.read_text(encoding="utf8")
    for family in ("expert", "MTP"):
        values = [int(value) for value in re.findall(r"post-residency " + family + r" source bytes=(\d+)", log)]
        assert values and all(value == 0 for value in values)
    result["post_residency_source_bytes"] = {"expert": 0, "MTP": 0}
    if args.mode == "mtp":
        result["batch_mtp_stats"] = [dict(zip(("slot", "accepted", "offered", "overlap_offered", "confidence_skips", "max_rows"), map(int, row)))
                                    for row in re.findall(r"safetensors batch MTP slot=(\d+) accepted=(\d+) offered=(\d+) overlap_offered=(\d+) confidence_skips=(\d+) max_rows=(\d+)", log)]
        for slot in (0, 1):
            stats = [row for row in result["batch_mtp_stats"] if row["slot"] == slot]
            assert any(row["overlap_offered"] > 0 and row["accepted"] > 0 and row["max_rows"] == 3 for row in stats)
            assert any(row["offered"] == 0 and row["confidence_skips"] > 0 and row["max_rows"] == 1 for row in stats), "1.0 floor still verified proposals"
    result["passed"] = True
    save()
    print(f"{args.mode}: median {result['median_output_tok_s']:.2f} output tok/s, passed", flush=True)


if __name__ == "__main__":
    main()
