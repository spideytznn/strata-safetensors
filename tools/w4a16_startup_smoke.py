"""Finite private-engine startup/request checks with copied expert profiles.

Uses the exact single and parallel trial profiles, including vision and context
budget. Opens no HTTP port and terminates all children before returning.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import time

from native_concurrency_acceptance import Protocol, ROOT, ChatTemplate, SafetensorsTokenizer
from serve.server import engine_args


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--profile", choices=("all", "single", "parallel"), default="all")
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    results = {"passed": False, "cases": []}
    result_path = a.output / "results.json"

    def save():
        result_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf8")

    save()
    for label, name in (("single", "rtx5090-262k-prefill-tc-bf16-w4a16.json"),
                        ("parallel", "rtx5090-262k-prefill-tc-bf16-parallel-w4a16.json")):
        if a.profile != "all" and a.profile != label:
            continue
        cfg = json.loads((ROOT / "config/native" / name).read_text(encoding="utf8"))
        cfg["args"] = engine_args(cfg)
        folder = a.output / label
        folder.mkdir()
        args = cfg["args"]
        profile_i = args.index("--expert-profile") + 1
        original_profile = Path(args[profile_i])
        copied_profile = folder.resolve() / "expert-profile.bin"
        shutil.copy2(original_profile, copied_profile)
        args[profile_i] = str(copied_profile)
        tok = SafetensorsTokenizer.from_directory(cfg["tokenizer"])
        template = ChatTemplate(Path(cfg["chat_template"]))

        def prompt(text):
            return tok.encode(template.render([{"role": "user", "content": text}],
                                             enable_thinking=False), parse_special=True)

        ids = prompt("Read the report below. Reply with a brief explanation of its main point.\n" +
                     "The schedule tracks completed work, planned work, and remaining risks. " * 650)
        assert 8192 < len(ids) < 12000, len(ids)
        case = {"profile": name, "prompt_tokens": len(ids),
                "exe_sha256": hashlib.sha256(Path(cfg["exe"]).read_bytes()).hexdigest()}
        results["cases"].append(case)
        save()
        print(f"{label}: starting exact desktop candidate with vision", flush=True)
        start = time.perf_counter()
        eng = Protocol(cfg, folder, vision=True)
        try:
            case["startup_s"] = time.perf_counter() - start
            case["info"] = eng.info
            if label == "parallel":
                assert any("batch_slots=2" in line for line in eng.info), eng.info
                assert any("native_batch_mtp=1" in line for line in eng.info), eng.info
            print(f"{label}: READY in {case['startup_s']:.1f}s; {len(ids)}-token prefill", flush=True)
            case["cold"] = eng.solo(ids, 32, "temperature=0")
            assert case["cold"]["ids"], "no generated tokens"
            case["cold"]["text"] = tok.decode(case["cold"]["ids"])
            case["warm"] = eng.solo(ids, 32, "temperature=0")
            assert case["cold"]["ids"] == case["warm"]["ids"], "warm replay differs"
            if label == "parallel":
                jobs = [prompt("Explain TCP congestion control in detail."),
                        prompt("写一个 Python 二分查找函数并解释边界条件。")]
                case["refs"] = [eng.solo(p, 32, "temperature=0") for p in jobs]
                case["batch"] = eng.batch([(p, 32, "temperature=0") for p in jobs])
                for i in range(2):
                    assert case["refs"][i]["ids"] == case["batch"]["jobs"][i]["ids"], f"slot {i} differs"
                    assert len(case["batch"]["jobs"][i]["ids"]) > 1
                ts = [j["times"] for j in case["batch"]["jobs"]]
                assert ts[0][1] < ts[1][-1] and ts[1][1] < ts[0][-1], "slots did not overlap"
        finally:
            eng.close()
            case["exit_code"] = eng.proc.returncode
            save()
        assert case["exit_code"] == 0, case
        log = (folder / "engine.log").read_text(encoding="utf8")
        assert "[prefill-only] W4A16:" in log
        file_reads = re.findall(r"expert_file_bytes=(\d+)\b", log)
        assert file_reads and all(int(n) == 0 for n in file_reads), "runtime expert file read"
        for source in ("expert", "MTP"):
            assert f"post-residency {source} source bytes=0" in log
        assert "expected w4a8" not in log
        case["passed"] = True
        save()
        print(f"{label}: startup, cold/warm prefill and requests passed; children closed", flush=True)
    results["passed"] = True
    save()


if __name__ == "__main__":
    main()
