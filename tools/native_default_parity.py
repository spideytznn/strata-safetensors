"""Check flag-off native arithmetic against a named released binary; private processes."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
from native_concurrency_acceptance import Protocol
from native_path_matrix import compare
from safetensors_tokenizer import SafetensorsTokenizer
from serve.frontend import ChatTemplate


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--exe", type=Path, required=True)
    p.add_argument("--reference-exe", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int, default=128)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    base = json.loads(a.config.read_text(encoding="utf-8-sig"))
    base["args"] = [arg for arg in base["args"] if arg != "--vision"]
    base["args"].extend(["--batch", "4", "--max-context", "32768"])
    base["env"].update(STRATA_NATIVE_EXPERT_DAG="0", STRATA_NATIVE_RAGGED_PREFILL="0",
                       STRATA_NATIVE_PREFIX_PAGES="0", STRATA_NATIVE_KV_TOKEN_BUDGET="0")
    tok = SafetensorsTokenizer.from_directory(base["tokenizer"])
    template = ChatTemplate(Path(base["chat_template"]))
    texts = ["Explain TCP and QUIC, including multiplexing.", "写一个 Python 二分查找函数，解释边界条件。",
             "Give three detailed tests for an LRU cache and explain eviction.",
             "A reference about sorted arrays. " * 340 + "Explain binary search.",
             "网络协议资料：TCP 提供可靠的有序字节流。" * 100 + "用中文概括。"]
    prompts = [tok.encode(template.render([{"role": "user", "content": text}], enable_thinking=False), parse_special=True) for text in texts]
    result = {"status": "running", "lengths": [len(ids) for ids in prompts], "runs": {}, "comparisons": []}
    def save():
        (a.output / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8")
    save()
    try:
        for name, exe in (("released", a.reference_exe), ("candidate", a.exe)):
            folder = a.output / name
            folder.mkdir()
            cfg = json.loads(json.dumps(base))
            cfg["exe"] = str(exe.resolve())
            cfg["env"]["STRATA_DUMP_FIRST_LOGITS"] = str((folder / "logits.f32").resolve())
            run = {"sha256": hashlib.sha256(exe.read_bytes()).hexdigest(), "config": cfg, "solo": []}
            result["runs"][name] = run
            engine = Protocol(cfg, folder)
            try:
                run["info"] = engine.info
                for ids in prompts:
                    run["solo"].append(engine.solo(ids, a.limit, "temperature=0 ckpt=0"))
                    save()
                run["batch"] = engine.batch([(ids, a.limit, "temperature=0") for ids in prompts[:4]])
            finally:
                engine.close()
                save()
            assert "post-residency expert source bytes=0" in engine.log_path.read_text(encoding="utf8")
            print(f"{name}: completed", flush=True)
        old, new = result["runs"]["released"], result["runs"]["candidate"]
        outputs_old = [row["ids"] for row in old["solo"]] + [row["ids"] for row in old["batch"]["jobs"]]
        outputs_new = [row["ids"] for row in new["solo"]] + [row["ids"] for row in new["batch"]["jobs"]]
        assert outputs_old == outputs_new, "flag-off output changed"
        for i in range(len(outputs_old)):
            x = np.fromfile(a.output / "released" / f"logits.f32.{i}", dtype="<f4")
            y = np.fromfile(a.output / "candidate" / f"logits.f32.{i}", dtype="<f4")
            row = compare(x, y)
            row["index"] = i
            result["comparisons"].append(row)
            save()
            assert np.array_equal(x, y), f"flag-off first logits changed: {row}"
        result["status"] = "passed"
        print(f"{len(outputs_old)} outputs and first-logit arrays identical to the released binary", flush=True)
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
