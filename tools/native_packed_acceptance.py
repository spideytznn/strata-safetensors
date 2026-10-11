"""Private-engine ragged prefill output/state regression. Never changes launch profiles."""
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
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--diagnose-single", action="store_true")
    parser.add_argument("--case-index", type=int, choices=range(5))
    parser.add_argument("--turn-checkpoints", action="store_true")
    parser.add_argument("--dump-moe", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = json.loads(args.config.read_text(encoding="utf-8-sig"))
    cfg["exe"] = str(args.exe.resolve())
    def setting(key, value):
        if key in cfg["args"]:
            cfg["args"][cfg["args"].index(key) + 1] = str(value)
        else:
            cfg["args"].extend([key, str(value)])
    setting("--batch", 4)
    setting("--max-context", 32768)
    cfg["args"] = [v for v in cfg["args"] if v != "--vision"]
    cfg["env"].update(STRATA_NATIVE_RAGGED_PREFILL="1", STRATA_NATIVE_EXPERT_DAG="0",
                      STRATA_NATIVE_PREFIX_PAGES="0",
                      STRATA_DUMP_FIRST_LOGITS=str(args.output / "logits.f32"))
    if args.dump_moe:
        cfg["env"].update(STRATA_DUMP_MOE_INPUT=str(args.output / "moe"), STRATA_DUMP_MOE_LAYER="all")
    tok = SafetensorsTokenizer.from_directory(cfg["tokenizer"])
    template = ChatTemplate(Path(cfg["chat_template"]))
    def prompt(text):
        return tok.encode(template.render([{"role": "user", "content": text}], enable_thinking=False), parse_special=True)
    sets = [
        [prompt("Explain TCP and QUIC, including multiplexing."), prompt("写一个 Python 二分查找函数，解释边界条件。")],
        [prompt("A reference about sorted arrays. " * 55 + "Explain binary search."),
         prompt("网络协议资料：TCP 提供可靠的有序字节流。" * 39 + "用中文概括。"),
         prompt("Give three detailed tests for an LRU cache and explain eviction.")],
    ]
    def sized(text, count, question):
        left, right = template.render([{"role": "user", "content": "STRATA_TEST_INSERT\n" + question}], enable_thinking=False).split("STRATA_TEST_INSERT")
        head, tail = tok.encode(left, parse_special=True), tok.encode(right, parse_special=True)
        unit = tok.encode(text)
        body = count - len(head) - len(tail)
        assert body > 0
        return head + (unit * ((body + len(unit) - 1) // len(unit)))[:body] + tail
    sets += [
        [sized("The notebook records units, dates and observations. ", 2049, "Explain why units matter."),
         sized("网络协议资料：TCP 提供可靠的有序字节流。", 3073, "用中文概括。"),
         sized("A cache stores recently used keys and evicts the oldest. ", 785, "Explain LRU eviction.")],
        [sized("A sorted array supports binary search with a low and a high bound. ", 8193, "Give a Python binary search and explain its bounds."),
         sized("Independent scientific records include dates and units. ", 2305, "Describe three useful tests.")],
    ]
    system = "You are a careful tutor. Check units, assumptions and boundary conditions. " * 210
    sets.append([tok.encode(template.render([
        {"role": "system", "content": system},
        {"role": "user", "content": text}], enable_thinking=False), parse_special=True) for text in (
            "A sorted array has a low bound and a high bound. " * 150 + "Explain binary search in Python.",
            "网络协议资料：TCP 提供可靠的有序字节流。" * 100 + "用中文概括。")])
    result = {"status": "running", "config": cfg, "exe_sha256": hashlib.sha256(args.exe.read_bytes()).hexdigest(), "cases": []}
    def save():
        (args.output / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8")
    save()
    engine = Protocol(cfg, args.output)
    index = 0
    def logits():
        nonlocal index
        path = args.output / f"logits.f32.{index}"
        index += 1
        values = np.fromfile(path, dtype="<f4")
        assert values.shape == (248320,) and np.isfinite(values).all(), str(path)
        return values
    try:
        result["info"] = engine.info
        assert any("ragged_prefill=1" in line for line in engine.info), engine.info
        engine.send("BPREF :1;1:2")
        assert engine.read() == "PREFDONE 0 0 0"
        for case, prompts in enumerate(sets):
            if args.case_index is not None and case != args.case_index:
                continue
            refs, ref_logits = [], []
            keys = "temperature=0" + ("" if args.turn_checkpoints else " ckpt=0")
            for ids in prompts:
                refs.append(engine.solo(ids, args.limit, keys))
                ref_logits.append(logits())
            singles = []
            if args.diagnose_single:
                for ids, reference in zip(prompts, ref_logits):
                    engine.send("BPREF 3:" + ','.join(map(str, ids[:-1])))
                    assert engine.read().startswith("PREFDONE 1 ")
                    engine.send("BGEN 3 1 temperature=0 " + ','.join(map(str, ids)))
                    while not engine.read().startswith("BADM 3 "):
                        pass
                    values = logits()
                    singles.append({"vs_standard": compare(reference, values), "logits": values})
            offsets = [0] * len(prompts)
            cuts = []
            for ids in prompts:
                last = next((i for i in range(len(ids) - 2, 0, -1) if ids[i] == 248045), len(ids) - 1)
                end = last if args.turn_checkpoints else len(ids) - 1
                root = next((i for i in range(1, last) if ids[i] == 248045), 0)
                cuts.append(([root] if root >= 2048 else []) + [end])
            chunks = []
            while any(offset < owner_cuts[-1] for offset, owner_cuts in zip(offsets, cuts)):
                active = [slot for slot in range(len(prompts)) if offsets[slot] < cuts[slot][-1]]
                quota = 4096 // len(active)
                if quota >= 256:
                    quota = quota // 256 * 256
                counts = {slot: min(quota, next(cut for cut in cuts[slot] if cut > offsets[slot]) - offsets[slot]) for slot in active}
                boundaries = {slot: offsets[slot] + counts[slot] in cuts[slot] for slot in active}
                engine.send("BPREF " + ';'.join(f"{slot}@{offsets[slot]}{'!' if boundaries[slot] else ''}:" + ','.join(map(str, prompts[slot][offsets[slot]:offsets[slot] + counts[slot]])) for slot in active))
                while True:
                    line = engine.read()
                    if line.startswith("PREFDONE "):
                        assert int(line.split()[1]) == len(active), line
                        chunks.append({"positions": list(offsets), "counts": counts, "done": line})
                        for slot, count in counts.items():
                            offsets[slot] += count
                        break
            batch = engine.batch([(ids, args.limit, keys) for ids in prompts])
            packed_logits = [logits() for ref in ref_logits]
            comparisons = [compare(ref, got) for ref, got in zip(ref_logits, packed_logits)]
            record = {"name": f"ragged-{case}", "lengths": [len(ids) for ids in prompts], "packed": line,
                      "cuts": cuts, "chunks": chunks, "references": refs, "batch": batch, "comparisons": comparisons,
                      "tokens_equal": [ref["ids"] == got["ids"] for ref, got in zip(refs, batch["jobs"])]}
            if singles:
                record["single_owner"] = [{"vs_standard": row["vs_standard"], "vs_packed": compare(row["logits"], got)}
                                           for row, got in zip(singles, packed_logits)]
            result["cases"].append(record)
            save()
            assert all(record["tokens_equal"]), record
            assert all(c["argmax_equal"] and c["kl_reference_candidate"] < 0.01 for c in comparisons), comparisons
            print(f"ragged-{case}: {len(prompts)} isolated owners, lengths {record['lengths']}, greedy equal", flush=True)
        result["status"] = "passed"
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        raise
    finally:
        engine.close()
        save()


if __name__ == "__main__":
    main()
