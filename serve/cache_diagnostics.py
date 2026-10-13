"""Opt-in cache diagnostics. Fingerprints and counts only; no prompt or token text.

Recent prompts are comparison candidates, not an inventory of the engine's KV.
Engine events come from its shared stderr and deliberately have no request ID.
"""
from array import array
from collections import deque
from datetime import datetime, timezone
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import secrets
import threading
import time


class CacheDiagnostics:
    BLOCK = 256
    RECENT = 32
    READ_MAX = 1 << 20
    MAX_BYTES = 16 << 20
    CACHE = re.compile(r"conversation cache: (parked|skipped|restored) (\d+) tokens.*?parked=(\d+) bytes=(\d+)(?: evictions=(\d+))?")
    SLOT = re.compile(r"strata batch: slot (\d+) gave back (\d+) tokens.*?\((.*?)\)")

    def __init__(self, path, turn_token=None, max_bytes=MAX_BYTES):
        self.path = Path(path)
        self.turn_token = turn_token
        self.max_bytes = max_bytes
        self.lock = threading.RLock()
        self.recent = deque(maxlen=self.RECENT)
        self.key = secrets.token_bytes(32)  # no cross-run/plaintext dictionary fingerprints
        self.run = secrets.token_hex(6)
        self.seq = 0
        self.log_key, self.log_pos = None, 0
        self.evictions = 0
        self.failed = False

    def _fingerprint(self, value, canonical=False):
        def binary(item):
            if isinstance(item, bytes):
                return {"bytes_fp": hmac.new(self.key, item, hashlib.sha256).hexdigest()[:24], "bytes": len(item)}
            raise TypeError("unsupported diagnostic value")
        raw = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=canonical,
                         default=binary).encode("utf-8")
        return hmac.new(self.key, raw, hashlib.sha256).hexdigest()[:24]

    def _token_fp(self, ids):
        return hmac.new(self.key, array("i", ids).tobytes(), hashlib.sha256).hexdigest()[:24]

    def _write(self, event, **fields):
        if self.failed:
            return
        row = {"time": datetime.now(timezone.utc).isoformat(), "event": event, "run": self.run,
               "pid": os.getpid(), **fields}
        data = json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and self.path.stat().st_size + len(data.encode("utf-8")) > self.max_bytes:
            self.path.replace(self.path.with_suffix(self.path.suffix + ".1"))
        with self.path.open("a", encoding="utf-8") as out:
            out.write(data)

    def _failure(self):
        if not self.failed:
            self.failed = True
            print("[strata] cache diagnostics disabled after a logging error; inference continues", flush=True)

    def _poll(self, engine):
        path, start = getattr(engine, "log_path", None), getattr(engine, "log_start", None)
        if not path or start is None:
            return
        key = (str(path), start, getattr(engine, "gen", None))
        if self.log_key != key:
            self.log_key, self.log_pos, self.evictions = key, start, 0
            info = getattr(engine, "info", {}) or {}
            self._write("engine_cache_reset", engine_generation=key[2],
                        budget_mib=info.get("conversation_cache_mib"), slots=info.get("conversation_cache_slots"),
                        floor_mib=info.get("conversation_cache_min_free_mib"))
        size = os.path.getsize(path)
        if size < self.log_pos:
            self.log_pos = 0
            self._write("engine_log_truncated")
        if size - self.log_pos > self.READ_MAX:
            self._write("engine_log_gap", skipped_bytes=size - self.log_pos - self.READ_MAX)
            self.log_pos = size - self.READ_MAX
            discard_partial = True
        else:
            discard_partial = False
        with open(path, "rb") as src:
            src.seek(self.log_pos)
            data = src.read(size - self.log_pos)
        end = data.rfind(b"\n") + 1
        self.log_pos += end
        lines = data[:end].decode("utf-8", "replace").splitlines()
        if discard_partial:
            lines = lines[1:]
        for line in lines:
            m = self.CACHE.search(line)
            if m:
                values = {"kind": m[1], "tokens": int(m[2]), "parked": int(m[3]), "bytes": int(m[4])}
                for field in ("snapshot_bytes", "reused_kv_bytes"):
                    number = re.search(rf"\b{field}=(\d+)", line)
                    if number:
                        values[field] = int(number[1])
                if m[5] is not None:
                    count = int(m[5])
                    values.update(evictions=count, eviction_delta=max(0, count - self.evictions))
                    self.evictions = count
                    # The counter includes RAM-floor and byte/slot evictions. Do not invent a victim or cause.
                if m[1] == "restored":
                    values["source"] = "checkpoint" if "(checkpoint)" in line else "live"
                self._write("engine_cache", **values)
            elif "conversation cache: evicted " in line:
                m = re.search(r"evicted (\d+).*?\((\d+) MiB plus (\d+) MiB floor\)", line)
                if m:
                    self._write("engine_cache_eviction", reason="physical_ram_floor", count=int(m[1]),
                                snapshot_mib=int(m[2]), floor_mib=int(m[3]))
            elif "conversation cache: skip parking" in line:
                reason = ("byte_budget" if "available budget" in line else
                          "physical_ram_floor" if "RAM" in line else
                          "minimum_tokens" if "minimum" in line else "capture_refused")
                numbers = {name: int(value) for name, value in re.findall(r"\b(parked|bytes)=(\d+)", line)}
                self._write("engine_cache_skip", reason=reason, **numbers)
            elif "conversation cache: dropped " in line:
                m = re.search(r"dropped (\d+).*?parked=(\d+)", line)
                if m:
                    self._write("engine_cache_drop", reason="superseded", count=int(m[1]), parked=int(m[2]))
            elif "conversation cache: discard " in line:
                self._write("engine_cache_discard", reason="invalid_snapshot")
            elif "after non-finite logits" in line:
                self._write("engine_cache_flush", reason="nonfinite")
            else:
                m = self.SLOT.search(line)
                if m:
                    self._write("engine_cache_restore", source="slot", slot=int(m[1]), tokens=int(m[2]),
                                checkpoint="checkpoint" in m[3])

    def prepare(self, ids, messages, tools, kwargs, image_sources, req, engine):
        if self.failed:
            return None
        try:
            t0 = time.perf_counter()
            roots = [i for i, token in enumerate(ids) if token == self.turn_token][:2] if self.turn_token is not None else []
            root_n = roots[1] if len(roots) > 1 else 0
            systems = [m for m in messages if m.get("role") in ("system", "developer")]
            fields = {"system": self._fingerprint(systems), "tools": self._fingerprint(tools),
                      "tools_canonical": self._fingerprint(tools, canonical=True),
                      "template_kwargs": self._fingerprint(kwargs), "image_sources": self._fingerprint(image_sources)}
            entry = {"tokens": len(ids), "prompt_fp": self._token_fp(ids),
                     "root_tokens": root_n, "root_fp": self._token_fp(ids[:root_n]) if root_n else None,
                     "fields": fields, "blocks": tuple(self._token_fp(ids[i:i + self.BLOCK])
                                for i in range(0, len(ids), self.BLOCK)),
                     "messages": tuple(self._fingerprint(m) for m in messages),
                     "hint": self._fingerprint({k: req[k] for k in ("prompt_cache_key", "conversation_id", "session_id")
                                                if req and req.get(k) is not None})}
            with self.lock:
                self._poll(engine)
                self.seq += 1
                entry["id"] = f"{self.run}-{self.seq}"
                best, best_rank = None, None
                for previous in reversed(self.recent):
                    common = 0
                    for left, right in zip(previous["blocks"], entry["blocks"]):
                        if left != right:
                            break
                        common += 1
                    msg_common = 0
                    for left, right in zip(previous["messages"], entry["messages"]):
                        if left != right:
                            break
                        msg_common += 1
                    rank = (previous["hint"] == entry["hint"], common, msg_common)
                    if best_rank is None or rank > best_rank:
                        best, best_rank = previous, rank
                compared = {}
                if best is not None:
                    lower = min(best_rank[1] * self.BLOCK, len(ids), best["tokens"])
                    same = best["prompt_fp"] == entry["prompt_fp"]
                    compared = {"compared_request": best["id"], "comparison_only": True,
                                "same_prompt": same, "common_prefix_tokens_at_least": lower,
                                "changed_fields": [k for k in fields if fields[k] != best["fields"][k]],
                                "root_changed": entry["root_fp"] != best["root_fp"]}
                    # Different-size final blocks may be a pure extension, not a token mismatch.
                    if not same and lower + self.BLOCK <= min(len(ids), best["tokens"]):
                        compared["first_differing_token_block"] = [lower, lower + self.BLOCK - 1]
                    elif not same:
                        compared["tail_comparison"] = "extension_or_tail_change"
                entry["comparison"] = compared
                self.recent.append(entry)
                self._write("prepared", request=entry["id"], prompt_tokens=len(ids), prompt_fp=entry["prompt_fp"],
                            root_tokens=root_n, root_fp=entry["root_fp"], fingerprints=fields,
                            message_count=len(messages), image_count=len(image_sources), tool_count=len(tools or []),
                            client_hint_fp=entry["hint"],
                            engine_generation=getattr(engine, "gen", None),
                            diagnostic_ms=round(1000 * (time.perf_counter() - t0), 3), **compared)
                return entry["id"]
        except Exception:
            self._failure()
            return None

    def finish(self, request, stats, finish, generated, engine):
        if not request or self.failed:
            return
        try:
            with self.lock:
                self._poll(engine)
                # Only this API request's DONE segments, never the engine's global last/another slot's figures.
                fields = {k: stats.get(k) for k in ("reused", "prompt_read", "prompt_ms", "decode_ms")}
                reused = stats.get("reused")
                result = "not_reported" if reused is None else "zero_reuse" if reused == 0 else "prefix_reused"
                self._write("finished", request=request, finish=finish, generated=generated,
                            reuse_result=result, **fields)
        except Exception:
            self._failure()
