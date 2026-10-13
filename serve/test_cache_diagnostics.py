"""Cache diagnostic attribution/privacy tests; no GPU or model downloads."""
from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest

from serve.cache_diagnostics import CacheDiagnostics
from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service


class CacheDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "trace.jsonl"
        self.trace = CacheDiagnostics(self.path, turn_token=10000)
        self.engine = SimpleNamespace(gen=1)

    def rows(self):
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines()]

    def prepare(self, ids, **kw):
        values = dict(messages=[{"role": "system", "content": "PRIVATE-SYSTEM"},
                                {"role": "user", "content": "PRIVATE-CHAT"}],
                      tools=[{"name": "secret-tool", "parameters": {"type": "object"}}],
                      kwargs={}, image_sources=[], req={}, engine=self.engine)
        values.update(kw)
        return self.trace.prepare(ids, **values)

    def test_root_change_and_field_changes_without_retaining_prompt(self):
        ids = [10000] + [1] * 599 + [10000] + [2] * 500
        self.prepare(ids)
        changed = list(ids)
        changed[300] = 3
        self.prepare(changed, messages=[{"role": "system", "content": "CHANGED-SYSTEM"}],
                     kwargs={"enable_thinking": False})
        row = self.rows()[-1]
        self.assertEqual(row["first_differing_token_block"], [256, 511])
        self.assertTrue(row["root_changed"])
        self.assertIn("system", row["changed_fields"])
        self.assertIn("template_kwargs", row["changed_fields"])
        self.assertTrue(row["comparison_only"])
        text = self.path.read_text(encoding="utf-8")
        for private in ("PRIVATE-SYSTEM", "PRIVATE-CHAT", "CHANGED-SYSTEM", "secret-tool"):
            self.assertNotIn(private, text)
        self.assertNotIn("ids", self.trace.recent[-1])

    def test_extension_does_not_claim_a_token_mismatch(self):
        self.prepare([1] * 600)
        self.prepare([1] * 700)
        row = self.rows()[-1]
        self.assertEqual(row["common_prefix_tokens_at_least"], 512)
        self.assertNotIn("first_differing_token_block", row)
        self.assertEqual(row["tail_comparison"], "extension_or_tail_change")

    def test_schema_key_order_and_bytes_images_are_distinguished(self):
        self.prepare([1] * 800, tools=[{"name": "x", "parameters": {"type": "object", "properties": {}}}],
                     image_sources=[b"private pixels"])
        self.prepare([1] * 800, tools=[{"parameters": {"properties": {}, "type": "object"}, "name": "x"}],
                     image_sources=[b"different pixels"])
        row = self.rows()[-1]
        self.assertIn("tools", row["changed_fields"])
        self.assertNotIn("tools_canonical", row["changed_fields"])
        self.assertIn("image_sources", row["changed_fields"])
        self.assertTrue(row["same_prompt"])
        self.assertNotIn("pixels", self.path.read_text())

    def test_concurrent_requests_keep_own_done_counts_and_bound_history(self):
        def request(n):
            request_id = self.prepare([n] * 600, req={"session_id": "private-session-" + str(n)})
            self.trace.finish(request_id, {"reused": n, "prompt_read": 600 - n, "prompt_ms": n + .5},
                              "cancel" if n % 2 else "stop", n * 2, self.engine)
            return request_id, n
        with ThreadPoolExecutor(max_workers=4) as pool:
            expected = dict(pool.map(request, range(48)))
        finished = {r["request"]: r for r in self.rows() if r["event"] == "finished"}
        self.assertEqual(len(finished), 48)
        for request_id, n in expected.items():
            self.assertEqual(finished[request_id]["reused"], n)
            self.assertEqual(finished[request_id]["generated"], n * 2)
        self.assertEqual(len(self.trace.recent), 32)
        self.assertNotIn("private-session", self.path.read_text())

    def test_shared_engine_events_and_partial_lines(self):
        engine_log = Path(self.tmp.name) / "engine.log"
        engine_log.write_text("strata serve: conversation cache: parked 100 tokens in 1 ms; parked=1 bytes=99 evictions=2 snapshot_bytes=99 reused_kv_bytes=0\n"
                              "strata serve: conversation cache: evicted 1 parked conversation to admit this snapshot (2923 MiB plus 2560 MiB floor)\n"
                              "strata batch: slot 1 gave back 80 tokens of this conversation (its turn checkpoint) in 2 ms\n"
                              "strata serve: conversation cache: restored 70 tokens (checkpoint) in 3 ms; parked=0 bytes=0", encoding="utf-8")
        self.engine.log_path, self.engine.log_start = str(engine_log), 0
        self.prepare([1] * 600)
        rows = self.rows()
        self.assertEqual(next(r for r in rows if r["event"] == "engine_cache")["eviction_delta"], 2)
        self.assertEqual(next(r for r in rows if r["event"] == "engine_cache_eviction")["reason"], "physical_ram_floor")
        slot = next(r for r in rows if r["event"] == "engine_cache_restore")
        self.assertEqual(slot["slot"], 1)
        self.assertTrue(slot["checkpoint"])
        self.assertNotIn("request", slot)  # shared stderr cannot be assigned to one concurrent request
        with engine_log.open("a") as out:
            out.write("\n")
        self.prepare([2] * 600)
        self.assertEqual(sum(r.get("kind") == "restored" for r in self.rows()), 1)
        self.prepare([3] * 600)
        self.assertEqual(sum(r.get("kind") == "restored" for r in self.rows()), 1)

    def test_logging_failure_and_rotation_do_not_stop_inference(self):
        self.path.mkdir()
        self.assertIsNone(self.prepare([1, 2, 3]))
        self.assertTrue(self.trace.failed)
        self.trace.finish("anything", {}, "stop", 0, self.engine)
        rotated = Path(self.tmp.name) / "rotated.jsonl"
        self.trace = CacheDiagnostics(rotated, max_bytes=1500)
        self.prepare([1] * 600)
        self.prepare([2] * 600)
        self.prepare([3] * 600)
        self.assertTrue(rotated.with_suffix(".jsonl.1").exists())
        self.assertLess(rotated.stat().st_size, 1500)

    def test_service_trace_preserves_prompt_output_and_handles_cancel(self):
        tok = ByteTokenizer()
        template = ChatTemplate(Path(__file__).with_name("chat_template.jinja"))
        messages = [{"role": "user", "content": "PRIVATE-CHAT"}]
        original = copy.deepcopy(messages)
        def run(diagnostic, cancel=False):
            svc = Service(MockEngine(tok, "answer"), tok, template)
            svc.cache_diagnostics = diagnostic
            ids, thinking, maximum = svc.prepare(messages, [], {"enable_thinking": False}, max_new=20)
            event = threading.Event()
            if cancel:
                event.set()
            result = list(svc.run(ids, thinking, [], maximum, {}, event))
            return ids, result
        plain_ids, plain = run(None)
        traced_ids, traced = run(self.trace)
        self.assertEqual(plain_ids, traced_ids)
        # Timings are wall-clock data, so compare the actual emitted model/parser events.
        self.assertEqual([v for k, v in plain if k == "event"], [v for k, v in traced if k == "event"])
        self.assertEqual(messages, original)
        finished = next(r for r in self.rows() if r["event"] == "finished")
        self.assertEqual(finished["reuse_result"], "not_reported")
        run(self.trace, cancel=True)
        self.assertEqual(sum(r["event"] == "finished" for r in self.rows()), 2)

    def test_parallel_http_requests_have_distinct_trace_ids_and_own_stats(self):
        from serve.test_parallel import ParallelService
        fixture = ParallelService()
        self.addCleanup(fixture.tearDown)
        fixture.start(2)
        fixture.svc.cache_diagnostics = self.trace
        replies = fixture._two_at_once("short question with different length")
        self.assertEqual(replies["a"][0], "ok")
        self.assertEqual(replies["b"][0], "ok")
        rows = self.rows()
        prepared = {r["request"]: r for r in rows if r["event"] == "prepared"}
        finished = [r for r in rows if r["event"] == "finished"]
        self.assertEqual(len(prepared), 2)
        self.assertEqual(len(finished), 2)
        history = {r["prompt_total"]: r for r in fixture.svc.history}
        for row in finished:
            own_history = history[prepared[row["request"]]["prompt_tokens"]]
            self.assertEqual(row["reused"], own_history["reused"])
            self.assertEqual(row["prompt_read"], own_history["prompt_read"])
            self.assertEqual(row["prompt_ms"], own_history["prompt_ms"])


if __name__ == "__main__":
    unittest.main()
