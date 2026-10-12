"""The native engine's shared M-RoPE table must never cross concurrent requests."""
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

from pathlib import Path
from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, Service, StrataEngine


class NativeVisionGateTests(unittest.TestCase):
    def setUp(self):
        self.engine = object.__new__(StrataEngine)
        self.engine.batch = 2
        self.engine.info = {"weight_source": "safetensors"}
        self.engine.vision_cv = threading.Condition()
        self.engine.slot_cv = threading.Condition()
        self.engine.vision_readers = 0
        self.engine.vision_writer = False
        self.engine.vision_waiting = 0
        self.engine.slot_busy = [False, False]
        self.engine.alive = lambda: True
        self.started = []
        self.releases = {name: threading.Event() for name in ("a", "b", "image", "late")}

        def generate(ids, max_new, sampling, cancel, embeddings=None):
            self.started.append(ids)
            yield 42
            while not self.releases[ids].wait(0.01) and not cancel.is_set():
                yield None

        self.engine.generate_batched = generate
        self.pool = ThreadPoolExecutor(max_workers=4)
        self.addCleanup(self.pool.shutdown)
        self.addCleanup(lambda: [ev.set() for ev in self.releases.values()])

    def run_request(self, name, image=False, cancel=None):
        return self.pool.submit(lambda: list(self.engine.generate(
            name, 32, {}, cancel or threading.Event(), "image-file" if image else None)))

    def until(self, predicate):
        end = time.monotonic() + 3
        while not predicate() and time.monotonic() < end:
            time.sleep(0.005)
        self.assertTrue(predicate())

    def test_text_parallel_image_exclusive_writer_preference(self):
        a = self.run_request("a")
        b = self.run_request("b")
        self.until(lambda: len(self.started) == 2)
        image = self.run_request("image", image=True)
        self.until(lambda: self.engine.vision_waiting == 1)
        late = self.run_request("late")
        self.releases["a"].set()
        a.result(timeout=3)
        self.assertNotIn("image", self.started)
        self.releases["b"].set()
        b.result(timeout=3)
        self.until(lambda: "image" in self.started)
        self.assertNotIn("late", self.started)
        self.releases["image"].set()
        image.result(timeout=3)
        self.until(lambda: "late" in self.started)
        self.releases["late"].set()
        late.result(timeout=3)
        self.assertEqual(self.engine.vision_readers, 0)
        self.assertFalse(self.engine.vision_writer)

    def test_cancel_waiting_image_does_not_block_new_text(self):
        a = self.run_request("a")
        self.until(lambda: "a" in self.started)
        cancel = threading.Event()
        image = self.run_request("image", image=True, cancel=cancel)
        self.until(lambda: self.engine.vision_waiting == 1)
        cancel.set()
        image.result(timeout=3)
        self.assertNotIn("image", self.started)
        b = self.run_request("b")
        self.until(lambda: "b" in self.started)
        self.releases["a"].set()
        self.releases["b"].set()
        a.result(timeout=3)
        b.result(timeout=3)

    def test_image_waits_for_background_slot_drain(self):
        self.engine.slot_busy[0] = True
        image = self.run_request("image", image=True)
        self.until(lambda: self.engine.vision_writer)
        self.assertNotIn("image", self.started)
        with self.engine.slot_cv:
            self.engine.slot_busy[0] = False
            self.engine.slot_cv.notify_all()
        self.until(lambda: "image" in self.started)
        self.releases["image"].set()
        image.result(timeout=3)

    def test_isolated_image_tables_allow_two_image_histories(self):
        self.engine.info["native_vision_slots"] = 1
        a = self.run_request("a", image=True)
        b = self.run_request("b", image=True)
        self.until(lambda: len(self.started) == 2)
        self.assertCountEqual(self.started, ["a", "b"])
        self.assertFalse(self.engine.vision_writer)
        self.assertEqual(self.engine.vision_waiting, 0)
        self.releases["a"].set()
        self.releases["b"].set()
        a.result(timeout=3)
        b.result(timeout=3)

    def test_isolated_image_tables_allow_image_and_text(self):
        self.engine.info["native_vision_slots"] = 1
        image = self.run_request("image", image=True)
        self.until(lambda: "image" in self.started)
        text = self.run_request("a")
        self.until(lambda: "a" in self.started)
        self.releases["image"].set()
        self.releases["a"].set()
        image.result(timeout=3)
        text.result(timeout=3)

    def test_request_progress_does_not_cross_threads(self):
        a, b = {}, {}
        ready = threading.Barrier(2)
        def record(status, total):
            self.engine.bind_request_status(status)
            ready.wait(timeout=3)
            self.engine._request_progress("reading", prompt_progress=(total - 1, total))
            self.engine.bind_request_status(None)
            self.engine._request_progress("queued")  # a finished generator cannot alter its old request
        fa = self.pool.submit(record, a, 141600)
        fb = self.pool.submit(record, b, 154172)
        fa.result(timeout=3)
        fb.result(timeout=3)
        self.assertEqual(a["prompt_progress"], (141599, 141600))
        self.assertEqual(b["prompt_progress"], (154171, 154172))
        self.assertEqual(a["engine_phase"], "reading")
        self.assertEqual(b["engine_phase"], "reading")

    def test_monitor_does_not_attach_other_requests_finished_prefill(self):
        engine = self.engine
        engine.progress = (141599, 141600)  # request A is decoding after finishing this read
        engine.prefill_tok_s_mean = 3300
        engine.slot_held = [[], []]
        engine.slot_live = [None, None]
        engine.waiting = engine.admitting = 0
        engine.max_context = 262144
        svc = Service(engine, ByteTokenizer(), ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        svc.loaded = lambda: True
        a = dict(busy=True, first_token=time.time(), prompt_tokens=141600, generated=25,
                 started=time.time(), phase="answering")
        b = dict(busy=True, first_token=None, prompt_tokens=154172, generated=0,
                 started=time.time(), engine_phase="queued", phase="waiting for the engine")
        svc.live_reqs = {1: (a, []), 2: (b, [])}
        view = svc.metrics()["live"]
        self.assertEqual(view["state"], "queued")
        self.assertEqual(view["prompt_tokens"], 154172)
        self.assertIsNone(view["prompt_read"])
        self.assertIsNone(view["prompt_total"])
        self.assertIsNone(view["prefill_tok_s_mean"])
        b.update(engine_phase="reading", prompt_progress=(4096, 154172), prefill_tok_s_mean=3000)
        view = svc.metrics()["live"]
        self.assertEqual((view["prompt_read"], view["prompt_total"]), (4096, 154172))

    def test_closed_generator_releases_permit(self):
        gen = self.engine.generate("a", 32, {}, threading.Event())
        self.assertEqual(next(gen), 42)
        gen.close()
        self.assertEqual(self.engine.vision_readers, 0)

    def test_closed_image_drain_releases_writer(self):
        self.engine.slot_busy[0] = True
        gen = self.engine.generate("image", 32, {}, threading.Event(), "image-file")
        self.assertIsNone(next(gen))
        self.assertTrue(self.engine.vision_writer)
        gen.close()
        self.assertFalse(self.engine.vision_writer)
        self.assertEqual(self.engine.vision_waiting, 0)


if __name__ == "__main__":
    unittest.main()
