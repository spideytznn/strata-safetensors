# Cache diagnostics

Set `"cache_diagnostics": true` at the top level of the server config. It is off
when absent. Restart the Python server to apply it. No engine rebuild is needed.
The desktop native single and parallel configs enable it for the next start.

The sidecar uses the engine log's name with `.cache-trace.jsonl` in place of
`.log`. For the desktop native parallel profile it is:

```text
logs/native-262k-parallel.cache-trace.jsonl
```

It rotates at 16 MiB, retaining one `.jsonl.1` file. A logging failure disables
diagnostics and leaves inference running. Prompt text, token IDs, tool schemas,
image bytes, client identifiers and credentials are not written. Fingerprints
use a random key for each server instance; compare them within that run only.

## Reading a repeat-prefill report

1. Find a `finished` event with `reuse_result: "zero_reuse"` or a large
   `prompt_read`. Its `request` identifies the matching `prepared` event.
2. `prepared.fingerprints` separates system/developer messages, the tool list,
   tool JSON with sorted object keys, template kwargs and image sources.
   `changed_fields` compares these with one recent prompt. A change only in
   `tools`, with `tools_canonical` unchanged, indicates JSON object key order
   rather than different schema content. The latter still preserves list order.
3. `root_tokens` and `root_fp` describe the rendered first system segment when
   the tokenizer has a single `<|im_start|>` token. A changed root shows that the
   rendered prefix changed, even when system/tool inputs themselves match.
4. `first_differing_token_block` gives a confirmed differing 256-token block.
   `common_prefix_tokens_at_least` is a lower bound. A different-size tail block
   is marked `extension_or_tail_change`, because it may simply append new tokens.
5. `engine_cache` records saves, restores, snapshot size and cumulative eviction
   counts. `engine_cache_restore` identifies a GPU slot restore. Explicit RAM
   admission evictions have `reason: "physical_ram_floor"`. The general eviction
   counter includes RAM, byte-budget and slot-limit evictions; it cannot identify
   every victim or separate these causes. `engine_cache_reset` includes the
   configured budget, slot ceiling and RAM floor when the engine reports them.

Recent prompts are comparison candidates, **not** a list of retained KV states.
The logger keeps fingerprints of at most 32 prompts, without their token arrays.
Two different conversations can have different roots legitimately. A client
conversation/session/cache key, if supplied, is hashed as `client_hint_fp`; it is
only a hint, not proof that two requests are the same conversation.

`finished` uses only the request's own engine DONE segments. Shared engine stderr
events deliberately carry no request ID: events from concurrent admissions can
interleave, so proximity alone does not prove which request caused an eviction.
Their `time` is when the server observed the event, not an engine event timestamp.
Engine stderr is polled at request preparation and completion; a late line may
appear in the next request's poll. A polling gap is recorded explicitly.
Image source fingerprints do not hash the computed embedding tensors. Repeated
prefill with identical recorded fingerprints still needs examination of engine
snapshot availability, image identity checks or other state changes.

## Validation

On 2026-10-13, the cache diagnostic, frontend, parallel, monitor, prompt encoder
and queue keepalive suites passed 71 tests on Windows without a GPU model run.
The tests cover token/output preservation, cancellation, concurrent request
counts, private data exclusion, log rotation and a failed log destination.
An additional HTTP test passed with two requests using the real server scheduler
and a fake engine, checking each trace against its own request history. A fresh
mock-server process also loaded the config switch and logged two HTTP requests
successfully. No real model process was started or restarted for these checks.

A CPU-only synthetic prompt of 188,462 token IDs, about 720,000 characters of
history, 40 tool definitions and 32 recent comparison candidates took a median
13.037 ms (maximum 14.098 ms over 13 measured samples) for preparation plus the
completion record on the Ryzen 9950X3D desktop. This measures diagnostic work,
not model prefill or decode throughput. No per-token logging is added. Evidence:
`logs/cache-audit-20261013-094503/diagnostic-overhead.json`.

Run the focused tests with:

```text
python -m unittest serve.test_cache_diagnostics
```
