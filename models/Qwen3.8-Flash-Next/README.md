# Qwen3.8 Flash Next on two R9700

[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) on two AMD Radeon AI PRO R9700, served by vLLM 0.29 on
ROCm 10 with tensor parallel 2 and the Paiton plugin. Two weight sets, one launcher: **3-bit** (fastest, everything in VRAM) and
**4-bit** (the 16-bit model's task scores; part of the experts in system RAM). No custom engine: regular vLLM with the Paiton
plugin and our own kernels.

## Which command do I run?

| I want … | Command | What I get | Host RAM |
|---|---|---|---|
| The fastest chat and code | `bash models/Qwen3.8-Flash-Next/run-flashnext.sh --fn-worker native` | 3-bit, 258.7 tok/s for one request, 687.5 tok/s for 8, 98K-token window | 128 GB+ |
| The most accurate answers | `bash models/Qwen3.8-Flash-Next/run-flashnext-4bit.sh --fn-worker native` | 4-bit, 245.3 tok/s for one request, 566.7 tok/s for 8, 98K-token window | 196 GB+ |
| Agents and coding tools (long conversations resent every turn) | `bash models/Qwen3.8-Flash-Next/run-flashnext-4bit.sh --fn-worker native --prefix-caching` | 4-bit, 262K-token window; a repeated 200K-token prompt starts answering in 0.85 s instead of 33.62 s; 206.9 tok/s for one request | 196 GB+ |
| Agents and coding tools on a smaller host | `bash models/Qwen3.8-Flash-Next/run-flashnext.sh --fn-worker native --prefix-caching` | 3-bit, 98K-token window; a repeated 90K-token prompt in 0.32 s instead of 10.53 s; 101 tok/s for one request (no speculative decoding) | 64 GB+ |
| The 4-bit on a 64-128 GB host | `bash models/Qwen3.8-Flash-Next/run-flashnext-4bit.sh` | 4-bit on the torch worker, about 190 tok/s for one request; the launcher picks the memory layout | 64 GB+ ([layouts](README-4bit.md#system-ram)) |
| Image input | `bash models/Qwen3.8-Flash-Next/run-flashnext-4bit.sh --vision` | 4-bit with images up to 4K (torch worker) | as above |
| 200K-token prompts on the 3-bit | `bash models/Qwen3.8-Flash-Next/run-flashnext-200k.sh` | 3-bit, 200,000-token window, speculative decoding off, about 100 tok/s | 128 GB+ |

Not sure your host fits? Add `--dry-run`: it checks RAM, GTT limit, GPU links and disk and prints what is missing. Without
`--fn-worker native` every command serves the torch worker exactly as in the previous releases. Pages per weight set:
[3-bit](README-3bit.md) · [4-bit](README-4bit.md) · [release notes](RELEASE-NOTES.md) · [all measurements](BENCHMARKS.md).

## Speed

BetterBench 0.6.0 standard profile on two R9700 through this launcher, unique prompts (no prefix-cache hits). Decode is
weighted over eight prompt categories. Native rows: 24 Oct 2026, one run each.

| Command | Decode, 1 request | 8 requests | Prefill, 64K prompt | Time to first token (p50) | Window |
|---|---:|---:|---:|---:|---:|
| 3-bit `--fn-worker native` | **258.7 tok/s** | **687.5 tok/s** | 8,433 tok/s | 56 ms | 98,304 |
| 4-bit `--fn-worker native` | **245.3 tok/s** | **566.7 tok/s** | 7,788 tok/s | 65 ms | 98,304 |
| 4-bit `--fn-worker native --prefix-caching` | 206.9 tok/s | 477.6 tok/s | 7,226 tok/s | 107 ms | 262,144 |
| 3-bit `--fn-worker native --prefix-caching` | 101.0 tok/s | 403.4 tok/s | 8,731 tok/s | 73 ms | 98,304 |
| 3-bit default (torch worker) | 216.3 tok/s | 565.1 tok/s | 8,161 tok/s ¹ | 117 ms | 98,304 |
| 4-bit default (torch worker) | 189.8 tok/s | 420.7 tok/s | 6,991 tok/s ¹ | 121 ms | 98,304 |

¹ measured in the 200K modes (`run-flashnext-200k.sh`, `run-flashnext-4bit-200k.sh`). The torch rows are from 10-11 Oct, the
3-bit cache row from 23 Oct. A second benchmark (LocalMaxxing) and all details: [BENCHMARKS.md](BENCHMARKS.md). How we
tested: [below](#how-we-tested).

## Accuracy

**Task scores** (the same questions for every model; [how we tested](#how-we-tested)):

| Task | 16-bit model | 3-bit | 4-bit |
|---|---:|---:|---:|
| GSM8K (grade-school math, 1,319 questions) | 96.9 % | 97.4 % | 96.8 % |
| HumanEval (Python coding, 164 problems) | 96.3 % | 94.5 % | 96.3 % |
| MMLU-Pro (knowledge and reasoning, 1,400 questions) | 84.7 % | 83.6 % | 83.7 % |

GSM8K asks school math word problems, HumanEval asks for Python functions that must pass tests, MMLU-Pro asks hard
multiple-choice questions across 14 subjects. No difference to the 16-bit model is statistically significant ("within noise");
for scale, the 16-bit model itself moves by up to 1.2 points between two runs with different random seeds.

**Closeness to the 16-bit model:**

| Measure | 3-bit | 4-bit |
|---|---:|---:|
| KL to the 16-bit model, free / routing locked (0 = identical; lower is better) | 0.0550 / 0.0367 | 0.0306 / 0.0166 |
| Same most likely next token as the 16-bit model | 91.18 % | 93.36 % |
| Long documents (8K-32K tokens): extra uncertainty against the 16-bit model | +0.35 % | +0.16 % |
| Needles found in a 131K-token prompt | 40 of 40 | 40 of 40 |
| AI2D science diagrams with images (16-bit: 85.3 %) | 81.8 % ² | – |

KL measures how far the model's next-word probabilities are from the 16-bit model's; two correct 16-bit implementations already
differ by about 0.008. "Routing locked" gives both models the same experts and isolates the rounding error. AI2D asks multiple-choice
questions about science diagrams.
² measured through an emulation of the 3-bit arithmetic; the 3-bit modes take text only.

The 3-bit weights of this release are Hugging Face revision `a100b3ae`, the default 3-bit download. On a host with older 3-bit
weights, the launcher downloads them into a fresh folder (`~/paiton-models/Qwen3.8-Flash-Next-W3A8-a100b3ae`) and says once
that the old folder can be deleted; an explicit `--weights DIR` with other weights is served as is, with a warning. Full
tables and intervals: the model cards ([W3A8](https://huggingface.co/EliovpAI/Qwen3.8-Flash-Next-W3A8-Paiton-RDNA4),
[W4A8](https://huggingface.co/EliovpAI/Qwen3.8-Flash-Next-W4A8-Paiton-RDNA4)).

## Prerequisites
- Two AMD Radeon AI PRO R9700 (32 GiB each) on a ROCm 10 host driver; Docker with GPU device access (`/dev/kfd`, `/dev/dri`;
  groups `video`, `render`).
- `python3` 3.10 or newer and `pip install huggingface_hub` (the launcher downloads the weights and the base model's tokenizer
  files through it).
- Disk: 116 GB (3-bit) or 133 GB (4-bit) of weights, plus the image: 41 GB for the 3-bit torch modes, about 56 GB for the 4-bit
  and every native mode.

**Host RAM per mode** (the launcher checks it and stops with the reason; `--dry-run` prints it):

| Mode | Host RAM |
|---|---|
| 3-bit `--fn-worker native` (also `--mode 3bit-long-cache-mtp-native`) | about 60 GiB of free GTT: 128 GB+ at the default GTT limit (half the RAM) |
| 3-bit `--fn-worker native --prefix-caching` (`3bit-cache-native`, `3bit-long-cache-native`) | from about 64 GB (RAM prefix tier from about 79 GiB) |
| 4-bit `--fn-worker native`, with or without `--prefix-caching` | a GTT limit of about 98 GiB: 196 GB+ at the default GTT limit; smaller hosts are refused and pointed at the modes that fit |
| 3-bit torch default | 128 GB+ |
| 4-bit torch default | 256 GB as benchmarked; 128, 96 and 64 GB with automatic layouts ([4-bit page](README-4bit.md#system-ram)) |

## Use the server
The first start of a mode downloads and verifies the weights, then takes about five to six minutes; wait for
`launcher: READY on port …`. Add `--detach` to run in the background (`docker logs -f <name>` follows the log).

**API:** `http://127.0.0.1:18982/v1` (or `http://<this-machine-ip>:18982/v1`) · **Model:** `Qwen3.8-Flash-Next` ·
**API key:** none unless you set one: `PAITON_API_KEY=<key> bash models/Qwen3.8-Flash-Next/run-flashnext.sh` (clients send it as
the bearer token; `--api-key KEY` works too but shows in the process list). The server listens on all interfaces;
`--host 127.0.0.1` keeps it local. Use a key or a firewall if the machine is reachable from untrusted networks.

```bash
curl --fail http://127.0.0.1:18982/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.8-Flash-Next","messages":[{"role":"user","content":"Write a Python function to remove duplicates from a list."}],"max_tokens":256,"stream":true}'
```

- **Thinking and tool calls:** the model thinks before it answers unless the request sets
  `"chat_template_kwargs": {"enable_thinking": false}`; the thinking comes back in `reasoning`, the answer in `content`, tool calls
  as `tool_calls`.
- **Stop** with `docker stop paiton-flashnext` (or the name you started it with) before switching mode.
- **Long prompts:** send prompts of 100K+ tokens from a file (`-d @request.json`) with a client timeout of a few minutes.

## Native worker (opt-in)
`--fn-worker native` runs the model in the native Paiton worker instead of the torch worker; vLLM keeps the API server, the
scheduler and the KV cache manager. It is the fastest path; every native mode runs one image:
`ghcr.io/eliovp/paiton-vllm-plugin:qwen38-flashnext-rocm10-vllm029-20261024-pc-rc3` @
`sha256:5bca558833ad230ab1e8db7e6a64f538a7396c44dd660b69716a5aec4e714640` (pinned; the launcher pulls it by digest). No
automatic fallback: on a host that does not fit, the launcher stops and says why. No image input (`--vision` is torch-only).
KV profiles, drafters, repeatability and the opt-in decode interleave: [BENCHMARKS.md](BENCHMARKS.md#native-worker-details).

### Request-API limits
The native worker answers these request fields with HTTP 400 and a message that names the field (the torch worker serves them):
`logprobs`, `top_logprobs`, `prompt_logprobs`; `response_format`, structured outputs, guided JSON; tool calls with `strict: true`
tools or `tool_choice` `"required"` / a named tool (plain `auto` tool calling works); `presence_penalty`, `frequency_penalty`,
`repetition_penalty`; `top_k` outside 1..32 at temperature > 0 (the server default 20 is fine); `min_tokens`,
`allowed_token_ids`, `bad_words`; `thinking_token_budget`. `min_p` and `logit_bias` are refused by vLLM itself with speculative
decoding, on both workers. We are working on these limits for a next release.

### Prefix caching on the native worker
An agent or coding tool resends the whole conversation every turn; with `--prefix-caching` the shared part is reused instead of
read again, with the same tokens as a cold start. Blocks evicted from VRAM go to a RAM prefix tier sized from the host (up to
16 GiB per GPU: about 2.2M tokens on the 4-bit) and are loaded back on a hit. Beyond that capacity the least recently used
sessions are dropped and come back cold (correct output, cold speed). Known issue on the 4-bit, under investigation: near that
capacity, sessions can come back cold even when they should fit ([details](BENCHMARKS.md#native-prefix-caching-details)).

| Repeated prompt, time to first token (cold → hit) | 4-bit `--fn-worker native --prefix-caching` | 3-bit `--fn-worker native --prefix-caching` |
|---|---:|---:|
| 32K tokens | 5.25 → 0.69 s | 3.70 → 0.27 s |
| 90K tokens | 14.68 → 0.78 s | 10.53 → 0.32 s |
| 200K-token prompt | 33.62 → 0.85 s | – (98K window; `--mode 3bit-long-cache-native`: 24.53 → 0.42 s) |
| Idle session resumed from the RAM tier | 145K-200K: 0.98-1.17 s (cold 21-30 s) | 90K: 10.48 → 0.42 s |

On the 4-bit, `--native-kv-tier off` is faster for single requests (232.6 instead of 206.9 tok/s, 24 Oct) when your sessions fit in
VRAM (about 311K tokens). Modes, tier sizing, tested host sizes and limits:
[BENCHMARKS.md](BENCHMARKS.md#native-prefix-caching-details).

### Host requirements
See **Host RAM per mode** under [Prerequisites](#prerequisites). `--memory GIB` caps the server container's RAM and sizes the
checks and the RAM tier as on a host with GIB of RAM.

## Troubleshooting
- **"It is slow" during a long prompt.** Every 10 s the server log prints `Avg prompt throughput … Avg generation throughput …`.
  Before the first token of a long request the server is reading the prompt (a 64K-token prompt takes about 8 s on the 3-bit and
  9 s on the 4-bit); generation throughput is near zero then by design.
- **Agents re-reading the whole conversation.** Use `--prefix-caching` (above); the log's *Prefix cache hit rate* shows the reuse.
- **Waiting requests.** `Waiting: N reqs` above 0 means the server is at its concurrency (8 requests) or its KV cache is full.
- **Pauses right after start.** After `/health` answers, the launcher drops the checkpoint from the page cache and waits (at most
  30 s) until the kernel has finished compacting host memory, which can pause the GPUs; `--settle-s 0` and `--keep-page-cache`
  turn this off.

## Options
`--mode decode|prefill-long|decode-nopf|prefill-long-nopf|4bit|4bit-200k|4bit-cache|4bit-long|4bit-xl|3bit-cache-native|3bit-long-cache-native|4bit-long-cache-native|4bit-long-cache-mtp-native|3bit-long-cache-mtp-native`,
`--fn-worker torch|native`, `--prefix-caching`, `--native-kv-tier auto|GIB|off`, `--memory GIB`, `--kv-profile capacity|speed`,
`--drafter distilled|stock`, `--vision`, `--vision-max-pixels N|2mp|4k`, `--host-pools auto|gtt|userptr`,
`--kv-cache-memory-bytes`, `--expert-vram-gib`, `--ple-host auto|pinned|pageable|filebacked`, `--ple-raw-dir DIR`,
`--expert-cold auto|off|SHARE`, `--expert-cold-dir DIR`, `--port`, `--host ADDRESS`, `--api-key KEY`, `--name`, `--weights DIR`,
`--cache DIR`, `--image REF`, `--devices GPU-...,GPU-...`, `--served-model-name`, `--detach`, `--settle-s`, `--keep-page-cache`,
`--dry-run`. The launcher picks the image per mode: the 3-bit torch modes run the 3-bit image (`...-20261010-r2`), the 4-bit
torch modes `...-20261017-r1`, every native mode `...-20261024-pc-rc3`.

<a id="how-the-numbers-were-taken"></a>

## How we tested

- **Hardware.** Two AMD Radeon AI PRO R9700 (32 GB each), PCIe 4.0 x16 host with 251 GiB of RAM. The 16-bit reference does
  not fit there (about 360 GB): it is the original checkpoint served with stock vLLM on a larger AMD GPU server.
- **Serving.** Every number comes from the server exactly as the launcher ships it, through its OpenAI-compatible API.
- **Task benchmarks.** Thinking on (the model's default chat template); temperature 1.0, top-p 0.95, top-k 20; replies capped
  at 16,384 tokens, and a capped reply counts as wrong. The same prompts and the same per-question seed for every model. Full
  GSM8K (1,319), HumanEval (164, run in a sandbox) and MMLU-Pro (1,400: 100 per subject, 14 subjects). Differences are paired
  per question: 95 % confidence intervals and an exact McNemar test.
- **Closeness to the 16-bit model.** KL, top-1 and long-document numbers come from an emulation of the served arithmetic on
  a 294,912-position evaluation corpus and 44 held-out documents of 8K-32K tokens; needles from the running server.
- **Speed.** BetterBench 0.6.0 standard profile: 20 passes per category after warm-up, unique prompts (no prefix-cache hits),
  1/2/4/8 concurrent requests x 48, prefill depths 2K-128K, exact bf16 activations on the wire. LocalMaxxing v0.1.54 (its
  matrix with 1 and 8 requests and 8,192 output tokens): one warm-up request, three runs on one server, the median run kept.
  Raw outputs: `benchmarks/`; tables: [BENCHMARKS.md](BENCHMARKS.md).

The launcher leaves `HSA_ENABLE_IPC_MODE_LEGACY` unset (ROCr's default IPC path, which also works with the in-tree amdgpu driver); set it in your environment (`HSA_ENABLE_IPC_MODE_LEGACY=1`) to force the legacy path.

## Files
`run-flashnext*.sh` (the wrappers), `launch-flashnext.py` (the launcher; also builds the runtime view `<weights>-view` once),
`modes.json` (environment and engine arguments of every mode), `README.md`, `README-3bit.md`, `README-4bit.md`,
`RELEASE-NOTES.md`, `BENCHMARKS.md`, `THIRD_PARTY_NOTICES.md`, `runtime.lock.json` and `checkpoint.lock.json` (what this release
was built from and tested with), `Dockerfile` + `prepare_image_context.py` + `build-context.lock.json` (rebuild the image, see
REPRODUCE.md), `release-audit.json`, `deployment-check.json`, `metrics.json` and `benchmarks/` (raw BetterBench outputs).
