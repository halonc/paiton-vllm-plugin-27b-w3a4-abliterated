# Qwen3.8 Flash Next 4-bit on two R9700: the accuracy release

[Overview: 3-bit or 4-bit?](README.md) · [Release notes](RELEASE-NOTES.md) · [Benchmarks](BENCHMARKS.md) ·
weights: [EliovpAI/Qwen3.8-Flash-Next-W4A8-Paiton-RDNA4](https://huggingface.co/EliovpAI/Qwen3.8-Flash-Next-W4A8-Paiton-RDNA4)

**The accuracy of the 16-bit model, on two R9700: the same GSM8K and HumanEval scores, and MMLU-Pro within one point.** The 16-bit
model is 335 GiB (360 GB) and can't run on this hardware at all. This is [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
with 4-bit experts, served by vLLM 0.29 on ROCm 10 with tensor parallel 2 and the Paiton plugin; part of the experts is served from
pinned system RAM: the benchmarked setup uses a host with 256 GB of RAM, and the launcher also runs it on 128 GB, 96 GB and 64 GB
hosts (same output, see [System RAM](#system-ram)). No custom engine: regular vLLM with the Paiton plugin and our own kernels.

## Accuracy

| Task benchmark | Questions | 16-bit | **4-bit** |
|---|---:|---:|---:|
| GSM8K (grade-school math) | 1,319 | 96.9 % | **96.8 %** |
| HumanEval (Python, pass@1) | 164 | 96.3 % | **96.3 %** |
| MMLU-Pro (100 per category × 14) | 1,400 | 84.7 % | **83.7 %** |

Same prompts, sampling seeds and thinking-on chat template for both. GSM8K and HumanEval are within the 16-bit model's own run-to-run
variation (0.0 and 1.2 points; paired McNemar p = 1.00). The MMLU-Pro gap of 1.0 point is not statistically significant (p = 0.16) and
comes from replies that reach the 16,384-token limit (the 4-bit model thinks about 12 % longer); on the 1,271 questions where neither
model reached the limit the scores are 91.1 % (4-bit) and 91.0 % (16-bit).

| Perplexity (context length) | 16-bit | **4-bit, as served** | change |
|---|---:|---:|---:|
| wikitext-2 test (512 / 4096) | 4.589 / 3.929 | **4.629 / 3.961** | +0.87 % / +0.80 % |
| unseen text (512 / 4096) | 6.938 / 6.163 | **6.954 / 6.187** | +0.23 % / +0.38 % |

Long documents (44 held-out documents of 8K-32K tokens): negative log-likelihood within 0.16 % of bf16. Needle retrieval at 131,072
tokens: 40 of 40, every bf16 answer reproduced token for token. Measured on the configuration the launcher serves (INT6 trunk, int8
prompt activations, fp8 KV cache). Method, KL, top-1 agreement and per-category scores: the
[model card](https://huggingface.co/EliovpAI/Qwen3.8-Flash-Next-W4A8-Paiton-RDNA4).

## Speed
Official numbers of the default (torch worker, the checkpoint's own drafter), measured on the r3 image through this launcher
(BetterBench 0.6.0 standard profile, exact bf16 wire, cold prefix cache, 2x R9700, PCIe 4.0 x16 host with 251 GiB of RAM):

| `--mode 4bit` / `--mode 4bit-200k` (default, torch worker) | |
|---|---:|
| Weighted single-stream decode (median of 3 runs) | 189.8 tok/s |
| Eight concurrent requests, aggregate | 420.7 tok/s (48/48) |
| Prefill, 64K-token prompt (BetterBench 64K row) | 6,991 tok/s |
| Prefill, exact 64,000-token prompt, cold cache | 7,162 tok/s |
| TTFT p50, short prompts | 121 ms |
| Update p99 | 17.7 ms per stream update |

Per-category decode (tok/s): chat 171.4, code 187.1, file_edit 207.5, json 217.5, math 217.2, prose 170.4, reasoning 170.0,
summarization 206.1. Prefill by depth (tok/s): 2K 5,564 / 8K 6,771 / 16K 6,641 / 32K 7,044 / 64K 6,991 / 128K 6,857.
Decode and concurrency: `--mode 4bit` (MTP depth 3, 98,304-token window, INT6 trunk). Prefill: `--mode 4bit-200k` (200,000-token
window, MTP off, 8-bit trunk). Both use an fp8 (E4M3) KV cache. How the numbers were taken: [overview](README.md#how-the-numbers-were-taken).

## Native worker (opt-in)
`--mode 4bit --fn-worker native` serves `--mode 4bit`'s configuration (98,304-token window, MTP depth 3, INT6 trunk, fp8 KV cache)
through the native Paiton worker. Two KV profiles: `--kv-profile capacity` (the default) = 2.5 GiB per GPU, about 305,338 tokens
(3.11 concurrent 98K contexts, as r3 `--mode 4bit`); `--kv-profile speed` = 1.5 GiB per GPU, about 183,202 tokens (1.86 contexts),
the freed VRAM holds more experts, faster.

**Repeatability:** Greedy output can vary from run to run while a long prompt (tens of thousands of tokens) is being read next to other requests; a request on its own, or several short ones together, repeat exactly. Errors and quality are not affected.

| `--mode 4bit --fn-worker native` | `--kv-profile speed`, distilled drafter B4gh5-mtpft | `--kv-profile capacity` (default), distilled drafter B4gh5-mtpft | `--drafter stock` (checkpoint's own) |
|---|---:|---:|---:|
| Weighted single-stream decode | 240.4 tok/s | 240.3 tok/s | 237.6 tok/s |
| TTFT p50, short prompts | 65 ms | 65 ms | 65 ms |
| Prefill, 64K-token prompt (in this mode) | 7,759 tok/s | 7,757 tok/s | 7,748 tok/s |
| Eight concurrent requests, aggregate | 566.7 tok/s | 561.9 tok/s | 545.4 tok/s |
| Four concurrent requests, aggregate | 456.0 tok/s | 456.9 tok/s | 443.4 tok/s |

Measured 17 Oct. Re-measured on 24 Oct with the release image (default `--kv-profile capacity`, distilled drafter): 245.3 tok/s,
65 ms, 7,788 tok/s, 566.7 tok/s, 450.0 tok/s ([BENCHMARKS.md](BENCHMARKS.md)).

It needs a GTT limit of at least about 98 GiB (about 196 GB of RAM at the default limit); smaller hosts keep the torch default and
its 128 GB / 96 GB / 64 GB layouts. No `--vision`, request-API limits (no logprobs, structured outputs, penalties, ...):
[overview](README.md#native-worker-opt-in). Image (every native mode): `ghcr.io/eliovp/paiton-vllm-plugin:qwen38-flashnext-rocm10-vllm029-20261026-pc-rc4`
@ `sha256:328be15c6f1bc3ec99b1b7a436c4a38eeb82ec36d0589b7bbef38f0e465e11cf`.

**Prefix caching on the native worker (new in 20261024-pc-rc2): `--mode 4bit --fn-worker native --prefix-caching`** switches to
`--mode 4bit-long-cache-mtp-native`: 262,144-token window, prefix caching with 2,048-token blocks, speculative decoding on (MTP depth
3), and a RAM prefix tier sized from the host (`--native-kv-tier auto`, up to 16 GiB per GPU). A repeated 200K-token document reaches
its first token in 0.85 s instead of 33.62 s (24 Oct); from the RAM tier 0.96 s instead of 30.39 s (rc2 gate run). Speed without prefix hits (BetterBench
standard profile): with the RAM tier on 206.9 tok/s single-request decode, 477.6 tok/s at 8 requests (24 Oct); with `--native-kv-tier off`
(hits from VRAM only) 232.6 / 487.5 tok/s, 7,270 tok/s prefill at 64K (`--mode 4bit --fn-worker native` without prefix caching:
245.3 / 566.7 / 7,788 on 24 Oct). The tier needs two executor settings that cost single-request decode, so `off` is the faster choice when
sessions fit in VRAM (311K tokens). Same host requirement as `--mode 4bit --fn-worker native`; a smaller host is refused and pointed
at the 3-bit cache modes. `--mode 4bit-long-cache-native` (speculative decoding off, 94.2 tok/s) stays available. Details, host sizes
and limits: [overview](README.md#prefix-caching-on-the-native-worker).

## Run it
Requirements: [Hardware](#hardware) below and the common [prerequisites](README.md#prerequisites). The first start downloads and
verifies the weights (about 133 GB) into `~/paiton-models/Qwen3.8-Flash-Next-W4A8` (`--weights DIR` to change it).

```bash
bash models/Qwen3.8-Flash-Next/run-flashnext-4bit.sh        # 98,304-token window, speculative decoding on (the speed table)
bash models/Qwen3.8-Flash-Next/run-flashnext-4bit-200k.sh   # 200,000-token window, speculative decoding off
bash models/Qwen3.8-Flash-Next/run-flashnext-4bit-cache.sh  # agents and coding tools: prefix caching + 8 GiB RAM prefix tier
bash models/Qwen3.8-Flash-Next/run-flashnext-4bit-long.sh   # 262,144-token window, speculative decoding on, ~461K tokens of KV cache
bash models/Qwen3.8-Flash-Next/run-flashnext-4bit-xl.sh     # 262,144-token window, speculative decoding on, ~595K tokens of KV cache
bash models/Qwen3.8-Flash-Next/run-flashnext-4bit.sh --fn-worker native   # opt-in: --mode 4bit on the native worker (--kv-profile speed: faster, less KV)
bash models/Qwen3.8-Flash-Next/run-flashnext-4bit.sh --fn-worker native --prefix-caching   # native prefix caching (4bit-long-cache-mtp-native)
```

By hand: `python3 models/Qwen3.8-Flash-Next/launch-flashnext.py --mode 4bit|4bit-200k|4bit-cache|4bit-long|4bit-xl [--vision] [--dry-run]`
(`--mode 4bit` also takes `--fn-worker native [--kv-profile capacity|speed] [--drafter stock] [--prefix-caching]`;
`--mode 4bit-long-cache-mtp-native` and `--mode 4bit-long-cache-native` take `--native-kv-tier auto|GIB|off` and `--memory GIB`).

**Image input: `--vision`** (opt-in, torch worker; `--mode 4bit`, `4bit-cache` and `4bit-long`). Images up to 4K (3840x2160;
`--vision-max-pixels N|2mp|4k` sets a lower cap), up to 8 images per request, no video. The vision tower takes its VRAM from the KV
cache: about 157,882 tokens in `4bit` (305,338 without images), about 170K in `4bit-cache`, about 301,871 in `4bit-long`
(461,566 without); about 1 GiB more RAM. Text answers stayed identical to text-only in our checks. AI2D: 84.7 % (bf16 85.3 %).
`--vision` is refused on `4bit-xl` and `4bit-200k` (not measured there) and with `--fn-worker native`.

**Agents and coding tools: `--mode 4bit-cache`.** An agent resends the whole conversation every turn. This mode is the 4-bit decode
mode with prefix caching (2,048-token blocks) and an 8 GiB RAM prefix tier that keeps prefix blocks evicted from VRAM in pinned system
memory. Repeated long prompts start in about 1 s instead of 6-16 s (32K 6.3 -> 1.0 s, 64K 12.3 -> 1.1 s, 90K 16.2 -> 1.1 s to the
first token); hits from the RAM tier give the same output as a cold start. The cost: about 22 % lower multi-user throughput than
`--mode 4bit` (8 concurrent requests 321.4 tok/s; single-stream 146.7) and about 0.1-0.2 s slower first token on new prompts. It needs
about 108 GiB of free GTT; without that the launcher leaves the RAM tier off, says so, and keeps prefix caching in VRAM.
`--prefix-caching` on `--mode 4bit` / `4bit-200k` is refused (not measured there).

**Long windows with speculative decoding: `--mode 4bit-long` / `--mode 4bit-xl`.** The 4-bit decode mode with a 262,144-token
window: it moves 1 or 2 GiB of experts per GPU from VRAM to the host tier to make room for a larger KV cache. Measured 12-13 October
on the released image with the same protocol for all three rows (allocator cap 0.925):

| Mode | Window | KV cache | Experts in VRAM per GPU | Decode, 1 request | 8 requests | 64K prefill on this server | Needles at ~258K tokens | Peak VRAM per card |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `4bit` | 98,304 | 305,338 tokens | 20.75 GiB | 189.4 tok/s | 416.1 tok/s | 4,527 tok/s | – | 30.6 GiB |
| `4bit-long` | 262,144 | 461,566 tokens | 19.75 GiB | 187.8 tok/s | 410.7 tok/s | 4,161 tok/s | 5 / 5 | 30.6 GiB |
| `4bit-xl` | 262,144 | 595,050 tokens | 18.75 GiB | 184.9 tok/s | 403.3 tok/s | 3,862 tok/s | 5 / 5 | 30.7 GiB |

Prompts are read faster in `--mode 4bit-200k` (speculative decoding off: 6,991 tok/s at 64K). `--prefix-caching` is refused in
both modes (not measured there); `--mode 4bit-cache` is the prefix-caching mode.

**Expert placement:** about 21.4 GiB of experts per GPU stay in VRAM (21.2 GiB with `--vision`; the most-used ones); the rest are read from
pinned system memory when a token needs them. `--expert-vram-gib` changes the split. `--host-pool off` is refused in the 4-bit modes: their expert
tier moves experts through the host copy, so it cannot be turned off.

The first start of a mode takes about five to six minutes; wait for `launcher: READY on port …`. **API:** `http://127.0.0.1:18982/v1`
(or `http://<this-machine-ip>:18982/v1`) · **Model:** `Qwen3.8-Flash-Next`. Network, API key, stopping and troubleshooting:
[overview](README.md#use-the-server).

## Hardware
- Two AMD Radeon AI PRO R9700 (32 GiB each), ROCm 10 host driver.
- **System RAM: 256 GB as benchmarked, 128 GB and 96 GB supported** (`--mode 4bit`, `4bit-cache`, `4bit-long`), **64 GB with the
  cold-expert tier** (`--mode 4bit`, `4bit-long`; next section).
- Native worker (`--fn-worker native`, also with `--prefix-caching`): a GTT limit of at least about 98 GiB (about 196 GB of RAM at
  the default limit); without it the launcher stops with the reason. Tested with native prefix caching on 252 GiB and under a
  200 GiB memory limit (RAM prefix tier 16 GiB per GPU on both).

## System RAM
The launcher picks the memory layout from the host's RAM and the GPU's GTT limit (half the RAM by default) and prints it at start
(`memory layout: ...`, with the reason). `--dry-run` shows the choice without starting anything. Every layout gives the same output.

| Host RAM | Layout the launcher picks | Host RAM in use while serving | Disk |
|---|---|---|---|
| 256 GB | per-layer embedding (PLE) tables and expert copies pinned in GTT memory (the benchmarked setup) | about 109 GiB | 133 GB of weights |
| 128 GB | PLE tables in pageable memory, about 47 GiB pinned (fits the GTT limit) | about 110 GiB | 133 GB of weights |
| 96 GB | PLE tables read from two raw files through the page cache, user-pointer host memory | about 61 GiB, plus page cache | 133 GB of weights + 49 GiB of raw PLE files |
| 64 GB | as 96 GB, plus the coldest share of the experts (0.2) read from one file per GPU (`--expert-cold`, picked automatically) | about 51 GiB, plus page cache | 133 GB of weights + 49 GiB of raw PLE files + the cold-expert files |

- **128 GB:** the launcher keeps about 6 GiB of free RAM beyond the table's figure; `--mode 4bit-cache` (whose 8 GiB RAM prefix
  tier needs more) uses the 96 GB layout there, without the RAM prefix tier.
- **96 GB:** the PLE tables (about 49 GiB) are laid out once as two raw files next to the weights (`<weights>-ple-raw`, about a
  minute on an NVMe disk) and reused at every start. The launcher checks before anything else that the disk has the 49 GiB plus
  10 GiB to spare, and otherwise stops with the numbers (`--ple-raw-dir` puts them on another disk). Free RAM beyond the figure in
  the table becomes page cache for the tables, so fewer reads go to the disk. Host memory runs as user-pointer memory in this
  layout, which can pause for a moment when the kernel compacts host memory.
- **64 GB:** `--expert-cold auto` (the default) turns on only when even the 96 GB layout does not fit the free RAM, with the smallest
  measured share that does: the coldest experts leave the pinned host pool for one file per GPU (`<weights>-expert-cold`, written at
  the first start and reused; `--expert-cold-dir` puts them elsewhere), read on demand. Same output, slower: measured under a 60 GiB
  memory limit, `--mode 4bit` about 85 tok/s single-stream and 181 at 8 requests, `--mode 4bit-long` about 89 / 237 with needles at
  131K tokens 5 / 5. `--expert-cold off` or a share (e.g. `0.2`) chooses explicitly.
- **Speed:** smaller hosts are slower in `--mode 4bit` (short prompt, 512 generated tokens, single-stream / 8 requests, same load
  client): 256 GB pinned 151.7 / 428.5 tok/s; 128 GB pageable, within a 128 GB host's memory budget, 130.1 / 458.3; 96 GB
  file-backed 128.4 / 304.7. One server start each; single-stream speed varies by up to about 15 % between starts.
- **Explicit choices:** `--ple-host pinned`, `pageable` or `filebacked` overrides the automatic choice (the RAM check still
  applies). `--mode 4bit-xl` and `--mode 4bit-200k` keep the pinned layout: on hosts whose GTT limit cannot hold it they fall back to
  ROCm's user-pointer memory and say so (186.0 vs 191.9 tok/s single-stream, 388.9 vs 423.1 at 8 requests in our validation run).
  Raising the GTT limit (`amdgpu.gttsize`, `ttm.pages_limit`) is a system setting the host's administrator may choose; the launcher
  never changes it.
- Disk: about 133 GB for the weights (plus 49 GiB on 96 GB and 64 GB hosts, above). Image of the torch 4-bit modes (the native
  modes run `...-20261026-pc-rc4`, above):
  `ghcr.io/eliovp/paiton-vllm-plugin:qwen38-flashnext-rocm10-vllm029-20261017-r1` @ `sha256:b801bafd3f4cbc04b99153180514845894f98554f49fccdbfe0d6344b47034d9` (digest pinned in modes.json; the launcher pulls it by digest).
- Peak VRAM per card in `--mode 4bit`: 30.6 GiB under a stream of long prompts, 30.0 GiB in the decode benchmark
  (allocator cap 0.925).

## Known issues and limitations
- 98,304-token window in `4bit` (MTP pool), 262,144 in `4bit-long` / `4bit-xl`, 200,000 in `4bit-200k` with MTP off. `--mode 4bit-200k` serves the same experts and fp8
  KV cache with the 8-bit trunk; its accuracy was not measured separately.
- Image input (`--vision`): `4bit`, `4bit-cache` and `4bit-long` on the torch worker only; not on `4bit-xl` or `4bit-200k`.
- `--mode 4bit-cache` is opt-in: the speed table is measured in `--mode 4bit`.
- The native worker is opt-in and has request-API limits and no image input ([overview](README.md#request-api-limits)).
- Native prefix caching with speculative decoding resumes one 2,048-token block before the end of the cached prefix (the draft
  layer's last block is recomputed), so a hit reads up to about 4K tokens again: a 200K re-hit takes 0.84-0.96 s (0.55-0.63 s with
  speculative decoding off in `4bit-long-cache-native`). Resuming an idle session takes longer the deeper it is (measured with
  `4bit-long-cache-native`: 0.6 s at 145K, 1.07 s at 260K). With the RAM tier on, single-request decode is 206.9 instead of 232.6
  tok/s. Greedy output of a hit equals a cold start when the request is served alone; next to other requests it can differ, as
  without the cache.
