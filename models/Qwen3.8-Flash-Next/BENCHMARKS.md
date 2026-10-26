# Benchmarks (measured on the shipped release image through this directory's launcher; BetterBench 0.6.0 standard profile through the vLLM OpenAI server, exact bf16 wire, cold prefix cache, 2x R9700)

<a id="release-20261024-24-oct-2026-before-and-now"></a>

## Release 20261024 (24 Oct 2026)

The release image through this launcher, 24 Oct 2026, one BetterBench run and one LocalMaxxing series per configuration.

| | 3-bit `--fn-worker native` | 4-bit `--fn-worker native` | 4-bit `--fn-worker native --prefix-caching` |
|---|---:|---:|---:|
| BetterBench decode, 1 request (tok/s) | 258.7 | 245.3 | 206.9 |
| BetterBench time to first token (p50) | 56 ms | 65 ms | 107 ms |
| BetterBench prefill, 64K prompt (tok/s) | 8,433 | 7,788 | 7,226 |
| BetterBench 8 requests (tok/s) | 687.5 | 566.7 | 477.6 |
| BetterBench 4 requests (tok/s) | 532.6 | 450.0 | 395.1 |
| LocalMaxxing 1 request (tok/s) | 220.3 | 200.5 | 188.9 |
| LocalMaxxing 8 requests (tok/s) | 781.3 | 680.2 | 629.3 |
| LocalMaxxing time to first token, 1 request | 59 ms | 67 ms | 101 ms |
| LocalMaxxing prefill 8K / 64K (tok/s) | 8,382 / 8,251 | 7,112 / 6,993 | 6,248 / 6,244 |
| LocalMaxxing 128K, 1 request (tok/s) | – (98K window) | – (98K window) | 163.2 |
| Repeated prompt, time to first token, 32K / 90K / 200K (cold → hit) | – | – | 5.25 → 0.69 / 14.68 → 0.78 / 33.62 → 0.85 s |
| Idle sessions resumed from the RAM tier | – | – | 145K-200K tokens: 0.98-1.17 s (cold 21-30 s) |

**BetterBench**: 0.6.0 standard profile, one run per configuration (method at the end of this page). **LocalMaxxing**: lmx
v0.1.54 with llm-inference-bench v0.7.7, `--concurrency 1,8 --contexts 0,128k --skip-oversize`, 8,192 output tokens per request
in 30 s decode cells, one fixed 119-token prompt (`ignore_eos`, server-default sampling); one warm-up request on the fresh server,
then three matrix runs on the same server, and the run with the median single-request tok/s is kept. The 128K cell is skipped on
98,304-token windows. LocalMaxxing's single fixed prompt gives a lower draft acceptance than BetterBench's mix of eight
categories, so its single-request number is lower and its 8-request number higher; compare rows within one benchmark.

## Native worker details
- **Modes.** `--mode decode --fn-worker native` (3-bit) and `--mode 4bit --fn-worker native` (4-bit) serve their torch mode's
  configuration (98,304-token window, MTP depth 3) through the native worker. 3-bit KV cache about 551K tokens. 4-bit:
  `--kv-profile capacity` (default) 2.5 GiB per GPU, about 305,338 tokens; `--kv-profile speed` 1.5 GiB per GPU, about 183,202
  tokens, more experts in VRAM (240.4 / 65 ms / 7,759 / 566.7 / 456.0, 17 Oct).
- **Drafter.** The native modes use the distilled MTP drafter B4gh5-mtpft, shipped inside the image. The launcher assembles
  `<checkpoint dir>-b4gh5-mtpft` next to the checkpoint (hard links plus links to the image's drafter files; the downloaded
  checkpoint is never modified); `--dry-run` checks the drafter files against their pinned sha256. `--drafter stock` uses the
  checkpoint's own drafter (4-bit, 17 Oct: 237.6 / 65 ms / 7,748 / 545.4 / 443.4).
- **Decoding next to long prompts (opt-in).** `--native-env PAITON_FN_DECODE_INTERLEAVE=1` adds one decode-only step after each
  2,048-token prompt chunk: in a burst of eight 17K-29K-token prompts a decoding request rose from about 9.5 to about 21 tok/s, at
  5 % less aggregate prompt throughput (`=2`: about 29 tok/s, -9 %). Off by default: with it on, the greedy tokens of concurrent
  requests can differ from run to run.
- **Repeatability (4-bit).** Greedy output can vary from run to run while a long prompt is read next to other requests; a request
  on its own, or several short ones together, repeat exactly. The 3-bit native worker repeats exactly in all of these cases.

## Native prefix-caching details
`--prefix-caching` with `--fn-worker native` switches to the mode's cache variant and prints the switch; the cache modes can also
be named with `--mode`:

| Mode | Reached by | Window | Speculative decoding | KV cache | Host |
|---|---|---:|---|---|---|
| `4bit-long-cache-mtp-native` | `--mode 4bit --fn-worker native --prefix-caching` (also from `4bit-long-native`) | 262,144 | on (MTP 3) | 2.5 GiB per GPU, about 311K tokens | GTT limit about 98 GiB |
| `4bit-long-cache-native` | `--mode 4bit-long-cache-native` | 262,144 | off | | GTT limit about 98 GiB |
| `3bit-cache-native` | `--mode decode --fn-worker native --prefix-caching` when the free GTT is below 60 GiB (64-120 GB hosts), or named | 98,304 | off | | from about 63 GiB of RAM |
| `3bit-long-cache-native` | `--mode 3bit-long-cache-native` | 262,144 | off | | from about 63 GiB of RAM |
| `3bit-long-cache-mtp-native` | `--mode decode --fn-worker native --prefix-caching` with at least 60 GiB of free GTT (128 GB+ at the default GTT limit) | 262,144 | on (MTP 3) | 4.5 GiB per GPU, about 561K tokens | as 3-bit native |

- **Blocks.** 2,048 tokens; a hit resumes on a block boundary and gives the same tokens as a cold start of the same request. With
  speculative decoding a hit resumes one block earlier (the draft layer's last block is recomputed), so a hit reads up to about
  4K tokens again. Next to other running requests the greedy tokens can differ, with or without a hit (the batch composition
  changes the arithmetic; quality is not affected).
- **RAM prefix tier.** `--native-kv-tier auto` (the default) sizes it at start and prints the formula with its numbers on the
  `native RAM prefix tier:` line:
  `tier per GPU = min(16, floor((available RAM - the mode's resident RAM - start margin - 8 GiB) / 2)) GiB, off below 2 GiB`.
  With the tier off, prefix caching stays on with hits from VRAM only. `--native-kv-tier GIB|off` overrides it. The tier is
  filled at start (10-27 s), so a RAM shortfall shows at start, not under load. Capacity: on `4bit-long-cache-mtp-native` 16 GiB per GPU is
  1,089 blocks of 2,048 tokens (7.7 KB per token per GPU, both ranks), and every session also keeps 14 blocks for its recurrent
  state (7 state groups, at the prompt end and one block earlier), so the tier holds about (1,089 - 14 x sessions) x 2,048 tokens:
  2.0M for 8 sessions, 1.9M for 11; a smaller auto-sized tier holds proportionally less. The server prints the numbers on its
  `kv tier: holds about` line. Sessions that do not fit come back cold (correct output, cold speed). When sessions taking turns
  need more than the tier holds, plain LRU would drop each session just before its turn (25 Oct, 11 sessions, 2.05M tokens:
  0 of 11 from the tier); the tier then keeps whole sessions and drops the most recently served other session instead: 9 of 11
  resumed from the tier at 0.87-1.33 s (cold 21-46 s), the oldest and the last one cold. Four 200K contexts re-hit at about 1 s
  as before.
- **Tier trade on the speculative cache modes.** With the tier on, the executor runs without early token return and without the
  MALL prefetch (both share the side stream the tier copies on); the launcher switches them and prints the trade. 4-bit:
  206.9 tok/s single-request decode with the tier (24 Oct release run), 232.6 with `--native-kv-tier off` (24 Oct, rc2 gate run:
  232.6 / 104 ms / 7,270 / 487.5 / 397.6); 3-bit `3bit-long-cache-mtp-native` 213.5 / 85 ms / 8,346 / 611.0 / 460.4 with the
  tier, 250.0 / 81 / 8,404 / 657.7 / 505.1 without.
- **Speed of the cache modes without speculative decoding** (BetterBench standard, 23 Oct; decode / TTFT p50 / prefill @64K /
  8 requests / 4 requests): `4bit-long-cache-native` 94.2 / 110 ms / 6,918 / 313.0 / 233.8; `3bit-cache-native` 101.0 / 73 ms /
  8,731 / 403.4 / 279.7; `3bit-long-cache-native` 100.5 / 73 ms / 8,725 / 401.5 / 279.0.
- **Repeated prompts, other modes** (cold → hit, one request at a time, 23-24 Oct): `4bit-long-cache-native` 90K 16.82 → 0.46 s,
  200K document 38.14 → 0.55 s, from the RAM tier 32.72 → 0.63 s; `3bit-long-cache-native` 90K 10.53 → 0.32 s, 200K document
  24.53 → 0.42 s; `3bit-long-cache-mtp-native` 90K 11.09 → 0.60 s, 200K document 25.81 → 0.68 s, from the RAM tier
  25.61 → 0.92 s. rc2 gate run of `4bit-long-cache-mtp-native`: 32K 5.26 → 0.69 s, 90K 14.72 → 0.78 s, 200K document
  33.62 → 0.84 s, from the RAM tier 30.39 → 0.96 s.
- **Idle sessions** (`4bit-long-cache-native`, 23 Oct): eight sessions of 145K-260K tokens (1.62M in total) left idle until
  their blocks had moved to the RAM tier, then resumed one at a time: 0.57-0.62 s at 145K up to 1.07 s at 260K (cold: 23-51 s);
  every session gave the same tokens as its cold twin. A resume reads the last 2.0-2.45K tokens again, which attend over the
  whole context; the copy from RAM takes about 0.15 s. A stream decoding while another session is restored keeps going
  (inter-token gap p50 10.5 → 12.5 ms, at most one gap of about 0.46 s).
- **`--memory GIB`** caps the server container's RAM (docker `--memory`, no swap); RAM checks, GTT limit (half of it) and tier
  are sized as on a host with GIB of RAM. Tested: 252 GiB, all cache modes, tier 16 GiB per GPU, all gates pass; 200 GiB,
  `4bit-long-cache-native`, tier 16 GiB per GPU, all gates pass; 192 / 128 / 64 GiB, 4-bit cache modes refused before start
  (`GTT limit ... < 98 GiB ...; this host fits --mode 3bit-cache-native or --mode 3bit-long-cache-native`); `3bit-cache-native`
  at 128 GiB tier 16 GiB per GPU and at 64 GiB tier off, all gates pass (tight: about 7 GiB left after the host tables).
- **Limits.** The 4-bit cache modes need their host tables pinned in GTT (no `--ple-host pageable|filebacked`, `--expert-cold`);
  no `--vision`; the request-API limits of the native worker.

Run of 10 Oct 2026 09:40-10:10 UTC on the release image with its baked compile caches: parts single-stream + concurrency in `--mode decode`,
prefill in `--mode prefill-long`.

## 3-bit torch worker, r3 (10 Oct 2026). Decode mode (`--mode decode`): single stream and concurrency
Decode per category (tok/s): chat 202.1, code 214.3, file_edit 236.0, json 259.2, math 255.5, prose 183.3, reasoning 190.1, summarization 239.8 (weighted 216.3)
Concurrency ladder 1/2/4/8 x 48 requests: 204.6 / 315.1 / 449.2 / 565.1 tok/s aggregate (48/48 completed at 8).
TTFT p50 short prompts 117 ms; update p99 13.9 ms per stream update (MTP: about 2.9 tokens per update).

## Prefill mode (`--mode prefill-long`): depth sweep (W8A8 prompt rows, default prefill path)
Prefill by depth, exact wire, 2,048-token chunks (tok/s): 2,000: 7,925; 8,000: 8,321; 16,000: 8,311; 32,000: 8,232; 64,000: 8,161; 128,000: 7,928
Prefill at exactly 64,000 prompt tokens (raw completions, cold prefix cache): 8,212 tok/s full (TTFT median 7.793 s); 8,204 tok/s steady with the first 2,048-token chunk removed; exact token count verified through the server's tokenizer.
Exact-arithmetic prefill (`--mode prefill-long-nopf`), development-stack measurement of 10 Oct 2026 (not re-run on the image): 2,000: 7,575; 8,000: 7,785; 16,000: 7,762; 32,000: 7,711; 64,000: 7,607; 128,000: 7,422; exactly 64,000 tokens 7,673 / 7,666.

## Long context and caching (served, 10 Oct 2026 gates)
Needles: 40/40 at 131,072 and at 200,000 tokens. Prefix caching (opt-in `--prefix-caching`, align mode, 2,048-token blocks): 64K repeat 9.6 s -> 0.35 s TTFT, identical output.
Single-stream decode in the 200K mode (prefill-long, speculation off): 105.3 / 100.6 / 99.9 tok/s after 32K / 100K / 190K-token prompts (TTFT 24.8 s at 190K).

## Host conditions
Quiet host (load average median ~5 on 48 threads; no concurrent builds or uploads).

## Method
20 passes per category after 3 warm-ups, temperature 0.7 / top-p 0.95 / top-k 20, unique nonce per request (no prefix-cache hits),
Under MTP the server streams several tokens per update, so "update p99" is per stream update.
