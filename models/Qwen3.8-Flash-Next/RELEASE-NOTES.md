# Qwen3.8-Flash-Next release notes

No custom engine was built: regular vLLM with the Paiton plugin and our own kernels. We keep tuning, we keep optimizing.
Pages: [overview](README.md) · [3-bit](README-3bit.md) · [4-bit](README-4bit.md).

## Next release (local): Paiton for coding agents
- **3-bit prefix caching keeps speculative decoding.** `--mode decode --fn-worker native --prefix-caching`
  (`run-flashnext.sh --fn-worker native --prefix-caching`) now serves `--mode 3bit-long-cache-mtp-native`: 262,144-token window
  (was 98,304), speculative decoding on, RAM prefix tier sized from the host. Same weights. Measured 25 Oct (BetterBench standard;
  decode / TTFT p50 / prefill @64K / 8 requests / 4 requests): 215.2 / 84 ms / 8,352 / 620.1 / 459.8 with the RAM tier (was
  101.0 / 73 ms / 8,731 / 403.4 / 279.7 in `3bit-cache-native`), 252.5 / 81 ms / 8,387 / 637.0 / 512.5 with `--native-kv-tier off`.
  Repeated prompt (cold → hit): 200K 25.84 → 0.68 s, 90K 11.07 → 0.60 s, 32K 3.93 → 0.54 s, 200K from the RAM tier 25.6 → 0.94 s;
  identical tokens on every hit.
- **Host check.** Speculative decoding needs the host tables in GTT pools (about 60 GiB of free GTT: 128 GB+ at the default GTT
  limit, the same as `--mode decode --fn-worker native`). With less, the launcher prints the reason and serves
  `--mode 3bit-cache-native` (98,304 tokens, speculative decoding off) instead of refusing the host: 64-120 GB hosts keep native
  prefix caching (tested under a 64 GiB memory limit: 90K 10.51 → 0.32 s).
- **RAM tier with many sessions taking turns.** The tier holds about (1,089 - 14 x sessions) x 2,048 tokens at 16 GiB per GPU
  (every session also keeps 14 blocks of recurrent state); the launcher and the server print it. When sessions taking turns need
  more than that, plain LRU dropped each session just before its turn (11 sessions, 2.05M tokens: 0 of 11 came back from the tier,
  21-46 s each). The tier now keeps whole sessions and drops the most recently served other one: 9 of 11 resumed from the tier in
  0.87-1.33 s. Sessions that fit behave as before. `PAITON_FN_KVT_POLICY=lru` (`--native-env`) restores plain LRU.
- **Decoding next to long prompts (opt-in, unchanged).** `--native-env PAITON_FN_DECODE_INTERLEAVE=1` on the speculative native
  modes, now including the 3-bit prefix-caching default: a request decoding during a burst of long prompts gets about 2.2x more
  tokens, at about 5 % less aggregate prompt throughput. Off by default (greedy tokens of concurrent requests can then differ from
  run to run); with it off the server output is unchanged.
- **Image 20261026-pc-rc4** (every native mode): the 20261024-pc-rc3 image plus one 49 kB layer with the RAM tier change above;
  executor and mode files unchanged.

## Release 20261024 (new 3-bit weights, clearer docs)
- **New 3-bit weights, the default 3-bit download** (Hugging Face revision `a100b3ae`). Same format, file sizes, kernels and
  speed as before. Against the 16-bit model: GSM8K / HumanEval / MMLU-Pro 97.4 / 94.5 / 83.6 % (16-bit 96.9 / 96.3 / 84.7 %, no
  significant difference); KL 0.0550 / 0.0367 free / routing locked; top-1 91.18 %; long-document NLL +0.35 %; AI2D with images
  (emulated) 81.8 % (16-bit 85.3 %). The default 3-bit weights folder is now keyed by the revision
  (`~/paiton-models/Qwen3.8-Flash-Next-W3A8-a100b3ae`): the launcher downloads the new weights into it, says once that an older
  folder can be deleted, and serves an explicit `--weights DIR` with other weights as is, with a warning.
- **Image 20261024-pc-rc3** (every native mode): `ghcr.io/eliovp/paiton-vllm-plugin:qwen38-flashnext-rocm10-vllm029-20261024-pc-rc3`
  @ `sha256:5bca558833ad230ab1e8db7e6a64f538a7396c44dd660b69716a5aec4e714640`: the pc-rc2 image plus one 6.15 MB layer with the
  executor plans for the new 3-bit weights, so `--drafter stock`, `3bit-cache-native`, `3bit-long-cache-native` and
  `3bit-long-cache-mtp-native --drafter stock` load them (the launcher picks the plan from the checkpoint's `manifest.json`;
  older weights keep theirs). Checked on rc3 with the new weights: all four modes load and repeat greedy output exactly (6/6); `3bit-cache-native`
  prefix-cache gate 16/16 (90K 10.62 → 0.33 s), BetterBench quick 101.0 tok/s, 8 requests 402.5; `--drafter stock` 253.6 / 673.8;
  default 3-bit 264.4 / 682.0 and 4-bit 240.0 / 561.7, as on rc2.
- **Measured** on the release image through this launcher, BetterBench standard and LocalMaxxing:
  [BENCHMARKS.md](BENCHMARKS.md#release-20261024-24-oct-2026-before-and-now).
- **Docs:** the overview starts with a "which command do I run" table, per-mode numbers, an accuracy table and the host RAM per
  mode; gates, formulas and measurement details moved to [BENCHMARKS.md](BENCHMARKS.md).

## Release 20261024-pc-rc2 (native prefix caching)

### Nothing changes for the default commands
- Every torch-worker command serves as in 20261017: the 4-bit torch modes keep `...-20261017-r1`, the 3-bit torch modes
  `...-20261010-r2`.
- Every native mode, including the two existing ones (`--mode decode --fn-worker native`, `--mode 4bit --fn-worker native`), now runs
  the new image: the 20261017-r1 image plus two layers (5.34 MB and 9.79 MB). Their executor and files are byte-identical; re-measured
  on the new image (decode / TTFT p50 / prefill @64K / 8 requests / 4 requests): 4-bit 239.7 / 65 ms / 7,802 / 562.4 / 467.4
  (20261017: 239.8-240.4 / 65-66 / 7,753-7,767 / 557.5-579.8 / 453.8-458.9); 3-bit 272.6 / 55 ms / 8,439 / 679.3 / 532.5 (20261017: 271.2-272.8 / 56-65 / 8,389-8,412 / 676.9-691.5 / 529.1-536.5); greedy repeat-identical 6/6.

### New: prefix caching on the native worker
- `--prefix-caching` with `--fn-worker native` switches to the mode's prefix-caching variant and prints the switch:
  `--mode 4bit` / `4bit-long-native` → `--mode 4bit-long-cache-mtp-native` (262,144-token window, speculative decoding on, MTP
  depth 3, KV cache 2.5 GiB per GPU ≈ 311K tokens); `--mode decode` → `--mode 3bit-cache-native` (98,304 tokens, speculative
  decoding off). `--mode 3bit-long-cache-native` (262,144 tokens) and `--mode 4bit-long-cache-native` (262,144 tokens, speculative
  decoding off) are named directly. All: 2,048-token blocks (a hit resumes on a block boundary, the same tokens as a cold start; with
  speculative decoding one block earlier). New named mode `--mode 3bit-long-cache-mtp-native`: 3-bit, 262,144 tokens, speculative
  decoding on, KV cache 4.5 GiB per GPU ≈ 561K tokens (same host requirement as `--mode decode --fn-worker native`; `--prefix-caching`
  on `--mode decode` keeps `3bit-cache-native`, which also runs on small hosts).
- **RAM prefix tier, sized from the host.** Blocks evicted from VRAM are kept in host memory and loaded back on a hit.
  `--native-kv-tier auto` (the default) sizes it at start: min(16, floor((available RAM - the mode's resident RAM - start margin -
  8 GiB) / 2)) GiB per GPU, off below 2 GiB (prefix caching then stays on, VRAM hits only); the start line prints the formula.
  `--native-kv-tier GIB|off` overrides it. On `4bit-long-cache-mtp-native` the tier needs two executor settings that cost
  single-request decode (no early token return, no MALL prefetch): 207.3 tok/s with the tier, 232.6 with `--native-kv-tier off`; the
  launcher switches them with the tier and prints the trade.
- **`--memory GIB`:** caps the server container's RAM (docker `--memory`, no swap); RAM checks, GTT limit and tier are sized as on a
  host with GIB of RAM.
- **Small hosts:** a 4-bit cache mode on a host whose GTT limit cannot hold the pinned tables is refused before start, with the
  cache modes that fit (`this host fits --mode 3bit-cache-native or --mode 3bit-long-cache-native`).
- **Time to first token on a repeated prompt** (cold → hit, one request at a time): 4-bit `4bit-long-cache-mtp-native` 200K
  document 33.62 → 0.84 s, 90K 14.72 → 0.78 s, 32K 5.26 → 0.69 s, 200K from the RAM tier 30.39 → 0.96 s; `4bit-long-cache-native`
  200K document 38.14 → 0.55 s, 90K 16.82 → 0.46 s, 200K from the RAM tier 32.72 → 0.63 s; 3-bit 90K 10.53 → 0.32 s, 32K 3.70 → 0.27 s, 90K from the RAM tier 10.48 → 0.42 s;
  3-bit long 200K document 24.53 → 0.42 s; `3bit-long-cache-mtp-native` 200K document 25.81 → 0.68 s, 90K 11.09 → 0.60 s, 200K from
  the RAM tier 25.61 → 0.92 s.
- **Idle sessions resumed from the RAM tier** (`4bit-long-cache-native`, eight sessions, 1.62M tokens, one at a time): 0.57-0.62 s at 145K, 0.67-0.70 at
  161K, 0.74-0.78 at 178K, 0.75-0.80 at 194K, 0.96-0.98 at 211K, 0.98-0.99 at 227K, 0.99 at 244K, 1.07 s at 260K (cold 23-51 s);
  identical tokens 8/8. A decoding stream next to a restore: inter-token gap p50 10.5 → 12.5 ms, at most one gap of about 0.46 s.
- **Speed without prefix hits** (BetterBench 0.6.0 standard profile; decode / TTFT p50 / prefill @64K / 8 requests / 4 requests;
  4-bit `4bit-long-cache-mtp-native` with the RAM tier 207.3 / 107 ms / 7,223 / 482.8 / 395.5, with `--native-kv-tier off` 232.6 /
  104 ms / 7,270 / 487.5 / 397.6; speculative decoding off: 4-bit `4bit-long-cache-native` 94.2 / 110 ms / 6,918 / 313.0 / 233.8; 3-bit `3bit-cache-native` 101.0 / 73 ms / 8,731 / 403.4 / 279.7; 3-bit `3bit-long-cache-native`
  100.5 / 73 ms / 8,725 / 401.5 / 279.0; 3-bit `3bit-long-cache-mtp-native` with the RAM tier 213.5 / 85 ms / 8,346 / 611.0 / 460.4, with `--native-kv-tier off`
  250.0 / 81 ms / 8,404 / 657.7 / 505.1. The speculative native modes without prefix caching stay the fastest for new prompts (4-bit 240.3, 3-bit 272.5 tok/s single-request decode).
- **Decoding next to long prompts (opt-in):** `--native-env PAITON_FN_DECODE_INTERLEAVE=1` on the speculative native modes adds a
  decode-only step after each step that mixed a 2,048-token prompt chunk with decoding requests. A request decoding during a burst of
  eight 17K-29K-token prompts: about 9.5 → 21 tok/s, aggregate prompt throughput -5 %. Off by default (greedy tokens of concurrent
  requests can then differ from run to run).
- **Hosts tested** (docker memory limits): 4-bit 252 GiB (`4bit-long-cache-mtp-native` tier on and off, `4bit-long-cache-native`)
  and 200 GiB (`4bit-long-cache-native`) (tier 16 GiB per GPU, all gates pass; 192 / 128 / 64 GiB
  refused with the hint); 3-bit 252 GiB and 128 GiB (tier 16 GiB per GPU), 64 GiB (tier off, VRAM hits). 0 out-of-memory kills, 0
  errors in the 5- and 15-minute soaks.

### Known limits
- Resuming an idle session takes longer the deeper it is (above): the last 2.0-2.45K tokens are read again over the whole context.
- The 3-bit cache modes run with speculative decoding off. With speculative decoding (`4bit-long-cache-mtp-native`) a hit resumes one
  2,048-token block before the end of the cached prefix, so it reads up to about 4K tokens again (200K re-hit 0.84-0.96 s instead of
  0.55-0.63 s with speculative decoding off).
- Greedy output of a hit equals a cold start when the request is served alone. Next to other running requests greedy tokens can
  differ, with or without a cache hit (batch composition, as in the published native modes); quality is not affected.
- 16 GiB per GPU of RAM tier holds about (1,089 - 14 x sessions) x 2,048 tokens of idle sessions (1.9M for 11 sessions); sessions
  beyond it are read again from the start. When sessions taking turns overflow it, the tier keeps whole sessions (9 of 11 resumed
  from the tier at 2.05M tokens, instead of 0 of 11 with plain LRU).
- 4-bit cache modes: host tables pinned in GTT only (no `--ple-host pageable|filebacked`, `--expert-cold`); no `--vision` on any
  native mode; the native request-API limits apply ([overview](README.md#request-api-limits)). A 64 GiB host is tight for the 3-bit
  cache modes (about 7 GiB left after the host tables; no RAM tier).

### Images
- Every native mode: `ghcr.io/eliovp/paiton-vllm-plugin:qwen38-flashnext-rocm10-vllm029-20261024-pc-rc2` @ `sha256:c5c3b398807fb12ccd34fc2fc12dacc11dc1aae5ad34868aa8f70d4b6ee9f2d0`.
- 4-bit torch modes: `...-20261017-r1` @ `sha256:b801bafd...` (unchanged); 3-bit torch modes: `...-20261010-r2` @ `sha256:ea84a147...`
  (unchanged).

## Release 20261017 (r6)

### Nothing changes for the default commands
- Every mode (`--mode decode`, `prefill-long`, `decode-nopf`, `prefill-long-nopf`, `4bit`, `4bit-cache`, `4bit-long`, `4bit-200k`,
  `4bit-xl`) runs the torch worker exactly as in r3; a published command serves as before. `--dry-run` prints which worker a command
  runs and how to get the other.

### New: the native Paiton worker (opt-in)
- `--fn-worker native` on `--mode decode` (3-bit) and on `--mode 4bit` (4-bit): the same configuration (98,304-token window, MTP
  depth 3) with the model running in the native Paiton worker; vLLM keeps the API server, scheduler and KV cache manager. The other
  modes have no native release path. Details: [overview](README.md#native-worker-opt-in).
- **Speed** (BetterBench 0.6.0 standard profile; decode / TTFT p50 / prefill @64K in the mode / 8 requests / 4 requests):
  - 3-bit `--mode decode --fn-worker native`, distilled drafter: 272.5 tok/s / 56 ms / 8,392 /
    685.4 / 531.0.
  - 4-bit `--mode 4bit --fn-worker native --kv-profile speed`, distilled drafter: 240.4 / 65 ms / 7,759 /
    566.7 / 456.0.
  - 4-bit `--mode 4bit --fn-worker native` (`--kv-profile capacity`, the default), distilled drafter: 240.3 /
    65 ms / 7,757 / 561.9 / 456.9.
  - With `--drafter stock` (the checkpoint's own drafter): 3-bit 249.9 / 56 ms / 8,401 / 659.4 /
    516.4; 4-bit 237.6 / 65 ms / 7,748 / 545.4 / 443.4 (4-bit at the default `--kv-profile capacity`).
  - For comparison, the default (torch worker, checkpoint drafter, r3): 3-bit 216.3 / 117 ms / 8,161 (`prefill-long`) / 565.1 /
    449.2; 4-bit 189.8 / 121 ms / 6,991 (`4bit-200k`) / 420.7.
- **Distilled drafter in the image.** The native modes use the distilled MTP drafter B4gh5-mtpft by default; one copy in the image
  serves 3-bit and 4-bit. The launcher assembles `<checkpoint dir>-b4gh5-mtpft` next to the downloaded checkpoint (hard links to the
  checkpoint's files, links to the drafter files in the image; the downloaded checkpoint is never modified), and `--dry-run` checks
  the image's drafter files against their pinned sha256. `--drafter stock` uses the checkpoint's own drafter.
- **KV profiles (4-bit native).** `--kv-profile capacity` (the default): 2.5 GiB per GPU, about 305,338 tokens (3.11 concurrent 98K
  contexts, as r3 `--mode 4bit`). `--kv-profile speed`: 1.5 GiB per GPU, about 183,202 tokens (1.86 contexts); the freed VRAM holds
  more experts, faster. 3-bit native: about 551K KV tokens (torch default: about 112K).

### 4-bit image update (torch modes)
- Every 4-bit mode now runs the new 4-bit image, which also carries the native worker. Torch-worker changes since r3:
  - **64 GB hosts:** the cold-expert tier (`--expert-cold`, picked automatically when even the 96 GB layout does not fit): the
    coldest share of the experts is read from one file per GPU. Same output, slower (`--mode 4bit` about 85 / 181 tok/s single /
    8 requests under a 60 GiB memory limit). See [System RAM](README-4bit.md#system-ram).
  - **Image input:** `--vision` (opt-in) on `--mode 4bit`, `4bit-cache` and `4bit-long`; images up to 4K, 8 per request, no video;
    the vision tower takes its VRAM from the KV cache. See [Run it](README-4bit.md#run-it).
  - **`--mode 4bit-200k`:** fixed a stall on prompts whose last 2,048-token chunk was 1 to 64 tokens long.
- The 3-bit torch modes keep the published 3-bit image (`...-20261010-r2` @ `sha256:ea84a147...`), unchanged.

### Known limits of the native worker
- 4-bit native: greedy output can vary from run to run while a long prompt (tens of thousands of tokens) is being read next to other requests; a request on its own, or several short ones together, repeat exactly. Errors and quality are not affected.
- Request fields answered with HTTP 400 (the torch default serves them): `logprobs` / `top_logprobs` / `prompt_logprobs`;
  `response_format`, structured outputs, guided JSON; tool calls with `strict: true` tools or `tool_choice` `"required"` / a named
  tool (plain `auto` tool calling works); presence / frequency / repetition penalties; `top_k` outside 1..32 at temperature > 0;
  `min_tokens`, `allowed_token_ids`, `bad_words`; `thinking_token_budget`. `min_p` and `logit_bias` are refused by vLLM itself with
  speculative decoding, on both workers. Full list: [overview](README.md#request-api-limits).
- No prefix caching: `--prefix-caching` keeps the torch worker; for agents and coding tools use `--prefix-caching` (3-bit) or
  `--mode 4bit-cache` (4-bit) on the torch worker. No image input: `--vision` stays on the torch worker (4-bit).
- Host requirements, with no automatic fallback when the native worker is asked for (the launcher prints the reason): 4-bit native
  needs a GTT limit of at least about 98 GiB (about 196 GB of RAM at the default limit; smaller hosts keep the torch default and its
  128 GB / 96 GB / 64 GB layouts; `--ple-host pageable|filebacked`, `--expert-cold` and `--host-pools userptr` are torch-worker
  layouts); 3-bit native needs about 60 GiB of free GTT (about 128 GB of RAM). Two R9700 (32 GiB), as before.

### Images
- 3-bit torch modes: `ghcr.io/eliovp/paiton-vllm-plugin:qwen38-flashnext-rocm10-vllm029-20261010-r2` @
  `sha256:ea84a147c6dd44f47a41836ace6b39864fab940095c4c0ba13e82537dd9e2a4b` (unchanged).
- Every 4-bit mode and both native modes: `ghcr.io/eliovp/paiton-vllm-plugin:qwen38-flashnext-rocm10-vllm029-20261017-r1` @ `sha256:b801bafd3f4cbc04b99153180514845894f98554f49fccdbfe0d6344b47034d9` (the launcher pins it by digest).

## Release 20261014-r3 (4-bit image)

- **4-bit on 128 GB and 96 GB hosts.** The launcher picks the memory layout from the host's RAM and prints it at start: the
  benchmarked pinned layout on 256 GB hosts, pageable embedding tables on 128 GB hosts, and on 96 GB hosts embedding tables read
  through the page cache from two raw files laid out once next to the weights (49 GiB of disk; checked before the start).
  `--mode 4bit`, `4bit-cache` and `4bit-long`; same output in every layout. See [System RAM](README-4bit.md#system-ram).
- **Decode in `--mode 4bit-200k`** (the mode without speculative decoding): batched steps no longer process padding rows; same output
  in our checks.
- New 4-bit image `...-20261011-r3` (the r2 image plus the two changes above); the 3-bit image stays r2.

## Launcher update (4-bit on smaller hosts)

- **Pageable embedding tables chosen automatically.** `--ple-host` now defaults to `auto`: on `--mode 4bit` and `--mode 4bit-cache`,
  when the GPU's GTT limit cannot hold the pinned tables (hosts with less than about 200 GB of RAM at the default limit), the
  launcher keeps the per-layer embedding tables in pageable memory and prints one line saying so. Hosts with enough GTT keep the
  pinned layout; `--ple-host pinned` / `pageable` choose explicitly. Before, these hosts fell back to ROCm's user-pointer memory,
  which was slower single-stream (130.2 vs 151.6 tok/s with RAM to spare).
- **`--host-pool off` is refused in the 4-bit modes** with a clear message; their expert tier needs the host copy, and the flag
  stopped the engine at start.

## Release 20261014-r2 (hotfix for both images)

### Fixes
- **Stability and correctness fixes** for prefix caching with several requests at once and for long prompts, in every mode
  (r2 images). Same speed as r1.
- **Thinking and tool calls.** Every mode now passes vLLM's reasoning and tool-call parsers: the reasoning comes back in
  `message.reasoning`, the answer in `content`, tool calls in `tool_calls` (works with the r1 images too).
- **3-bit memory caps tuned.** With `--prefix-caching`, `--mode decode` and `--mode decode-nopf` cap the memory pool at 94 % (was 95 %): in our
  test with 8 concurrent requests plus long prompts the cards stayed at or below 31.4 GiB (was up to 31.7 GiB), at 170.2 / 493.0 tok/s
  with 1 / 8 requests. The prefill-long modes, which need the room for 131K-token prompts, and every run without prefix caching keep
  95 %. No need to edit `PYTORCH_ALLOC_CONF` by hand any more (works with the r1 images too).
- **12 October: prefix caching.** `--prefix-caching` (3-bit modes) now sets 2,048-token cache blocks. A repeated conversation then
  gives exactly the same output as a fresh one, and peak VRAM was about 1 GiB lower per card in our 65K-token test.
- **13 October: 4-bit VRAM.** `--mode 4bit` and `--mode 4bit-cache` cap the memory pool at 92.5 %, so a stream of long prompts stays
  at about 30.6 GiB per card (was up to 31.3 GiB), at the same speed.

### New
- **128 GB hosts for the 4-bit modes:** `--ple-host pageable` (opt-in, `--mode 4bit` and `--mode 4bit-cache`, r2 4-bit image) keeps
  the per-layer embedding tables in pageable memory: about 47 GiB pinned instead of 96 GiB, so the GTT pools fit the default limit of
  a 128 GB host. Same output; with RAM to spare as fast as the default, within a 128 GB host's memory budget about 14 % slower
  single-stream and as fast at 8 requests.

### Images
- 3-bit: `qwen38-flashnext-rocm10-vllm029-20261010-r2`; 4-bit: `qwen38-flashnext-rocm10-vllm029-20261011-r2` (the launcher pins both
  by digest). The r1 images stay available; the fixes above need r2.

## Release 20261011

### 4-bit (new): the accuracy release
- **Weights** `EliovpAI/Qwen3.8-Flash-Next-W4A8-Paiton-RDNA4`, modes `--mode 4bit` (98,304-token window, MTP depth 3, INT6 trunk) and
  `--mode 4bit-200k` (200,000-token window, MTP off, 8-bit trunk). Per layer the 410 most-used experts are INT4 (g128) and the 102
  least-used use the 3-bit codebook format; about 20.8 GiB of experts per GPU stay in VRAM, the rest are served from pinned system
  memory (expert host tier); fp8 KV cache (about 305,000 tokens in `4bit`).
- **Accuracy** (as served) against the 16-bit model:
  - GSM8K 96.8 vs 96.9 %, HumanEval 96.3 vs 96.3 %, MMLU-Pro 83.7 vs 84.7 %. The MMLU-Pro gap is not significant (p = 0.16) and comes
    from replies at the 16,384-token limit; uncapped: 91.1 vs 91.0 %.
  - Perplexity +0.87 % on wikitext, +0.38 % on unseen text (4096).
  - Long-document NLL within 0.16 % of bf16; needles 40/40 identical to bf16.
- **Speed** on the shipped image: decode 189.8 tok/s (median of 3 runs) / @8 420.7 / prefill @64K 6,991 (an exact 64,000-token
  prompt 7,162) / TTFT 121 ms / update p99 17.7 ms.
- **Cache mode for agents** (opt-in, `--mode 4bit-cache`): prefix caching plus an 8 GiB RAM prefix tier; repeated long prompts start
  in about 1 s instead of 6-16 s (64K 12.3 -> 1.1 s), identical output from the tier. It needs about 108 GiB of free GTT, else the tier
  stays off (with a warning) and prefix caching runs in VRAM only. The cost: about 22 % lower multi-user throughput than `--mode 4bit`
  (8 concurrent requests 321.4 tok/s; single-stream 146.7) and about 0.1-0.2 s slower first token on new prompts. `--prefix-caching`
  on `--mode 4bit` / `4bit-200k` is refused (not measured there).
- **Host memory**: about 96 GiB pinned (55 GiB host tables + 41 GiB experts) in GTT-backed host pools when the free GTT holds them
  (about 98 GiB: a 256 GB host), with an automatic fallback to ROCm's user-pointer memory and a warning.
- **Image**: `qwen38-flashnext-rocm10-vllm029-20261011-r1` (the launcher picks it for the 4-bit modes).

### 3-bit (launcher update)
- Same image (`qwen38-flashnext-rocm10-vllm029-20261010-r1`), weights, modes, speed and accuracy as the 10 Oct release.
- Launcher: a short settle wait before it reports READY, the peer-to-peer and GTT preflight checks, and the opt-in `--host-pools gtt`
  (removes the occasional memory pauses under load, costs ~1 GiB VRAM in decode); the published user-pointer pools stay the default.
- **Agents and coding tools: add `--prefix-caching`.** An agent resends the whole conversation every turn; with the prefix cache the
  shared part is reused instead of prefilled again: a repeated 64K-token prefix reaches its first token in 0.35 s instead of 9.6 s,
  with identical output. The speed tables are measured without it (the benchmark never repeats a prefix).
- **Known issue:** in very long agentic sessions (beyond ~30K tokens) the 3-bit can end replies early. Use `--prefix-caching`, and
  prefer the 4-bit for long coding sessions; a fix is planned for the next 3-bit release.

### Both
- The server listens on all interfaces (0.0.0.0) by default; `--host 127.0.0.1` keeps it local. API key: `PAITON_API_KEY=<key>` in
  the environment (or `--api-key`, visible in `ps`); it is handed to Docker by variable name, never in the container's command line.
- The launcher warns when the KV cache is small relative to the window with prefix caching on, and its READY line explains prompt vs
  generation throughput in the server log (README Troubleshooting).
- `--dry-run` reports host RAM, the GTT limit and what it chooses, and warns when the GPUs have no peer-to-peer link.

## Next release
- SSD KV-cache tier for the 4-bit cache mode.
- Native worker: the request-API limits above, prefix caching and image input.

## Release 20261010-r1 (previous: 3-bit)

### Stack (image rc8)
- Image: `ghcr.io/eliovp/paiton-vllm-plugin:qwen38-flashnext-rocm10-vllm029-20261010-r1` @ `sha256:c7e76bc0d7d9f8a3d575b940d23f9b5219eada13b1960e10f3aaa996b5ad9191`.
- Plugin commit 8582563; runtime bundles tp-collective-m27h-981f701e, moe-grouped-71dad0d2, fn-mixers-9f92291e (scorer layout 3);
  W8A16 drafter view (MTP depth 3). Prompt rows run W8A8; accuracy numbers: the model card's table.
- Official numbers, measured on the shipped image through the launcher: decode 216.3 weighted / 565.1 @8 (48/48; ladder 204.6 / 315.1 / 449.2 / 565.1),
  TTFT 117 ms, update p99 13.9 ms; prefill @64K 8,161 (exact 64,000 tokens 8,212 / 8,204; depth rows 7,925 / 8,321 / 8,311 / 8,232 / 8,161 / 7,928).

### Two prefill paths (both in the image; the launcher mode selects)
- `--mode decode` / `--mode prefill-long` (default): native GDN prefill path; prefill @64K 8,161 tok/s on the shipped image (depth rows 7,925 / 8,321 / 8,311 / 8,232 / 8,161 / 7,928).
- `--mode decode-nopf` / `--mode prefill-long-nopf`: exact-arithmetic prefill, about 7 % slower (the two native-prefill flags off). Our long-document control
  found the native path is not bit-reproducible run to run while the exact path reproduces itself; prefill
  @64K on the exact path: 7,607 tok/s (development-stack measurement, not re-run on the image; exact 64,000 tokens 7,673; depth rows 7,575 / 7,785 /
  7,762 / 7,711 / 7,607 / 7,422), about 7 % below the default path. Decode rows are identical in both.
- `--prefix-caching` (opt-in, the 3-bit modes; the 4-bit modes use `--mode 4bit-cache` and refuse the flag): align-mode prefix caching of the recurrent state per 2,048-token block; 64K repeat 9.6 s -> 0.35 s TTFT, identical output.

### Long context
- `decode`: 98,304-token window with speculative decoding (MTP depth 3); `prefill-long`: 200,000-token window, speculative decoding off.
- Tested end to end at 200K: needles 40/40 at 200,000 tokens (identical to bf16); prefix caching opt-in with byte-identical hits (128K repeat
  20 s -> 0.41 s TTFT); prefill measured to 128K; single-stream decode in the 200K mode (prefill-long, speculation off):
  105.3 / 100.6 / 99.9 tok/s after 32K / 100K / 190K-token prompts (TTFT 24.8 s at 190K); 200K with speculation does not fit the stock pool.

### Host memory (3-bit)
- The host-resident tables (about 55 GiB over both GPUs) are allocated as driver-owned GTT memory (`HSA_USERPTR_FOR_PAGED_MEM=0`, set by the
  launcher) when the GPUs' GTT size holds them; the default GTT size is half the system RAM, so this needs about 120 GiB of RAM or more.
  ROCm's default user-pointer memory can be moved when the kernel compacts host memory, which paused the GPU queues briefly under load
  (2.7 s in total over a 3-minute run at 8 concurrent requests; 0.07 s with GTT memory, speed unchanged). On smaller hosts the launcher keeps
  the default and says so; `--host-pools userptr` keeps it everywhere.

## Reference (both releases)

### Host memory (4-bit)
- `--mode 4bit` / `--mode 4bit-200k` keep part of the experts in pinned system memory. Together with the host tables this pins about
  96 GiB (about 55 GiB of tables plus about 41 GiB of experts, both GPUs together; measured 95.5 GiB in the decode mode, 93.0 GiB in the
  200K mode). The launcher checks the available RAM first (pinned memory plus 16 GiB for the system).
- The benchmarked configuration places all of it in GTT memory (`HSA_USERPTR_FOR_PAGED_MEM=0`), which is charged to one GPU's GTT. The
  usable GTT is limited per GPU and by the kernel's TTM page limit, both half the RAM by default, so **the benchmarked 4-bit
  configuration needs about 98 GiB of free GTT: in practice a host with 256 GB of RAM** (a 200 GB host has about 93 GiB).
- On hosts with less GTT the launcher keeps ROCm's default user-pointer memory and prints a warning: the model starts and runs, but
  slower (186.0 vs 191.9 tok/s single-stream, 388.9 vs 423.1 tok/s at 8 requests in our validation run) and the GPUs can pause for
  several seconds when the kernel compacts host memory. Raising the GTT limit (`amdgpu.gttsize`, `ttm.pages_limit`) is a system setting
  the host's administrator may choose; the launcher never changes it.
- `--dry-run` prints the host RAM, the GTT limit and what it would choose, and warns when the two GPUs have no peer-to-peer link.

### Checkpoint repo (3-bit)
- The 3-bit modes load the W8 drafter file `mtp/mtp-draft-e4m3.safetensors` shipped in EliovpAI/Qwen3.8-Flash-Next-W3A8-Paiton-RDNA4 (verified by
  SHA256SUMS together with the other files on the first start).
- The 3-bit modes use the bf16 KV cache (fp8 KV measured there, not adopted); the 4-bit modes use an fp8 (E4M3) KV cache.

### Launcher behaviour
- The checkpoint repos (EliovpAI/Qwen3.8-Flash-Next-W3A8-Paiton-RDNA4 and -W4A8-Paiton-RDNA4) are Paiton container exports: a
  descriptor `config.json`, `manifest.json`, `layers/`, `common/`, `mtp/`, `SHA256SUMS` (the 4-bit repo adds `trunk-int6/` and
  `expert-format-map.json`). They carry no base-model config or tokenizer.
- `launch-flashnext.py` builds the vLLM runtime view itself, once, next to the weights (`<weights>-view`): the base model's
  config + tokenizer files at the pinned revision `Qwen/Qwen3.8-Flash-Next@de4b8e4d43b917e7706784d8bb445c9af86a3540` (downloaded
  with `hf download`, non-safetensors files only), `language_model_only` + the Paiton `quantization_config` block, and links to
  the checkpoint files as the container sees them (`/models/ck/...`). The view is mounted read-only at `/models/view`; the server
  loads `--model/--tokenizer /models/view`.
- This reproduces, file for file, the view the official numbers were measured with (the benchmark harness built it with the
  same tool chain); the only omitted entry is an unused alternative drafter export (the decode mode uses the int2 draft head in
  `mtp/`). Bit-neutral: same files, same engine arguments, same environment as the measured modes (`modes.json`).
- Re-running the launcher reuses an existing complete view; delete `<weights>-view` to rebuild it.

### Runtime bundles
- The three Flash Next runtime bundles ship with symbol tables stripped (as every bundle of the base image); their manifests and
  SHA256SUMS are pinned to the shipped files. Stripping removes `.symtab/.strtab` only: `.text`, `.rodata` and `.dynsym` are
  byte-identical to the benchmarked libraries.
### Prerequisites
- Docker with GPU device access, python3 >= 3.10, `pip install huggingface_hub` (downloads go through the Python API; the `hf` command is only a
  fallback), two AMD GPUs visible, disk for the weights (about 116 GB for the 3-bit, 133 GB for the 4-bit). `launch-flashnext.py --dry-run` prints what is missing.

### How the numbers were taken
BetterBench 0.6.0 standard profile on this machine against our OpenAI-compatible server; prefill with exact bf16 activations (no compressed
wire).
