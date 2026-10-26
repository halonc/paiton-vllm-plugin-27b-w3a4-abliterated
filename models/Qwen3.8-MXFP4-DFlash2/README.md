# Qwen3.8 27B on one R9700

Qwen3.8 27B with DFlash2 speculative decoding in vLLM, on one Radeon AI PRO R9700 (32 GB). Two weight choices:

- **MXFP4**: the most accurate, from the base download alone.
- **3-bit W3A4**: the fastest, with the most context (up to 262,144 tokens per request, 524,288 experimental) and images
  in every long mode except the 524K one. An extra 9.55 GB download. MMLU-Pro 60.6 against MXFP4's 62.6 with the first 3-bit weights; the
  updates since recover about 0.5 to 1 point.

[What you get](#what-you-get) · [Set up](#set-up) · [Ways to run](#ways-to-run) · [Options](#advanced-options) ·
[Benchmarks](#current-benchmark-results) · [Earlier releases](#run-an-earlier-release)

**24 October: 3-bit accuracy update.** New 3-bit weights (revision `383fa4b221a992fc2437500328056ef05355c8eb`), closer to
the 16-bit model. Same format, image, launcher and speed.

| Against bf16 | Change |
| --- | --- |
| KL divergence, served on the R9700 | −3.0 % [−6.4, −0.4] |
| Long documents, 8K-32K tokens (emulated) | closer in every length bucket |
| 131K-token documents, served | none worse, closer from 64K on |
| GSM8K, HumanEval, MMLU-Pro | unchanged within noise |
| Speed (BetterBench) | identical: 169.1 vs 169.3 tok/s (`--mode long-kv4`), 155.5 vs 155.9 (default) |

Already downloaded? Run the [download](#optional-the-3-bit-weights) again, then `sha256sum -c SHA256SUMS`. Earlier revisions
stay downloadable: `3ef7eb149ec0bd9a0f7927491bf30f53f54ced01` (12 October) and `7a2e3d702144f744c1c5a46eaa42c7a2309d5cf7`
(first release). The 3-bit task tables below were measured with the first weights (`7a2e3d70`). Details: the
[model card](https://huggingface.co/EliovpAI/Qwen3.8-27B-W3Rot-INT3-Paiton-RDNA4).

<a id="what-you-get"></a><a id="pick-how-to-run-it"></a><a id="want-longer-context-run-this"></a><a id="long-context-200k-and-220k"></a>

## What you get

Each row is one command (all of them start with `bash models/Qwen3.8-MXFP4-DFlash2/`); run one at a time.
**Per request** is the longest prompt plus output one request may have. **Cache** is the KV cache that all running
requests share: it sets how many long requests fit together and, with **prefix caching**, how much already-read
text stays ready, so a follow-up question about it starts in seconds instead of minutes.

| You run | Per request | Cache (text) | Cache with `--vision` | Full-length requests that fit together | Prefix caching |
| --- | ---: | ---: | ---: | --- | --- |
| `run-3bit.sh --mode long-kv4`: **coding agents, long documents, screenshots** | 262,144 | **628,877** | **555,128** | 2 (plus one of 100K) | yes |
| `run-3bit.sh --mode long-kv4 --extend-cache` | 262,144 | 628,877 on the GPU, plus about 0.7M to 2.8M in system RAM or on an NVMe/SSD | – | 2 | yes, also after a restart |
| `run-3bit.sh --mode long` | 262,144 (245,000 with images) | 316,325 | 253,298 | 1, plus short ones | yes |
| `run-3bit.sh --mode long-512k` (experimental) | 524,288 | 594,290 | – | 1, plus short ones | yes |
| `run-3bit.sh`: fast everyday chat | 65,536 | 393,216 | 278,050 | 6 | no |
| `run-mxfp4.sh --mode long`: most accurate, one long document | 200,000 | one request | – | 1 | yes |
| `run-mxfp4.sh`: most accurate | 65,536 | 174,634 | 120,277 | 2 | no |

Every row except MXFP4 `--mode long` takes up to 8 requests at once; short requests share the cache with the long
ones. A request longer than its row allows is refused (HTTP 400).

What each row gives up:

| You run | You give up |
| --- | --- |
| `run-3bit.sh --mode long-kv4` | 2.4 GiB of pinned system RAM ([system RAM](#system-ram)); reading a new prompt is about 3 % slower than without prefix caching (7 to 13 % for short prompts, a fraction of a second) |
| `… --extend-cache` | system RAM, or up to 64 GiB of NVMe/SSD space; no images. On hosts with 16 or 32 GB of RAM the GPU part holds 451,879 tokens ([what each host gets](#extend-cache)) |
| `run-3bit.sh --mode long` | about 3 to 4 % speed on short requests against `run-3bit.sh` ([measured](REFERENCE.md#mode-long-short-requests)); half the cache of `--mode long-kv4` |
| `run-3bit.sh --mode long-512k` | the model's long-context scaling on every request, short ones included; 2.4 GiB of pinned system RAM with no fallback; a cold 500K-token read takes about 6 minutes |
| `run-3bit.sh` | a little accuracy: MMLU-Pro 60.6 instead of 62.6 (previous weights `7a2e3d70`; the 12 October update recovers about 0.5 to 1 point) |
| `run-mxfp4.sh --mode long` | one request at a time; no images |
| `run-mxfp4.sh` | speed and room against 3-bit: 156 instead of 180 tok/s, 174,634 instead of 393,216 cache tokens |

**Speed and accuracy at a glance** (one R9700, thinking off; all numbers in [Benchmarks](#current-benchmark-results)):

| | `run-mxfp4.sh` | `run-3bit.sh` | `run-3bit.sh --mode long-kv4` |
| --- | ---: | ---: | ---: |
| Decode, one request, short prompts | 156 tok/s | 180 tok/s | 173 tok/s |
| Eight requests at once, total output | 428 tok/s | 489 tok/s | 479 tok/s |
| Decode with 120K / 228K tokens of context, one request | – | – | 120 / 100 tok/s |
| New question about a 258K-token document already read | – | – | 2.7 s to the first token (134 s for the first read) |
| GSM8K / HumanEval / MMLU-Pro (3-bit columns: previous weights `7a2e3d70`) | 95.68 / 95.12 / 62.57 | 95.22 / 94.51 / 60.57 | 95.45 / 92.68 / 59.93 |

The coding mode's short-prompt speed, accuracy and cached-document rows were measured before its FP8 output head
became the default (6 October); the FP8 head measured the same text prediction and the same or faster decode. The
long-context decode row is an upper range (a prompt the drafter predicts well); long real text decodes at about
72 to 76 tok/s at 258K.

**Long coding sessions** (6 October, previous weights `7a2e3d70`): in 20-turn sessions that grow from 100K to 181K tokens, with a new coding task
every turn, `--mode long-kv4` solved **87.5 %** of the tasks with **no decline from the first turns to the last**,
started cached turns in 3.1 to 4.1 s and decoded code at about 210 tok/s (MXFP4 `--mode long`: 83.75 %, 3.3 to 4.4 s,
about 165 tok/s). [The full table](#long-coding-sessions).

<a id="set-up"></a><a id="before-you-start"></a>

## Set up

You need:

- Linux x86-64, Python 3, the [Hugging Face CLI (`hf`)](https://huggingface.co/docs/huggingface_hub/guides/cli) or only
  its Python package (`huggingface_hub`, see [without `hf`](#download-without-hf)), and Docker usable without `sudo`.
- One R9700 with 32 GB VRAM that does not drive your desktop (for a shared card, see `--profile desktop` in
  [Options](#advanced-options)), and AMD device access (`/dev/kfd` and `/dev/dri`).
- About **75 GB free disk**, including the container, target, drafter and optional 3-bit weights; the optional
  [`--extend-cache`](#extend-cache) SSD folder takes up to 64 GiB more.
- **16 GB of system RAM** is enough for every row (tested). Two rows pin 2.4 GiB of it; see
  [system RAM](#system-ram).

Clone the repository, then run every command below from its root, in the same terminal:

```bash
git clone --depth 1 https://github.com/Eliovp-BV/paiton-vllm-plugin.git
cd paiton-vllm-plugin
```

Already cloned it? Run `git pull` in your checkout. The launcher pins the **5 October r1** image,
`ghcr.io/eliovp/paiton-vllm-plugin:qwen38-rocm10-vllm029-20261005-r1@sha256:245c71d54f046f89b74dbfbd4c894003754561e55c9d215a8d3ce65364fb0f80`, and Docker pulls it on the first
start ([what changed](REFERENCE.md#release-notes-5-october-2026); the launcher's own updates since:
[6 October](REFERENCE.md#launcher-update-6-october-2026-evening),
[11 October](REFERENCE.md#launcher-update-11-october-2026),
[26 October](REFERENCE.md#launcher-update-26-october-2026)). Earlier releases stay runnable:
[run an earlier release](REFERENCE.md#run-an-earlier-release).

<a id="model-weights-and-existing-downloads"></a>

### Download the weights

Every row needs these two downloads, the base download: the target, an NVFP4 checkpoint that the server converts to
MXFP4 when it starts, and the DFlash2 drafter. The launcher finds them, and its runtime cache (files the server
builds on its first start and reuses later), through these variables: set them again in every new terminal; the
downloads and the runtime cache are reused.

```bash
export PAITON_TARGET_DIR="$PWD/model-cache/qwen38-nvfp4"
export PAITON_DRAFT_DIR="$PWD/model-cache/qwen38-dflash2"
export PAITON_CACHE_DIR="$PWD/runtime-cache/qwen38-rocm10-20261005"
mkdir -p "$PAITON_TARGET_DIR" "$PAITON_DRAFT_DIR" "$PAITON_CACHE_DIR"

hf download unsloth/Qwen3.8-27B-NVFP4 \
  --revision f0b7c9e722f5565102fff8481c99e4d86ae099c7 \
  --local-dir "$PAITON_TARGET_DIR"
hf download tcclaviger/Qwen3.8-27B-DFlash2-FP8 \
  --revision ee0cb26a8279b7910cc28d82a8a3e15e4728d56f \
  --local-dir "$PAITON_DRAFT_DIR"
```

<a id="download-without-hf"></a>

**Without `hf` on your PATH** (`pip install --user huggingface_hub` puts the `hf` command in `~/.local/bin`, which
some shells do not search), the Python package downloads the same pinned revisions into the same folders:

```bash
python3 -c 'import os; from huggingface_hub import snapshot_download as get
get("unsloth/Qwen3.8-27B-NVFP4", revision="f0b7c9e722f5565102fff8481c99e4d86ae099c7", local_dir=os.environ["PAITON_TARGET_DIR"])
get("tcclaviger/Qwen3.8-27B-DFlash2-FP8", revision="ee0cb26a8279b7910cc28d82a8a3e15e4728d56f", local_dir=os.environ["PAITON_DRAFT_DIR"])'
```

Already downloaded these revisions? Set the variables to your complete local
folders instead. Each needs its own `config.json` and weights; keep the target's
tokenizer too. For linked Hugging Face cache snapshots, use the
[cache setup instructions](REFERENCE.md#already-in-the-hugging-face-cache).

<a id="optional-3-bit-w3a4-weights"></a>

### Optional: the 3-bit weights

For the `run-3bit.sh` rows, also download these **9.55 GB**. They replace part of the target's weights, so the
target and drafter above stay required. `run-3bit.sh` finds them through `PAITON_W3ROT_DIR`; set it again in a new
terminal too.

```bash
export PAITON_W3ROT_DIR="$PWD/model-cache/qwen38-w3rot-int3"
mkdir -p "$PAITON_W3ROT_DIR"
hf download EliovpAI/Qwen3.8-27B-W3Rot-INT3-Paiton-RDNA4 \
  --revision 383fa4b221a992fc2437500328056ef05355c8eb \
  --local-dir "$PAITON_W3ROT_DIR"
(cd "$PAITON_W3ROT_DIR" && sha256sum -c SHA256SUMS)
```

Without `hf`, replace the `hf download` command with:

```bash
python3 -c 'import os; from huggingface_hub import snapshot_download as get
get("EliovpAI/Qwen3.8-27B-W3Rot-INT3-Paiton-RDNA4", revision="383fa4b221a992fc2437500328056ef05355c8eb", local_dir=os.environ["PAITON_W3ROT_DIR"])'
```

<a id="ways-to-run"></a><a id="start-the-server"></a>

## Ways to run

### Start the server

Pick a row in [What you get](#what-you-get) and run its command with the paths above set, for example:

```bash
bash models/Qwen3.8-MXFP4-DFlash2/run-3bit.sh --mode long-kv4
```

The server runs in the foreground; add `--detach` to run it in the background (`docker logs -f paiton-qwen38`
follows its log). The first start takes about **three to four minutes** (the image ships a seeded compile cache; later starts about
two minutes); wait for `/health` to succeed.

**API:** `http://127.0.0.1:18982/v1` (or `http://<this-machine-ip>:18982/v1` from another system) · **Model:** `Qwen3.8` · **API key:** none unless you set `--api-key` (enter any value if a client asks)

The server listens on all interfaces (0.0.0.0); use `--host 127.0.0.1` to keep it local; add `--api-key KEY` or a firewall if the machine is reachable from untrusted networks.

From another terminal:

```bash
curl --fail http://127.0.0.1:18982/health
curl --fail http://127.0.0.1:18982/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.8","messages":[{"role":"user","content":"Write a Python function to remove duplicates from a list."}],"max_tokens":256,"stream":true,"chat_template_kwargs":{"enable_thinking":false}}'
```

- **Thinking** is off by default with the 3-bit weights (`run-3bit.sh`, every mode) and in every `--mode long*` row;
  it stays on by default in the MXFP4 65,536-token rows. Each request can set it
  with `"chat_template_kwargs":{"enable_thinking":true}` (or `false`); `--thinking on|off` changes the server
  default. All benchmarks use thinking off.
- **Long prompts:** send a prompt of hundreds of thousands of tokens from a file (`-d @request.json`) and give the
  client a timeout of several minutes: reading 500K tokens for the first time takes about 6 minutes.
- **Stop** with `docker stop paiton-qwen38` before switching to another row.

### Coding agents

Start the server with `--mode long-kv4` and point the agent at `http://127.0.0.1:18982/v1`, model `Qwen3.8`, any
API key, context window 262,144 tokens (prompt plus output). An agent sends its whole conversation again on every
turn; the prefix cache reads only what is new. In 20-turn sessions that grew from 100K to 181K tokens, later turns
started in 3.1 to 4.1 s instead of 32 s for the first, cold turn ([measurements](#long-coding-sessions)).

- Cache hits need an unchanged start of the prompt: the same system prompt and earlier turns, no timestamp at the
  top. Each response's `usage.prompt_tokens_details.cached_tokens` shows how many prompt tokens came from the cache.
- **Short questions during a long read** (text only): the server reads a long prompt in 2,048-token steps, so a
  short request from another user is answered meanwhile, in about 0.7 s instead of about 5.9 s during a 64K read; the
  long read takes about 1 % longer. `--long-prefill-threshold off` restores the previous behaviour. With images
  (`--vision`) the long read keeps its previous steps.
- For MXFP4 accuracy with one agent, `run-mxfp4.sh --mode long` also keeps the prefix cache (200,000 tokens, one
  request at a time, no images).
- To keep more documents than the GPU cache holds, also across restarts, add [`--extend-cache`](#extend-cache).

<a id="images-and-vision"></a>

### Images

Add `--vision` to a row whose "Cache with `--vision`" column has a number. Images up to 4K are read at full
resolution; larger ones are downscaled. Send them as OpenAI-style `image_url` content; a 1920 × 1080 screenshot costs
about 2,000 tokens ([how to send one](REFERENCE.md#images-and-vision)).

```bash
bash models/Qwen3.8-MXFP4-DFlash2/run-3bit.sh --mode long-kv4 --vision
```

For screenshots of code prefer `--mode long-kv4 --vision`: the FP8-cache `--mode long --vision` can misread single
characters of tiny (about 15 px) text in 4K screenshots.

<a id="extend-cache"></a>

### More cache for coding agents: `--extend-cache`

`--mode long-kv4` keeps its prefix cache on the GPU. When that is full, the oldest documents are dropped and the
next question about them reads them again. Add `--extend-cache` to keep them in system RAM or on an NVMe/SSD
instead: a document comes back from there in seconds, with the same output
([measured restores](#cache-tier-results)).

```bash
bash models/Qwen3.8-MXFP4-DFlash2/run-3bit.sh --mode long-kv4 --extend-cache
```

| Option | What it does |
| --- | --- |
| `--extend-cache` (the same as `--extend-cache auto`) | Uses system RAM when the host has enough of it for RAM to add capacity beyond the GPU cache; otherwise an NVMe/SSD folder behind a small RAM staging tier. |
| `--extend-cache ram` | Always system RAM, also where it adds little capacity (the start line then says so). |
| `--extend-cache disk` | Always the NVMe/SSD folder: the most capacity (about 2.8M tokens at the 64 GiB cap), somewhat slower restores. |

What `auto` picks (one line at start states the choice and the capacities):

| Host RAM | `auto` picks | What you get |
| --- | --- | --- |
| 16 GB | NVMe/SSD with 4 GiB of RAM staging; the embedding table moves to the GPU (GPU cache 451,879 tokens) | documents up to about 220K tokens restore from disk |
| 24 GB | NVMe/SSD with 4.5 GiB of RAM staging | documents up to about 250K tokens restore from disk |
| 32 GB | RAM, a tier of about 15.5 GiB; the embedding table moves to the GPU | about 677K tokens of cache |
| 48 GB | RAM, a tier of about 21 GiB | about 917K tokens |
| 64 GB | RAM, about 29 GiB | about 1.27M tokens |
| 96 GB and more | RAM | about 1.9M tokens and up |

`auto` takes RAM only where the tier holds at least 1.25 times the GPU cache: the RAM tier keeps a copy of what the
GPU cache holds, so a tier only slightly larger serves little. The token counts are computed from the 24 KB a cached
token takes in a tier.

- **Same output.** A reopened document gives output bit-identical to the same document served from the GPU cache,
  at every length tested (32K to about 258K tokens). A hit the server cannot place is read again instead.
- **RAM tier size.** A RAM tier of N GiB restores a recently read document of up to about N × 55K tokens; with
  several large documents competing, older ones are read again, correctly, just not faster. A RAM tier smaller than
  the GPU cache (chosen with `--extend-cache ram` or `--host-cache-gib`) gets a warning at start; `--extend-cache
  disk` keeps documents across the session instead.
- **The disk folder** is `PAITON_CACHE_DIR/kv-disk`, or `--disk-cache-dir DIR`. It is capped at 64 GiB or a quarter
  of the free space, whichever is smaller, checked at start; a folder that grew past its cap is removed at the next
  start. It stores about 24 KB per newly read prompt token (64 GiB holds about 2.8M new tokens). A restore passes
  through the RAM staging tier, which sets the largest document it restores (the table above) and how many big
  documents come back at the same moment (4 GiB stages about two 100K-token documents). Spinning disks are refused.
- **First reads are a little slower:** every new prompt block is also written to the tier, about 8 % for a
  131K-token prompt in our test. Decode speed is unchanged within noise.
- **16 GB hosts:** a RAM tier with the embedding table in system RAM needs 32 GB, so the launcher moves the table to
  the GPU there. It refuses to start when free memory is short, naming the shortfall, and warns when memory is tight.
- **Only with `--mode long-kv4`**, without images.

More: [cache tiers in the reference](REFERENCE.md#cache-tiers).

<a id="system-ram"></a>

### System RAM

`--mode long-kv4` and `--mode long-512k` keep the 2.4 GiB embedding table in pinned RAM. The launcher warns when less
than 6 GiB of RAM is available at start (the server and the table need about that much): close programs, or with
`--mode long-kv4` add `--no-system-memory-weights` (`--mode long-512k` needs the RAM and refuses that flag). A host
with less than about 14.25 GiB of RAM in total cannot pin the table: there `--mode long-kv4` keeps it on the GPU and
prints a `Note:` line at start. Requests still get 262,144 tokens and the prefix cache, but the cache holds 451,879
tokens (one full-length request plus short ones). `--mode long-512k` refuses to start there.

### Longer than 262,144 tokens: `--mode long-512k`

Experimental. It extends the model's position range to 524,288 tokens with the model's official long-context
scaling, applied to every request in this mode, short ones included. Short-question scores stayed within noise of
`--mode long-kv4`, and a small needle test found 4 of 4 planted facts at 300K and at 500K tokens. Its cache is only
4 % larger than that of `--mode long-kv4`: use it only for single requests above 262,144 tokens. With a `--context`
of 262,144 or less, the launcher serves exactly `--mode long-kv4` (without the scaling).

<a id="advanced-options"></a><a id="common-options"></a>

## Options

Every row already sets its context, cache and request limits. Add these only to tune; the launcher refuses flags that
contradict the chosen `--mode`. `--help` lists every option and `--dry-run` prints the Docker command without
starting it.

| Option | What it changes |
| --- | --- |
| `--context TOKENS` | A smaller context than the row's (input plus output). Only MXFP4 `--mode long` goes higher: up to 220,000, the largest tested. |
| `--max-num-seqs COUNT` | Fewer concurrent requests than the row's (1 to 8; above the row's default, such as the 1 of MXFP4 `--mode long` and `--profile desktop`, is untested). |
| `--long-prefill-threshold TOKENS\|off` | Cap on the prefill tokens a long prompt takes per step, so short requests from other users are answered while it is processed. 2048 by default in `--mode long-kv4` (without `--vision`), off elsewhere; `off` restores the previous read behaviour. |
| `--lm-head bf16` | 3-bit rows: the bf16 output head instead of the checkpoint's FP8 head (the default on the 5 October image and later); `--mode long-kv4` then holds 569,878 instead of 628,877 tokens, `--mode long` 281,665 instead of 316,325, and the start command is the previous launcher's (in `--mode long-kv4` and `long-512k` together with `--host-pools userptr`). |
| `--host-pools userptr` | `--mode long-kv4` and `long-512k`: ROCm's default user-pointer memory for the pinned embedding table instead of GTT memory (the default where the GPU's GTT size holds it; [why](REFERENCE.md#launcher-update-11-october-2026)). |
| `--thinking on` / `--thinking off` | The server default for thinking (off with the 3-bit weights and in the `--mode long*` rows, on in the MXFP4 65,536-token rows). Requests can override it. |
| `--kv-cache fp8` | `run-3bit.sh` without `--mode`: the FP8 instead of the 4-bit KV cache (250,578 tokens). |
| `--no-system-memory-weights` | `--mode long-kv4` with nothing pinned, for a host short of free RAM: the embedding table stays on the GPU and the cache holds 451,879 tokens (one full-length request plus short ones); prefix caching stays on. |
| `--extend-cache [auto\|ram\|disk]` | `--mode long-kv4`: keep prefix-cache blocks that leave the GPU in system RAM or on an NVMe/SSD; see [more cache](#extend-cache). |
| `--kv-cache-memory-bytes BYTES` / `--gpu-memory-utilization FRACTION` | Your own KV budget instead of the measured one, also with `--vision`. |
| `--profile desktop` | For an R9700 that also drives your desktop: 32,768 context, one request, 2 GiB KV. |
| `--port PORT` / `--name NAME` / `--detach` | API port (default 18982), container name, run in the background. |
| `--host ADDRESS` / `--api-key KEY` | Listen address (default 0.0.0.0, every interface; 127.0.0.1 keeps it local); opt-in key the server then requires on `/v1` (handed to Docker by variable name, never printed). |
| `--devices N` / `--list-gpus` | The GPU to run on (default: the first R9700) and the list to choose from. |
| `--ignore-vram-check` | Start although VRAM is already in use on the selected GPU (the launcher otherwise waits up to 30 s for a stopping container, then refuses with the amount in use). |

Experimental options, such as setting the cache tiers by hand (`--host-cache-gib`, `--disk-cache-dir`), are in
the [reference](REFERENCE.md#system-memory-and-experimental-options); no row needs them.
See the reference for [GPU selection and memory budgets](REFERENCE.md#gpu-context-and-memory-controls) and
[long-context details](REFERENCE.md#long-context-200k-and-220k).

<a id="current-benchmark-results"></a>

## Benchmarks

Measured on **one 300 W R9700**, thinking off. Speed depends on prompt length, workload and concurrency; use these
as reference measurements for each row.

### The 65,536-token rows

Speed from BetterBench; accuracy with greedy decoding on the same questions for each column.

| | `run-mxfp4.sh` | `run-3bit.sh --kv-cache fp8` | `run-3bit.sh` |
| --- | ---: | ---: | ---: |
| KV cache | FP8 | FP8 | 4-bit |
| Speed measured on | 26 September image | 26 September image | 2 October r1 image |
| Weighted single-stream decode | 156.1 tok/s | 184.4 tok/s | 179.7 tok/s |
| Eight concurrent requests, aggregate output | 428.0 tok/s | 492.1 tok/s | 488.5 tok/s |
| Prefill, approximately 5.9K input tokens | 3,831 tok/s | 4,165 tok/s | – |
| GSM8K 5-shot (1,319) | 95.68 | 95.30 | 95.22 |
| HumanEval pass@1 (164) | 95.12 | 93.90 | 94.51 |
| MMLU-Pro subset, 0-shot (14 × 100) | 62.57 | 59.71 | 60.57 |

3-bit accuracy rows: previous weights (`7a2e3d70`). Paired MXFP4 against 3-bit (both FP8 KV): only the MMLU-Pro gap (−2.86 points, 95 % interval −4.81 to −0.90) is
beyond the uncertainty; math and coding are within it. The 4-bit KV cache scores the same as FP8.
[Full settings and results](REFERENCE.md#current-benchmark-results) ·
[Quality results](REFERENCE.md#faster-decode-and-prefill-3-bit-w3a4-weights-optional).

### One long document: `run-3bit.sh --mode long`

Measured 1 October.

| | 198,989-token prompt | 257,992-token prompt |
| --- | ---: | ---: |
| New prompt, time to first token | 85 s | 126 s |
| Follow-up question on the cached document, time to first token | 1.4 s | 1.7 s |
| Decode at this depth | 73–78 tok/s | 72–76 tok/s |

Accuracy of the 3-bit weights (previous weights `7a2e3d70`) with every question placed after a 257K-token document, against the same weights at
short context (paired per question; all differences are within the 95 % intervals):

| Benchmark | 3-bit, short context | 3-bit, after 257K tokens |
| --- | ---: | ---: |
| GSM8K 5-shot (1,319) | 95.30 | 95.83 |
| HumanEval pass@1 (164) | 93.90 | 91.46 |
| MMLU-Pro subset, 0-shot (14 × 100) | 59.71 | 60.29 |

[Reports and numbers](benchmarks/2026-10-01-262k/README.md).

<a id="coding-mode-results"></a>

### Coding agents: `run-3bit.sh --mode long-kv4`

3-bit weights, thinking off. Measured on the 3 October release with the bf16 output head unless noted; later images
give identical outputs with `--lm-head bf16`, and the FP8 head (the default since 6 October) measured the same text
prediction.

**Decode at long context** (6 October launcher, FP8 output head; our decode load is a shared prompt the drafter
predicts well, so these are an upper range; real text at 258K decodes at 72 to 76 tok/s):

| Context already in the cache | One request | Eight requests, aggregate |
| --- | ---: | ---: |
| 120K tokens | 120 tok/s | 165 tok/s |
| 228K tokens | 100 tok/s | 147 tok/s |

| Short prompts (BetterBench, full 20-pass run) | `--mode long-kv4` |
| --- | ---: |
| Weighted single-stream decode | 173.4 tok/s |
| Aggregate output at 1 / 2 / 4 / 8 requests | 143.9 / 246.2 / 355.4 / 478.5 tok/s |
| Time to first token p50 at 1 / 8 requests | 87.1 / 147.8 ms |
| Cold prefill at 2K / 8K / 32K / 64K depth | 3,641 / 3,856 / 3,824 / 3,536 tok/s |

- **A new question about a 258K-token document already read:** 2.7 s to the first token instead of 134 s for the
  first read, with the same output as a cold read.
- **Accuracy** with standard-length questions (greedy; previous weights `7a2e3d70`): GSM8K 95.45, HumanEval 92.68, MMLU-Pro 59.93, needle at
  61,440 tokens 80/80, within noise of the 65,536-token rows; the MMLU-Pro gap to MXFP4 is the 3-bit weights' own.
  An answer from the cache scores like a cold read.
- **`--mode long-512k`:** reading a 300K-token document cold takes 158 s, a 500K-token one 361 s; follow-up
  questions take 2.8 to 2.9 s and 4.2 to 4.4 s. Needles at 300K and 500K: 4/4 each.

[Coding mode details](REFERENCE.md#coding-mode) · [`--mode long-512k` details](REFERENCE.md#mode-long-512k) ·
[comparisons with earlier images](REFERENCE.md#earlier-coding-mode-comparisons).

<a id="long-coding-sessions"></a>

### Long coding sessions

Measured 6 October on the 5 October image and launcher, thinking off, with the previous weights (`7a2e3d70`). Eight sessions of 20 turns per mode, identical
prompts in both: each session starts from about 95K tokens of real source code, and every turn adds one more source
file and a new HumanEval+ task, so the prompt grows from about 100K to 181K tokens. Each turn's answer runs against
the task's tests.

| | `run-mxfp4.sh --mode long` | `run-3bit.sh --mode long-kv4` |
| --- | ---: | ---: |
| Tasks passed (HumanEval+ base and extra tests) | 83.75 % | **87.5 %** |
| Tasks passed (base tests) | 90.0 % | 92.5 % |
| Change per turn over the 20 turns | −0.51 points (95 % interval −1.47 to +0.48) | **0.00** (−0.71 to +0.64) |
| Last five turns against the first five | −15.0 points | −2.5 points |
| First token, first turn (100K tokens, cold) | 40.7 s | 32.1 s |
| First token, later turns (100K to 185K, 95 to 97 % from the cache) | 3.3 to 4.4 s | 3.1 to 4.1 s |
| Decode, code | 172 to 160 tok/s | 217 to 207 tok/s |

The 3-bit coding mode keeps its pass rate over all 20 turns; its lead over MXFP4 (+3.75 points, 95 % interval
+0.62 to +7.50) is small with 160 tasks per mode. The tasks are independent of each other, so this measures how well
each mode works deep into a long context, not errors that carry over from turn to turn. Decode includes DFlash2 on
code, which the drafter predicts well.

<a id="cache-tier-results"></a>

### Cache tiers: `--extend-cache`

On a 16 GB host with a SATA SSD, `--extend-cache` (auto), after a full server restart:

| Document | Read cold | Restored from the SSD |
| --- | ---: | ---: |
| 130K tokens | 46.9 s | **6.8 s** |
| 200K tokens | 86.4 s | **9.0 s** |

Both outputs were identical to a fresh read. In a session that keeps more documents than the GPU cache holds (six
100K-token documents asked in turn, twice, 16 GB host), every re-read came back from the SSD: 4.3 s to the first
token instead of 33.5 s, all 12 answers identical to a fresh read. NVMe should be faster; we have not measured it.

### Images with long context, and MXFP4 long context

`run-3bit.sh --mode long --vision` (245,000 tokens): a 200K-token prompt with a chart, 87 s cold and 1.3 s from the
cache. `run-mxfp4.sh --mode long` (200,000 tokens, one request, 28 September): a 199K-token prompt, 102 s cold, 1.2 s
from the cache, 63–68 tok/s. `run-3bit.sh --mode long-kv4 --vision` (6 October): the 29-case image set passed, also
with eight image requests at once and repeated cold 4K images; the first token on a cold 4K image came after 4.75 s.
[Long-context details and accuracy](REFERENCE.md#long-context-200k-and-220k) ·
[Vision checks and how to send an image](REFERENCE.md#images-and-vision).

<a id="advanced-setup-and-release-history"></a><a id="run-an-earlier-release"></a><a id="native-serving"></a>

## More

- [Release reference](REFERENCE.md): image pins, release notes, detailed benchmarks, long-context checks and KV
  tuning.
- [Run an earlier release](REFERENCE.md#run-an-earlier-release) with `--image`, and the command lines of earlier
  releases.
- Native serving without the container: `paiton serve qwen38-nvfp4` serves a separate preset (text only, 65K, FP8 KV,
  without DFlash2 or the 3-bit weights; bundle `qwen38-rocm10-native-20260921`, profile `qwen38-nvfp4-w4a8-text-65k`,
  the same pinned target revision). [Native setup](REFERENCE.md#native-serving).

[Third-party notices](THIRD_PARTY_NOTICES.md) ·
[3-bit weight license and calibration](https://huggingface.co/EliovpAI/Qwen3.8-27B-W3Rot-INT3-Paiton-RDNA4)
