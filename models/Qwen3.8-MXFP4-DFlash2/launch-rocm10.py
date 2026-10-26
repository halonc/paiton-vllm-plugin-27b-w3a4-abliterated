#!/usr/bin/env python3
"""Start Qwen3.8 on one R9700 from the pinned ROCm 10 image.

Choose the weights and a mode; everything else is optional."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time


IMAGES = {
    # The 5 October release: the 4 October image plus a seeded compile cache and kernel warm-up at start, native long-prompt
    # attention and 4-bit decode kernels (identical outputs), images in the coding mode and 24 KB/token cache tiers. The
    # previous release images (PREVIOUS_IMAGES) stay usable through --image with their previous behaviour.
    '65k': 'ghcr.io/eliovp/paiton-vllm-plugin:qwen38-rocm10-vllm029-20261005-r1@sha256:245c71d54f046f89b74dbfbd4c894003754561e55c9d215a8d3ce65364fb0f80',
    '200k': 'ghcr.io/eliovp/paiton-vllm-plugin:qwen38-rocm10-vllm029-200k-20260918-r2@sha256:32dab97330ea84b86967537d25f91878c30f21ff844f71369508c5a049b89178',
}
PREVIOUS_IMAGES = {
    # the 4 October release (coding mode with RAM/SSD cache tiers at 40 KB/token)
    '20261004-r1': 'ghcr.io/eliovp/paiton-vllm-plugin:qwen38-rocm10-vllm029-20261004-r1@sha256:6a97d65fda17c1c48b36d3423c6f3709bd65a3849227e45a8a24a552d4c81b9d',
    # the 3 October release (coding mode, opt-in 512K); its 4-bit cache refuses the RAM/SSD cache tiers
    '20261003-r1': 'ghcr.io/eliovp/paiton-vllm-plugin:qwen38-rocm10-vllm029-20261003-r1@sha256:fb71b59eb29f3341dd10e9972920e75073a03fefc7bf6d2f2e1966b91a730f53',
    # the 29 September r2 image with the new 4-bit KV page format (the 65K preset and --mode long-kv4) and the
    # VRAM-headroom overlay
    '20261002-r1': 'ghcr.io/eliovp/paiton-vllm-plugin:qwen38-rocm10-vllm029-20261002-r1@sha256:82a24a1926bc01a134b106401390650b9e0ddb0aa8cf6a613ba3a615ce46b840',
    # r1s: a rebuild of the 2 October r1 image with the same runtime behaviour (rollback image)
    '20261002-r1s': 'ghcr.io/eliovp/paiton-vllm-plugin:qwen38-rocm10-vllm029-20261002-r1s@sha256:a1c1025052f84a009428709bfe7e9431281ab5d5c0f723a49d79eecafe519dad',
    '20260929-r2': 'ghcr.io/eliovp/paiton-vllm-plugin:qwen38-rocm10-vllm029-20260929-r2@sha256:1195f31329966b3dc4e8e2d17327d339827b3d2b09165f9969b053a6fc2db045',
}
# Images that carry the native 3-bit (W3A4) runtime. Its flags default on inside
# the image and read the rotated weights from /models/w3rot.
W3_RELEASES = frozenset(('65k',))
W3_FLAGS = ('PAITON_W3_DECODE', 'PAITON_W3_PREFILL', 'PAITON_W3_A4')
# Release KV budget plus 2.65 GiB of the 3.29 GiB the 3-bit weights free: four
# 61K-token requests fit and peak at the MXFP4 release VRAM (31.39 vs 31.37 GiB).
W3_KV_CACHE_BYTES = 9381235631
# Long-context mode with the 3-bit weights (fp8 cache, prefix caching): eight sequences, the release prefill budget
# and graph set, and a KV budget sized for the 262,144-token model limit plus short concurrent requests.
# Measured 1 Oct 2026 (262,144 context, 8 sequences, 4096 budget): 281,665 fp8
# tokens, startup zero-check OK, 1.29 GiB idle headroom; 10.95 GB (302,381 tokens) left only 0.59 GiB and 11.6 GB
# (320,309) 0.14 GiB, so the smaller budget keeps the margin for a full-context request plus short ones.
W3_LONG_KV_CACHE_BYTES = 10200000000
# MXFP4 keeps the one-request chat profile: its 8 GiB fp8 cache holds 231,067 tokens, so 262,144 cannot start. --mode
# long serves the measured 200,000 (the profile's default context); 220,000 is the largest tested.
MXFP4_LONG_CONTEXT = 200000
MXFP4_LONG_MAX_CONTEXT = 220000
# Vision in the long-context mode (3-bit weights): the 0.88 GiB vision encoder comes out of the KV budget. Measured
# 1 Oct 2026: 253,560 cache tokens with the encoder loaded, zero-check OK,
# 1.08 GiB idle headroom; the largest --context is that capacity minus 8,192 tokens, rounded down to thousands.
W3_LONG_VISION_KV_CACHE_BYTES = W3_LONG_KV_CACHE_BYTES - 944000000
W3_LONG_VISION_MAX_CONTEXT = 245000
# With the 4-bit KV cache (capacity mode) the same pool holds 1.8x the attention tokens. The mode needs about 0.16 GiB
# more working memory (prefill workspace, decode scratch) and admits more concurrent requests, so the pool has 618
# blocks of 14,336,000 B: 0.44 GiB more free VRAM at idle than the fp8 release budget, for 1.70x its 8-sequence
# attention capacity (the same from 618 to 632 blocks). Under full load the allocator's cache grows into free memory
# at any budget: peaks 31.65-31.76 GiB vs the release's 31.63 GiB, without OOM.
W3_KV4_CACHE_BYTES = 8859648000
# Images that carry the 4-bit KV cache (dense KV4 pages published to the allocator). It is qualified with the 3-bit
# weights: in the 65K preset and, on request (--mode long-kv4), in the long-context mode, both without prefix caching;
# every other configuration keeps the fp8 KV cache.
KV4_RELEASES = frozenset(('65k',))
KV4_FLAGS = ('PAITON_KV4', 'PAITON_KV4_CAPACITY')
# The 4-bit decode path is qualified up to this context (prompt + generated tokens): the model's native 262,144 tokens
# with KV4 bundle kv4-v5. The 28 and 29 September images carry bundle kv4-v4, whose decode stops at 200,000 and whose
# adapter predates the prefix-caching check: they keep that limit and stay out of the long-context mode. Without an
# explicit --kv-cache kv4, the launcher selects the 4-bit cache only in the 65K preset, where it was measured end to end.
KV4_MAX_CONTEXT = 262144
KV4_V4_MAX_CONTEXT = 200000
# The 28 and 29 September images, by reference or by tag (with or without a digest).
KV4_V4_SUFFIXES = ('qwen38-rocm10-vllm029-20260929-r2', 'qwen38-rocm10-vllm029-20260928-r1')
KV4_V4_IMAGES = frozenset((
    'ghcr.io/eliovp/paiton-vllm-plugin:qwen38-rocm10-vllm029-20260929-r2@sha256:1195f31329966b3dc4e8e2d17327d339827b3d2b09165f9969b053a6fc2db045',
    'ghcr.io/eliovp/paiton-vllm-plugin:qwen38-rocm10-vllm029-20260928-r1@sha256:487c97d51e5b4a3fcd0a206e53d842a52dd56a199d8ee3e884f48815093a80d4',
))
KV4_AUTO_MAX_CONTEXT = 65536
# Long-context mode with the 4-bit cache (3-bit weights, bundle kv4-v5, --mode long-kv4, no prefix caching): the fp8
# mode's budget less the 4-bit mode's prefill workspace (one fp8 page per 16 tokens of the context, 512 MiB at
# 262,144), rounded down to whole pool blocks of 14,336,000 B. On the 2 October image, through this launcher's own
# command (262,144 context, 8 sequences, no prefix caching): 458,922 KV4 tokens, 1.63x the fp8 mode's 281,665. The
# same budget measured 2 Oct with prefix caching (an earlier layout of this mode, 451,879 tokens): startup zero-check
# OK, 1.03 GiB idle headroom once the workspace exists (fp8: 1.29 GiB); 700 blocks would leave 0.91 GiB.
W3_LONG_KV4_CACHE_BYTES = 9662464000
# The 3-bit profiles cap PyTorch's caching allocator at 95 % of the card (the 65K default since 2 Oct as well: on
# the 2 October release candidate it reached the KFD eviction edge within seconds of BetterBench's concurrency
# phase). Uncapped, the allocator returns memory only after a failed allocation and climbs to the VRAM edge during
# long prefills; the KFD admits allocations up to ~31.79 GiB per process although they no longer fit, and the
# process's buffers are then evicted into system memory (2 Oct 2026, 4-bit long mode, two ~199K documents:
# evict/restore loops, then the KV pool in GTT and the 16 GB host out of memory). 95 % leaves ~1 GiB for memory
# outside PyTorch (runtime, code objects, scratch) plus ~0.5 GiB headroom; hipMemGetInfo is no guide (it reported
# 327 MiB free with ~1.6 GiB of the card unused). The 3-bit --vision modes are capped too (6 Oct: without the cap,
# --mode long --vision on the 5 October image evicted under 8 concurrent image requests followed by a large image and
# did not start with the drafter FP8 GEMM; capped: no eviction); the MXFP4 weights keep the previous setting.
W3_MEMORY_FRACTION = 0.95
# With the FP8 output head the memory outside PyTorch grows by ~0.28 GiB (the native head's runtime), and the start's fp8
# GEMM warm-up fills the allocator up to its cap: the card peaked at 31,975 MiB in --mode long against 31,693 MiB with
# the bf16 head (11 Oct 2026, gate q27head). One point less (-326 MiB of allocator room) keeps the start peak under the bf16
# head's level (gates q27cap / q27smk, 11 Oct: --mode long 31,587, long-kv4 31,407, 65k 31,767, 65k --vision 31,397,
# long --vision 31,340 MiB); serving peaked at 30,138 (long) and 30,113 MiB (long-kv4) with 258K needles 4/4.
W3_MEMORY_FRACTION_LMHEAD_W8 = 0.94
# 3 Oct 2026: in the 4-bit long-context mode a prefix-cache hit on a shared-prefix (junction) checkpoint ended in a
# GDN-norm nonfinite engine error (a repeated 32,758-token request after related requests). Until that is fixed the
# 4-bit long mode runs without prefix caching; measured on one R9700 (repeat of that request):
KV4_LONG_NO_PREFIX_CACHING = ('this image runs the 4-bit long-context mode (--mode long-kv4) without prefix caching: '
                              'it predates the fix of that mode\'s prefix-cache-hit fault, so every request prefills '
                              'its full prompt again (a repeated 32K-token prompt takes 10.2 s instead of 2.9 s from the '
                              'fp8 cache). The fp8 long-context mode (--mode long) and the current release image keep '
                              'prefix caching.')
KV4_LONG_PREFIX_CACHING_OFF = ('the 4-bit long-context cache runs without prefix caching in this spelling: every '
                               'request prefills its full prompt again; --mode long-kv4 serves it with prefix caching.')
# Prefix caching in the 4-bit long mode needs the Mamba seed-column compat overlay: without it a resumed request seeded
# its GDN state from the drafter's block grid instead of the GDN block grid, so a prefix-cache hit could continue from
# the wrong state. The 2 October release image (r1 and its rebuild r1s) and older images predate it; any other image
# may enable it explicitly (experimental, --prefix-caching on).
KV4_PREFIX_CACHE_UNFIXED_SUFFIXES = ('qwen38-rocm10-vllm029-20261002-r1', 'qwen38-rocm10-vllm029-20261002-r1s',
                                     'qwen38-rocm10-vllm029-20260929-r2', 'qwen38-rocm10-vllm029-20260928-r1')
# With prefix caching, vLLM keeps a GDN state at every block of a cached prompt when an EAGLE-class drafter is present
# (dense checkpoints, several pool blocks per block, so one long prompt can cycle the whole pool and evict every other
# document). The 4-bit long mode with prefix caching retains them every 32,000 tokens instead, plus
# (PAITON_PC_EAGLE_TAIL, compat overlays) the one block before each prompt's end, where the next turn of the same
# conversation hits with the EAGLE tail-block drop.
KV4_PREFIX_RETENTION_INTERVAL = 32000
# --mode long-512k (experimental, up to 524,288 tokens): factor-2 position scaling on the target (the model card's route past 262,144
# tokens; only the full-attention layers use RoPE), the drafter's position table extended to match (its sliding window
# keeps relative distances short), an image with KV4 bundle kv4-v6, prefix caching, and --system-memory-weights for the
# pool (815 blocks; one 524,288-token request takes 719). Static position scaling also changes short prompts slightly: opt-in only.
KV4_YARN_CONTEXT = 524288
W3_LONG_KV4_YARN_CACHE_BYTES = 815 * 14336000
YARN_FACTOR_2 = {'rope_type': 'yarn', 'factor': 2.0, 'original_max_position_embeddings': 262144}
# Prefill chunking under sparse retention (compat overlay PAITON_PC_SPARSE_ALIGN): with prefix caching in the 4-bit
# long modes a prompt's chunks stop only at the states the retention keeps (and the tail checkpoint) instead of every
# 1,600-token block: fewer, longer prefill steps.
KV4_SPARSE_ALIGN = True
# --mode long-kv4 --vision: the embedding in system memory, the vision encoder on the GPU, the allocator capped as
# without --vision, 740 pool blocks (496,129 KV tokens with prefix caching), images up to VISION_MAX_PIXELS and a
# VISION_MM_CACHE_GIB processor cache. Measured on the 5 October image, 6 Oct (vision suite: cold and cached images,
# 1,600-token boundaries, a 200K prompt with a chart, 3 x 8 concurrent image requests, the cold 4096^2 image 5 times in
# one server, a 4K screenshot of 15 px code read back exactly): no KFD eviction, peak 31.32 GiB. Without the pixel cap
# a cold 4096^2 image (65,536 patches in one encoder pass) peaked at 31.85 GiB, past the 31.79 GiB the KFD admits, and
# evicted on every repeat; the default 4 GiB processor cache held each distinct large image in host memory.
KV4_LONG_VISION = True
W3_LONG_KV4_VISION_SYSMEM_CACHE_BYTES = 740 * 14336000
W3_LONG_KV4_VISION_SYSMEM_LMHEAD_W8_CACHE_BYTES = (740 + 88) * 14336000   # the FP8 head's freed VRAM as pool blocks (gate V, 6 Oct: 555,128 tokens)
# Image limits of every --vision mode: a 4K UHD screenshot (3840 x 2160) is read at full resolution, larger images
# are downscaled to that many pixels (the checkpoint allows 4096 x 4096); the multimodal processor cache keeps 1 GiB of
# preprocessed images in host memory instead of vLLM's 4 GiB (hosts with 16-32 GB). The native drafter FP8 GEMM (T2,
# bit-identical, ~240 MiB at the startup peak) stays on unless KV4_LONG_VISION_DRAFT_FP8_GEMM is False (measured at 740
# blocks: 29/29 cases, five cold 4K-capped images without new eviction, card peak 31.17 GiB).
VISION_MAX_PIXELS = 3840 * 2160
VISION_MM_CACHE_GIB = 1
KV4_LONG_VISION_DRAFT_FP8_GEMM = True
# The host KV tier (--host-cache-gib) with the 4-bit cache needs the connector compat overlay that ends every hit on a
# recurrent-state block (1,600 tokens); on older images a hit can end on a drafter block (800 tokens) and resume from
# the wrong state, so there the tier stays with the fp8 cache. --mode long-512k keeps it off.
KV4_HOST_CACHE_UNFIXED_SUFFIXES = ('qwen38-rocm10-vllm029-20261003-r1',)
# --mode long-kv4 --vision was qualified on the 5 October image only (VRAM headroom and the vision suite under load).
KV4_LONG_VISION_UNQUALIFIED_SUFFIXES = ('qwen38-rocm10-vllm029-20261004-r1',)
# Published image digests, so a reference by digest alone (without the tag) is recognised like its tag.
IMAGES_BY_RELEASE_DIGEST = {
    '20261004-r1': 'sha256:6a97d65fda17c1c48b36d3423c6f3709bd65a3849227e45a8a24a552d4c81b9d',
    '20261003-r1': 'sha256:fb71b59eb29f3341dd10e9972920e75073a03fefc7bf6d2f2e1966b91a730f53',
    '20261002-r1s': 'sha256:a1c1025052f84a009428709bfe7e9431281ab5d5c0f723a49d79eecafe519dad',
    '20261002-r1': 'sha256:82a24a1926bc01a134b106401390650b9e0ddb0aa8cf6a613ba3a615ce46b840',
    '20260929-r2': 'sha256:1195f31329966b3dc4e8e2d17327d339827b3d2b09165f9969b053a6fc2db045',
    '20260928-r1': 'sha256:487c97d51e5b4a3fcd0a206e53d842a52dd56a199d8ee3e884f48815093a80d4',
}
KV4_HOST_CACHE_REFUSAL = ('--host-cache-gib with the 4-bit KV cache needs an image with the host-tier alignment fix; '
                          'this image predates it (the host tier works with --mode long)')
KV4_512K_HOST_CACHE_REFUSAL = ('--host-cache-gib is not qualified with --mode long-512k; it works with --mode long-kv4 '
                               'on an image with the host-tier alignment fix')
# The serving process plus the pinned embedding need this much available system memory at launch; below it the
# launcher warns (the host may swap or reclaim pinned pages under pressure).
SYSTEM_MEMORY_MIN_AVAILABLE_GIB = 6.0
# --mode: one named preset per serving mode of the 65k release image. Each stands for the legacy flags listed in MODES
# (3-bit weights, text) and produces exactly their Docker argv; --mode long picks the context its configuration holds:
# --context 200000 with the MXFP4 weights, --context 245000 with --vision. --mode long-kv4 falls back to the embedding on
# the GPU (674 pool blocks, one printed note) where the host cannot pin it, and keeps its previous form (no prefix
# caching) on images that predate the prefix-cache fix. Explicit flags that contradict a preset are refused, compatible
# refinements (a smaller --context, --max-num-seqs, --thinking, --port, memory budgets, ...) pass through.
LONG_CONTEXT = 262144
MODES = {
    '65k': 'no flags',
    'long': '--context 262144',
    'long-kv4': '--context 262144 --kv-cache kv4 --prefix-caching on --system-memory-weights',
    'long-512k': '--context 524288 --kv-cache kv4 --prefix-caching on --system-memory-weights',
}
MODE_HELP = (
    'how to serve; without --mode: 65k\n'
    '65k: 65,536 context, up to 8 requests, 4-bit KV cache with the 3-bit weights: the fast everyday default\n'
    'long: 262,144 context (MXFP4: 200,000, one request), fp8 KV cache with prefix caching: one long document at a '
    'time, fast follow-ups\n'
    'long-kv4: 262,144 context per request, 4-bit KV cache with prefix caching and the embedding in system memory: '
    'about 629,000 tokens of reusable cache, for coding agents and many long conversations (3-bit weights only)\n'
    'long-512k: experimental, up to 524,288 context per request (long-context position scaling), otherwise as long-kv4; needs 2.4 GiB '
    'of free system memory and an image with KV4 bundle kv4-v6')
# VRAM left unclaimed by PyTorch's caching allocator after warm-up in the 3-bit profiles (worker compat overlay; it
# only ever lowers the launcher's fraction above): the KFD admits allocations past the physically free VRAM and evicts
# to system memory instead of failing. 1 GiB = ~0.5 GiB margin to the KFD admission limit plus the ~270 MiB that
# memory outside PyTorch grew during the 2 October validation run of the 4-bit long mode (with 512 MiB the card
# peaked 0.17 GiB below the limit).
VRAM_HEADROOM_MIB = 1024
# Image input (--vision) also serves the checkpoint's vision encoder (0.88 GiB), which the release command leaves out
# with --language-model-only. Its weights, its encoder cache (one 16,384-token image) and its startup profiling come
# out of the KV budget (the MXFP4 release budget runs out of memory at KV allocation). Each weights / KV cache pair has
# a budget measured on one R9700 with a 4096 x 4096 image, a ~58K-token prompt plus an image and eight concurrent
# image requests, then lowered by 0.5 GiB where the peak came within 0.1 GiB of the card (3-bit weights: 437 pool
# blocks with the 4-bit cache). The startup self-check (a one-time 2.37 GiB allocation) passes with every budget.
VISION_RELEASES = frozenset(('65k',))
VISION_KV_CACHE_BYTES = {('mxfp4', 'fp8'): 4500000000, ('w3a4', 'fp8'): 6760000000, ('w3a4', 'kv4'): 6264832000}
SYS_DRM = Path('/sys/class/drm')
# VRAM preflight (a real start only): the mode budgets assume an idle card. VRAM already in use above the idle
# allowance (a display server, a container that is still releasing, another server) made allocate_kv_cache fail with
# torch.OutOfMemoryError although the free figure looked sufficient (Hugging Face report, 5 Oct 2026). The launcher waits a
# little for a just-stopped container, then refuses with the figures; --ignore-vram-check skips it.
VRAM_PREFLIGHT_IDLE_BYTES = 1 * 1024**3
VRAM_PREFLIGHT_WAIT_S = 30.0
VRAM_PREFLIGHT_POLL_S = 2.0
SYS_KFD = Path('/sys/class/kfd/kfd/topology/nodes')
# --host-cache-gib (experimental, opt-in): evicted prefix-cache blocks kept in pinned system memory (vLLM's
# OffloadingConnector). The launcher sets PAITON_HOST_KV_PRIVATE=1: the tier is private hipHostMalloc (GTT) memory, as in
# upstream #57160, instead of a /dev/shm region registered as a user pointer, which the kernel can reclaim under memory
# pressure and so stop the GPU queues (restore_userptr). GTT memory counts against the TTM limit (ttm.pages_limit,
# 50 % of RAM by default), shared with the other pinned consumers; the tier also keeps clear of the host's working set:
# at most MemTotal less 11 GiB for the serving process, the OS and the model load.
# --system-memory-weights (3-bit long modes; the default of --mode long-kv4 where the host can pin it, required by
# --mode long-512k, opt-in elsewhere; images with the host-memory modules only): the bf16 input embedding
# (2,542,796,800 B) lives in pinned system memory (one hipHostMalloc table, rows read zero-copy) and the freed VRAM goes
# to the KV cache. Budgets: long-kv4 850 pool blocks (two 262,144-token requests at 385 blocks each); fp8 long +2.5 GB.
# The vision encoder always stays on the GPU: streaming its blocks from system memory (PAITON_HOST_VISION) is not
# qualified, so --mode long --vision does not take --system-memory-weights.
SYSTEM_MEMORY_EMBEDDING_BYTES = 2542796800
W3_LONG_KV4_SYSMEM_CACHE_BYTES = 850 * 14336000
# FP8 lm_head (--lm-head fp8; the compiler's lmhead-w8 bundle, PAITON_LMHEAD_W8=1): the checkpoint's FP8 head stays
# as loaded (1.19 GiB: e4m3 rows and one scale per row, served by a native W8A16 kernel) instead of the 2.37 GiB bf16
# copy. In --mode long-kv4 the freed 1.18 GiB go to the pool: LMHEAD_W8_POOL_BLOCKS more blocks of 14,336,000 B.
# Measured 4 Oct 2026 on one R9700 (20261003-r1 image + bundle, --mode long-kv4, 938 blocks): 628,877 KV4 tokens
# (+59,000), idle 4.09 GiB free (the release: 1.71; the fp8 head also drops the startup zero check's 2.37 GiB
# temporary), peak 30.84 GiB, GSM8K-200 188/200 as the release, DFlash2 acceptance unchanged; sampled C1 decode +6.9 %,
# C8 +14.5 % (those take the full-vocab head). Since the 6 October launcher update --mode long-kv4 (text) selects it by
# default on the images that carry the bundle (--lm-head auto; gate D, 6 Oct 2026, published 5 October image: 131K NLL
# equal on every bucket, needles 131K/258K 4/4, pool 569,878 -> 628,877 tokens); --lm-head bf16 restores the bf16 head
# and the 850-block budget. Images without the bundle refuse fp8.
# Since the 10 October launcher update every 3-bit mode selects it by default on those images (--lm-head auto): the head
# is the checkpoint's own FP8 tensor in both forms (the bf16 copy is its dequantized form), so the change is the kernel
# that applies it. --mode long (text) gives the freed VRAM to its fp8 KV cache like long-kv4 (W3_LONG_LMHEAD_W8_POOL_BYTES);
# the other 3-bit modes keep their measured KV budgets and gain the 1.18 GiB as headroom (65k with and without --vision,
# --mode long --vision, long-512k, --system-memory-weights in --mode long, the desktop profile). MXFP4 keeps the bf16 head.
LMHEAD_W8_IMAGE_SUFFIXES = ('qwen38-rocm10-vllm029-20261005-r1',)
LMHEAD_W8_POOL_BLOCKS = 88
LMHEAD_W8_POOL_BYTES = LMHEAD_W8_POOL_BLOCKS * 14336000          # 1,261,568,000 B: what the FP8 head frees, as pool bytes
W3_LONG_KV4_SYSMEM_LMHEAD_W8_CACHE_BYTES = (850 + LMHEAD_W8_POOL_BLOCKS) * 14336000
W3_LONG_SYSMEM_CACHE_BYTES = W3_LONG_KV_CACHE_BYTES + 2500000000
# --mode long (text, 3-bit, fp8 KV, embedding on the GPU) with the FP8 head: the release budget plus the freed bytes
# (gate q27head, 10 Oct 2026; see REFERENCE.md for the measured pool, VRAM peak and BetterBench rows).
W3_LONG_LMHEAD_W8_KV_CACHE_BYTES = W3_LONG_KV_CACHE_BYTES + LMHEAD_W8_POOL_BYTES
# --disk-cache-dir (experimental, with --host-cache-gib): evicted host-tier blocks also go to files under DIR (vLLM's
# TieringOffloadingSpec, file-system tier; every block passes through the host tier first). vLLM names its folder
# after the served model path, dtype and cache groups only, which are the same for both weight sets of this launcher,
# so the launcher adds a fingerprint folder (image id, weight folders, KV format, context, GDN state). The file tier has
# no size limit of its own: the launcher refuses to start when the folder exceeds --disk-cache-gib (default 64) and
# --wipe-disk-cache removes it (through Docker: the files are written by the container's user).
# The tiering spec keeps the host tier in a /dev/shm file that it unlinks only on a clean shutdown (it ignores
# PAITON_HOST_KV_PRIVATE), so in the host's IPC namespace a killed server (OOM killer, docker kill) leaves the whole tier
# in the host's /dev/shm until reboot. With a disk tier the container gets a private IPC namespace instead: its own
# /dev/shm, the tier plus 1 GiB, which Docker unmounts when the container stops, however the server exits.
DISK_CACHE_FORMAT = 1
DISK_CACHE_DEFAULT_GIB = 64.0
DISK_TIER_SHM_HEADROOM_GIB = 1.0
# --extend-cache [auto|ram|disk] (experimental, --mode long-kv4, off unless given): sizes the prefix-cache tiers itself.
# The host tier is inclusive (it keeps a copy of what the GPU pool holds), so system memory adds capacity only when its
# tier holds clearly more than the GPU pool: auto picks system memory when the tier holds at least 1.25 x the pool's
# tokens (a RAM tier only slightly larger than the pool serves almost nothing to a cycling working set: 64 GiB vs 1.92M
# tokens gave 0 of 32 re-asks a hit on 6 Oct, while the SSD tier's capacity caught it; with the FP8 head's 628,877-token
# pool the 5 October image's 24 KB per token still puts 32 GB hosts and up on system memory), otherwise NVMe/SSD behind a staging tier that restores a whole 256K document (4.5 GiB, the embedding on the
# GPU where needed) or the largest that fits down to 2 GiB (4 GiB: documents up to ~220K tokens, 2 GiB: ~110K).
# Token capacities use the tier space a stored token takes and the bytes a restored token loads (~19 KB), per image
# (TIER_BYTES_BY_IMAGE_SUFFIX), so an image that stores less per token gets the larger capacities without other changes.
# Every stored key (one chunk of one KV group) takes a whole tier slot of ~14.36 MB, the size of a block across the
# cache's layers (a 4 GiB tier holds 299 slots), and a 4-bit prompt stores ~4.3 keys per 1,600 tokens (two attention
# chunks, two 800-token drafter chunks, the recurrent states at the retained boundaries): ~40 KB of tier per token
# (330 slots for 120,001 tokens on the 20261004 images; the transferred bytes, 31-33 KB, undercount the padded slots).
# A prompt that needs more slots than the tier holds evicts its own oldest keys while it is stored, its recurrent
# checkpoints first, and gets no tier hit. The disk tier lives under PAITON_CACHE_DIR/kv-disk (or --disk-cache-dir) on
# NVMe/SSD, at most 64 GiB or a quarter of the free space there (with what the folder already holds). The tier has no
# size limit while it runs (~40 KB per newly prefilled token: a 64 GiB cap holds ~1.7M new tokens; continuous prefill
# fills it in minutes, a coding session in hours), so a run can overshoot the cap: the next start removes an over-full
# folder of its configuration. A full disk is safe for the tier: a failed store is logged and skipped (atomic files).
EXTEND_CACHE_RAM_FACTOR = 1.25
# Pinning the RAM tier at start costs ~0.65 s per GiB (6 Oct, Threadripper 3960X, 251 GiB: 64 GiB +40 s, the 122.5 GiB auto
# tier +80 s over 110.7 s). Auto keeps the whole pinnable tier: a working set that exceeds the tier even slightly gets no
# hits when it cycles (64 GiB with 14 % too many tokens: 0 of 32 re-asks), one inside it re-reads a 120K prompt in 1.2 s
# instead of ~41 s, so the larger tier pays for its start after about two re-asked large documents. The note says what
# the start costs; --host-cache-gib sets a smaller tier.
HOST_PIN_SECONDS_PER_GIB = 0.65
TIER_STORED_BYTES_PER_TOKEN = 40960
TIER_LOADED_BYTES_PER_TOKEN = 19000
TIER_BYTES_BY_IMAGE_SUFFIX = {       # image name suffix: (stored, loaded) bytes per token, for images that store less
    # rc-next (host-tier stores only the draft-group chunks a hit can use): measured 340 slots for 200,002 new tokens =
    # 24,371 B/token on a disk run, stored rounded up to 24 KiB; loaded unchanged
    'qwen38-rocm10-vllm029-20261005-rcnext-dev1': (24576, 19000),
    'qwen38-rocm10-vllm029-20261005-r1': (24576, 19000),       # the published 5 October image (same build)
}
KV4_POOL_TOKENS = {True: 569878, False: 451879}    # long-kv4 with prefix caching, keyed by: embedding in system memory
KV4_POOL_TOKENS_LMHEAD_W8 = {True: 628877, False: 451879}   # the same with the FP8 lm_head (measured 6 Oct; the GPU-embedding budget is unchanged)


def kv4_pool_tokens(args):
    """The long-kv4 GPU pool in tokens, keyed by the embedding placement, for the head the start will use."""
    return KV4_POOL_TOKENS_LMHEAD_W8 if lm_head_mode(args) == 'fp8' else KV4_POOL_TOKENS
EXTEND_CACHE_STAGING_GIB = (4.5, 4.0, 3.5, 3.0, 2.5, 2.0)     # the largest that fits; 4.5 restores a 256K document
EXTEND_CACHE_DISK_MAX_GIB = 64.0
EXTEND_CACHE_DISK_FREE_SHARE = 0.25
SYS_DEV_BLOCK = Path('/sys/dev/block')
PROC_MEMINFO = Path('/proc/meminfo')
DEV_SHM = '/dev/shm'
TTM_PAGES_LIMIT = Path('/sys/module/ttm/parameters/pages_limit')
# Pinned system memory (the embedding with --system-memory-weights, the host KV tier) is bounded so the whole server
# fits: the serving processes' unpinned peak during model load and graph capture (measured on the 16 GiB testbench,
# MemAvailable drop less the pinned buffers: 4.3-4.65 GiB with the embedding in system memory; 4.0-4.2 GiB with it on
# the GPU next to a 3 GiB tier and at least 4.5 GiB next to a 4.5 GiB tier, 4 Oct; rounded up) plus the pinned buffers
# plus a reserve for the system and a desktop (3.5 GiB) and the 3.5 GiB that must stay available, at most MemTotal. On a
# 16 GiB host that keeps the coding mode's embedding in system memory and allows a RAM tier only with the embedding on
# the GPU (up to ~4 GiB); the RAM tier with the embedding in system memory needs about 32 GB of RAM.
SERVER_UNPINNED_PEAK_GIB = {True: 4.75, False: 4.5}     # keyed by: embedding in system memory
HOST_RESERVE_GIB = 7.0
HOST_CACHE_SHM_MARGIN_GIB = 0.5
# At launch, with a host KV tier: refuse below the server's need plus 1 GiB (the start would hit the OOM killer or swap
# hard), warn below its need plus 3.5 GiB.
LAUNCH_REFUSE_MARGIN_GIB = 1.0
LAUNCH_WARN_MARGIN_GIB = 3.5
RAM_TIER_WITH_EMBEDDING_MIN_GB = 32
# --host-pools (the pinned host tables: the embedding with --system-memory-weights, the host KV tier). ROCm allocates host
# memory as user pointers by default; the kernel's host-memory compaction can move such pages, which invalidates them and
# pauses the process's GPU queues until they are restored (KFD evictions: short stalls under load; 2.7 s over a 3-minute
# 8-request run on the two-GPU Flash Next image, 0.07 s with the setting below, speed unchanged, 10 Oct 2026).
# HSA_USERPTR_FOR_PAGED_MEM=0 makes them driver-owned GTT buffers, which compaction leaves alone. They count against the
# GPU's GTT size (the pin limit above already keeps the tables under it), so auto sets it only when that size holds the
# pinned tables plus HOST_POOLS_GTT_MARGIN_GIB (runtime buffers that move along: staging, queues); without host tables,
# or with an unknown GTT size, ROCm's default stays. On the 27B (11 Oct, gate q27cap, long-kv4 with the 2.4 GiB embedding
# pinned): same speed and greedy outputs, GTT use +2.7 GiB, and no KFD evictions after the start with either setting.
HOST_POOLS_ENV = ('HSA_USERPTR_FOR_PAGED_MEM', '0')
HOST_POOLS_GTT_MARGIN_GIB = 2.0


# Graph capture sizes (--capture-sizes). The image's command captures steps of up to 64 tokens: eight requests decoding
# with seven draft tokens each fill 64, so a step that also admits a new request (its prompt, ~50-250 tokens for a chat
# turn, plus the running requests' 8 tokens each) or reads a short prompt alone ran without a graph. wide adds
# piecewise graphs up to 256 tokens (mixed steps pad to the next size); one-request profiles keep [1, 2, 4, 8] and the
# --vision modes the release sizes (their VRAM budgets were measured with them). The image's compile-cache seed does
# not depend on the sizes (vLLM compiles one graph for every step size up to --max-num-batched-tokens); the opt-in
# --compile-cache keys its saved graphs on them, so its first start with wide compiles once.
CAPTURE_SIZES_RELEASE = [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64]
CAPTURE_SIZES_WIDE = CAPTURE_SIZES_RELEASE + [96, 128, 160, 192, 256]
# Measured 11 Oct 2026 (gate q27cap, long-kv4, published 5 October image, BetterBench concurrency): wide against release
# -0.1 / +0.0 / +1.9 / +0.2 % aggregate at 1 / 2 / 4 / 8 requests, time to first token within 3 ms, graph capture 1 s
# longer, +0.1 GiB VRAM at health: no clear gain, so auto keeps the release sizes; wide stays an opt-in.
CAPTURE_SIZES_WIDE_DEFAULT = False


# The API listens on every interface (the container shares the host network); --host 127.0.0.1 keeps it local.
DEFAULT_HOST = '0.0.0.0'


def release_command(release):
    """Exact Config.Cmd of each immutable image; Docker overrides replace CMD."""
    context, sequences, cache, capture, mamba = {
        '65k': (65536, 8, 6535819798, [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64], 'align'),
        '200k': (200000, 1, 6979321856, [1, 2, 4, 8], 'none'),
    }[release]
    compilation = {'cudagraph_capture_sizes': capture,
                   'pass_config': {'fuse_norm_quant': True, 'fuse_act_quant': True}}
    speculative = {'method': 'dflash', 'model': '/models/draft',
                   'num_speculative_tokens': 7, 'draft_tensor_parallel_size': 1,
                   'attention_backend': 'TRITON_ATTN', 'max_model_len': context,
                   'disable_padded_drafter_batch': True, 'draft_sample_method': 'greedy'}
    return [
        'serve', '/models/target', '--tokenizer', '/models/target',
        '--served-model-name', 'Qwen3.8', '--host', '127.0.0.1', '--port', '18982',
        '--tensor-parallel-size', '1', '--dtype', 'bfloat16',
        '--max-model-len', str(context), '--max-num-seqs', str(sequences),
        '--max-num-batched-tokens', '4096', '--kv-cache-dtype', 'fp8',
        '--kv-cache-memory-bytes', str(cache), '--gpu-memory-utilization', '0.98',
        '--no-enable-prefix-caching', '--enable-chunked-prefill', '--language-model-only',
        '--safetensors-load-strategy', 'lazy', '--attention-backend', 'R4D',
        '--compilation-config', json.dumps(compilation),
        '--speculative-config', json.dumps(speculative), '--mamba-cache-mode', mamba,
        '--mamba-cache-dtype', 'bfloat16', '--mamba-ssm-cache-dtype', 'float16',
        '--no-async-scheduling', '--enable-auto-tool-choice',
        '--tool-call-parser', 'qwen3_coder', '--reasoning-parser', 'qwen3',
        '--override-generation-config', json.dumps({'temperature': 0.7, 'top_p': 0.95, 'top_k': 20}),
        '--seed', '42',
    ]


def long_prefill_threshold_value(value):
    """--long-prefill-threshold: a positive token count, or 'off' (no cap even where the mode has a default)."""
    return 'off' if value == 'off' else positive_integer(value)


LONG_PREFILL_THRESHOLD_CODING = 2048   # the default of --mode long-kv4 (text) since the 6 October launcher update: a long
                                       # prompt is read in 2,048-token steps, so a short request from another user is
                                       # answered in ~0.7 s instead of ~5.9 s during a 64K read, and the long read takes
                                       # about 1 % longer (gate L, 6 Oct 2026, R9700, published 5 October image: 131K NLL
                                       # equal on every bucket, needles 4/4, GSM8K/HumanEval identical, 20-turn coding
                                       # session first tokens 20/20). 'off' restores the previous launcher's 4,096-token
                                       # steps; long-512k and --vision keep their settings (not measured).


def coding_mode_long_prefill_threshold(requested, default=None):
    """The threshold a 4-bit long mode serves with: the user's value, None for 'off', the mode's default when unset."""
    if requested == 'off':
        return None
    return default if requested is None else requested


def text_long_kv4(args):
    """--mode long-kv4, with or without --vision (and the same settings spelled out): 4-bit cache, prefix caching, up to
    262,144 per request; the long-512k mode (context above 262,144) is not it. (Named for the text mode it started with;
    gate V on 6 Oct 2026 qualified the same defaults with images.)"""
    mode = getattr(args, 'mode', None)
    if mode is not None:                      # before apply_mode merged the preset's flags (the tier sizing runs there)
        return mode == 'long-kv4'
    return (args.profile == 'chat' and args.kv_cache == 'kv4'
            and (args.context is None or args.context <= 262144))


def positive_integer(value):
    try:
        result = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError('must be a positive integer') from None
    if result <= 0:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return result


def utilization(value):
    try:
        result = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError('must be a number greater than 0 and less than 1') from None
    if not math.isfinite(result) or not 0 < result < 1:
        raise argparse.ArgumentTypeError('must be greater than 0 and less than 1')
    return result


def host_cache_gib(value):
    try:
        result = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError('must be a number of GiB greater than 0') from None
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError('must be a number of GiB greater than 0')
    return result


def _image_parts(args):
    """(name without digest, digest or '') of the selected image."""
    image = args.image or IMAGES[args.release]
    name, _, digest = image.partition('@')
    return name, digest


def _released_digests(*keys):
    return {IMAGES_BY_RELEASE_DIGEST[key] for key in keys}


def image_is_kv4_v4(args):
    """The 28 and 29 September images: KV4 bundle kv4-v4 (decode up to 200,000 tokens, no prefix-caching check)."""
    name, digest = _image_parts(args)
    return name.endswith(KV4_V4_SUFFIXES) or digest in _released_digests('20260929-r2', '20260928-r1')


def image_predates_kv4_host_fix(args):
    """Images whose connector overlay can end a 4-bit host-tier hit between two recurrent states."""
    name, digest = _image_parts(args)
    return (image_predates_prefix_fix(args) or name.endswith(KV4_HOST_CACHE_UNFIXED_SUFFIXES)
            or digest in _released_digests('20261003-r1'))


def image_predates_kv4_long_vision(args):
    """Images on which --mode long-kv4 --vision was never qualified: the 4 October image and older."""
    name, digest = _image_parts(args)
    return (image_predates_kv4_host_fix(args) or name.endswith(KV4_LONG_VISION_UNQUALIFIED_SUFFIXES)
            or digest in _released_digests('20261004-r1'))


def image_predates_prefix_fix(args):
    """The 2 October release image and older: no seed-column overlay and no KV4 bundle kv4-v6."""
    name, digest = _image_parts(args)
    return (image_is_kv4_v4(args) or name.endswith(KV4_PREFIX_CACHE_UNFIXED_SUFFIXES)
            or digest in _released_digests('20261002-r1', '20261002-r1s'))


def image_has_lmhead_w8(args):
    """Images that carry the native FP8 lm_head bundle (/opt/paiton/runtime/lmhead-w8)."""
    name, _ = _image_parts(args)
    return bool(LMHEAD_W8_IMAGE_SUFFIXES) and name.endswith(LMHEAD_W8_IMAGE_SUFFIXES)


def lm_head_mode(args, weights=None):
    """'fp8' or 'bf16'. auto: fp8 with the 3-bit weights on images that carry the native FP8 head, bf16 otherwise.
    weights: the resolved weights ('w3a4' or 'mxfp4'); callers without it are the 4-bit cache's paths, which serve the
    3-bit weights only."""
    choice = getattr(args, 'lm_head', 'auto') or 'auto'
    if choice == 'fp8' and not image_has_lmhead_w8(args) and not args.image:
        raise ValueError('--lm-head fp8 needs an image with the native FP8 head (pass it with --image)')
    if choice == 'auto':
        three_bit = weights == 'w3a4' if weights is not None else (text_long_kv4(args) or args.kv_cache == 'kv4')
        return 'fp8' if image_has_lmhead_w8(args) and three_bit else 'bf16'
    return choice


def system_memory_weights_bytes(args):
    return SYSTEM_MEMORY_EMBEDDING_BYTES


def system_memory_weights_fit(args):
    """(fits, limit GiB): whether this host can pin the embedding (plus any --host-cache-gib) within its limit."""
    limit = host_cache_limit_gib(True)
    need = system_memory_weights_bytes(args) / 2 ** 30 + (getattr(args, 'host_cache_gib', None) or 0)
    return limit is not None and need <= limit, limit


def mem_available_gib():
    try:
        return next(int(line.split()[1]) for line in PROC_MEMINFO.read_text().splitlines()
                    if line.startswith('MemAvailable:')) / 2 ** 20
    except (OSError, StopIteration, ValueError):
        return None


def gtt_limit_gib():
    """The smallest GTT size of the AMD GPUs (mem_info_gtt_total, GiB), or None when no AMD GPU reports one."""
    sizes = [read_number(path / 'device' / 'mem_info_gtt_total') for path in sorted(SYS_DRM.glob('renderD*'))
             if read_number(path / 'device' / 'vendor') == 0x1002]
    sizes = [size for size in sizes if size > 0]
    return min(sizes) / 2 ** 30 if sizes else None


def host_cache_limit_gib(embedding_in_ram=True):
    """Largest total of pinned system memory (embedding and host KV tier, 0.5 GiB steps) this host allows, or None
    when it cannot be determined: MemTotal less the server's unpinned peak and the reserve, at most the TTM (GTT) limit
    less a margin. PAITON_HOST_PIN_LIMIT_GIB overrides it for a host whose memory is known to be free."""
    override = os.environ.get('PAITON_HOST_PIN_LIMIT_GIB')
    try:
        if override:
            return float(override)
        total = next(int(line.split()[1]) for line in PROC_MEMINFO.read_text().splitlines()
                     if line.startswith('MemTotal:')) / 2 ** 20
        try:
            ttm = int(TTM_PAGES_LIMIT.read_text()) * 4096 / 2 ** 30
        except (OSError, ValueError):
            ttm = 0.0
        # 0 is the kernel default (half of system memory): newer kernels report the default as 0 instead of the
        # computed limit. The amdgpu driver's GTT size is the limit for the GPU's pinned buffers; take the smaller
        limits = [limit for limit in (ttm, gtt_limit_gib()) if limit and limit > 0]
        ttm = min(limits) if limits else total / 2
    except (OSError, StopIteration, ValueError):
        return None
    room = total - SERVER_UNPINNED_PEAK_GIB[embedding_in_ram] - HOST_RESERVE_GIB
    return max(0.0, math.floor(2 * min(room, ttm - HOST_CACHE_SHM_MARGIN_GIB)) / 2)


def host_pools(args):
    """(choice 'gtt' | 'userptr', GTT size GiB or None, pinned host tables GiB) for --host-pools; raises ValueError for
    an explicit gtt that this host's GTT size cannot hold."""
    pinned = ((SYSTEM_MEMORY_EMBEDDING_BYTES / 2 ** 30 if getattr(args, 'system_memory_weights', False) else 0.0)
              + (getattr(args, 'host_cache_gib', None) or 0.0))
    choice = getattr(args, 'host_pools', 'auto') or 'auto'
    gtt = gtt_limit_gib()
    fits = gtt is not None and gtt >= pinned + HOST_POOLS_GTT_MARGIN_GIB
    if choice == 'auto':
        choice = 'gtt' if pinned > 0 and fits else 'userptr'
    elif choice == 'gtt' and gtt is not None and not fits:
        raise ValueError(f'--host-pools gtt: the GPU\'s GTT size ({gtt:.1f} GiB) does not hold the {pinned:.1f} GiB of pinned '
                         f'host tables plus {HOST_POOLS_GTT_MARGIN_GIB:g} GiB; use --host-pools userptr or a smaller tier')
    return choice, gtt, pinned


def pin_limit_text(embedding_in_ram):
    return (f'MemTotal less the server ({SERVER_UNPINNED_PEAK_GIB[embedding_in_ram]:g} GiB with the embedding '
            f'{"in system memory" if embedding_in_ram else "on the GPU"}) and a {HOST_RESERVE_GIB:g} GiB reserve for '
            f'the system and desktop, at most the TTM (GTT) limit less {HOST_CACHE_SHM_MARGIN_GIB:g} GiB')


def tier_bytes_per_token(args):
    """(stored, loaded) bytes per token of the selected image's host tier."""
    name, _ = _image_parts(args)
    return next((value for suffix, value in TIER_BYTES_BY_IMAGE_SUFFIX.items() if name.endswith(suffix)),
                (TIER_STORED_BYTES_PER_TOKEN, TIER_LOADED_BYTES_PER_TOKEN))


def existing_parent(path):
    path = Path(path)
    while not path.exists() and path != path.parent:
        path = path.parent
    return path


def storage_is_rotational(path):
    """True on a spinning disk, False on NVMe/SSD, None when the device cannot be determined (network, virtual or
    pooled file systems)."""
    device = os.stat(existing_parent(path)).st_dev
    try:
        node = (Path(SYS_DEV_BLOCK) / f'{os.major(device)}:{os.minor(device)}').resolve(strict=True)
    except OSError:
        return None
    for folder in (node, node.parent):              # a partition's queue belongs to its disk
        try:
            return (folder / 'queue' / 'rotational').read_text().strip() == '1'
        except OSError:
            continue
    return None


def extend_cache_settings(args, environment):
    """--extend-cache: (the tier flags it stands for, the note that explains the choice)."""
    choice = args.extend_cache
    if getattr(args, 'host_cache_gib', None):
        raise ValueError('--extend-cache sizes the host tier itself; drop --host-cache-gib, or drop --extend-cache and '
                         'set --host-cache-gib and --disk-cache-dir by hand')
    if choice == 'ram' and getattr(args, 'disk_cache_dir', None):
        raise ValueError('--extend-cache ram keeps the cache in system memory; drop --disk-cache-dir')
    stored, loaded = tier_bytes_per_token(args)
    if getattr(args, 'system_memory_weights', False):
        placements = (True,)
    elif getattr(args, 'no_system_memory_weights', False):
        placements = (False,)
    else:
        placements = (True, False)
    tier = {}
    for embedding_in_ram in placements:
        limit = host_cache_limit_gib(embedding_in_ram)
        if limit is None:
            raise ValueError('--extend-cache cannot read the size of this host\'s memory; set --host-cache-gib by hand')
        pinned = system_memory_weights_bytes(args) / 2 ** 30 if embedding_in_ram else 0
        tier[embedding_in_ram] = max(0.0, math.floor(2 * (limit - pinned)) / 2)

    def tokens(gib):
        return int(gib * 2 ** 30 / stored)

    def where(embedding_in_ram):
        return 'in system memory' if embedding_in_ram else 'on the GPU'
    gpu = kv4_pool_tokens(args)
    adds = [e for e in placements if tokens(tier[e]) >= EXTEND_CACHE_RAM_FACTOR * gpu[e]]
    if choice == 'ram' or (choice == 'auto' and adds):
        # the embedding stays in system memory (the larger GPU pool) whenever the tier adds capacity that way; a forced
        # ram choice that adds little takes the largest tier
        e = adds[0] if adds else max(placements, key=lambda e: (tokens(tier[e]), e))
        if tier[e] < EXTEND_CACHE_STAGING_GIB[-1]:
            raise ValueError(f'--extend-cache ram: this host can pin {tier[e]:g} GiB for the cache tier '
                             f'({pin_limit_text(e)}), less than {EXTEND_CACHE_STAGING_GIB[-1]:g} GiB; use '
                             '--extend-cache disk')
        note = (f'--extend-cache {choice}: system memory, a {tier[e]:g} GiB tier (~{tokens(tier[e]):,} tokens) next to '
                f'the GPU pool\'s {gpu[e]:,} tokens, the embedding {where(e)}; no disk tier; pinning it adds about '
                f'{round(HOST_PIN_SECONDS_PER_GIB * tier[e])} s to every start (--host-cache-gib for a smaller tier)')
        return {'extend_cache': None, 'host_cache_gib': tier[e], 'system_memory_weights': e,
                'no_system_memory_weights': not e}, note
    for staging in EXTEND_CACHE_STAGING_GIB:         # the staging tier every disk hit passes through
        e = next((e for e in placements if tier[e] >= staging), None)
        if e is not None:
            break
    else:
        raise ValueError(f'--extend-cache: this host can pin at most {max(tier.values()):g} GiB, not the '
                         f'{EXTEND_CACHE_STAGING_GIB[-1]:g} GiB staging tier the disk cache needs '
                         f'({pin_limit_text(placements[-1])})')
    folder = getattr(args, 'disk_cache_dir', None)
    if not folder:
        cache = environment.get('PAITON_CACHE_DIR')
        if not cache:
            raise ValueError('--extend-cache keeps its disk tier under PAITON_CACHE_DIR/kv-disk; set PAITON_CACHE_DIR '
                             'or --disk-cache-dir')
        folder = str(Path(cache).expanduser().resolve() / 'kv-disk')
    rotational = storage_is_rotational(folder)
    if rotational and not getattr(args, 'disk_cache_allow_hdd', False):
        raise ValueError(f'--extend-cache: {folder} is on a spinning disk; the disk tier needs NVMe or SSD storage '
                         '(point --disk-cache-dir at one), or pass --disk-cache-allow-hdd')
    cap = getattr(args, 'disk_cache_gib', None)
    if cap is None:
        used = folder_bytes(folder) / 2 ** 30 if Path(folder).is_dir() else 0.0
        free = shutil.disk_usage(existing_parent(folder)).free / 2 ** 30
        cap = float(min(EXTEND_CACHE_DISK_MAX_GIB, math.floor(EXTEND_CACHE_DISK_FREE_SHARE * (free + used))))
        if tokens(cap) < EXTEND_CACHE_RAM_FACTOR * gpu[e]:
            raise ValueError(f'--extend-cache: {free:.0f} GiB free under {folder}; a quarter of it holds ~{tokens(cap):,} '
                             f'tokens, under {EXTEND_CACHE_RAM_FACTOR:g}x the GPU pool\'s {gpu[e]:,}. Free space or '
                             'point --disk-cache-dir at a larger NVMe/SSD')
    restore = int(staging * 2 ** 30 / loaded) // 10000 * 10000
    best = max(placements, key=lambda e: tokens(tier[e]))
    note = (f'--extend-cache {choice}: NVMe/SSD under {folder}, up to {cap:g} GiB (~{tokens(cap):,} tokens; checked at '
            'start, an over-full folder is removed then)'
            + (f' (system memory would hold ~{tokens(tier[best]):,} tokens, under {EXTEND_CACHE_RAM_FACTOR:g}x the GPU '
               'pool)' if choice == 'auto' else '')
            + f'; GPU pool {gpu[e]:,} tokens with the embedding {where(e)}; a {staging:g} GiB system-memory staging '
            f'tier: documents up to ~{restore:,} tokens restore from disk'
            + ('; the storage type of that folder is unknown: the disk tier wants NVMe or SSD' if rotational is None
               else ''))
    return {'extend_cache': None, 'host_cache_gib': staging, 'disk_cache_dir': folder, 'disk_cache_gib': cap,
            'disk_cache_auto_wipe': True, 'system_memory_weights': e, 'no_system_memory_weights': not e}, note


def small_ram_tier_warning(args):
    """The warning for an explicit RAM-only tier (--extend-cache ram, or --host-cache-gib without --disk-cache-dir) in the
    4-bit coding mode that holds less than 1.25x the GPU pool: the tier keeps a copy of what the pool holds, so with
    several large documents in turn it rarely serves a hit (5 x 120K documents, 3 GiB tier: none), while the disk
    tier served every re-read. None otherwise (the fp8 tier's space per token is not measured)."""
    if not getattr(args, 'host_cache_gib', None) or getattr(args, 'disk_cache_dir', None) or args.kv_cache != 'kv4':
        return None
    embedding_in_ram = bool(getattr(args, 'system_memory_weights', False))
    tokens = int(args.host_cache_gib * 2 ** 30 / tier_bytes_per_token(args)[0])
    pool = kv4_pool_tokens(args)[embedding_in_ram]
    if tokens >= EXTEND_CACHE_RAM_FACTOR * pool:
        return None
    return (f'a RAM tier smaller than the GPU cache rarely helps with several large documents (this one holds '
            f'~{tokens:,} tokens next to the GPU pool\'s {pool:,}); --extend-cache disk keeps them across the session')


def cache_bytes(value):
    return 'auto' if value == 'auto' else positive_integer(value)


class HelpFormatter(argparse.HelpFormatter):
    """Wraps help texts as usual but keeps their explicit line breaks (one line per --mode preset)."""

    def _split_lines(self, text, width):
        return [line for part in text.splitlines() for line in super()._split_lines(part, width)]


def parser():
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False, formatter_class=HelpFormatter, epilog=(
        'Context includes prompt and generated tokens. '
        'Requests should set chat_template_kwargs.enable_thinking=false to match the reported benchmarks.'))
    result.add_argument('--release', choices=IMAGES, default='65k', help=argparse.SUPPRESS)
    choose = result.add_argument_group('Choose how to run')
    choose.add_argument('--weights', choices=('auto', 'w3a4', 'mxfp4'), default='auto',
                        help='mxfp4 gives you the most accurate weights; w3a4 the 3-bit weights, fastest with the most '
                             'context (extra download in PAITON_W3ROT_DIR); auto (default): w3a4 when PAITON_W3ROT_DIR '
                             'is set, mxfp4 otherwise')
    choose.add_argument('--mode', choices=tuple(MODES), help=MODE_HELP)
    choose.add_argument('--vision', action='store_true',
                        help='gives you image input; works with --mode 65k, long (up to 245,000 context) and long-kv4 '
                             '(262,144 per request, ~496,000 cached tokens, the embedding in system memory), not with '
                             'long-512k. Images up to 3840 x 2160 (4K UHD) are read at full resolution, larger ones are '
                             'downscaled to that many pixels')
    server = result.add_argument_group('Server')
    server.add_argument('--port', type=positive_integer, help='API port (default: 18982)')
    server.add_argument('--host', metavar='ADDRESS',
                        help='address the API listens on (default: 0.0.0.0, every interface; 127.0.0.1 keeps it on '
                             'this machine)')
    server.add_argument('--api-key', metavar='KEY',
                        help='opt-in: the server then requires this key on /v1 requests (handed to Docker as '
                             'VLLM_API_KEY by variable name, never printed); default: no key')
    server.add_argument('--name', help='Docker container name (run-3bit.sh and run-mxfp4.sh: paiton-qwen38)')
    server.add_argument('--detach', action='store_true', help='run Docker in the background')
    server.add_argument('--dry-run', action='store_true',
                        help='print Docker argv as JSON; do not pull or start the image')
    server.add_argument('--ignore-vram-check', action='store_true',
                        help='start even when VRAM is already in use on the selected GPU (the launcher otherwise '
                             'waits up to 30 s for a stopped container to release it and then refuses)')
    server.add_argument('--list-gpus', action='store_true',
                        help='list physical render devices and their GPU numbers without starting Docker')
    server.add_argument('--devices', type=device_list, metavar='GPU',
                        help='the GPU number from --list-gpus to run on (default: the first R9700); replaces the '
                             'ROCR/HIP/CUDA_VISIBLE_DEVICES masks')
    server.add_argument('--image', help='compatible runtime image override; preserves the selected release settings')
    advanced = result.add_argument_group('Advanced tuning',
                                         'Each --mode sets these; flags that contradict it are refused.')
    advanced.add_argument('--context', type=positive_integer, metavar='TOKENS',
                          help='context limit of target and drafter; a smaller value than the mode\'s works. Without '
                               '--mode, above 65536 selects the long-context mode')
    advanced.add_argument('--max-num-seqs', type=positive_integer, metavar='COUNT',
                          help='maximum concurrent requests (1 to 8; the long-context mode defaults to 8 with the '
                               '3-bit weights, 1 with MXFP4)')
    advanced.add_argument('--kv-cache', choices=('auto', 'kv4', 'fp8'), default='auto',
                          help='auto: the 4-bit KV cache with the 3-bit weights in the 65k mode, fp8 otherwise. kv4 '
                               'with a context above 65536 without --mode: the 2 October form of long-kv4 (no prefix '
                               'caching)')
    advanced.add_argument('--prefix-caching', choices=('on', 'off'),
                          help='prefix reuse with materialized recurrent state: on in --mode long, long-kv4 and '
                               'long-512k; off in 65k')
    advanced.add_argument('--thinking', choices=('on', 'off'),
                          help='server default for enable_thinking (off in the long modes and with the 3-bit weights); individual requests may '
                               'override it')
    advanced.add_argument('--long-prefill-threshold', type=long_prefill_threshold_value, metavar='TOKENS|off',
                          help='cap the prefill tokens a long prompt takes per step so short requests answer while it '
                               'is processed. 2048 by default in --mode long-kv4 without --vision (6 Oct 2026, R9700: a '
                               'short request sent during a 64K read answers in 0.7 s instead of 5.9 s, the long read '
                               'takes about 1%% longer, accuracy unchanged); off elsewhere (a long prompt is read in '
                               '4,096-token steps); off restores the previous read behaviour (with --lm-head bf16 the '
                               'start command of the previous launcher)')
    advanced.add_argument('--gdn-state', choices=('auto', 'lazy', 'eager'), default='auto',
                          help='recurrent-state snapshots of the linear-attention layers during speculative decoding: '
                               'lazy keeps one stash per request (fewer cache blocks per request, more KV tokens), '
                               'eager one snapshot per draft token. auto: lazy in the 65k mode, eager in the long '
                               'modes. lazy needs prefix caching off (--mode long-kv4 --prefix-caching off); '
                               'experimental')
    advanced.add_argument('--host-cache-gib', type=host_cache_gib, metavar='GIB',
                          help='experimental: keep up to GIB of evicted prefix-cache blocks in pinned system memory, '
                               'so re-reading a long document restores it over PCIe instead of recomputing it. '
                               'With --mode long, and with --mode long-kv4 on an image with the host-tier alignment '
                               'fix; limited by system memory. Off by default')
    advanced.add_argument('--disk-cache-dir', metavar='DIR',
                          help='experimental, with --host-cache-gib: also keep evicted prefix-cache blocks in files '
                               'under DIR (one folder per image, weights and KV format), so long documents survive a '
                               'restart; bounded by --disk-cache-gib, removed with --wipe-disk-cache')
    advanced.add_argument('--disk-cache-gib', type=host_cache_gib, metavar='GIB',
                          help=f'refuse to start when the --disk-cache-dir folder of this configuration exceeds GIB '
                               f'(default {DISK_CACHE_DEFAULT_GIB:g})')
    advanced.add_argument('--extend-cache', nargs='?', const='auto', choices=('auto', 'ram', 'disk'),
                          help='experimental, --mode long-kv4: extend the prefix cache beyond the GPU. auto (also the '
                               'flag alone) uses system memory where the host has enough of it to add capacity, '
                               'otherwise NVMe/SSD under PAITON_CACHE_DIR/kv-disk (or --disk-cache-dir); ram or disk '
                               'force one. Sizes the host tier, the disk cap and the embedding placement itself and '
                               'prints its choice; a RAM tier adds ~0.65 s per GiB to the start; off by default')
    advanced.add_argument('--disk-cache-allow-hdd', action='store_true',
                          help='let --extend-cache put its disk tier on a spinning disk (slow restores)')
    advanced.add_argument('--host-pools', choices=('auto', 'gtt', 'userptr'), default='auto',
                          help='pinned host tables (the embedding in system memory, the host KV tier): gtt allocates them '
                               'as driver-owned GTT memory, which host-memory compaction cannot move (no brief GPU '
                               'pauses under load); userptr keeps ROCm\'s default. auto: gtt when the GPU\'s GTT size '
                               'holds them')
    advanced.add_argument('--lm-head', choices=('auto', 'bf16', 'fp8'), default='auto',
                          help='fp8: keep the checkpoint\'s FP8 output head (1.19 GiB instead of a 2.37 GiB bf16 '
                               'copy; --mode long-kv4 and --mode long give the difference to the KV cache, long-kv4: '
                               '628,877 instead of 569,878 tokens; the other modes keep it as headroom); needs an image '
                               'with the native FP8 head. auto: fp8 with the 3-bit weights on such images (the 5 '
                               'October image and later), bf16 otherwise')
    advanced.add_argument('--capture-sizes', choices=('auto', 'release', 'wide'), default='auto',
                          help='step sizes that run as captured graphs: release up to 64 tokens (the image\'s '
                               'setting); wide up to 256, so a step that admits a new request runs captured too '
                               '(multi-request text modes). auto: release')
    advanced.add_argument('--compile-cache', action='store_true',
                          help='keep compiled graphs under PAITON_CACHE_DIR so later starts of the same image, weights '
                               'and settings skip compilation; off by default')
    advanced.add_argument('--wipe-disk-cache', action='store_true',
                          help='remove every configuration folder under --disk-cache-dir and exit')
    advanced.add_argument('--no-system-memory-weights', action='store_true',
                          help='keep the input embedding on the GPU in --mode long-kv4 (smaller KV cache)')
    advanced.add_argument('--system-memory-weights', action='store_true',
                          help='keep the input embedding (2.4 GiB) in pinned system memory and give the VRAM to the KV '
                               'cache: the default of --mode long-kv4 and long-512k (two concurrent 262,144-token '
                               'requests); experimental in --mode long without --vision (3-bit long modes only)')
    advanced.add_argument('--profile', choices=('release', 'desktop', 'chat'),
                          help='desktop: 32768 context, 2 GiB KV, one request, 1024 prefill chunks, for a GPU shared '
                               'with a desktop. chat: the long-context mode (--mode long); MXFP4: 200000 context '
                               '(tested to 220000), one request, 1024 prefill chunks, 8 GiB KV. Default: release, or '
                               'chat when --context exceeds 65536 on the 65k image')
    advanced.add_argument('--kv-cache-memory-bytes', type=cache_bytes, metavar='BYTES|auto',
                          help='fixed KV budget in bytes, or automatic sizing from GPU memory utilization')
    advanced.add_argument('--gpu-memory-utilization', type=utilization, metavar='FRACTION',
                          help='automatic memory budget; implies automatic KV sizing unless explicit bytes are '
                               'supplied')
    advanced.add_argument('--max-num-batched-tokens', type=positive_integer, metavar='TOKENS',
                          help='prefill tokens per step (4096; the MXFP4 long mode and desktop: 1024)')
    return result


def read_text(path):
    try:
        return path.read_text().strip()
    except OSError:
        return ''


def read_number(path):
    try:
        return int(read_text(path), 0)
    except ValueError:
        return 0


def kfd_gpu_nodes():
    """{render minor: (architecture, runtime ordinal, uuid)} from the KFD topology, without initializing a GPU
    runtime. The runtime numbers GPU agents in KFD node order, skipping CPU nodes; that ordinal is what
    ROCR_VISIBLE_DEVICES indexes. The uuid is the runtime's 'GPU-%016x' form of unique_id, or None without one."""
    nodes = []
    for path in SYS_KFD.glob('*/properties'):
        properties = dict(line.split(maxsplit=1) for line in read_text(path).splitlines()
                          if len(line.split(maxsplit=1)) == 2)
        try:
            minor = int(properties.get('drm_render_minor', 0))
            architecture = int(properties.get('gfx_target_version', 0))
            simds = int(properties.get('simd_count', 0))
            unique = int(properties.get('unique_id', 0))
        except ValueError:
            continue
        node = int(path.parent.name) if path.parent.name.isdigit() else None
        nodes.append((node, minor, architecture, simds, unique))
    gpus = sorted((n for n in nodes if n[2] > 0 or n[3] > 0), key=lambda n: (n[0] is None, n[0] or 0))
    result = {minor: (architecture, None, None) for _, minor, architecture, _, _ in nodes}
    numbered = all(n[0] is not None for n in gpus)
    for ordinal, (_, minor, architecture, _, unique) in enumerate(gpus):
        result[minor] = (architecture, ordinal if numbered else None, f'GPU-{unique:016x}' if unique else None)
    return result


def discover_gpus():
    """Match DRM and KFD by render minor, without initializing a GPU runtime."""
    runtime = kfd_gpu_nodes()
    devices = []
    for path in sorted(SYS_DRM.glob('renderD*')):
        if not re.fullmatch(r'renderD\d+', path.name):
            continue
        device = path / 'device'
        vendor, identifier = read_number(device / 'vendor'), read_number(device / 'device')
        properties = dict(line.split('=', 1) for line in read_text(device / 'uevent').splitlines() if '=' in line)
        architecture, ordinal, uuid = runtime.get(int(path.name[7:]), (0, None, None))
        vram = read_number(device / 'mem_info_vram_total')
        # The qualified card is a 32 GiB gfx1201 device. The same architecture's
        # smaller consumer cards cannot hold this release's model and cache.
        supported = vendor == 0x1002 and architecture == 120001 and vram >= 30 * 1024**3
        devices.append({'path': '/dev/dri/' + path.name,
                        'pci': properties.get('PCI_SLOT_NAME', device.resolve().name),
                        'vendor': vendor, 'device': identifier, 'vram': vram,
                        'gfx': architecture, 'supported': supported,
                        'ordinal': ordinal if vendor == 0x1002 else None, 'uuid': uuid})
    return devices


def describe_gpu(gpu):
    label = 'R9700 / gfx1201' if gpu['supported'] else {
        0x1002: 'AMD (not supported by this release)',
        0x8086: 'Intel (not supported by this release)',
        0x10de: 'NVIDIA (not supported by this release)',
    }.get(gpu['vendor'], 'not supported by this release')
    vram = f"{gpu['vram'] / 1024**3:.1f} GiB" if gpu['vram'] else 'unknown VRAM'
    ordinal = f"GPU {gpu['ordinal']}" if gpu.get('ordinal') is not None else '-'
    return (f"{ordinal:6s} {gpu['path']}  PCI {gpu['pci']}  {gpu['vendor']:04x}:{gpu['device']:04x}  "
            f"{vram}  {label}")


VISIBILITY_VARIABLES = ('ROCR_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES')


def device_list(value):
    """--devices: comma-separated runtime GPU ordinals as --list-gpus prints them."""
    try:
        result = [int(part) for part in value.split(',')]
    except ValueError:
        raise argparse.ArgumentTypeError('must be GPU numbers from --list-gpus, separated by commas') from None
    if any(item < 0 for item in result):
        raise argparse.ArgumentTypeError('must be GPU numbers from --list-gpus, separated by commas')
    if len(set(result)) != len(result):
        raise argparse.ArgumentTypeError('lists a GPU twice')
    return result


def selected_gpus(requested, devices):
    """The GPUs a start runs on: --devices N, else the first qualified R9700 ([] when the topology has none)."""
    numbered = {gpu['ordinal']: gpu for gpu in devices if gpu.get('ordinal') is not None}
    if requested is None:
        qualified = [numbered[ordinal] for ordinal in sorted(numbered) if numbered[ordinal]['supported']]
        return qualified[:1]
    for ordinal in requested:
        if ordinal not in numbered:
            raise ValueError(f'--devices {ordinal}: no such GPU; --list-gpus prints the GPU numbers')
        if not numbered[ordinal]['supported']:
            raise ValueError(f'--devices {ordinal}: {numbered[ordinal]["path"]} is not a 32 GiB R9700 / gfx1201')
    if len(requested) != 1:
        raise ValueError('--devices takes one GPU: this launcher runs one GPU per server')
    return [numbered[ordinal] for ordinal in requested]


def vram_in_use_bytes(gpu):
    """Bytes of VRAM in use on the GPU per sysfs (mem_info_vram_used of its DRM render device), None when unreadable."""
    try:
        return int(read_text(SYS_DRM / Path(gpu['path']).name / 'device' / 'mem_info_vram_used').strip())
    except (OSError, ValueError, AttributeError, TypeError):
        return None


def vram_preflight(args, environment, command, devices=None, sleep=time.sleep, now=time.monotonic):
    """None when the selected GPU is idle enough for this mode's budget, else the refusal text (after waiting up to
    VRAM_PREFLIGHT_WAIT_S for a just-stopped container to release its memory, announced once on stderr). Skipped when
    the selection follows the visibility environment (no sysfs mapping), when no qualified GPU is known, when sysfs does
    not report the figure, and with --ignore-vram-check."""
    if getattr(args, 'ignore_vram_check', False) or any(v in environment for v in VISIBILITY_VARIABLES):
        return None
    selected = selected_gpus(getattr(args, 'devices', None), discover_gpus() if devices is None else devices)
    if len(selected) != 1:
        return None
    gpu = selected[0]
    used = vram_in_use_bytes(gpu)
    if used is None or used <= VRAM_PREFLIGHT_IDLE_BYTES:
        return None
    print(f"Waiting up to {VRAM_PREFLIGHT_WAIT_S:.0f} s for {used / 1024**3:.1f} GiB of VRAM in use on GPU {gpu['ordinal']} "
          f"({gpu['path']}) to be released (a container that just stopped?)", file=sys.stderr, flush=True)
    deadline = now() + VRAM_PREFLIGHT_WAIT_S
    while now() < deadline:
        sleep(VRAM_PREFLIGHT_POLL_S)
        used = vram_in_use_bytes(gpu)
        if used is None or used <= VRAM_PREFLIGHT_IDLE_BYTES:
            return None
    budget = int(command[command.index('--kv-cache-memory-bytes') + 1]) if '--kv-cache-memory-bytes' in command else None
    if budget is not None:
        fix = (f'pass --kv-cache-memory-bytes {max(budget - used, 0)} (about {used / 1024**3:.1f} GiB less than the '
               f'{budget / 1024**3:.1f} GiB this mode assumes)')
    else:
        fix = 'lower --gpu-memory-utilization'
    return (f"{used / 1024**3:.1f} GiB of VRAM is in use by another process on GPU {gpu['ordinal']} ({gpu['path']}): a display "
            f"or another container? This mode expects an idle card. Stop it, {fix}, or add --ignore-vram-check")


def gpu_selection(args, environment, devices=None):
    """(Docker -e arguments, stderr note) that make exactly the GPUs this server runs on visible in the container.

    The image sets ROCR_VISIBLE_DEVICES=0 and HIP_VISIBLE_DEVICES=0; a bare '-e NAME' for a variable the host does
    not set deletes that default, and a two-GPU host then shows both cards (the one-GPU runtime refuses to start).
    - --devices N: GPU N of --list-gpus (one GPU per server).
    - host visibility variables set, no --devices: forwarded exactly; unset ones lose the image default, so the host
      masks apply to every GPU (unchanged behaviour).
    - neither: the first qualified R9700. Without one (no KFD data), the image default (runtime GPU 0) applies.
    ROCR_VISIBLE_DEVICES takes the GPU's UUID where KFD reports one, otherwise its runtime ordinal; HIP and CUDA then
    number the remaining GPUs from 0."""
    requested = getattr(args, 'devices', None)
    inherited = [variable for variable in VISIBILITY_VARIABLES if variable in environment]
    if requested is not None and inherited:
        raise ValueError('--devices replaces ' + ', '.join(inherited) + '; unset ' +
                         ('it' if len(inherited) == 1 else 'them') + ' or drop --devices')
    if requested is None and inherited:
        result = []
        for variable in VISIBILITY_VARIABLES:
            result += ['-e', variable + '=' + environment[variable] if variable in environment else variable]
        return result, 'GPU selection follows your visibility environment and runtime.'
    selected = selected_gpus(requested, discover_gpus() if devices is None else devices)
    if not selected:
        return [], 'No qualified R9700 found in the KFD topology; the image default (runtime GPU 0) applies.'
    uuids = all(gpu['uuid'] for gpu in selected)
    rocr = ','.join(gpu['uuid'] if uuids else str(gpu['ordinal']) for gpu in selected)
    hip = ','.join(str(index) for index in range(len(selected)))
    note = 'Using ' + ', '.join(f"GPU {gpu['ordinal']} ({gpu['path']}, PCI {gpu['pci']})" for gpu in selected) + \
        (' (the first qualified R9700; choose another with --devices).' if requested is None else '.')
    return ['-e', 'ROCR_VISIBLE_DEVICES=' + rocr, '-e', 'HIP_VISIBLE_DEVICES=' + hip,
            '-e', 'CUDA_VISIBLE_DEVICES=' + hip], note


def replace_value(command, flag, value):
    command[command.index(flag) + 1] = str(value)


def apply_mode(args, environment):
    """--mode as the legacy flags it stands for (MODES), so the Docker argv is identical. An explicit flag that
    contradicts the preset raises ValueError; compatible refinements pass through to the usual checks."""
    mode = getattr(args, 'mode', None)
    if getattr(args, 'extend_cache', None) and mode != 'long-kv4':
        raise ValueError('--extend-cache extends the prefix cache of the coding mode; use it with --mode long-kv4')
    if mode is None:
        return args
    name = f'--mode {mode}'
    if args.release != '65k':
        raise ValueError(f'{name} is a preset of the 65k release image; drop --release {args.release}')
    profile = 'release' if mode == '65k' else 'chat'
    if args.profile not in (None, profile):
        raise ValueError(f'{name} contradicts --profile {args.profile}; use one of them')
    settings = {'mode': None, 'profile': profile}
    if mode == '65k':
        if args.context is not None and args.context > KV4_AUTO_MAX_CONTEXT:
            raise ValueError(f'{name} serves up to --context {KV4_AUTO_MAX_CONTEXT}; for --context {args.context} '
                             'use --mode long or --mode long-kv4')
        if args.prefix_caching == 'on':
            raise ValueError(f'{name} runs without prefix caching; for prefix caching use --mode long')
        return argparse.Namespace(**{**vars(args), **settings})
    weights = weights_mode(args, environment)
    if mode == 'long':
        if args.kv_cache == 'kv4':
            raise ValueError(f'{name} uses the fp8 KV cache; for the 4-bit KV cache use --mode long-kv4')
        if args.prefix_caching == 'off':
            raise ValueError(f'{name} serves with prefix caching, which makes re-reading a document fast; drop '
                             '--prefix-caching off (without prefix caching: --mode long-kv4 --prefix-caching off)')
        if weights != 'w3a4' and args.vision:
            raise ValueError(f'{name} --vision needs the 3-bit W3A4 weights (set PAITON_W3ROT_DIR or use run-3bit.sh); '
                             'with the MXFP4 weights use --vision without --mode long')
        context, limit = mode_context(args, weights)
        if context > limit:
            if weights != 'w3a4':
                reason = (f'with the MXFP4 weights serves one request up to --context {limit} (tested); for more '
                          'context use the 3-bit W3A4 weights (set PAITON_W3ROT_DIR or use run-3bit.sh)')
            elif args.vision:
                reason = (f'--vision serves up to --context {limit}: the vision encoder takes its memory from the KV '
                          'cache. Drop --context or lower it')
            else:
                reason = f'serves up to --context {limit}, the model\'s limit'
            raise ValueError(f'{name} {reason}')
        return argparse.Namespace(**{**vars(args), **settings, 'context': context})
    if args.kv_cache == 'fp8':
        raise ValueError(f'{name} uses the 4-bit KV cache; for the fp8 KV cache use --mode long')
    if getattr(args, 'system_memory_weights', False) and getattr(args, 'no_system_memory_weights', False):
        raise ValueError(f'{name}: --system-memory-weights and --no-system-memory-weights contradict each other')
    if mode == 'long-512k':
        return long_512k_settings(args, weights, name, settings)
    notes = []
    # An image that predates the prefix-cache fix and the host-memory overlays keeps the previous form of the mode:
    # no prefix caching, the embedding on the GPU (an explicit --prefix-caching on is refused by long_kv4_refusal).
    old_image = image_predates_prefix_fix(args)
    if args.prefix_caching is None:
        prefix = None if old_image else 'on'
    else:
        prefix = args.prefix_caching
    if getattr(args, 'extend_cache', None):
        if prefix != 'on':
            raise ValueError(f'{name} --extend-cache keeps evicted prefix-cache blocks; it needs prefix caching (drop '
                             '--prefix-caching off)')
        if image_predates_kv4_host_fix(args):
            raise ValueError(f'{name} --extend-cache: {KV4_HOST_CACHE_REFUSAL}')
        tiers, note = extend_cache_settings(args, environment)
        args = argparse.Namespace(**{**vars(args), **tiers})
        notes.append(note)
    if getattr(args, 'host_cache_gib', None) and weights == 'w3a4' and image_predates_kv4_host_fix(args):
        raise ValueError(f'{name}: {KV4_HOST_CACHE_REFUSAL}')
    if args.vision and not (KV4_LONG_VISION and not image_predates_kv4_long_vision(args)) and weights == 'w3a4':
        raise ValueError(f'{name} is not qualified with --vision; for images use --mode long --vision (up to '
                         f'{W3_LONG_VISION_MAX_CONTEXT:,} tokens)')
    sysmem = getattr(args, 'system_memory_weights', False)
    if not sysmem and not getattr(args, 'no_system_memory_weights', False) and not old_image:
        fits, limit = system_memory_weights_fit(args)
        sysmem = fits
        if not fits and not args.vision:
            need = (f'the 2.4 GiB embedding plus the {args.host_cache_gib:g} GiB host cache together'
                    if getattr(args, 'host_cache_gib', None) else 'the 2.4 GiB the embedding needs')
            notes.append(f'{name}: this host can pin {format(limit, "g") if limit is not None else "an unknown amount of"} GiB of '
                         f'system memory, not {need}; the embedding stays on the GPU and the KV cache is smaller '
                         '(about 452,000 tokens)')
        elif not fits:
            raise ValueError(f'{name} --vision needs {system_memory_weights_bytes(args) / 2 ** 30:.1f} GiB of pinned '
                             f'system memory for the embedding; this host can pin '
                             f'{format(limit, "g") if limit is not None else "an unknown amount of"} GiB. For images use --mode long '
                             f'--vision (up to {W3_LONG_VISION_MAX_CONTEXT:,} tokens)')
    refusal = long_kv4_refusal(argparse.Namespace(**{**vars(args), 'prefix_caching': prefix,
                                                     'system_memory_weights': sysmem}), weights)
    if refusal:
        raise ValueError(f'{name} {refusal}')
    if args.context is not None and args.context > KV4_MAX_CONTEXT:
        raise ValueError(f'{name} serves up to --context {KV4_MAX_CONTEXT} per request, the model\'s native limit; '
                         f'for up to {KV4_YARN_CONTEXT} use --mode long-512k')
    settings['context'] = args.context if args.context is not None else LONG_CONTEXT
    return argparse.Namespace(**{**vars(args), **settings, 'kv_cache': 'kv4', 'prefix_caching': prefix,
                                 'system_memory_weights': sysmem, 'launcher_notes': notes})


def long_512k_settings(args, weights, name, settings):
    """--mode long-512k: long-kv4 with factor-2 position scaling up to 524,288 tokens; system-memory weights are required (the
    pool must hold one full-length request)."""
    if weights != 'w3a4':
        raise ValueError(f'{name} needs the 3-bit W3A4 weights (set PAITON_W3ROT_DIR or use run-3bit.sh)')
    if args.vision:
        raise ValueError(f'{name} is not qualified with --vision; for images use --mode long --vision')
    if args.prefix_caching == 'off':
        raise ValueError(f'{name} serves with prefix caching; drop --prefix-caching off')
    if getattr(args, 'host_cache_gib', None):
        raise ValueError(f'{name}: {KV4_512K_HOST_CACHE_REFUSAL}')
    if getattr(args, 'no_system_memory_weights', False):
        raise ValueError(f'{name} needs the embedding in system memory for its KV cache; drop '
                         '--no-system-memory-weights (or use --mode long-kv4)')
    if image_predates_prefix_fix(args):
        raise ValueError(f'{name} needs an image with KV4 bundle kv4-v6 and the prefix-cache fix, such as the pinned '
                         'release image; this image predates them')
    fits, limit = system_memory_weights_fit(args)
    if not fits:
        raise ValueError(f'{name} needs 2.4 GiB of pinned system memory for the embedding'
                         f'; this host can pin {format(limit, "g") if limit is not None else "an unknown amount of"} GiB '
                         f'({pin_limit_text(True)}). Use --mode long-kv4 (262,144 tokens)')
    context = args.context if args.context is not None else KV4_YARN_CONTEXT
    if context > KV4_YARN_CONTEXT:
        raise ValueError(f'{name} serves up to --context {KV4_YARN_CONTEXT}')
    return argparse.Namespace(**{**vars(args), **settings, 'kv_cache': 'kv4', 'prefix_caching': 'on',
                                 'system_memory_weights': True, 'context': context, 'launcher_notes': []})


def mode_context(args, weights):
    """(context, largest --context) of --mode long: the measured configuration of the weights, the same argv as the
    explicit --context spelling. 3-bit: the model's 262,144 tokens, 245,000 with --vision (the encoder takes its memory
    from the KV cache); MXFP4: one request of 200,000 tokens, tested up to 220,000."""
    if weights != 'w3a4':
        default, limit = MXFP4_LONG_CONTEXT, MXFP4_LONG_MAX_CONTEXT
    elif args.vision:
        default = limit = W3_LONG_VISION_MAX_CONTEXT
    else:
        default = limit = LONG_CONTEXT
    return (args.context if args.context is not None else default), limit


def long_kv4_refusal(args, weights):
    """Why the 4-bit long-context mode cannot be served, or None. The one check behind both spellings of that mode:
    --mode long-kv4 (apply_mode) and the legacy --kv-cache kv4 with a context above 65536 or --profile chat
    (kv4_refusal), with or without --prefix-caching off. The reason reads as the continuation of the mode's name."""
    if weights != 'w3a4':
        return ('needs the 3-bit W3A4 weights, the only weights the 4-bit KV cache is qualified with (set '
                'PAITON_W3ROT_DIR or use run-3bit.sh); with MXFP4 use --mode long (200,000 tokens, one request)')
    if args.prefix_caching == 'on' and image_predates_prefix_fix(args):
        return ('runs without prefix caching on this image: it predates the prefix-cache-hit fix of that mode; drop '
                '--prefix-caching on (--mode long keeps prefix caching), or use an image with the fix')
    if args.vision and not (KV4_LONG_VISION and not image_predates_kv4_long_vision(args)):
        return (f'is not qualified with --vision; for images use --mode long --vision (up to '
                f'{W3_LONG_VISION_MAX_CONTEXT:,} tokens)')
    if args.vision and not getattr(args, 'system_memory_weights', False):
        return ('--vision keeps the embedding and the vision encoder in system memory; drop '
                f'--no-system-memory-weights, or use --mode long --vision (up to {W3_LONG_VISION_MAX_CONTEXT:,} tokens)')
    if image_is_kv4_v4(args):
        return ('needs an image with KV4 bundle kv4-v5, such as the pinned release image; this image carries kv4-v4 '
                f'(decode up to {KV4_V4_MAX_CONTEXT} tokens, no prefix-caching check)')
    return None


def selected_profile(args):
    """--profile as given; without one, a context above the 65k preset selects the long-context chat profile."""
    if args.profile is not None:
        return args.profile
    if args.release == '65k' and args.context is not None and args.context > 65536:
        return 'chat'
    return 'release'


def prefix_caching_enabled(args):
    if args.kv_cache == 'kv4' and args.profile == 'chat':
        return args.prefix_caching == 'on'              # off unless asked for: KV4_LONG_NO_PREFIX_CACHING
    return args.prefix_caching == 'on' or (args.prefix_caching is None and args.profile == 'chat')


def kv4_refusal(args, weights):
    """Why the 4-bit KV cache cannot be served in this configuration, or None where it is qualified: the 3-bit weights
    on the 65k release, without prefix caching up to the image's decode limit, or the long-context mode where
    long_kv4_refusal qualifies it (the same check as --mode long-kv4, with or without --prefix-caching off)."""
    long_mode = args.profile == 'chat'
    if args.release not in KV4_RELEASES or (weights != 'w3a4' and not long_mode):   # long mode: its own reason
        return 'the 4-bit KV cache needs the 65k release with the 3-bit W3A4 weights'
    if long_mode:
        refusal = long_kv4_refusal(args, weights)
        return refusal and 'the 4-bit long-context mode (--mode long-kv4) ' + refusal
    if prefix_caching_enabled(args):
        return ('the 4-bit KV cache runs without prefix caching; drop --prefix-caching on, or use --kv-cache fp8 '
                '(--mode long serves the long-context mode with prefix caching)')
    image = args.image or IMAGES[args.release]
    limit = KV4_V4_MAX_CONTEXT if image_is_kv4_v4(args) else KV4_MAX_CONTEXT
    if args.context is not None and args.context > limit:
        return f'the 4-bit decode path of this image is qualified up to --context {limit}'
    return None


def kv_cache_mode(args, weights):
    """'kv4' or 'fp8'. auto picks kv4 only where it was measured end to end: the 65K release preset with the 3-bit
    weights. An explicit kv4 request is allowed wherever kv4_refusal finds it qualified and refused elsewhere."""
    refusal = kv4_refusal(args, weights)
    if args.kv_cache == 'kv4' and refusal:
        raise ValueError(f'--kv-cache kv4: {refusal}')
    if args.kv_cache in ('kv4', 'fp8'):
        return args.kv_cache
    context = args.context if args.context is not None else 0
    measured = (refusal is None and args.profile == 'release' and not prefix_caching_enabled(args)
                and context <= KV4_AUTO_MAX_CONTEXT)
    return 'kv4' if measured else 'fp8'


def engine_command(args, weights='mxfp4'):
    yarn = (args.context is not None and args.context <= KV4_YARN_CONTEXT and args.kv_cache == 'kv4'
            and args.profile == 'chat' and args.prefix_caching == 'on' and getattr(args, 'system_memory_weights', False))
    if args.context is not None and args.context > 262144 and not yarn:
        raise ValueError('--context exceeds this checkpoint\'s 262144-token model limit')
    if args.max_num_seqs is not None and args.max_num_seqs > 8:
        raise ValueError('--max-num-seqs must be between 1 and 8 for this release')
    if args.port is not None and args.port > 65535:
        raise ValueError('--port must be between 1 and 65535')
    command = release_command(args.release)
    desktop = args.profile == 'desktop'
    chat = args.profile == 'chat'
    long_w3 = chat and weights == 'w3a4'
    compact_graphs = desktop or (chat and not long_w3)
    default_context = 200000 if chat else (32768 if desktop else None)
    context = args.context if args.context is not None else default_context
    if args.vision and args.release not in VISION_RELEASES:
        raise ValueError(f'--vision is not available for the {args.release} release')
    if args.vision and prefix_caching_enabled(args) and not chat:
        raise ValueError('--vision is not qualified with the experimental --prefix-caching on outside the long-context mode')
    if args.vision and chat and not long_w3:
        raise ValueError('--vision in the long-context mode needs the 3-bit W3A4 weights (set PAITON_W3ROT_DIR or '
                         '--weights w3a4); with MXFP4 use it in the 65K mode')
    sysmem = getattr(args, 'system_memory_weights', False)
    if sysmem and not long_w3:
        raise ValueError('--system-memory-weights is qualified in the 3-bit long-context modes only (--mode long, '
                         '--mode long-kv4, --mode long-512k)')
    if sysmem and args.vision and kv_cache_mode(args, weights) != 'kv4':
        raise ValueError('--system-memory-weights is not qualified with --vision in --mode long; drop one of them '
                         f'(--mode long --vision serves up to {W3_LONG_VISION_MAX_CONTEXT:,} tokens)')
    if args.vision and long_w3 and not sysmem and context > W3_LONG_VISION_MAX_CONTEXT:
        raise ValueError(f'--vision in the long-context mode holds up to --context {W3_LONG_VISION_MAX_CONTEXT} '
                         f'(measured cache capacity with the vision encoder loaded)')
    if chat and not long_w3 and context > MXFP4_LONG_MAX_CONTEXT:
        raise ValueError(f'--context {context} needs the 3-bit W3A4 weights: the MXFP4 long-context mode keeps an '
                         f'8 GiB fp8 cache of 231,067 tokens and is tested up to --context {MXFP4_LONG_MAX_CONTEXT}')
    sequences = args.max_num_seqs if args.max_num_seqs is not None else (1 if compact_graphs else None)
    batched_tokens = args.max_num_batched_tokens if args.max_num_batched_tokens is not None else (1024 if compact_graphs else None)
    default_budget = 0.98 if chat else (0.90 if desktop else None)
    budget = args.gpu_memory_utilization if args.gpu_memory_utilization is not None else default_budget
    cache = args.kv_cache_memory_bytes
    if cache is None:
        if args.gpu_memory_utilization is not None:
            cache = 'auto'
        elif args.vision and long_w3 and sysmem and kv_cache_mode(args, weights) == 'kv4':
            cache = (W3_LONG_KV4_VISION_SYSMEM_LMHEAD_W8_CACHE_BYTES if lm_head_mode(args, weights) == 'fp8'
                     else W3_LONG_KV4_VISION_SYSMEM_CACHE_BYTES)
        elif args.vision and long_w3:
            cache = W3_LONG_VISION_KV_CACHE_BYTES
        elif chat and long_w3 and sysmem and context is not None and context > KV4_MAX_CONTEXT:
            cache = W3_LONG_KV4_YARN_CACHE_BYTES
        elif chat and long_w3 and sysmem:
            if kv_cache_mode(args, weights) != 'kv4':
                cache = W3_LONG_SYSMEM_CACHE_BYTES
            elif lm_head_mode(args, weights) == 'fp8':
                cache = W3_LONG_KV4_SYSMEM_LMHEAD_W8_CACHE_BYTES
            else:
                cache = W3_LONG_KV4_SYSMEM_CACHE_BYTES
        elif desktop:
            cache = 2 * 1024**3          # the desktop profile keeps its 2 GiB budget, with or without --vision
        elif chat and long_w3:
            if kv_cache_mode(args, weights) == 'kv4':
                cache = W3_LONG_KV4_CACHE_BYTES
            elif lm_head_mode(args, weights) == 'fp8':      # --mode long (text): the FP8 head's freed VRAM as KV cache
                cache = W3_LONG_LMHEAD_W8_KV_CACHE_BYTES
            else:
                cache = W3_LONG_KV_CACHE_BYTES
        elif chat:
            cache = 8 * 1024**3
        elif args.vision:
            cache = VISION_KV_CACHE_BYTES[weights, kv_cache_mode(args, weights)]
        elif weights == 'w3a4':
            cache = W3_KV4_CACHE_BYTES if kv_cache_mode(args, weights) == 'kv4' else W3_KV_CACHE_BYTES
    for flag, value in (('--max-model-len', context), ('--max-num-seqs', sequences),
                        ('--gpu-memory-utilization', budget), ('--port', args.port),
                        ('--host', getattr(args, 'host', None) or DEFAULT_HOST),
                        ('--max-num-batched-tokens', batched_tokens)):
        if value is not None:
            replace_value(command, flag, value)
    if context is not None:
        index = command.index('--speculative-config') + 1
        speculative = json.loads(command[index])
        speculative['max_model_len'] = context
        command[index] = json.dumps(speculative)
    capture = getattr(args, 'capture_sizes', 'auto') or 'auto'
    if capture == 'wide' and compact_graphs:
        raise ValueError('--capture-sizes wide is for the multi-request modes; this one-request profile captures steps '
                         'of up to 8 tokens')
    if capture == 'wide' and args.vision:
        raise ValueError('--capture-sizes wide is not qualified with --vision (the vision budgets were measured with the '
                         'release sizes); drop one of them')
    if compact_graphs:
        index = command.index('--compilation-config') + 1
        compilation = json.loads(command[index])
        compilation['cudagraph_capture_sizes'] = [1, 2, 4, 8]
        command[index] = json.dumps(compilation)
    elif capture == 'wide' or (capture == 'auto' and CAPTURE_SIZES_WIDE_DEFAULT and not args.vision):
        index = command.index('--compilation-config') + 1
        compilation = json.loads(command[index])
        compilation['cudagraph_capture_sizes'] = CAPTURE_SIZES_WIDE
        command[index] = json.dumps(compilation)
    if cache == 'auto':
        index = command.index('--kv-cache-memory-bytes')
        del command[index:index + 2]
    elif cache is not None:
        replace_value(command, '--kv-cache-memory-bytes', cache)
    if args.vision:
        command.remove('--language-model-only')
        # every vision mode: a cold 4096^2 image (65,536 patches in one encoder pass) took the card to the KFD edge
        # in 65k and long-kv4 alike; 4K UHD screenshots stay at full resolution
        command += ['--mm-processor-kwargs', json.dumps({'max_pixels': VISION_MAX_PIXELS}),
                    '--mm-processor-cache-gb', f'{VISION_MM_CACHE_GIB:g}']
    if prefix_caching_enabled(args):
        command[command.index('--no-enable-prefix-caching')] = '--enable-prefix-caching'
        replace_value(command, '--mamba-cache-mode', 'align')
        if args.kv_cache == 'kv4' and chat:
            command += ['--prefix-cache-retention-interval', str(KV4_PREFIX_RETENTION_INTERVAL)]
    # 3-bit 65K defaults to thinking off too: its published speed rows are measured that way
    thinking = args.thinking if args.thinking is not None else ('off' if chat or weights == 'w3a4' else None)
    if thinking is not None:
        command += ['--default-chat-template-kwargs',
                    json.dumps({'enable_thinking': thinking == 'on'})]
    if chat:
        command.append('--enable-prompt-tokens-details')
    # --mode long-kv4 (text) defaults to the measured threshold; long-512k and the vision modes only take an explicit one
    threshold = (coding_mode_long_prefill_threshold(args.long_prefill_threshold,
                                                    LONG_PREFILL_THRESHOLD_CODING if (text_long_kv4(args) and not args.vision) else None)
                 if (args.kv_cache == 'kv4' and chat)
                 else (None if args.long_prefill_threshold == 'off' else args.long_prefill_threshold))
    if threshold is not None:
        command += ['--long-prefill-token-threshold', str(threshold)]
    if context is not None and context > KV4_MAX_CONTEXT:
        command += ['--hf-overrides', json.dumps({'text_config': {'rope_parameters': YARN_FACTOR_2}})]
    if getattr(args, 'host_cache_gib', None):
        command += ['--kv-offloading-size', f'{args.host_cache_gib:g}', '--kv-offloading-backend', 'native']
    if getattr(args, 'disk_cache_dir', None):
        # blocks on disk must match after a restart: a cryptographic block hash with vLLM's fixed seed (a
        # non-cryptographic one takes a random per-process seed unless PYTHONHASHSEED is set)
        command += ['--prefix-caching-hash-algo', 'sha256']
        command += ['--kv-transfer-config', json.dumps({
            'kv_connector': 'OffloadingConnector', 'kv_role': 'kv_both',
            'kv_connector_extra_config': {'spec_name': 'TieringOffloadingSpec',
                                          'secondary_tiers': [{'type': 'fs', 'root_dir': '/kvdisk'}]}})]
    return command


def weights_mode(args, environment):
    if args.release not in W3_RELEASES:
        if args.weights == 'w3a4':
            raise ValueError(f'--weights w3a4 is not available for the {args.release} release')
        return 'mxfp4'
    if args.weights == 'auto':
        return 'w3a4' if environment.get('PAITON_W3ROT_DIR') else 'mxfp4'
    return args.weights


def disk_cache_fingerprint(args, environment, weights, kv_mode, image):
    """Folder name for --disk-cache-dir: everything that changes the bytes of a stored block."""
    image_id = image
    if not args.dry_run:     # a dry run prints the command without touching Docker
        try:
            image_id = subprocess.run(['docker', 'image', 'inspect', '--format', '{{.Id}}', image],
                                      capture_output=True, text=True, timeout=30).stdout.strip() or image
        except (OSError, subprocess.SubprocessError):
            pass
    folders = {}
    for variable in ('PAITON_TARGET_DIR', 'PAITON_DRAFT_DIR') + (('PAITON_W3ROT_DIR',) if weights == 'w3a4' else ()):
        path = Path(environment.get(variable, '')).expanduser().resolve()
        listing = sorted((f.name, f.stat().st_size, int(f.stat().st_mtime)) for f in path.iterdir()
                         if f.is_file()) if path.is_dir() else []
        folders[variable] = [str(path), listing]
    value = {'format': DISK_CACHE_FORMAT, 'image': image_id, 'weights': weights, 'folders': folders,
             'kv_cache': kv_mode, 'context': args.context, 'release': args.release,
             'gdn_state': getattr(args, 'gdn_state', 'auto'), 'system_memory_weights': args.system_memory_weights}
    digest = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:16]
    return f'qwen38-{digest}', value


def folder_bytes(path):
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += os.lstat(os.path.join(root, name)).st_size
            except OSError:
                pass
    return total


def model_mounts(environment, weights):
    mounts = []
    for variable, destination, writable in (
        ('PAITON_TARGET_DIR', '/models/target', False),
        ('PAITON_DRAFT_DIR', '/models/draft', False),
        ('PAITON_CACHE_DIR', '/cache', True),
    ) + ((('PAITON_W3ROT_DIR', '/models/w3rot', False),) if weights == 'w3a4' else ()):
        setting = environment.get(variable)
        if not setting:
            raise ValueError(f'Set {variable} to an existing {"writable cache" if writable else "checkpoint"} directory')
        path = Path(setting).expanduser().resolve()
        if not path.is_dir():
            raise ValueError(f'{variable} is not an existing directory: {path}')
        if ':' in str(path) or '\n' in str(path):
            raise ValueError(f'{variable} must not contain colons or newlines (Docker volume syntax)')
        if not os.access(path, os.R_OK | os.X_OK | (os.W_OK if writable else 0)):
            raise ValueError(f'{variable} is not {"writable" if writable else "readable"}: {path}')
        mounts += ['-v', f'{path}:{destination}:{"rw" if writable else "ro"}']
    return mounts


def docker_command(args, environment):
    args = apply_mode(args, environment)
    args = argparse.Namespace(**{**vars(args), 'profile': selected_profile(args)})
    name = args.name or f'paiton-qwen38-{args.release}'
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_.-]*', name):
        raise ValueError('--name must be a valid Docker container name')
    image = args.image or IMAGES[args.release]
    if image.startswith('-') or any(c.isspace() for c in image):
        raise ValueError('--image must be a Docker image reference')
    weights = weights_mode(args, environment)
    # validates an explicit --kv-cache kv4 for every release, before the engine's own checks: the legacy spelling of
    # the 4-bit long-context mode gives the same refusal as --mode long-kv4 (long_kv4_refusal)
    kv_mode = kv_cache_mode(args, weights)
    engine = engine_command(args, weights)
    command = ['docker', 'run', '--rm', '--name', name, '--network', 'host',
               '--device', '/dev/kfd', '--device', '/dev/dri',
               '--group-add', 'video', '--ipc', 'host']
    if not args.detach and sys.stdin.isatty() and sys.stdout.isatty():
        command.append('-it')
    command += gpu_selection(args, environment)[0]
    if getattr(args, 'compile_cache', False):
        # the image's compat layer keys the compile-cache folders on every PAITON_ setting and the installed runtime
        command += ['-e', 'PAITON_COMPILE_CACHE=1']
    for variable in ('PAITON_NGRAM_CODRAFT', 'PAITON_NGRAM_CODRAFT_HOT_MATCH'):
        # Opt-in n-gram co-drafting. Forwarded only when set on the host, so the
        # image default (off) applies otherwise.
        if variable in environment:
            command += ['-e', variable + '=' + environment[variable]]
    pinned_weights = 0
    if (getattr(args, 'system_memory_weights', False) or getattr(args, 'host_cache_gib', None)) and \
            image_predates_prefix_fix(args):
        raise ValueError('--system-memory-weights and --host-cache-gib need the compat overlays of the current release '
                         'image; this image predates them')
    if getattr(args, 'system_memory_weights', False):
        pinned_weights = system_memory_weights_bytes(args)
        limit = host_cache_limit_gib(True)
        if limit is None or pinned_weights / 2 ** 30 + (getattr(args, 'host_cache_gib', None) or 0) > limit:
            tier = getattr(args, 'host_cache_gib', None)
            gpu_limit = host_cache_limit_gib(False)
            raise ValueError(f'--system-memory-weights pins {pinned_weights / 2 ** 30:.1f} GiB'
                             + (f' and --host-cache-gib {tier:g} GiB more' if tier else '')
                             + f'; this host can pin {format(limit, "g") if limit is not None else "an unknown amount of"} GiB safely '
                             f'({pin_limit_text(True)})'
                             + (f'. The RAM tier with the embedding in system memory needs about '
                                f'{RAM_TIER_WITH_EMBEDDING_MIN_GB} GB of RAM; here use --no-system-memory-weights (host '
                                f'tier up to {gpu_limit:g} GiB)' if tier and gpu_limit is not None else ''))
        command += ['-e', 'PAITON_HOST_EMBED=1']
    if getattr(args, 'disk_cache_dir', None) and not getattr(args, 'host_cache_gib', None):
        raise ValueError('--disk-cache-dir needs --host-cache-gib: every block passes through the host tier')
    if getattr(args, 'host_cache_gib', None):
        if not prefix_caching_enabled(args):
            raise ValueError('--host-cache-gib keeps evicted prefix-cache blocks; it needs prefix caching (--mode long)')
        if kv_mode == 'kv4' and image_predates_kv4_host_fix(args):
            raise ValueError(KV4_HOST_CACHE_REFUSAL)
        if kv_mode == 'kv4' and args.context is not None and args.context > KV4_MAX_CONTEXT:
            raise ValueError(KV4_512K_HOST_CACHE_REFUSAL)
        embedding_in_ram = bool(getattr(args, 'system_memory_weights', False))
        limit = host_cache_limit_gib(embedding_in_ram)
        if limit is None or args.host_cache_gib + pinned_weights / 2 ** 30 > limit:
            raise ValueError(f'--host-cache-gib {args.host_cache_gib:g} exceeds what this host can pin safely '
                             f'({format(limit, "g") if limit is not None else "unknown"} GiB: {pin_limit_text(embedding_in_ram)})')
        # the region is registered page by page with the GPU driver; the default container memlock limit is too small
        limit_bytes = int((args.host_cache_gib + 1) * 2 ** 30)
        command += ['--ulimit', f'memlock={limit_bytes}:{limit_bytes}', '-e', 'PAITON_HOST_KV_PRIVATE=1']
    if getattr(args, 'disk_cache_dir', None):
        base = Path(args.disk_cache_dir).expanduser().resolve()
        if ':' in str(base) or '\n' in str(base):
            raise ValueError('--disk-cache-dir must not contain colons or newlines (Docker volume syntax)')
        name, value = disk_cache_fingerprint(args, environment, weights, kv_mode, args.image or IMAGES[args.release])
        folder = base / name
        cap = args.disk_cache_gib or DISK_CACHE_DEFAULT_GIB
        used = folder_bytes(folder) / 2 ** 30 if folder.is_dir() else 0.0
        if used > cap:
            if not getattr(args, 'disk_cache_auto_wipe', False):
                raise ValueError(f'--disk-cache-dir {folder} holds {used:.1f} GiB, more than --disk-cache-gib {cap:g}; '
                                 'remove it with --wipe-disk-cache or raise --disk-cache-gib')
            # --extend-cache: the tier has no size limit while it runs, so an over-full folder of this configuration
            # (a cache the launcher owns) is removed at the start; other configurations' folders stay
            print(f'Note: --extend-cache: {folder} holds {used:.1f} GiB, more than its {cap:g} GiB cap; '
                  + ('a real start removes it first' if args.dry_run else 'removing it before the start'),
                  file=sys.stderr)
            if not args.dry_run and subprocess.run(
                    ['docker', 'run', '--rm', '-v', f'{base}:/kvdisk:rw', '--entrypoint', 'rm', image, '-rf',
                     f'/kvdisk/{name}']).returncode != 0:
                raise ValueError(f'could not remove {folder}; remove it with --wipe-disk-cache')
        if not args.dry_run:
            base.mkdir(mode=0o700, parents=True, exist_ok=True)
            folder.mkdir(mode=0o700, exist_ok=True)
            (folder / 'FINGERPRINT.json').write_text(json.dumps(value, indent=1, sort_keys=True) + '\n')
        command += ['-v', f'{folder}:/kvdisk:rw']
        ipc = command.index('--ipc')
        command[ipc + 1:ipc + 2] = ['private', '--shm-size',
                                    f'{math.ceil(args.host_cache_gib + DISK_TIER_SHM_HEADROOM_GIB)}g']
    if host_pools(args)[0] == 'gtt':
        command += ['-e', '='.join(HOST_POOLS_ENV)]      # pinned host tables as GTT buffers (host-memory compaction)
    if args.vision and kv_mode == 'kv4' and args.profile == 'chat' and not KV4_LONG_VISION_DRAFT_FP8_GEMM:
        command += ['-e', 'PAITON_DRAFT_FP8_GEMM=0']          # --mode long-kv4 --vision: ~240 MiB more headroom
    if kv_mode == 'kv4' and args.profile == 'chat' and prefix_caching_enabled(args):
        command += ['-e', 'PAITON_PC_EAGLE_TAIL=1']
        if KV4_SPARSE_ALIGN:
            command += ['-e', 'PAITON_PC_SPARSE_ALIGN=1']
    gdn_state = getattr(args, 'gdn_state', 'auto')
    if gdn_state == 'lazy':
        if prefix_caching_enabled(args):
            raise ValueError('--gdn-state lazy runs without prefix caching (prefix reuse needs a snapshot per draft '
                             'token); drop --gdn-state lazy, or use --mode long-kv4 --prefix-caching off')
        # opt-in: one stash block per request instead of one per draft token (radiance_gdn_lazy.py)
        command += ['-e', 'RADIANCE_GDN_LAZY=1']
    elif (gdn_state == 'eager' or prefix_caching_enabled(args)
          or (args.kv_cache == 'kv4' and args.profile == 'chat')):
        # the 4-bit long mode keeps its measured GDN path while it runs without prefix caching
        command += ['-e', 'RADIANCE_GDN_LAZY=0']
    if args.profile == 'chat' or weights == 'w3a4' or args.vision:
        # The allocator setting the chat profile, the W3A4 KV budget and the vision budgets were measured with.
        allocator = 'max_split_size_mb:64'
        if weights == 'w3a4':                                   # every 3-bit mode, --vision included (6 Oct)
            fraction = W3_MEMORY_FRACTION_LMHEAD_W8 if lm_head_mode(args, weights) == 'fp8' else W3_MEMORY_FRACTION
            allocator += f',per_process_memory_fraction:{fraction}'
        command += ['-e', 'PYTORCH_ALLOC_CONF=' + allocator]
    if weights == 'mxfp4' and args.release in W3_RELEASES:
        # All three flags: the runtime rejects W3A4 prefill without W3 decode.
        for variable in W3_FLAGS:
            command += ['-e', variable + '=0']
    if args.release in KV4_RELEASES:
        state = '1' if kv_mode == 'kv4' else '0'
        for variable in KV4_FLAGS:
            command += ['-e', variable + '=' + state]
    if weights == 'w3a4' and not args.vision:
        command += ['-e', f'PAITON_VRAM_HEADROOM_MIB={VRAM_HEADROOM_MIB}']
    if lm_head_mode(args, weights) == 'fp8':
        command += ['-e', 'PAITON_LMHEAD_W8=1']
    if getattr(args, 'api_key', None):
        # by name only: Docker reads the value from the launcher's environment (main sets it), so it never appears in
        # the argv, a dry run or the process list
        command += ['-e', 'VLLM_API_KEY']
    if args.detach:
        command.append('--detach')
    mounts = model_mounts(environment, weights)
    if args.context is not None and args.context > KV4_MAX_CONTEXT:
        # the drafter's rotary table covers its config's max_position_embeddings: a copy of its config.json with the
        # extended length, mounted over the original (written to the writable cache folder, not in a dry run)
        draft = Path(environment['PAITON_DRAFT_DIR']).expanduser().resolve()
        target = Path(environment['PAITON_CACHE_DIR']).expanduser().resolve() / 'paiton-launcher' / \
            f'draft-config-{args.context}.json'
        if not args.dry_run:
            config = json.loads((draft / 'config.json').read_text())
            config['max_position_embeddings'] = max(int(config.get('max_position_embeddings', 0)), args.context)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(config, indent=2) + '\n')
        mounts += ['-v', f'{target}:/models/draft/config.json:ro']
    return command + mounts + [image] + engine


def main(argv=None):
    arguments = parser()
    args = arguments.parse_args(argv)
    if args.list_gpus:
        devices = discover_gpus()
        print('\n'.join(describe_gpu(gpu) for gpu in devices) or 'No DRM render devices found.')
        return 0
    if args.wipe_disk_cache:
        if not args.disk_cache_dir:
            arguments.error('--wipe-disk-cache needs --disk-cache-dir')
        base = Path(args.disk_cache_dir).expanduser().resolve()
        targets = sorted(p.name for p in base.glob('qwen38-*') if p.is_dir()) if base.is_dir() else []
        if not targets:
            print(f'Nothing to remove under {base}.', file=sys.stderr)
            return 0
        command = ['docker', 'run', '--rm', '-v', f'{base}:/kvdisk:rw', '--entrypoint', 'rm',
                   args.image or IMAGES[args.release], '-rf'] + [f'/kvdisk/{t}' for t in targets]
        if args.dry_run:
            print(json.dumps(command, indent=2))
            return 0
        return subprocess.run(command).returncode
    try:
        given = args
        args = apply_mode(args, os.environ)
        command = docker_command(args, os.environ)
    except ValueError as error:
        arguments.error(str(error))
    print(gpu_selection(args, os.environ)[1], file=sys.stderr)
    if given.mode == 'long' and given.context is None and args.context != LONG_CONTEXT:
        print(f'--mode long: context {args.context:,} tokens ' + (
            '(the vision encoder takes its memory from the KV cache).' if args.vision else
            '(MXFP4 weights, one request; the 3-bit weights serve 262,144).'), file=sys.stderr)
    if args.weights == 'auto' and args.release in W3_RELEASES and weights_mode(args, os.environ) == 'mxfp4':
        print('Serving MXFP4 weights; set PAITON_W3ROT_DIR to the downloaded 3-bit weights for faster decode and prefill.',
              file=sys.stderr)
    if args.kv_cache == 'kv4' and selected_profile(args) == 'chat' and args.prefix_caching is None:
        print('Note: ' + (KV4_LONG_NO_PREFIX_CACHING if image_predates_prefix_fix(args) else
                          KV4_LONG_PREFIX_CACHING_OFF), file=sys.stderr)
    for note in getattr(args, 'launcher_notes', None) or ():
        print('Note: ' + note, file=sys.stderr)
    pools, gtt, pinned = host_pools(args)
    # a note only when GTT buffers are chosen or the GTT size is known to be too small for them (unknown size or an
    # explicit --host-pools userptr: ROCm's default, silently)
    too_small = gtt is not None and pinned > 0 and gtt < pinned + HOST_POOLS_GTT_MARGIN_GIB
    if pools == 'gtt':
        print(f'Note: host pools: {pinned:.1f} GiB of pinned host tables as GTT buffers (HSA_USERPTR_FOR_PAGED_MEM=0; '
              f'GTT size {"unknown" if gtt is None else f"{gtt:.1f} GiB"})', file=sys.stderr)
    elif too_small:
        print(f'Note: host pools: {pinned:.1f} GiB of pinned host tables stay ROCm user pointers (GTT size {gtt:.1f} GiB, '
              f'under the tables plus {HOST_POOLS_GTT_MARGIN_GIB:g} GiB); host-memory compaction can pause the GPU '
              'briefly', file=sys.stderr)
    warning = small_ram_tier_warning(args)
    if warning:
        print('Warning: ' + warning + '.', file=sys.stderr)
    available = mem_available_gib()
    if getattr(args, 'host_cache_gib', None) and available is not None:
        embedding_in_ram = bool(getattr(args, 'system_memory_weights', False))
        need = (SERVER_UNPINNED_PEAK_GIB[embedding_in_ram] + args.host_cache_gib
                + (SYSTEM_MEMORY_EMBEDDING_BYTES / 2 ** 30 if embedding_in_ram else 0))
        fixes = ('close other programs, lower --host-cache-gib'
                 + (', or add --no-system-memory-weights' if embedding_in_ram else ''))
        # a dry run starts nothing: the memory of the moment is a warning there (validators build arms that way)
        if available < need + LAUNCH_REFUSE_MARGIN_GIB and not args.dry_run:
            arguments.error(f'only {available:.1f} GiB of system memory is available; the server with this host tier '
                            f'needs about {need:.1f} GiB plus {LAUNCH_REFUSE_MARGIN_GIB:g} GiB to start '
                            f'({need + LAUNCH_REFUSE_MARGIN_GIB - available:.1f} GiB short): {fixes}')
        if available < need + LAUNCH_WARN_MARGIN_GIB:
            print(f'Warning: only {available:.1f} GiB of system memory is available; the server with this host tier '
                  f'needs about {need:.1f} GiB and should leave {LAUNCH_WARN_MARGIN_GIB:g} GiB free '
                  f'({need + LAUNCH_WARN_MARGIN_GIB - available:.1f} GiB short): {fixes}.', file=sys.stderr)
    elif getattr(args, 'system_memory_weights', False):
        if available is not None and available < SYSTEM_MEMORY_MIN_AVAILABLE_GIB:
            advice = ('Close other programs (--mode long-512k needs the embedding in system memory).'
                      if given.mode == 'long-512k' else
                      'Close other programs, or with --mode long-kv4 add --no-system-memory-weights.')
            print(f'Warning: only {available:.1f} GiB of system memory is available; the server and its pinned '
                  f'buffers want about {SYSTEM_MEMORY_MIN_AVAILABLE_GIB:g} GiB. {advice}', file=sys.stderr)
    if args.dry_run:
        print(json.dumps(command, indent=2))
        return 0
    problem = vram_preflight(args, os.environ, command)
    if problem:
        arguments.error(problem)
    if getattr(args, 'api_key', None):
        os.environ['VLLM_API_KEY'] = args.api_key
    try:
        os.execvp(command[0], command)
    except FileNotFoundError:
        arguments.error('Docker is not installed or is not on PATH')


if __name__ == '__main__':
    sys.exit(main())
