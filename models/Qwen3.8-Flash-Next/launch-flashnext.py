#!/usr/bin/env python3
"""Qwen3.8 Flash Next on two AMD R9700 (ROCm 10, vLLM 0.29, tensor parallel 2) with the Paiton plugin image.

Modes (modes.json, from the measured runs on this image):
  decode             default: MTP depth 3, 98,304-token window, KV 1.6 GiB per card              (decode / concurrency numbers)
  prefill-long       200,000-token window, MTP off, 3 GiB KV per card                            (prefill numbers)
  decode-nopf / prefill-long-nopf   the same with exact-arithmetic prefill (about 7 % slower)
  --prefix-caching   opt-in on the 3-bit modes: align-mode prefix caching (byte-identical hits); 4-bit: --mode 4bit-cache
Host pools: 3-bit modes keep ROCm's user-pointer memory for the host-resident tables (~55 GiB, both ranks; the published behaviour);
  --host-pools gtt (opt-in) allocates them as driver-owned GTT memory (HSA_USERPTR_FOR_PAGED_MEM=0): no memory pauses from host memory
  compaction under load, ~1 GiB more VRAM in decode. 4-bit modes use GTT pools when the GPU's GTT size holds them.
  4bit / 4bit-200k   4-bit experts (W4A8 repo) with the expert host tier: experts beyond --expert-vram-gib per GPU live in
                     pinned host memory (about 41 GiB with the default host pool; the launcher checks the RAM first)
  4bit-cache         opt-in for agents and coding tools: 4bit + prefix caching + an 8 GiB RAM prefix tier (needs ~8 GiB more GTT)
  4bit-long / 4bit-xl  4bit with a 262,144-token window (MTP on): KV 3.5 / 4.5 GiB per GPU (~461K / ~595K tokens), experts
                     19.75 / 18.75 GiB per GPU in VRAM; --prefix-caching refused (not measured)
  --ple-host auto|pinned|pageable  4bit / 4bit-cache: the PLE tables (~49 GiB, both ranks) pinned in GTT (as benchmarked) or in
                     pageable host memory (pinned host memory ~96 -> ~47 GiB, so the GTT pools fit a 128 GB host; needs an image with that
                     runtime). auto (default): pageable when the GPU's GTT limit cannot hold the pinned tables (a host with less than
                     ~200 GB of RAM at the default GTT limit), pinned otherwise; the launcher prints which one it chose and why.
  --vision           opt-in image input on the 3-bit modes and 4bit / 4bit-cache / 4bit-long (off by default: the defaults serve text
                     only, --language-model-only; refused on 4bit-200k and 4bit-xl, not validated there, and on the published images
                     before the vision runtime): the bf16 vision tower (0.42 GiB per GPU), up to 8 images per request, video off, a
                     per-mode pixel cap (4K = 3840x2160 in the 4-bit modes, 2 MP = 1920x1080 in the 3-bit modes; --vision-max-pixels lowers it);
                     >= 1 GiB free VRAM per card is kept: the room comes from the KV pool (3-bit decode serves a 65,536-token window,
                     3-bit prefill-long a 131,072-token window with --vision; modes.json 'vision' blocks); image accuracy
                     (AI2D): bf16 85.3 %, 4-bit 84.7 %, 3-bit 77.7 % (the 3-bit experts cost 7 points on images)
Prerequisites: python3 >= 3.10, docker with GPU device access (/dev/kfd, /dev/dri), the huggingface_hub Python package
  (`pip install huggingface_hub`; the `hf` / `huggingface-cli` binaries are used only as a fallback), two AMD GPUs visible.
Usage: launch-flashnext.py [--mode ...] [--prefix-caching] [--port 18982] [--host 0.0.0.0] [--api-key KEY | env PAITON_API_KEY] [--name paiton-flashnext]
       [--detach] [--dry-run] [--ple-host auto|pinned|pageable] [--vision [--vision-max-pixels 2mp|4k|N]]
       [--weights DIR] [--cache DIR] [--image REF] [--devices GPU-xxx,GPU-yyy] [--served-model-name Qwen3.8-Flash-Next]
Weights: downloaded once from the Hugging Face repo in modes.json (pinned revision, SHA256SUMS verified) into --weights.
"""
import argparse, json, os, pathlib, re, shlex, shutil, subprocess, sys

HERE = pathlib.Path(__file__).resolve().parent
CFG = json.loads((HERE / 'modes.json').read_text()); COMMON = CFG['common']; MODES = CFG['modes']
W3 = COMMON['weights_3bit']   # the 3-bit weights: one pinned revision (common), a default dir keyed by it
W3.update(hf_repo=COMMON['hf_repo'], hf_revision=COMMON['hf_revision'], default_dir=f"{W3['default_dir']}-{COMMON['hf_revision'][:8]}")

SITE_PACKAGES = '/usr/local/lib/python3.12/dist-packages'
DRAFTER_ROOT = '/opt/paiton/drafter'   # D1 (17 Oct): drafter files baked into the image (one layer, before the code)
# verbatim copy of paiton_vllm_plugin/flashnext/zz_paiton_fn_kv_tier.pth (a no-op unless PAITON_FN_KV_TIER=1)
ENV_OVERRIDES = ('PAITON_FN_ROPE_TRIM', 'PAITON_FN_OUTSIDE_RESERVE_MIB', 'HSA_ENABLE_IPC_MODE_LEGACY')   # caller's value wins; unset = the mode or executor default
KV_TIER_PTH = ("import os; os.environ.get('PAITON_FN_KV_TIER', '0') == '1' and os.path.exists('/usr/local/lib/python3.12/dist-packages/paiton_vllm_plugin/flashnext/kv_tier_boot.py') "
               "and __import__('runpy').run_path('/usr/local/lib/python3.12/dist-packages/paiton_vllm_plugin/flashnext/kv_tier_boot.py', run_name='paiton_fn_kv_tier_boot')['install']()")

# admission timing (paiton_vllm_plugin/flashnext/fnadmit.py) at interpreter start, when PAITON_FN_ADMIT is set and not off; an image
# without the module (and no --plugin-overlay of it) logs one line and serves without it
ADMIT_PTH = ("import os, sys; os.environ.get('PAITON_FN_ADMIT', 'off') != 'off' and (os.path.exists('/usr/local/lib/python3.12/dist-packages/paiton_vllm_plugin/flashnext/fnadmit.py') "
             "and __import__('paiton_vllm_plugin.flashnext.fnadmit') or sys.stderr.write('[fnadmit] PAITON_FN_ADMIT is set but this image has no "
             "paiton_vllm_plugin/flashnext/fnadmit.py (pass it with --plugin-overlay): admission timing off\\n'))")

def amd_gpus():
    """ROCR ids of every AMD GPU (vendor 0x1002) from sysfs; no card numbers or PCI addresses are assumed."""
    out = []
    for dev in sorted(pathlib.Path('/sys/class/drm').glob('card[0-9]*/device')):
        try:
            if (dev / 'vendor').read_text().strip().lower() != '0x1002': continue
            uid = (dev / 'unique_id').read_text().strip().lower().replace('0x', '')
            if uid and uid not in out: out.append(uid)
        except OSError: continue
    return ['GPU-' + u for u in out]

def render_gid():
    try: return subprocess.run(['getent', 'group', 'render'], capture_output=True, text=True).stdout.split(':')[2]
    except Exception: return None

def hf_download(repo, revision, dest, patterns=None, dry=False):
    """Download repo files (all, or the allow_patterns) at a pinned revision into dest: huggingface_hub API first, the hf /
    huggingface-cli binaries as a fallback, otherwise one clear line instead of a traceback."""
    what = f'{repo}@{revision[:12]}' + (f' ({len(patterns)} files)' if patterns else ' (all files)')
    if dry: print(f'download (dry run): {what} -> {dest}'); return
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        exe = shutil.which('hf') or shutil.which('huggingface-cli')
        if not exe: sys.exit('missing dependency: the huggingface_hub Python package (pip install huggingface_hub) or the hf command on PATH')
        cmd = [exe, 'download', repo, '--revision', revision, '--local-dir', str(dest)] + sum((['--include', f] for f in (patterns or [])), [])
        print('download:', ' '.join(cmd)); subprocess.run(cmd, check=True); return
    print(f'download: {what} -> {dest}')
    snapshot_download(repo_id=repo, revision=revision, local_dir=str(dest), allow_patterns=patterns)

def sha256_file(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''): h.update(chunk)
    return h.hexdigest()

def verify_sums(d):
    bad, n = [], 0
    for line in (d / 'SHA256SUMS').read_text().splitlines():
        parts = line.split(None, 1)
        if len(parts) != 2: continue
        want, rel = parts[0], parts[1].strip().lstrip('*'); n += 1
        if not (d / rel).exists() or sha256_file(d / rel) != want: bad.append(rel)
    if bad: sys.exit(f'weights: {len(bad)} of {n} files fail SHA256SUMS in {d}: {bad[:5]}')
    print(f'weights: SHA256SUMS verified ({n} files)')

def ensure_weights(d, dry, repo=None, revision=None):
    d = pathlib.Path(d)
    if (d / 'SHA256SUMS').exists() and (d / 'model.safetensors').exists():
        print(f'weights: {d} (present; delete SHA256SUMS to force a re-download, or re-verify with `sha256sum -c SHA256SUMS` there)'); return d
    if repo and not revision:   # a local-only checkpoint (modes.json local_only): never fall back to another repo's revision
        if dry: print(f'weights: {d} (local-only checkpoint; not present)'); return d
        sys.exit(f'weights: {d} has no SHA256SUMS + model.safetensors, and this checkpoint is not downloadable yet (a local dir)')
    hf_download(repo or COMMON['hf_repo'], revision or COMMON['hf_revision'], d, dry=dry)
    if not dry: verify_sums(d)
    return d

def w3_revision_note(d, explicit):
    """3-bit weights only: an explicit --weights dir whose SHA256SUMS is not the pinned revision's gets a warning (its files are
    served as they are); on the default dir, an older un-keyed 3-bit dir gets one line saying it is unused and can be deleted."""
    d = pathlib.Path(d); sums = d / 'SHA256SUMS'
    if explicit:
        if sums.exists() and sha256_file(sums) != COMMON['hf_sha256sums_sha256']:
            print(f'WARNING weights: {d} is not the pinned 3-bit revision {COMMON["hf_revision"][:8]} of {COMMON["hf_repo"]} '
                  f'(its SHA256SUMS differs); serving its files as they are')
        return
    old = pathlib.Path(os.path.expanduser(W3['previous_default_dir']))
    if old != d and (old / 'SHA256SUMS').exists():
        gib = sum(f.stat().st_size for f in old.rglob('*') if f.is_file() and not f.is_symlink()) / 2**30
        print(f'weights: using the pinned 3-bit revision {COMMON["hf_revision"][:8]} in {d}; the older 3-bit weights in {old} '
              f'({gib:.0f} GiB, with its {old.name}-* folders next to it) are no longer used and can be deleted')

def ensure_overlay(d, wspec, dry):
    """weights spec with an 'overlay' (D1 b): the base checkpoint (overlay.of) + the files of the repo folder overlay.subdir at
    wspec's revision, assembled in d: every base file hard-linked (copied across file systems), the folder's files replacing their
    paths, SHA256SUMS = the base list with those lines replaced. The base dir is never modified."""
    ov = wspec['overlay']; d = pathlib.Path(d)
    if (d / 'SHA256SUMS').exists() and (d / 'model.safetensors').exists():
        print(f'weights: {d} (present; assembled from {ov["of"]} + {ov["subdir"]}/)'); return d
    base = COMMON[ov['of']]; bd = ensure_weights(os.path.expanduser(base['default_dir']), dry, base['hf_repo'], base['hf_revision'])
    dl = d.parent / (d.name + '.download'); sub = ov['subdir'].strip('/')
    hf_download(wspec['hf_repo'], wspec['hf_revision'], dl, patterns=[f'{sub}/*', f'{sub}/**'], dry=dry)
    if dry: print(f'weights (dry run): {d} = {bd} + {dl / sub}'); return d
    src = dl / sub; over = {}
    for ln in (src / 'SHA256SUMS').read_text().splitlines():
        h, _, f = ln.strip().partition('  ')
        if f: over[f.removeprefix('./')] = h
    for f, h in over.items():
        if not (src / f).is_file() or sha256_file(src / f) != h: sys.exit(f'weights: {sub}/{f} fails its SHA256SUMS line')
    lines = []
    for ln in (bd / 'SHA256SUMS').read_text().splitlines():
        h, _, f = ln.strip().partition('  ')
        if not f: continue
        rel = f.removeprefix('./'); s_ = src / rel if rel in over else bd / rel
        t = d / rel; t.parent.mkdir(parents=True, exist_ok=True)
        if t.exists() or t.is_symlink(): t.unlink()
        try: os.link(s_, t)
        except OSError: shutil.copy2(s_, t)
        lines.append(f'{over.get(rel, h)}  ./{rel}')
    missing = sorted(set(over) - {l.split('  ./', 1)[1] for l in lines})
    if missing: sys.exit(f'weights: {sub}/ files not in the base checkpoint: {missing[:4]}')
    (d / 'SHA256SUMS').write_text('\n'.join(lines) + '\n')
    print(f'weights: {d} assembled from {bd} + {sub}/ ({len(over)} files replaced, hard links)')
    return d

def baked_target(p):
    """a checkpoint path that is a link to a drafter file baked into the image (resolves inside the container only)"""
    p = pathlib.Path(p)
    return p.is_symlink() and os.readlink(p).startswith(DRAFTER_ROOT + '/')

def ensure_baked(base, wspec, dry):
    """weights spec with 'baked' (D1, 17 Oct): the published checkpoint at base + the drafter files the image carries under
    DRAFTER_ROOT, assembled in <base>-<baked.id>: every base file hard-linked (copied across file systems), each baked file a link
    to its path in the image (it resolves inside the container), SHA256SUMS = the base list with the baked files' lines. The base
    dir is never modified; the image's copies are checked against the same sha256 at build time and by --dry-run."""
    bk = wspec['baked']; base = pathlib.Path(base); d = base.parent / (base.name + '-' + bk['id'])
    if dry and not (base / 'SHA256SUMS').exists():
        print(f'weights (dry run): {d} = {base} + the image drafter {bk["id"]}'); return d
    want = wspec['drafter_sha256']; lines = []; seen = set()
    for ln in (base / 'SHA256SUMS').read_text().splitlines():
        h, _, f = ln.strip().partition('  ')
        if not f: continue
        rel = f.removeprefix('./'); t = d / rel
        if dry: lines.append(rel); seen.add(rel); continue
        t.parent.mkdir(parents=True, exist_ok=True)
        if rel in bk['files']:
            tgt = f'{DRAFTER_ROOT}/{bk["id"]}/{bk["files"][rel]}'
            if not (t.is_symlink() and os.readlink(t) == tgt):
                if t.exists() or t.is_symlink(): t.unlink()
                os.symlink(tgt, t)
            lines.append(f'{want[rel]}  ./{rel}')
        else:
            if not t.exists():
                try: os.link(base / rel, t)
                except OSError: shutil.copy2(base / rel, t)
            lines.append(f'{h}  ./{rel}')
        seen.add(rel)
    missing = sorted(set(bk['files']) - seen)
    if missing:
        msg = f'weights: {base} lacks {missing[:4]} (not the {COMMON[bk["of"]]["hf_repo"]} checkpoint?)'
        if dry: print('WARNING:', msg)
        else: sys.exit(msg)
    if dry: print(f'weights (dry run): {d} = {base} + the image drafter {bk["id"]} ({len(bk["files"])} files)'); return d
    for f in ('config.json',):   # small non-listed files the loaders read next to the shards
        if (base / f).exists() and not (d / f).exists(): shutil.copy2(base / f, d / f)
    (d / 'SHA256SUMS').write_text('\n'.join(lines) + '\n')
    print(f'weights: {d} = {base} + the image drafter {bk["id"]} ({len(bk["files"])} files from {DRAFTER_ROOT}/{bk["id"]}, hard links)')
    return d

def baked_verify(image, wspec):
    """--dry-run: sha256 of every baked drafter file inside the image against modes.json drafter_sha256 (no GPU, ~20 s)"""
    bk = wspec['baked']; want = wspec['drafter_sha256']
    sums = ''.join(f'{want[rel]}  {DRAFTER_ROOT}/{bk["id"]}/{p}\n' for rel, p in bk['files'].items())
    r = subprocess.run(['docker', 'run', '--rm', '-i', '--network', 'none', '--entrypoint', 'sha256sum', image, '-c', '-'],
                       input=sums, capture_output=True, text=True)
    ok = r.returncode == 0
    print(f'drafter: image {image}: {bk["id"]} files {"match" if ok else "DO NOT match"} modes.json drafter_sha256'
          + ('' if ok else ': ' + (r.stdout + r.stderr).strip().replace('\n', '; ')[:300]))
    return ok

def drafter_check(d, wspec, dry, soft=False):
    """Weights with a pinned drafter (modes.json drafter_sha256): the dir's SHA256SUMS must list exactly those drafter files, so a
    stock checkpoint never runs under the distilled drafter's executor manifest (or the other way round)."""
    want = wspec.get('drafter_sha256') if wspec else None
    if not want: return
    sums = pathlib.Path(d) / 'SHA256SUMS'
    have = {}
    if sums.exists():
        for ln in sums.read_text().splitlines():
            h, _, f = ln.strip().partition('  ')
            if f: have[f.removeprefix('./')] = h
    bad = [f for f, h in want.items() if have.get(f) != h]
    if not bad: print(f'drafter: {wspec["drafter"].split(" (")[0]} ({len(want)} files match SHA256SUMS)'); return True
    if soft: return False
    msg = (f'weights {d}: not the {wspec["drafter"].split(" (")[0]} checkpoint ({", ".join(bad)} differ from modes.json drafter_sha256 '
           f'or are missing in SHA256SUMS); pass --weights with that checkpoint, or --drafter stock')
    if dry: print('WARNING:', msg)
    else: sys.exit(msg)


def ple_raw_plan(weights):
    """(meta, shards) of the E4M3 PLE export <weights>/common/ple-e4m3/shard_*.safetensors (codes [n, dim] F8_E4M3 + row_scale [n]
    F32 each): what --ple-host filebacked lays out as two raw files (codes.e4m3 = rows x dim bytes, row_scale.f32 = rows x 4)."""
    src = pathlib.Path(weights) / 'common' / 'ple-e4m3'
    files = sorted(src.glob('shard_*.safetensors'), key=lambda q: int(q.name[6:].split('.')[0]))
    if not files: return None, []
    shards = []; rows = 0; dim = None
    for f in files:
        with open(f, 'rb') as fh:
            n = int.from_bytes(fh.read(8), 'little'); h = json.loads(fh.read(n)); base = 8 + n
        c, r = h['codes'], h['row_scale']
        if c['dtype'] != 'F8_E4M3' or r['dtype'] != 'F32' or c['shape'][0] != r['shape'][0] or (dim is not None and c['shape'][1] != dim):
            sys.exit(f'{f}: unexpected PLE export layout ({c["dtype"]} {c["shape"]}, {r["dtype"]} {r["shape"]})')
        dim = c['shape'][1]; rows += c['shape'][0]
        shards.append((f, base + c['data_offsets'][0], c['data_offsets'][1] - c['data_offsets'][0],
                       base + r['data_offsets'][0], r['data_offsets'][1] - r['data_offsets'][0]))
    meta = {'format': 1, 'rows_total': rows, 'dim': dim, 'shards': [[f.name, f.stat().st_size] for f, *_ in shards]}
    return meta, shards

def ensure_ple_raw(weights, out, dry):
    """--ple-host filebacked: the PLE export as two raw files in <out> (built once, about 50 GB; reused while meta.json matches)."""
    out = pathlib.Path(out); meta, shards = ple_raw_plan(weights)
    if meta is None: sys.exit(f'--ple-host filebacked: no PLE export in {weights}/common/ple-e4m3')
    size = meta['rows_total'] * (meta['dim'] + 4)
    try: have = json.loads((out / 'meta.json').read_text())
    except Exception: have = None
    if have == meta and (out / 'codes.e4m3').stat().st_size == meta['rows_total'] * meta['dim'] \
            and (out / 'row_scale.f32').stat().st_size == meta['rows_total'] * 4:
        print(f'PLE raw files: {out} ({size / 2**30:.1f} GiB, reused)'); return out
    if dry: print(f'PLE raw files: would build {out} ({size / 2**30:.1f} GiB, once)'); return out
    out.mkdir(parents=True, exist_ok=True); free = shutil.disk_usage(out).free
    if free < size + (10 << 30): sys.exit(f'--ple-host filebacked: {out} needs {size / 2**30:.0f} GiB + 10 GiB free disk, has {free / 2**30:.0f} GiB')
    (out / 'meta.json').unlink(missing_ok=True)
    print(f'PLE raw files: building {out} ({size / 2**30:.1f} GiB from {len(shards)} export shards, once)', flush=True)
    for name, pick in (('codes.e4m3', lambda s: s[1:3]), ('row_scale.f32', lambda s: s[3:5])):
        tmp = out / (name + '.tmp')
        with open(tmp, 'wb') as dst:
            for i, sh in enumerate(shards):
                off, n = pick(sh)
                with open(sh[0], 'rb') as src:
                    while n:
                        k = os.copy_file_range(src.fileno(), dst.fileno(), n, off)
                        if k <= 0: sys.exit(f'PLE raw files: copy from {sh[0]} failed')
                        off += k; n -= k
            dst.flush(); os.fsync(dst.fileno())
        tmp.rename(out / name)
    (out / 'meta.json').write_text(json.dumps(meta) + '\n')
    return out

def ensure_base_cfg(d, dry):
    """Base model's config + tokenizer files (pinned revision) next to the weights; the container export does not carry them."""
    d = pathlib.Path(d)
    if all((d / f).exists() for f in COMMON['base_cfg_files']): return d
    hf_download(COMMON['base_model'], COMMON['base_model_revision'], d, patterns=list(COMMON['base_cfg_files']), dry=dry)
    return d

def gtt_limits_gib(rocr_ids):
    """(limit, free) GiB of GTT for the host pools: per chosen AMD GPU (vendor 0x1002, matched by unique id) mem_info_gtt_total, capped
    by TTM's global page limit (ttm.pages_limit when set, else the kernel default of half of MemTotal); free = limit minus the GTT that
    is already in use on that GPU (other processes). The smallest over the GPUs; (None, None) when unreadable."""
    want = {g.lower().replace('gpu-', '') for g in rocr_ids}; devs = []
    for dev in pathlib.Path('/sys/class/drm').glob('card[0-9]*/device'):
        try:
            if (dev / 'vendor').read_text().strip() != '0x1002': continue
            if (dev / 'unique_id').read_text().strip().lower().replace('0x', '') in want:
                devs.append((int((dev / 'mem_info_gtt_total').read_text()), int((dev / 'mem_info_gtt_used').read_text())))
        except (OSError, ValueError): continue
    if not devs: return None, None
    try: ttm = int(pathlib.Path('/sys/module/ttm/parameters/pages_limit').read_text()) * os.sysconf('SC_PAGE_SIZE')
    except (OSError, ValueError): ttm = 0
    if ttm <= 0:
        try: ttm = int(re.search(r'MemTotal:\s+(\d+)', open('/proc/meminfo').read()).group(1)) * 1024 // 2
        except Exception: ttm = 0
    lim = [min(t, ttm) if ttm > 0 else t for t, _ in devs]
    if MEM_CAP_GIB: lim = [min(l, int(MEM_CAP_GIB * (1 << 30)) // 2) for l in lim]   # the kernel default: half the RAM
    return min(lim) / (1 << 30), min(l - u for l, (_, u) in zip(lim, devs)) / (1 << 30)

def host_pool_env(choice, gpus, tier_gib=0.0, pinned_gib=None, ram_tier_gib=0.0, ram_tier_overhead_gib=0.0):
    """Host pools of a start; returns (env, RAM-tier GiB that fits). All pinned host memory of both TP workers is charged to one GPU's GTT
    when HSA_USERPTR_FOR_PAGED_MEM=0. Modes with a measured pinned_gib (4-bit: host tables + expert host tier, GTT at ready) use the ladder:
      GTT free >= pinned + RAM tier + 2 GiB -> GTT pools + the RAM tier;  >= pinned + 2 GiB -> GTT pools, RAM tier shrunk to what fits
      (>= 4 GiB) or off;  less -> ROCm user-pointer pools, no RAM tier, with a warning.
    Other modes: GTT pools when the host tables (+ margin) fit, else user-pointer pools (the published 3-bit behaviour)."""
    hp = COMMON['host_pools']; lim, free = gtt_limits_gib(gpus)
    margin = 2.0 if pinned_gib else hp['gtt_margin_gib']; base = (pinned_gib or hp['gtt_need_gib'] + tier_gib) + margin
    fits = free is not None and free >= base
    if choice == 'gtt' and not fits:
        sys.exit(f'host pools: --host-pools gtt needs {base:.1f} GiB of free GTT on the first GPU, this host has '
                 f'{"an unknown amount" if free is None else f"{free:.1f} GiB (limit {lim:.1f} GiB)"} (use --host-pools userptr)')
    if choice == 'auto': choice = 'gtt' if fits else 'userptr'
    rt = 0.0
    if choice == 'gtt' and ram_tier_gib:
        head = free - base - ram_tier_overhead_gib   # the tier's GTT at load: its size plus the measured overhead (cache mode: ~106 GiB at 8 GiB)
        rt = ram_tier_gib if head >= ram_tier_gib else (head if head >= 4.0 else 0.0)
        if rt < ram_tier_gib: print(f'RAM tier {"shrunk to " + f"{rt:.1f} GiB" if rt else "disabled"}: GTT free {free:.1f} GiB < needed {base + ram_tier_overhead_gib + ram_tier_gib:.1f} GiB')
    print(f'host pools: {choice} (GTT limit {"unknown" if lim is None else f"{lim:.1f} GiB"}, free {"unknown" if free is None else f"{free:.1f} GiB"}, '
          f'needed {base:.1f} GiB for GTT pools)' + ('' if choice == 'gtt' else '; user-pointer pools can stall when the kernel compacts host memory'))
    if choice == 'userptr' and pinned_gib and not fits:   # a chosen userptr layout (--ple-host filebacked, --host-pools userptr) is not a GTT shortfall
        busy = '' if free is None or lim is None or lim - free < 1 else (f' ({lim - free:.0f} GiB of the {lim:.0f} GiB GTT limit is in use by other '
                                                                         'processes: stop other GPU servers first if that is not intended)')
        print(f'WARNING host pools: host GTT {"unknown" if free is None else f"{free:.0f} GiB"} < {base:.0f} GiB{busy}: this mode was benchmarked with its '
              f'~{pinned_gib:.0f} GiB of pinned host memory in GTT (by default the GTT limit is half the RAM, so in practice a '
              f'{128 if pinned_gib < 60 else 256} GB host). Using '
              'pageable pinned (user-pointer) memory instead, without a RAM tier: it starts and runs, but slower than the benchmark (186 vs 192 tok/s, '
              '389 vs 423 tok/s at 8 requests in our validation) and expect GPU stalls of up to several seconds when the kernel compacts host memory. '
              'Raising the GTT limit (amdgpu.gttsize / ttm.pages_limit) is a system setting you may choose; this launcher never changes it.')
    return (dict(hp['env']) if choice == 'gtt' else {}), rt

def pgmigrate_success():
    try: return int(re.search(r'^pgmigrate_success (\d+)$', open('/proc/vmstat').read(), re.M).group(1))
    except Exception: return None

def drop_page_cache(root):
    """posix_fadvise(DONTNEED) on every checkpoint file: drops their clean page-cache pages after the load (frees host memory, so the
    kernel compacts less while the server is serving). Read-only, nothing is written; files in use elsewhere are simply re-read."""
    n = b = 0
    for p in pathlib.Path(root).rglob('*'):
        try:
            if not p.is_file() or p.is_symlink(): continue
            fd = os.open(p, os.O_RDONLY)
            try: os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); n += 1; b += os.fstat(fd).st_size
            finally: os.close(fd)
        except (OSError, AttributeError): continue
    return n, b

def wait_ready(port, proc, name, timeout_s=1800, settle_s=30.0, quiet_s=5.0, rate_max=10000, weights=None, drop_cache=True,
               host='0.0.0.0'):
    """Ready = /health answers, then (after dropping the checkpoint's page cache) the host's page migration rate (/proc/vmstat
    pgmigrate_success) stays below rate_max pages/s for quiet_s seconds, capped at settle_s: the kernel compacts memory right after
    the engine is ready, which can pause the GPU queues for seconds; the first requests then land after it. Returns True when ready."""
    import time, urllib.request
    wildcard = host in ('0.0.0.0', '::', '')            # every interface, loopback included
    probe = '127.0.0.1' if wildcard else (f'[{host}]' if ':' in host else host)
    t0 = time.time()
    while True:
        try:
            with urllib.request.urlopen(f'http://{probe}:{port}/health', timeout=3) as r:
                if r.status == 200: break
        except Exception: pass
        if proc is not None and proc.poll() is not None: return False
        if proc is None and subprocess.run(['docker', 'inspect', '-f', '{{.State.Running}}', name], capture_output=True, text=True).stdout.strip() != 'true': return False
        if time.time() - t0 > timeout_s: print(f'launcher: /health did not answer within {timeout_s} s', flush=True); return False
        time.sleep(2)
    t_health = time.time()
    if drop_cache and weights is not None:
        n, b = drop_page_cache(weights); print(f'launcher: dropped the page cache of {n} checkpoint files ({b / 2**30:.1f} GiB)', flush=True)
    if settle_s > 0 and pgmigrate_success() is not None:
        t1 = time.time(); last = pgmigrate_success(); tl = t1; calm = 0.0; peak = 0.0
        while time.time() - t1 < settle_s:
            time.sleep(0.5); now = time.time(); cur = pgmigrate_success(); rate = (cur - last) / max(now - tl, 1e-3); last, tl = cur, now
            peak = max(peak, rate); calm = calm + 0.5 if rate < rate_max else 0.0
            if calm >= quiet_s: break
        print(f'launcher: settle wait {time.time() - t1:.1f} s (host page migration peak {peak:,.0f} pages/s; quiet = below {rate_max:,} pages/s '
              f'for {quiet_s:g} s, cap {settle_s:g} s)', flush=True)
    print(f'launcher: READY on port {port} ({time.time() - t0:.0f} s after start, {time.time() - t_health:.0f} s after /health)', flush=True)
    print(f'launcher: API http://{probe}:{port}/v1' + (f' (or http://<this-machine-ip>:{port}/v1 from another system)' if wildcard else ''),
          flush=True)
    print('launcher: every 10 s the server log shows "Avg prompt throughput" (reading prompts, prefill) and "Avg generation throughput" '
          '(writing tokens); a long prompt shows up as prompt throughput, not as slow generation (README: Troubleshooting)', flush=True)
    return True

def tier_params(spec):
    """'static:vram_gib=21.6,counts=...,pool=1,core=0.3' -> (kind, {key: value})"""
    kind, _, rest = spec.partition(':')
    return kind, dict(kv.split('=', 1) for kv in rest.split(',') if '=' in kv)

def tier_spec(spec, vram_gib=None, host_pool=None, cold=None):
    kind, kv = tier_params(spec)
    if vram_gib is not None: kv['vram_gib'] = f'{vram_gib:g}'
    if host_pool is not None: kv['pool'] = '1' if host_pool == 'on' else '0'
    if cold: kv['cold'] = f'{cold:g}'
    return kind + ':' + ','.join(f'{k}={v}' for k, v in kv.items())

def pinned_host_gib(spec, expert_gib_total, ranks):
    """Pinned host memory the expert tier allocates (both ranks): pool mode keeps a host copy of every non-core expert;
    without the pool only the experts beyond the VRAM budget live on the host."""
    _, kv = tier_params(spec); per_rank = expert_gib_total / ranks
    if kv.get('pool', '0') == '1': host = per_rank * (1 - float(kv.get('core', '0')) - float(kv.get('cold', '0')))   # cold: in a file
    else: host = max(0.0, per_rank - float(kv.get('vram_gib', per_rank)))
    return host * ranks

MEM_CAP_GIB = None   # --memory: the container's RAM limit; every RAM / GTT figure below is read as if the host had this much

def meminfo_gib(field):
    """/proc/meminfo field (MemTotal, MemAvailable) in GiB, at most --memory; None when unreadable."""
    try: v = int(re.search(rf'{field}:\s+(\d+)', open('/proc/meminfo').read()).group(1)) / (1 << 20)
    except Exception: return None
    return min(v, MEM_CAP_GIB) if MEM_CAP_GIB else v

def kv_tier_auto(kvt, need_gib, margin_gib):
    """--native-kv-tier auto: the RAM prefix tier per GPU from this host's free RAM (MemAvailable, at most --memory) after the mode's
    own resident RAM, its start margin and the executor's host reserve, split over the two ranks; whole GiB, capped at the measured
    size, off below the floor. Returns (GiB per GPU, the formula line)."""
    au = kvt['auto']; avail = meminfo_gib('MemAvailable')
    if avail is None: return 0.0, 'off (/proc/meminfo unreadable)'
    spare = avail - need_gib - margin_gib - au['host_reserve_gib']
    t = float(min(au['max_gib_per_gpu'], max(0, int(spare // COMMON['tp']))))
    f = (f'(available {avail:.0f} - resident {need_gib:.0f} - margin {margin_gib:g} - reserve {au["host_reserve_gib"]:g}) / {COMMON["tp"]} '
         f'= {spare / COMMON["tp"]:.1f}, cap {au["max_gib_per_gpu"]:g}, floor {au["min_gib_per_gpu"]:g}')
    return (t, f) if t >= au['min_gib_per_gpu'] else (0.0, f'off: {f}')

def ram_need_gib(tier_gib, pinned_gib=None, pageable_gib=0.0, filebacked_gib=0.0):
    """Host memory a start keeps resident: pinned (host tables + expert host tier, or the measured pinned total) + pageable PLE tables;
    file-backed PLE tables live in the page cache (evictable) and are not counted."""
    tables = COMMON['host_pools']['gtt_need_gib'] - pageable_gib - filebacked_gib
    return max(tables + tier_gib, pinned_gib or 0.0) + pageable_gib

def ram_check(tier_gib, margin_gib=8.0, pinned_gib=None, pageable_gib=0.0, filebacked_gib=0.0, resident=None):
    """Pinned host memory of a start: the host tables (every mode) plus the expert host tier (4-bit modes), against MemAvailable;
    with --ple-host pageable the PLE tables (part of the host tables) move from pinned to pageable host memory and still need the RAM."""
    tables = COMMON['host_pools']['gtt_need_gib'] - pageable_gib - filebacked_gib
    need_gib = ram_need_gib(tier_gib, pinned_gib, pageable_gib, filebacked_gib)
    avail = meminfo_gib('MemAvailable')
    if resident is not None and avail is not None:
        # measured: what the running server keeps in host RAM with this layout (pinned / GTT + pageable + its processes)
        ok = avail >= resident + RESIDENT_MARGIN_GIB
        print(f'ram: about {resident:.0f} GiB of host RAM in use while serving (measured with this layout; the PLE tables '
              + ('in the page cache are extra and evictable' if filebacked_gib else 'included') + f'); available {avail:.1f} GiB; needed '
              f'{resident + RESIDENT_MARGIN_GIB:.0f} GiB with a {RESIDENT_MARGIN_GIB:g} GiB margin -> {"ok" if ok else "NOT ENOUGH"}')
        return [] if ok else [f'{resident + RESIDENT_MARGIN_GIB:.0f} GiB of available RAM (have {avail:.0f} GiB; with the PLE tables '
                              f'{"file-backed, the smallest measured 4-bit layout" if filebacked_gib else "resident"} this mode keeps about '
                              f'{resident:.0f} GiB in use while serving)']
    if avail is None: print('ram: /proc/meminfo unreadable; not checked'); return []
    ok = avail >= need_gib + margin_gib
    print(f'ram: pinned host memory {need_gib - pageable_gib:.1f} GiB (host tables {tables:g}' + (f' + expert host tier {tier_gib:.1f}' if tier_gib else '') + ', both GPUs)'
          + (f' + pageable PLE tables {pageable_gib:.1f} GiB' if pageable_gib else '')
          + (f'; PLE tables {filebacked_gib:.1f} GiB file-backed (page cache: free RAM beyond this check means fewer disk reads)' if filebacked_gib else '') + '; '
          f'available {avail:.1f} GiB; needed {need_gib + margin_gib:.1f} GiB with a {margin_gib:g} GiB margin -> {"ok" if ok else "NOT ENOUGH"}')
    return [] if ok else [f'{need_gib + margin_gib:.0f} GiB of available RAM for the pinned host memory (have {avail:.0f} GiB'
                          + ')']

def p2p_check(rocr_ids):
    """Warn (never refuse) when the chosen GPUs have no peer-to-peer link in the KFD topology: tensor-parallel traffic then goes
    through host memory and decode/prefill run slower than measured. Uses /sys/class/kfd topology (gpu_id per node, p2p_links) and,
    when readable, the kernel's CONFIG_HSA_AMD_P2P; nothing host-specific is assumed."""
    import gzip
    topo = pathlib.Path('/sys/class/kfd/kfd/topology/nodes')
    try:
        gpu_nodes = {n.name for n in topo.iterdir() if (n / 'gpu_id').read_text().strip() not in ('', '0')}
    except OSError:
        print('p2p: KFD topology unreadable; not checked'); return
    linked = set()
    for n in gpu_nodes:
        for l in (topo / n / 'p2p_links').glob('*/properties'):
            try:
                kv = dict(line.split() for line in l.read_text().splitlines() if len(line.split()) == 2)
                if kv.get('node_to') in gpu_nodes: linked.add(n)
            except OSError: continue
    cfg = None
    for src in (f'/boot/config-{os.uname().release}', '/proc/config.gz'):
        try:
            txt = gzip.open(src, 'rt').read() if src.endswith('.gz') else open(src).read()
            cfg = 'y' if 'CONFIG_HSA_AMD_P2P=y' in txt else 'not set'; break
        except OSError: continue
    if len(gpu_nodes) >= 2 and linked >= gpu_nodes:
        print(f'p2p: GPU peer-to-peer links present ({len(gpu_nodes)} GPU nodes)' + (f'; kernel CONFIG_HSA_AMD_P2P={cfg}' if cfg else ''))
    else:
        print(f'p2p: WARNING: no GPU peer-to-peer link between the GPUs in the KFD topology' + (f' (kernel CONFIG_HSA_AMD_P2P={cfg})' if cfg else '')
              + '; tensor-parallel traffic goes through host memory and both GPUs run slower than our numbers. Check the kernel option, '
              'the BIOS (Above 4G decoding, Resizable BAR) and that both GPUs share a PCIe root without ACS blocking peer access.')

def prerequisites(tp):
    """What a user needs; prints exactly what is missing (also in --dry-run) and stops unless everything is there."""
    missing = []
    if sys.version_info < (3, 10): missing.append(f'python3 >= 3.10 (this is {sys.version.split()[0]})')
    if not shutil.which('docker'): missing.append('docker (the docker command is not on PATH)')
    else:
        r = subprocess.run(['docker', 'info', '--format', '{{.ServerVersion}}'], capture_output=True, text=True)
        if r.returncode != 0: missing.append('docker daemon access (`docker info` failed: ' + (r.stderr.strip().splitlines() or ['no details'])[-1][:100] + ')')
    try: import huggingface_hub  # noqa: F401
    except ImportError:
        if not (shutil.which('hf') or shutil.which('huggingface-cli')): missing.append('the huggingface_hub Python package: pip install huggingface_hub')
    if not os.path.exists('/dev/kfd'): missing.append('/dev/kfd (ROCm kernel driver not loaded)')
    gpus = amd_gpus()
    if len(gpus) < tp: missing.append(f'{tp} AMD GPUs visible (found {len(gpus)}: {gpus})')
    print('prerequisites:', 'all present' if not missing else 'MISSING -> ' + '; '.join(missing))
    return missing

def ensure_view(weights, view, base_cfg, dry, wanted_optional=(), int6=False, vision=False):
    """Runtime view for vLLM (as the benchmarks used): base config with the Paiton quantization block + tokenizer files, and links
    to the container's safetensors as the container will see them (/models/ck/...). Written next to the weights, never inside them."""
    weights, view = pathlib.Path(weights), pathlib.Path(view); ck = COMMON['mounts']['checkpoint']; vdir = COMMON['mounts']['view']
    layers = sorted(p.name for p in (weights / 'layers').glob('L[0-9][0-9].safetensors'))
    links = {n: f'{ck}/layers/{n}' for n in layers}
    links.update({'model.safetensors': f'{ck}/model.safetensors', 'manifest.json': f'{ck}/manifest.json', 'ple-e4m3': f'{ck}/common/ple-e4m3',
                  'mtp-vision.safetensors': f'{ck}/common/bf16-mtp-vision.safetensors', 'mtp-experts.safetensors': f'{ck}/mtp/mtp-experts.safetensors',
                  'mtp-draft-int2/draft-head-int2.safetensors': f'{ck}/mtp/draft-head-int2.safetensors', 'mtp-draft-int2/draft-head-int2.json': f'{ck}/mtp/draft-head-int2.json'})
    # INT6 trunk overlay (top-level i6-LNN files): only for modes that read it (PAITON_FN_INT6_EXTRA). Other modes never see it: a loader
    # without INT6 support fails on those tensors (11 Oct: the published 3-bit image on a weights dir that also holds trunk-int6/)
    for f in (sorted((weights / 'trunk-int6' / 'layers').glob('L[0-9][0-9].safetensors')) if int6 else ()):
        links['i6-' + f.name] = f'{ck}/trunk-int6/layers/{f.name}'
    for k, rel in COMMON.get('view_links_if_present', {}).items():   # files of some exports only (the hybrid 4-bit format map)
        if (weights / rel).exists(): links[k] = f'{ck}/{rel}'
    for k, rel in COMMON.get('view_links_optional', {}).items():   # files only the modes that list them in 'requires' load
        if rel in wanted_optional and ((weights / rel).exists() or baked_target(weights / rel)): links[k] = f'{ck}/{rel}'
    missing = [tgt for tgt in links.values() if not (weights / tgt[len(ck) + 1:]).exists() and not baked_target(weights / tgt[len(ck) + 1:])]
    if missing and dry: print(f'view (dry run): built after the download ({len(missing)} files not present yet)'); return view
    if missing: sys.exit(f'weights incomplete, missing: {missing[:4]}')
    have = {str(x.relative_to(view)) for x in view.rglob('*') if x.is_symlink()} if view.exists() else set()
    if (view / 'config.json').exists() and have == set(links): print(f'view: {view} (present)'); return view
    print(f'view: building {view} ({len(layers)} layers)')
    if dry: return view
    if view.exists(): shutil.rmtree(view)
    view.mkdir(parents=True); (view / 'mtp-draft-int2').mkdir()
    cfg = json.loads((base_cfg / 'config.json').read_text()); cfg['language_model_only'] = not vision   # --vision: its own view dir
    cfg['quantization_config'] = {'quant_method': 'paiton_fnq', 'view_dir': vdir}
    (view / 'config.json').write_text(json.dumps(cfg, indent=2))
    for f in COMMON['base_cfg_files']:
        if f != 'config.json' and (base_cfg / f).exists(): shutil.copy(base_cfg / f, view / f)
    for k, tgt in links.items(): os.symlink(tgt, view / k)
    return view

def set_alloc_fraction(env, frac):
    """PyTorch's per-process VRAM cap in PYTORCH_ALLOC_CONF (per_process_memory_fraction), other allocator options kept."""
    parts = [x for x in env.get('PYTORCH_ALLOC_CONF', '').split(',') if x and not x.startswith('per_process_memory_fraction:')]
    env['PYTORCH_ALLOC_CONF'] = ','.join(parts + [f'per_process_memory_fraction:{frac:g}'])

def kv_pool_check(mode, m, engine_args):
    """Warn when the KV cache is small for how the mode is used (GitHub issue #22): with prefix caching, a cache smaller than twice
    the window means agent requests queue and an evicted conversation is re-processed from the start; a lowered
    --kv-cache-memory-bytes shrinks the cache (and below one window vLLM refuses to start)."""
    g = lambda f: int(engine_args[engine_args.index(f) + 1]) if f in engine_args else None  # noqa: E731
    window, kv_bytes = g('--max-model-len'), g('--kv-cache-memory-bytes')
    mode_bytes = int(m['engine_args'][m['engine_args'].index('--kv-cache-memory-bytes') + 1])
    base = m.get('kv_pool_tokens')
    if not (window and kv_bytes and base):
        return None
    tokens = int(base * kv_bytes / mode_bytes)
    if kv_bytes < mode_bytes:
        print(f'WARNING KV cache: --kv-cache-memory-bytes {kv_bytes:,} is below the {mode} value {mode_bytes:,}: the cache holds about '
              f'{tokens:,} tokens instead of {base:,}' + (f'; that is less than one {window:,}-token window, so the server will refuse to start'
                                                            if tokens < window else ''))
    if '--enable-prefix-caching' in engine_args and tokens < 2 * window:
        print(f'WARNING prefix caching: the KV cache holds about {tokens // 1000}K tokens < 2 x the {window // 1000}K window: agent requests '
              'will queue, and a conversation evicted from the cache re-processes the whole conversation on its next turn'
              + (' (--mode 4bit-cache keeps evicted prefixes in a RAM tier)' if mode.startswith('decode') or mode.startswith('prefill') else ''))
    return tokens


RESIDENT_MARGIN_GIB = 6.0

def resident_gib(m, layout, cold=0.0):
    """Measured host RAM a running server keeps (pinned / GTT + pageable + the processes' own memory) for one PLE layout, or None;
    with cold experts (file-backed layout only) the measured value for that share."""
    if cold:
        return ((m.get('expert_cold') or {}).get('resident_gib') or {}).get(f'{cold:g}')
    return (m.get('resident_gib') or {}).get(layout)

def vision_setup(a, m):
    """--vision: the mode's vision block merged with the common one (None without --vision); the pixel cap after --vision-max-pixels."""
    if not a.vision:
        if a.vision_max_pixels: sys.exit('--vision-max-pixels needs --vision')
        return None
    vm = m.get('vision')
    if not vm: sys.exit(f'--vision is not prepared for --mode {a.mode}')
    if vm.get('refused'): sys.exit(f'--vision is not available on --mode {a.mode}: {vm["refused"]}')
    vis = dict(COMMON['vision']); vis.update(vm)
    cap = int(vm['max_pixels'])
    if a.vision_max_pixels:
        named = {'2mp': 1920 * 1080, '4k': 3840 * 2160}
        want = named.get(str(a.vision_max_pixels).lower()) or (int(a.vision_max_pixels) if str(a.vision_max_pixels).isdigit() else None)
        if not want or want < 64 * 64 * 16: sys.exit(f'--vision-max-pixels {a.vision_max_pixels}: expected 2mp, 4k or a pixel count >= 65536')
        if want > cap: sys.exit(f'--vision-max-pixels {want:,} is above the --mode {a.mode} cap {cap:,} (measured VRAM budget); not raised')
        cap = want
    vis['max_pixels'] = cap
    tokens = cap // 1024                     # smart_resize keeps h x w <= cap on a 32-pixel grid: one token per 32 x 32 pixels
    print(f'image input: on (--vision); at most {cap:,} pixels per image (about {tokens:,} image tokens), {vis["max_images_per_request"]} images per '
          f'request, video off; mode {a.mode}: {vis.get("status", "")}')
    return vis

def vision_image_check(a, image):
    """--vision needs the plugin's vision runtime in the image (modes.json common.vision.image_requirement): the published images
    listed in images_without_vision serve text only, so --vision is refused there; an --image without a digest is not checked."""
    old = COMMON['vision'].get('images_without_vision') or {}
    dig = image.split('@sha256:', 1)[1][:12] if '@sha256:' in image else None
    if dig in old:
        sys.exit(f'--vision is not available with the image {image} ({old[dig]}): {COMMON["vision"]["image_requirement"]}')
    if a.image and dig is None:
        print(f'note image input: --image {a.image} is not checked for the vision runtime; {COMMON["vision"]["image_requirement"]}')

def vision_apply(a, m, vis, engine_args, env):
    """engine args + env for image input; returns (engine_args, the mode dict seen by the KV check)."""
    engine_args = [x for x in engine_args if x not in vis['engine_args_drop']] + list(vis['engine_args_add'])
    engine_args += ['--mm-processor-kwargs', json.dumps({'max_pixels': vis['max_pixels']})]
    pc = '--enable-prefix-caching' in engine_args
    if pc and '--enable-prefix-caching' not in m['engine_args']:
        # --prefix-caching (align mode, 2048-token blocks) holds ~8 % fewer tokens per byte: the window needs its own KV size
        if not vis.get('kv_cache_memory_bytes_prefix_caching'):
            sys.exit(f'--vision with --prefix-caching is not sized for --mode {a.mode} (the window would not fit the KV cache); '
                     'use --vision without --prefix-caching')
        vis = dict(vis, kv_cache_memory_bytes=vis['kv_cache_memory_bytes_prefix_caching'], kv_pool_tokens=vis['kv_pool_tokens_prefix_caching'],
                   kv_delta_gib=vis['kv_delta_gib_prefix_caching'], kv_tokens_lost=vis['kv_tokens_lost_prefix_caching'])
    i = engine_args.index('--kv-cache-memory-bytes'); engine_args[i + 1] = str(vis['kv_cache_memory_bytes'])
    if vis.get('max_model_len'):   # the window the vision tower leaves room for at >= 1 GiB free VRAM (3-bit decode / prefill-long)
        j = engine_args.index('--max-model-len'); text_win = engine_args[j + 1]; engine_args[j + 1] = str(vis['max_model_len'])
        print(f'image input: context window {vis["max_model_len"]:,} tokens (text-only {int(text_win):,}): the vision tower takes the room')
    env.update(vis.get('env', {})); env['PAITON_FN_VISION_MAX_PIXELS'] = str(vis['max_pixels'])
    for k in vis.get('env_drop', ()): env.pop(k, None)   # text-only switches (rope trim: the vision positions need the full rotary cache)
    if pc and vis.get('alloc_fraction_prefix_caching'): set_alloc_fraction(env, vis['alloc_fraction_prefix_caching'])
    elif vis.get('alloc_fraction'): set_alloc_fraction(env, vis['alloc_fraction'])
    if vis.get('nospec_alloc_fraction'): env['PAITON_FN_NOSPEC_ALLOC_FRACTION'] = str(vis['nospec_alloc_fraction'])
    if vis.get('nospec_gib') and m.get('expert_tier'):   # no-spec prefill server (4bit-200k): fewer resident experts give the vision tower room in
        # the allocator pool; the experts beyond the VRAM budget are staged from the host copy (same output, prefill a little slower)
        te = m['expert_tier']['env']; spec = env.get(te, m['expert_tier']['default']); kind, kv = tier_params(spec); was = kv.get('nospec_gib')
        kv['nospec_gib'] = f"{vis['nospec_gib']:g}"; env[te] = kind + ':' + ','.join(f'{k}={v}' for k, v in kv.items())
        print(f'image input: no-spec expert VRAM {vis["nospec_gib"]:g} GiB per GPU (text-only {was}): room for the vision tower in the allocator pool')
    m2 = dict(m); m2['engine_args'] = list(engine_args); m2['kv_pool_tokens'] = vis['kv_pool_tokens']
    print(f'image input: KV cache {vis["kv_cache_memory_bytes"]:,} bytes per GPU (about {vis["kv_pool_tokens"]:,} tokens, {vis["kv_delta_gib"]:+.3f} GiB / '
          f'-{vis["kv_tokens_lost"]:,} tokens vs text-only); allocator {env.get("PYTORCH_ALLOC_CONF", "")}'
          + (f'; no-spec cap {env["PAITON_FN_NOSPEC_ALLOC_FRACTION"]}' if vis.get('nospec_alloc_fraction') else ''))
    return engine_args, m2

def cold_plan(a, m, image, avail):
    """--expert-cold auto on a file-backed start: off when the file-backed layout fits MemAvailable; else the smallest measured share
    that fits (the largest when none does: the RAM check then refuses). Explicit shares must be measured on the mode."""
    ec = m.get('expert_cold') or {}; ci = COMMON.get('expert_cold') or {}
    shares = sorted(float(k) for k in (ec.get('resident_gib') or {}))
    ok_image = any(ref in image for ref in ci.get('images', []))
    if a.expert_cold == 'off': a.expert_cold = 0.0; return
    if a.expert_cold != 'auto':
        x = float(a.expert_cold)
        if x not in shares:
            sys.exit(f'--expert-cold {a.expert_cold} is not measured on --mode {a.mode}'
                     + (f' (measured: {", ".join(f"{v:g}" for v in shares)})' if shares else ''))
        if not ok_image:
            msg = f'--expert-cold needs an image with the cold-expert runtime ({", ".join(ci.get("images", [])) or "none published yet"})'
            if a.dry_run: print('WARNING:', msg)
            else: sys.exit(msg)
        if a.ple_host != 'filebacked':
            sys.exit('--expert-cold needs the file-backed PLE tables (--ple-host filebacked or auto)')
        a.expert_cold = x; a.layout_reason = (a.layout_reason or 'as requested'); return
    a.expert_cold = 0.0
    if a.ple_host != 'filebacked' or not shares or not ok_image or avail is None: return
    if avail >= resident_gib(m, 'filebacked') + RESIDENT_MARGIN_GIB: return
    pick = next((x for x in shares if avail >= resident_gib(m, 'filebacked', x) + RESIDENT_MARGIN_GIB), shares[-1])
    a.expert_cold = pick
    a.layout_reason = (a.layout_reason or '') + (f'; MemAvailable {avail:.0f} GiB < {resident_gib(m, "filebacked") + RESIDENT_MARGIN_GIB:.0f} GiB '
                                                 f'with every expert copy in RAM: the coldest {pick:.0%} of the experts are read from a file')

def pinned_fits(m, gpus):
    """(fits, why): the host holds mode m with the PLE tables pinned (the GTT limit >= the measured pinned total + 2 GiB and
    MemAvailable >= the measured resident RAM + 6 GiB), as low_ram_plan's first choice; unknown limits count as fitting."""
    lim, _ = gtt_limits_gib(gpus); avail = meminfo_gib('MemAvailable')
    need_gtt = m['gtt_pinned_gib'] + 2.0; res = resident_gib(m, 'pinned')
    if lim is not None and lim < need_gtt: return False, f'GTT limit {lim:.0f} GiB < {need_gtt:.0f} GiB'
    if avail is not None and res is not None and avail < res + RESIDENT_MARGIN_GIB: return False, f'MemAvailable {avail:.0f} GiB < {res + RESIDENT_MARGIN_GIB:.0f} GiB'
    return True, ''

def low_ram_plan(a, m, image, gpus, vis=None):
    """--ple-host auto on the 4-bit modes, in this order, the first that fits:
      pinned      as benchmarked: the GTT limit (half of MemTotal by default) holds the measured pinned total + 2 GiB and
                  MemAvailable holds the measured resident RAM + 6 GiB
      pageable    PLE tables in ordinary host memory (~49 GiB less pinned): modes measured with it, an image with the runtime,
                  MemAvailable holds its measured resident RAM + 6 GiB
      filebacked  PLE tables mmapped from raw files (page cache), with user-pointer host pools when GTT is short: modes marked
                  ple_filebacked, an image with the runtime (the RAM check below refuses when even this does not fit)
    With --vision: the vision block's ram_extra_gib on top of every resident figure, and no filebacked (not measured with image input).
    An explicit flag is kept. Sets a.layout_reason (printed once, with the host pools, as the 'memory layout:' line)."""
    a.layout_reason = 'as requested' if a.ple_host != 'auto' else None
    if not m.get('expert_tier'):
        if a.ple_host == 'auto': a.ple_host = 'pinned'; a.layout_reason = None
        return
    if a.ple_host != 'auto': return
    ph = COMMON['ple_host']; fb = ph.get('filebacked') or {}
    lim, _ = gtt_limits_gib(gpus); avail = meminfo_gib('MemAvailable'); tot = meminfo_gib('MemTotal')
    host = f'{tot:.0f} GiB RAM' if tot is not None else 'RAM unknown'
    need_gtt = m['gtt_pinned_gib'] + 2.0
    extra = vis['ram_extra_gib'] if vis else 0.0   # --vision: the vision tower's host memory on top of the measured text-only figures
    fits = lambda layout: avail is None or resident_gib(m, layout) is None or avail >= resident_gib(m, layout) + RESIDENT_MARGIN_GIB + extra
    if (lim is None or lim >= need_gtt) and fits('pinned'):
        a.ple_host = 'pinned'; a.layout_reason = None; return
    why = (f'GTT limit {lim:.0f} GiB < {need_gtt:.0f} GiB for the pinned tables' if lim is not None and lim < need_gtt
           else f'MemAvailable {avail:.0f} GiB < {resident_gib(m, "pinned") + RESIDENT_MARGIN_GIB + extra:.0f} GiB with the tables pinned')
    cand = []
    if 'gtt_pinned_gib_pageable' in m and any(ref in image for ref in ph['images']): cand.append('pageable')
    if m.get('ple_filebacked') and any(ref in image for ref in fb.get('images', [])) and not vis: cand.append('filebacked')
    for layout in cand:
        if fits(layout) or layout == cand[-1]:
            a.ple_host = layout
            more = '' if layout == 'pageable' or 'pageable' not in cand else \
                f'; MemAvailable {avail:.0f} GiB < {resident_gib(m, "pageable") + RESIDENT_MARGIN_GIB + extra:.0f} GiB with them resident'
            a.layout_reason = f'{host}: {why}{more}'
            return
    a.ple_host = 'pinned'
    a.layout_reason = (f'{host}: {why}, but ' + (f'--mode {a.mode} has no measured low-RAM layout yet (use --mode 4bit, 4bit-cache or 4bit-long)'
                       if not ('gtt_pinned_gib_pageable' in m or m.get('ple_filebacked')) else f'{image} has no low-RAM PLE runtime') + '; keeping them pinned')

def cold_preflight(out, share, dry):
    """--expert-cold: the per-GPU cold-expert files (written at the first start, reused after) must fit the disk under <out>."""
    out = pathlib.Path(out); need = share * COMMON['expert_cold']['file_gib_full'] + 10.0
    have = sum(f.stat().st_size for f in out.glob('cold-r*.bin')) / 2 ** 30 if out.is_dir() else 0.0
    d = out
    while not d.exists() and d != d.parent: d = d.parent
    free = shutil.disk_usage(d).free / 2 ** 30 + have
    if free >= need: return
    msg = (f'--expert-cold {share:g}: the cold experts are written once to {out} ({share * COMMON["expert_cold"]["file_gib_full"]:.0f} GiB '
           f'plus 10 GiB headroom = {need:.0f} GiB of free disk); {d} has {free:.0f} GiB. Free disk space there or pass --expert-cold-dir')
    if dry: print('WARNING:', msg)
    else: sys.exit(msg)

def ple_raw_preflight(out, dry):
    """--ple-host filebacked: before anything is downloaded or started, the disk under <out> must hold the raw PLE files (built once)."""
    out = pathlib.Path(out)
    try:
        meta = json.loads((out / 'meta.json').read_text())
        if (out / 'codes.e4m3').stat().st_size == meta['rows_total'] * meta['dim']: return
    except Exception: pass
    need = COMMON['ple_host']['filebacked']['raw_gib'] + 10.0; d = out
    while not d.exists() and d != d.parent: d = d.parent
    free = shutil.disk_usage(d).free / 2 ** 30
    if free >= need: return
    msg = (f'--ple-host filebacked: the PLE tables are read from raw files built once next to the weights '
           f'({COMMON["ple_host"]["filebacked"]["raw_gib"]:.0f} GiB at {out}, plus 10 GiB headroom = {need:.0f} GiB of free disk); '
           f'{d} has {free:.0f} GiB free. Free disk space there or pass --ple-raw-dir <dir on a disk with {need:.0f} GiB free>')
    if dry: print('WARNING:', msg)
    else: sys.exit(msg)

NATIVE_GUARD_PTH = "import os; os.environ.get('PAITON_FN_NATIVE', '0') == '1' and __import__({mod!r})"

def native_spec_tokens(a, m, engine_args):
    """--spec-tokens K (development, SPECN SPEC2 16 Oct): num_speculative_tokens of a native MTP mode (3 = the mode; 4 / 5 = Kmax for the
    per-batch K steps). The executor's kv-spec / kv-meta next to --fnexec-lib must be derived at the same K (the lib refuses a KV spec of
    the other K); refused on prefix-caching modes (align 2048: the K5 ring of 10 does not divide the block)."""
    if a.spec_tokens is None: return
    if '--speculative-config' not in engine_args: sys.exit(f'--spec-tokens: --mode {a.mode} has no MTP')
    if a.spec_tokens != 3 and ('--enable-prefix-caching' in engine_args or '--enable-prefix-caching' in m.get('engine_args', [])):
        sys.exit(f'--spec-tokens: --mode {a.mode} serves prefix caching (2048-token blocks): no K other than the mode\'s 3 yet')
    i = engine_args.index('--speculative-config') + 1
    engine_args[i] = json.dumps(dict(json.loads(engine_args[i]), num_speculative_tokens=a.spec_tokens))
    print(f'MTP: num_speculative_tokens {a.spec_tokens} (development; the lib dir must carry kv files derived at K {a.spec_tokens})')

def native_async_scheduling(a, engine_args):
    """Native worker with MTP: vLLM async scheduling is required (the worker keeps its drafts and hands none to a synchronous scheduler,
    so a sync engine would silently decode with MTP off). Refuses anything that turns it off and passes --async-scheduling explicitly,
    so vLLM fails at start on an incompatible setting instead of quietly falling back. PAITON_FN_ASYNC=0 stays a valid sync-step arm."""
    if '--speculative-config' not in engine_args: return
    spec = json.loads(engine_args[engine_args.index('--speculative-config') + 1])
    backend = engine_args[engine_args.index('--distributed-executor-backend') + 1] if '--distributed-executor-backend' in engine_args else 'mp'
    why = ('--no-async-scheduling' if '--no-async-scheduling' in engine_args else
           'disable_padded_drafter_batch in --speculative-config' if spec.get('disable_padded_drafter_batch') else
           f'--distributed-executor-backend {backend}' if backend not in ('mp', 'uni') else None)
    if why: sys.exit(f'--mode {a.mode}: the native worker with MTP needs vLLM async scheduling; {why} turns it off (refused)')
    if '--async-scheduling' not in engine_args: engine_args.append('--async-scheduling')

def native_manifest_check(a, env, lib, weights=None):
    """The executor manifest next to --fnexec-lib (manifest-<executor mode>.json) is hashed from one checkpoint: its
    container.manifest_sha256 must be the sha256 of <weights>/manifest.json (e.g. the distilled drafter's manifest on the stock
    checkpoint would load the wrong drafter hashes). Skipped when either file is absent (the image's own fnexec dir, a dry run)."""
    if lib is None: return
    # weights: the resolved checkpoint dir (with the image drafter: <published>-b4gh5-mtpft, whose manifest.json links into the image)
    man = lib.parent / f'manifest-{env.get("PAITON_FN_MODE_NAME", a.mode)}.json'; ck = pathlib.Path(weights or a.weights) / 'manifest.json'
    if not (man.is_file() and ck.is_file()): return
    want = (json.loads(man.read_text()).get('container') or {}).get('manifest_sha256')
    import hashlib; have = hashlib.sha256(ck.read_bytes()).hexdigest()
    if not want or want == have: return
    msg = (f'--fnexec-lib {man.name} was built from a checkpoint whose manifest.json is {want[:8]}, but {ck.parent} holds {have[:8]}: '
           f'pass the matching --weights (or --drafter), or the manifest built from this checkpoint')
    if a.dry_run: print('WARNING:', msg)
    else: sys.exit(msg)

def native_apply(a, m, env, engine_args, cache, weights=None):
    """--mode with a native block: --fn-worker native (default) wires the native worker (DESIGN 4.1 / 5): worker class, the three
    guard layers, libpaiton_fnexec and the init args through the environment. --fn-worker torch keeps the torch worker with the
    same configuration (the reference). Edits env / engine_args in place; returns the extra docker mounts."""
    nat = m['native']; worker = a.fn_worker or 'native'
    ref = nat.get('torch_reference_mode')
    if worker == 'torch' and ref:   # long native modes: the torch worker cannot hold their KV at that window
        ref_m = MODES[ref]; ref_kv = int(ref_m['engine_args'][ref_m['engine_args'].index('--kv-cache-memory-bytes') + 1])
        if not (a.kv_cache_memory_bytes and a.kv_cache_memory_bytes <= ref_kv):   # same contract, the reference mode's KV: a gate's like-for-like reference
            sys.exit(f"--mode {a.mode} --fn-worker torch: no torch path at this KV size; the torch reference is --mode {ref} "
                     f"(or this mode's contract with --kv-cache-memory-bytes <= {ref_kv})")
    if worker == 'torch' and nat.get('native_only'):
        sys.exit(f'--mode {a.mode} --fn-worker torch: native only (no torch path for this mode)')
    if worker == 'torch':
        engine_args += list(nat.get('engine_args_torch', []))
        print(f'worker: torch (the reference for --mode {a.mode}; native worker off)')
        return []
    in_image = COMMON[m.get('image_key', 'image')].get('fnexec_in_image', False)   # the image carries /opt/paiton/runtime/fnexec (lib + per-mode files)
    if not a.fnexec_lib and not a.dry_run and not in_image:
        sys.exit(f'--mode {a.mode}: the native worker needs --fnexec-lib <libpaiton_fnexec.so> (this image carries no native executor; '
                 'or --fn-worker torch for the reference)')
    fm = COMMON[m.get('image_key', 'image')].get('fnexec_modes')   # the PAITON_FN_MODE_NAMEs whose manifest / kv files the image carries
    if not a.fnexec_lib and in_image and fm is not None and nat.get('env_native', {}).get('PAITON_FN_MODE_NAME') not in fm:
        sys.exit(f'--mode {a.mode}: the image carries the native executor files of {", ".join(fm)} only; this mode needs --fnexec-lib '
                 '<libpaiton_fnexec.so> with its mode files (or --fn-worker torch)')
    if not a.fnexec_lib and in_image: print(f'native executor: the image\'s {nat["lib_dir_mount"]} (--fnexec-lib overrides it)')
    lib = pathlib.Path(a.fnexec_lib).resolve() if a.fnexec_lib else None
    if lib is not None and not lib.is_file() and not a.dry_run: sys.exit(f'--fnexec-lib {lib}: no such file')
    engine_args += list(nat.get('engine_args_native', [])) + ['--worker-cls', a.native_worker_cls or nat['worker_cls']]
    env.update(nat.get('env_native', {}))
    mbd = (nat.get('manifest_by_drafter') or {}).get(a.drafter or nat.get('drafter_default'))
    if mbd and lib is None and in_image:   # the image's executor dir holds one manifest per drafter (the executor default is manifest-<mode>.json)
        env['PAITON_FN_MANIFEST'] = f"{nat['lib_dir_mount']}/{mbd}"
    alias = COMMON[m.get('image_key', 'image')].get('fnexec_manifest_alias') or {}
    if alias and lib is None and in_image and weights:   # a drop-in checkpoint the image carries its own executor manifests for
        try: ck = sha256_file(pathlib.Path(weights) / 'manifest.json')
        except OSError: ck = None   # the image-drafter view links manifest.json into the image: that plan is the mode's own
        man = env.get('PAITON_FN_MANIFEST', f"{nat['lib_dir_mount']}/manifest-{env.get('PAITON_FN_MODE_NAME', a.mode)}.json")
        if ck in alias and man.rsplit('/', 1)[-1] in alias[ck]['files']:
            env['PAITON_FN_MANIFEST'] = f"{man[:-5]}-{alias[ck]['suffix']}.json"
            print(f'native executor manifest: {env["PAITON_FN_MANIFEST"]} (checkpoint manifest.json {ck[:8]})')
    if a.native_kv_tier:   # opt-in RAM prefix tier through the native executor (fnexec_kvt_*): vLLM's OffloadingConnector
        kvt = nat['kv_tier']; cfg = json.loads(json.dumps(kvt['kv_transfer_config']))   # scheduler side unchanged, the glue's spec swaps the worker
        cfg['kv_connector_extra_config']['cpu_bytes_to_use'] = int(round(2 * a.native_kv_tier * (1 << 30)))   # total of both ranks, as CPUOffloadingSpec
        engine_args += ['--kv-transfer-config', json.dumps(cfg)]; env.update(kvt['env']); env['PAITON_FN_KVT_LAYOUT'] = a.native_kv_tier_layout
        print(f'native RAM prefix tier: {a.native_kv_tier:g} GiB per GPU ({2 * a.native_kv_tier:g} GiB of host RAM, {a.native_kv_tier_layout}); '
              + (f'auto {a.native_kv_tier_why}' if getattr(a, 'native_kv_tier_why', None) else 'as requested'))
        if kvt.get('trade'): print(f'native RAM prefix tier: {kvt["trade"]}')
    native_spec_tokens(a, m, engine_args)
    native_async_scheduling(a, engine_args)
    native_manifest_check(a, env, lib, weights)
    real = env.get('ROCR_VISIBLE_DEVICES', '')   # layer 1: no process sees a GPU; the executor restores the real list before hipInit
    env['PAITON_FN_GPU_MASK'] = real; env['ROCR_VISIBLE_DEVICES'] = nat['mask_invalid']
    budget = a.card_budget_gib if a.card_budget_gib is not None else float(nat['card_budget_gib'])
    env['PAITON_FN_CARD_BUDGET_GIB'] = f'{budget:g}'; env['PAITON_FN_GUARD'] = a.native_guard or nat['guard']
    # the library's directory is mounted whole: the per-mode files (manifest-<mode>.json, kv-spec / kv-meta-<mode>.json) travel with the .so
    env['PAITON_FN_FNEXEC_LIB'] = f"{nat['lib_dir_mount']}/{lib.name if lib else 'libpaiton_fnexec.so'}"; env['PAITON_FN_NATIVE_LOG'] = nat['log_mount']
    env['PAITON_FN_HOST_LAYOUT'] = a.ple_host if a.ple_host in ('pinned', 'pageable', 'filebacked') else 'pinned'   # fnexec_load_args.host_layout
    for kv in a.native_env:   # development switches of the executor (opt-ins); only PAITON_FN_* names
        k, sep, v = kv.partition('=')
        if not sep or not re.fullmatch(r'PAITON_FN_[A-Z0-9_]+', k): sys.exit(f'--native-env {kv!r}: expected PAITON_FN_<NAME>=<value>')
        env[k] = v
    log = pathlib.Path(a.native_log_dir).resolve() if a.native_log_dir else cache.with_name(cache.name + '-native-log')
    hook = cache.with_name(cache.name + '-native-guard.pth')   # layer 2 at interpreter start in every process (next to the cache, as the KV-tier hook)
    if not a.dry_run:
        log.mkdir(parents=True, exist_ok=True); hook.write_text(NATIVE_GUARD_PTH.format(mod=nat['guard_module']) + '\n')
    print(f'worker: native ({a.native_worker_cls or nat["worker_cls"]}); libpaiton_fnexec {lib or (nat['lib_dir_mount'] + '/libpaiton_fnexec.so (in the image)' if in_image else '(dry run: not given)')}; card budget {budget:g} GiB; '
          f'guard {env["PAITON_FN_GUARD"]}; GPUs masked for every process (real list {real} in PAITON_FN_GPU_MASK); logs {log}')
    return (['-v', f'{lib.parent}:{nat["lib_dir_mount"]}:ro'] if lib else []) + ['-v', f'{log}:{nat["log_mount"]}',
            '-v', f'{hook}:{SITE_PACKAGES}/zzzz_paiton_fn_native_guard.pth:ro']

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mode', default='decode', choices=sorted(MODES)); ap.add_argument('--port', type=int, default=18982); ap.add_argument('--name', default='paiton-flashnext')
    ap.add_argument('--host', default='0.0.0.0', help='address the API listens on (default 0.0.0.0, every interface; 127.0.0.1 keeps it on this machine)')
    ap.add_argument('--api-key', default=None, metavar='KEY', help='opt-in: the server then requires this key on /v1 requests (handed to Docker as VLLM_API_KEY by '
                    'variable name, never printed). Preferred: set PAITON_API_KEY in the environment instead (the default here); a key '
                    'given with --api-key is visible in the process list (ps). Default: no key')
    ap.add_argument('--weights', default=None, help='weights dir (default per mode: ~/paiton-models/Qwen3.8-Flash-Next-W3A8, 4-bit modes ~/paiton-models/Qwen3.8-Flash-Next-W4A8)');
    ap.add_argument('--kv-cache-memory-bytes', type=int, default=None, help='KV cache size per GPU in bytes (default: the mode value); '
                    'lower values shrink the cache the prefix cache and long conversations live in')
    ap.add_argument('--expert-vram-gib', type=float, default=None, help='4-bit modes: VRAM per GPU for resident experts (default from modes.json)'); ap.add_argument('--host-pool', choices=('on', 'off'), default=None, help='4-bit modes: keep a pinned host copy of every non-core expert (on, default) or only of the non-resident ones (off)'); ap.add_argument('--cache', default=None, help='compile cache dir (default ~/.cache/paiton/flashnext/<mode>)')
    ap.add_argument('--vision', action='store_true', help='opt-in: image input (vision tower on, video off, per-mode pixel cap; modes.json vision blocks)')
    ap.add_argument('--vision-max-pixels', default=None, metavar='N|2mp|4k', help='with --vision: a lower pixel cap than the mode\'s (2mp = 1920x1080, 4k = 3840x2160)')
    ap.add_argument('--image', default=None, help='image reference (default: the pinned release image)'); ap.add_argument('--devices', default=None, help='comma list of ROCR ids (default: all AMD GPUs; two are required)')
    ap.add_argument('--view', default=None, help='runtime view dir (default: <weights>-view)'); ap.add_argument('--base-cfg', default=None, help='dir with the base model config + tokenizer files (default: <weights>-base-cfg, downloaded at the pinned revision)')
    ap.add_argument('--ple-host', choices=('auto', 'pinned', 'pageable', 'filebacked'), default='auto', help='4-bit modes 4bit, 4bit-cache and 4bit-long: where the '
                    'per-layer embedding tables live on the host: pinned (as benchmarked); pageable (~49 GiB less pinned host memory and GTT, '
                    'so the GTT pools fit a 128 GB host; same output); filebacked (the tables mmapped from raw files next to the weights, in the '
                    'page cache, with user-pointer host pools: a 96 GB host; same output; the raw files are built once, about 49 GiB of disk); '
                    'auto (default): the first of pinned, pageable, filebacked that fits the GTT limit and the free RAM (the start line '
                    '"memory layout:" says which and why)')
    ap.add_argument('--expert-cold', default='auto', help='4-bit modes on hosts below the file-backed layout (about 67 GiB of free RAM): '
                    'the coldest share of the experts is read from a file per GPU instead of RAM (same output, slower); auto (default): '
                    'only when nothing else fits, the smallest measured share that does; off; or a measured share, e.g. 0.2')
    ap.add_argument('--expert-cold-dir', default=None, help='--expert-cold: where the cold-expert files live (default <weights>-expert-cold)')
    ap.add_argument('--ple-raw-dir', default=None, help='--ple-host filebacked: where the raw PLE files live (default <weights>-ple-raw)')
    ap.add_argument('--prefix-caching', action='store_true', help='opt-in: align-mode prefix caching (recurrent state cached per 2048-token block); see RELEASE-NOTES.md'); ap.add_argument('--served-model-name', default='Qwen3.8-Flash-Next'); ap.add_argument('--host-pools', choices=('auto', 'gtt', 'userptr'), default='auto', help='host-resident tables: driver-owned GTT memory or ROCm user-pointer memory; auto = the mode default (3-bit: userptr; 4-bit: gtt when the GTT size holds them)'); ap.add_argument('--detach', action='store_true'); ap.add_argument('--settle-s', type=float, default=30.0, help='after /health: wait (at most this long) for the host page migration to calm down before reporting READY; 0 = off'); ap.add_argument('--keep-page-cache', action='store_true', help='do not drop the checkpoint files from the page cache after the load'); ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--fn-worker', choices=('native', 'torch'), default=None, help='native = the Paiton native worker: on --mode decode / --mode 4bit the opt-in (= --mode 3bit-mtp-native / '
                    '4bit-mtp-native; faster, with request-API limits, see RELEASE-NOTES.md), on the *-native modes the default; torch = the '
                    'torch worker (the default of --mode decode / 4bit, as the r3 release)')
    ap.add_argument('--fnexec-lib', default=None, help='native worker: libpaiton_fnexec.so on the host; its directory is mounted read-only at '
                    'the mode\'s lib_dir_mount (/opt/paiton/runtime/fnexec), so the per-mode manifest and KV files next to it travel along')
    ap.add_argument('--card-budget-gib', type=float, default=None, help='native worker: card VRAM budget per GPU (sysfs), handed to fnexec_init (default: the mode value)')
    ap.add_argument('--native-guard', choices=('enforce', 'record'), default=None, help='native worker: torch HIP guard mode (default: the mode value)')
    ap.add_argument('--native-worker-cls', default=None, help='native worker: the vLLM worker class (default: the mode value)')
    ap.add_argument('--native-log-dir', default=None, help='native worker: host dir for the guard / worker logs (default: next to the cache dir)')
    ap.add_argument('--spec-tokens', type=int, choices=(3, 4, 5), default=None, metavar='K', help='native MTP modes (development): '
                    'num_speculative_tokens K (default: the mode, 3); 4 / 5 = Kmax for the per-batch K steps; needs a lib and kv files built for that K')
    ap.add_argument('--drafter', choices=('distilled', 'stock'), default=None, help='native MTP modes (--fn-worker native on --mode decode / 4bit, 3bit-mtp-native, 4bit-mtp-native, 4bit-cache-mtp-native): '
                    'the MTP drafter checkpoint: distilled (the default) = B4gh5-mtpft, the published 4-bit checkpoint with a distilled drafter '
                    '(measured: decode +6.0 %%, greedy tokens identical to the stock drafter; a local dir, default ~/paiton-models/'
                    'Qwen3.8-Flash-Next-W4A8-mtpft); stock = the published checkpoint\'s drafter. The executor manifest next to --fnexec-lib '
                    'must be the one built from that checkpoint')
    ap.add_argument('--kv-profile', choices=('speed', 'capacity'), default=None, help='native MTP mode (--mode 4bit = 4bit-mtp-native): capacity (the default) = '
                    'KV 2.5 GiB per GPU, about 305,338 cached tokens (as the r3 release); speed = KV 1.5 GiB per GPU, about 183,202 '
                    'cached tokens, more experts in VRAM (faster n=8 / n=4 and prefill, fewer concurrent long contexts)')
    ap.add_argument('--native-env', action='append', default=[], metavar='PAITON_FN_X=V', help='native worker (development): extra PAITON_FN_* switch for the executor, repeatable (e.g. PAITON_FN_PF_TRUE_MIXED=1); lands in fnexec_init_args.mode_json')
    ap.add_argument('--plugin-overlay', action='append', default=[], metavar='FILE', help='development: mount FILE read-only over '
                    'paiton_vllm_plugin/flashnext/<basename> in the image (repeatable; e.g. the native glue before an image carries it)')
    ap.add_argument('--dev-env', action='append', default=[], metavar='PAITON_FN_X=V', help='development: an extra PAITON_FN_* variable for any mode (e.g. PAITON_FN_NLL_DIR for a gate\'s NLL tap)')
    ap.add_argument('--dev-mount', action='append', default=[], metavar='SRC:DST[:ro]', help='development: an extra bind mount (e.g. a gate\'s schedule hook); SRC must exist')
    ap.add_argument('--native-kv-tier', type=lambda v: v if v == 'auto' else 0.0 if v == 'off' else float(v), default=None, metavar='GIB|auto|off',
                    help='native cache modes: RAM prefix tier of GIB per GPU (host RAM = 2 x GIB): prefix-cache blocks evicted from VRAM are kept in '
                    'host memory by the native executor and loaded back on a hit. auto (the default): sized from this host\'s free RAM after the '
                    'mode\'s own memory (the start line "native RAM prefix tier:" prints the formula), off when RAM is short; off: VRAM hits only')
    ap.add_argument('--memory', type=float, default=None, metavar='GIB', help='cap the server container\'s RAM (docker --memory, no swap); '
                    'the RAM checks, the GTT limit (half of it) and the RAM prefix tier are then sized as on a host with GIB of RAM')
    ap.add_argument('--native-kv-tier-layout', choices=('pageable', 'pinned'), default='pageable', help='--native-kv-tier host layout (default pageable: '
                    'private anonymous memory through a small pinned staging ring; pinned = direct DMA, charged to the GPU\'s GTT)')
    ap.add_argument('--prefix-cache-retention-interval', type=int, default=None, metavar='TOKENS', help='prefix-caching modes (development): keep the '
                    'recurrent-state checkpoint every TOKENS tokens (a multiple of the 2048-token block; host-tier hits end on a kept checkpoint). '
                    'Default: vLLM\'s (MTP modes: dense; MTP off: only the latest replay boundary and shared-prefix junctions)')
    a = ap.parse_args(); m = MODES[a.mode]
    global MEM_CAP_GIB; MEM_CAP_GIB = a.memory
    alias = next((k for k, v in MODES.items() if (v.get('native') or {}).get('serves_mode') == a.mode), None)
    # the native worker is the default of a mode whose native alias says so (native.is_default) once the alias's image carries the
    # executor (fnexec_in_image); --fn-worker torch keeps the torch worker, and --vision does too (no image input on the native worker yet),
    # and --prefix-caching (the native worker serves without cross-request hits: the torch worker's prefix caching, as published)
    native_default = bool(alias) and not m.get('native') and a.fn_worker is None and not a.vision and not a.prefix_caching and \
        MODES[alias]['native'].get('is_default', False) and COMMON[MODES[alias].get('image_key', 'image')].get('fnexec_in_image', False)
    cold = a.expert_cold not in ('auto', 'off')   # an explicit cold share (auto picks one only when even file-backed does not fit)

    def native_unmeasured():
        """(why, measured-with) when the native worker is not measured on this host / layout, else None. The native worker is measured
        with the PLE tables pinned (4-bit: the GTT limit holds the pinned total) and the host tables in GTT pools (3-bit: the free GTT
        holds them); the torch worker has the pageable / file-backed / user-pointer layouts."""
        gp = (a.devices.split(',') if a.devices else amd_gpus())[:COMMON['tp']]
        if a.ple_host not in ('auto', 'pinned') or cold or (a.ple_host == 'auto' and MODES[alias].get('expert_tier')):
            ok, why = (False, f'--ple-host {a.ple_host}') if a.ple_host not in ('auto', 'pinned') else \
                (False, f'--expert-cold {a.expert_cold}') if cold else pinned_fits(MODES[alias], gp)
            if not ok: return why, 'with the PLE tables pinned'
        why = gtt_pools_short(gp, free_check=not MODES[alias].get('expert_tier'))
        return (why, 'with GTT host pools') if why else None

    def gtt_pools_short(gp, free_check=True):
        """why the 3-bit host tables cannot go in GTT pools here (--host-pools userptr, or the free GTT is short), else None."""
        hp = COMMON['host_pools']; need = hp['gtt_need_gib'] + hp['gtt_margin_gib']; _, free = gtt_limits_gib(gp) if free_check else (None, None)
        return '--host-pools userptr' if a.host_pools == 'userptr' else \
            f'free GTT {free:.0f} GiB < {need:g} GiB for the host tables' if free is not None and free < need else None
    if native_default:   # the default path: a host or layout the native worker is not measured on keeps the torch worker
        nu = native_unmeasured()
        if nu:
            native_default = False
            print(f'mode {a.mode}: the torch worker ({nu[0]}; the native worker --mode {alias} is measured {nu[1]} only)')
    elif a.fn_worker == 'native' and alias and not m.get('native'):   # an explicit opt-in: refused with the reason (a dry run warns)
        nu = native_unmeasured()
        if nu and not (a.prefix_caching and MODES[alias]['native'].get('cache_mode_small_host')):   # else the small-host cache mode below
            msg = (f'--mode {a.mode} --fn-worker native: {nu[0]}; the native worker --mode {alias} is measured {nu[1]} only '
                   f'(the default torch worker runs on this host: drop --fn-worker native)')
            if a.dry_run: print('WARNING:', msg)
            else: sys.exit(msg)
    if alias and a.fn_worker is None and not m.get('native') and not MODES[alias]['native'].get('is_default', False) and \
            COMMON[MODES[alias].get('image_key', 'image')].get('fnexec_in_image', False):
        # r6 (user 17 Oct ~21:02): every mode defaults to the torch worker, as the r3 release; the native worker is the opt-in
        print(f'mode {a.mode}: the torch worker (the default); --fn-worker native = --mode {alias}: faster, with request-API limits '
              '(no logprobs, structured output, penalties; RELEASE-NOTES.md)')
    elif a.vision and alias and a.fn_worker is None and not m.get('native'):
        print(f'mode {a.mode} --vision: the torch worker (image input is not on the native worker yet; without --vision: --mode {alias})')
    elif a.prefix_caching and alias and a.fn_worker is None and not m.get('native') and MODES[alias]['native'].get('is_default', False):
        print(f'mode {a.mode} --prefix-caching: the torch worker (prefix caching is not on the native worker yet; without --prefix-caching: --mode {alias})')
    if (a.fn_worker == 'native' or native_default) and not m.get('native') and alias:   # e.g. --mode 4bit = --mode 4bit-mtp-native
        print(f'mode {a.mode}{" --fn-worker native" if a.fn_worker else ""}: --mode {alias} (the native worker in this mode\'s configuration'
              f'{"; the default, --fn-worker torch for the torch worker" if native_default else ""})'); a.mode = alias; m = MODES[a.mode]
    cm = (m.get('native') or {}).get('cache_mode'); small = (m.get('native') or {}).get('cache_mode_small_host')
    if a.prefix_caching and small:   # 3-bit: the MTP cache mode needs the GTT host pools (as the speculative native worker is measured)
        why = gtt_pools_short((a.devices.split(',') if a.devices else amd_gpus())[:COMMON['tp']])
        if why:
            print(f'mode {a.mode} --prefix-caching: {why} (speculative decoding with prefix caching, --mode {cm}, needs the GTT host pools: '
                  f'about 120 GiB of RAM or more at the default GTT limit); this host gets --mode {small} (speculative decoding off)')
            cm = small
    if a.prefix_caching and cm:   # native prefix caching: the mode's cache variant (align mode, 2048-token blocks; MTP on, 3-bit small hosts: MTP off)
        print(f'mode {a.mode} --prefix-caching: --mode {cm} ({MODES[cm]["note"]})'); a.mode = cm; m = MODES[a.mode]
    if a.fn_worker == 'torch' and alias and not m.get('native'):   # --mode 4bit --fn-worker torch: this mode's torch worker (as the r3 release)
        print(f'mode {a.mode} --fn-worker torch: the torch worker (the native default is --mode {alias})'); a.fn_worker = None
    if (a.fn_worker or a.fnexec_lib or a.card_budget_gib or a.native_guard or a.native_worker_cls or a.native_log_dir or a.native_env or a.spec_tokens) and not m.get('native'):
        sys.exit(f'--fn-worker / --fnexec-lib / --card-budget-gib / --native-* / --spec-tokens: --mode {a.mode} has no native worker (use --mode 4bit-native, 4bit-long-native or 4bit-200k-native)')
    kv_dflt = (m.get('native') or {}).get('kv_profile_default')
    if a.kv_profile is None and kv_dflt and a.kv_cache_memory_bytes is None and (a.fn_worker or 'native') == 'native':
        a.kv_profile = kv_dflt   # r6 (user 17 Oct): the mode's default profile (4-bit: capacity = the r3 KV); an explicit KV size skips it
    else:
        kv_dflt = None
    if a.kv_profile:   # KV profile (16 Oct): speed = a smaller KV, the freed VRAM to the dynamic expert tier
        prof = ((m.get('native') or {}).get('kv_profiles') or {}).get(a.kv_profile)
        if not prof or (a.fn_worker or 'native') != 'native':
            sys.exit(f'--kv-profile {a.kv_profile}: --mode {a.mode} has no such KV profile (native modes with one: '
                     + ', '.join(k for k, v in MODES.items() if a.kv_profile in ((v.get('native') or {}).get('kv_profiles') or {})) + ')')
        if a.kv_cache_memory_bytes is not None:
            sys.exit('--kv-profile and --kv-cache-memory-bytes: give one of the two')
        a.kv_cache_memory_bytes = prof['kv_cache_memory_bytes']
        given = {e.split('=', 1)[0] for e in a.native_env}
        a.native_env = [f'{k}={v}' for k, v in prof.get('env_native', {}).items() if k not in given] + a.native_env
        other = [k for k in (m.get('native') or {}).get('kv_profiles', {}) if k != a.kv_profile]
        print(f"KV profile {a.kv_profile}{' (the default)' if kv_dflt else ''}: KV cache {prof['kv_cache_memory_bytes']:,} bytes per GPU "
              f"(about {prof['kv_pool_tokens']:,} tokens)" + ''.join(f"; --kv-profile {o}: about {m['native']['kv_profiles'][o]['kv_pool_tokens']:,} tokens"
              f" ({m['native']['kv_profiles'][o]['kv_cache_memory_bytes'] / 2**30:g} GiB per GPU)" for o in other) + f"; {prof['note']}")
    kvt = (m.get('native') or {}).get('kv_tier')
    if kvt and kvt.get('auto') and (a.fn_worker or 'native') == 'native' and a.native_kv_tier is None: a.native_kv_tier = 'auto'
    if a.native_kv_tier == 'auto' and not (kvt or {}).get('auto'): sys.exit(f'--native-kv-tier auto: --mode {a.mode} has no sizing rule; give GIB')
    if a.native_kv_tier is not None:
        if not kvt or (a.fn_worker or 'native') != 'native':
            sys.exit(f'--native-kv-tier: --mode {a.mode} has no native RAM prefix tier (native cache modes: '
                     + ', '.join(k for k, v in MODES.items() if (v.get('native') or {}).get('kv_tier')) + '; the native worker only)')
        if a.native_kv_tier != 'auto' and 0 < a.native_kv_tier < kvt['min_gib_per_gpu']: sys.exit(f'--native-kv-tier {a.native_kv_tier:g}: at least {kvt["min_gib_per_gpu"]:g} GiB per GPU')
    if a.prefix_cache_retention_interval is not None:
        ea = list(m['engine_args']) + (COMMON['prefix_caching']['engine_args_add'] if a.prefix_caching else [])
        bs = next((int(ea[i + 1]) for i, x in enumerate(ea) if x == '--block-size'), None)
        if '--enable-prefix-caching' not in ea or not bs:
            sys.exit(f'--prefix-cache-retention-interval: --mode {a.mode} has no prefix caching with an explicit block size')
        if a.prefix_cache_retention_interval < 0 or a.prefix_cache_retention_interval % bs:
            sys.exit(f'--prefix-cache-retention-interval {a.prefix_cache_retention_interval}: 0 or a multiple of the {bs}-token block')
    vis = vision_setup(a, m)
    if a.api_key:
        print('note: --api-key is visible in the process list (ps) of this machine; prefer PAITON_API_KEY=<key> in the environment')
    a.api_key = a.api_key or os.environ.get('PAITON_API_KEY') or None
    if a.api_key: print('API key: required on /v1 requests (value not printed)')
    drafters = (m.get('native') or {}).get('drafters')
    if a.drafter and not drafters:
        sys.exit(f'--drafter: only {", ".join(k for k, v in MODES.items() if (v.get("native") or {}).get("drafters"))} have a drafter choice')
    wkey = drafters[a.drafter or m['native']['drafter_default']] if drafters else m.get('weights_key')
    wspec = COMMON.get(wkey) if wkey else None
    baked = bool(drafters and wspec and wspec.get('baked') and
                 COMMON[m.get('image_key', 'image')].get('drafter_in_image') == wspec['baked']['id'])
    if drafters and a.drafter is None and wspec and wspec.get('local_only') and a.weights is None and not baked and \
            not (pathlib.Path(os.path.expanduser(wspec['default_dir'])) / 'SHA256SUMS').exists() and 'stock' in drafters:
        # the default drafter is not downloadable (yet) and not present: the published checkpoint's drafter, with a note
        print(f'drafter: stock (the {wspec["drafter"].split(" (")[0]} checkpoint is not at {wspec["default_dir"]}; '
              f'it is the mode default once present, or --weights <dir>)')
        a.drafter = 'stock'; wkey = drafters['stock']; wspec = COMMON.get(wkey)
    if drafters: print(f'drafter: {a.drafter or m["native"]["drafter_default"]} ({wkey}{"" if a.drafter else ", the mode default"})')
    if wspec and wspec.get('local_only') and a.weights is None and not a.dry_run and not baked and \
            not (pathlib.Path(os.path.expanduser(wspec['default_dir'])) / 'SHA256SUMS').exists():
        sys.exit(f'--mode {a.mode}: the {wspec["drafter"].split(" (")[0]} checkpoint is not downloadable yet; put it at '
                 f'{wspec["default_dir"]} or pass --weights <dir> (or --drafter stock for the published checkpoint)')
    explicit_weights = a.weights is not None
    if a.weights is None:
        a.weights = os.path.expanduser(COMMON[wspec['baked']['of']]['default_dir'] if baked and not
                                       (pathlib.Path(os.path.expanduser(wspec['default_dir'])) / 'SHA256SUMS').exists()
                                       else (wspec or W3)['default_dir'])
    if (wspec or W3).get('hf_repo') == COMMON['hf_repo']: w3_revision_note(a.weights, explicit_weights)
    if m.get('status', '').startswith('PENDING') : print(f'mode {a.mode}: {m["status"]}')
    ispec = COMMON[m.get('image_key', 'image')]
    image = a.image or (ispec['tag'] + ('@' + ispec['digest'] if ispec['digest'] != 'PIN-ME' else ''))
    if vis: vision_image_check(a, image)
    gpus = a.devices.split(',') if a.devices else amd_gpus()
    if (m.get('native') or {}).get('kv_tier') and m.get('gtt_pinned_gib') and (a.fn_worker or 'native') == 'native':
        # 4-bit native cache modes are measured with the host tables pinned in GTT only; a smaller host gets the mode that fits it
        ok, why = pinned_fits(m, gpus[:COMMON['tp']])
        if not ok:
            fit = [k for k, v in MODES.items() if (v.get('native') or {}).get('kv_tier') and v.get('status', '').startswith('release') and
                   (pinned_fits(v, gpus[:COMMON['tp']])[0] if v.get('gtt_pinned_gib')
                    else (meminfo_gib('MemAvailable') or 0.0) >= COMMON['host_pools']['gtt_need_gib'] + 8.0)]
            msg = (f'--mode {a.mode}: {why} (the 4-bit native worker is measured with its host tables pinned in GTT); '
                   + (f'this host fits --mode {" or --mode ".join(fit)}' if fit else 'no native prefix-caching mode fits this host'))
            if a.dry_run: print('WARNING:', msg)
            else: sys.exit(msg)
    low_ram_plan(a, m, image, gpus[:COMMON['tp']], vis)
    if vis and a.ple_host == 'filebacked':
        sys.exit(f'--ple-host filebacked is not measured with --vision (--mode {a.mode}); image input runs with the PLE tables pinned or '
                 '--ple-host pageable (measured on 4bit, 4bit-cache and 4bit-long)')
    if m.get('expert_tier'): cold_plan(a, m, image, meminfo_gib('MemAvailable'))
    else: a.expert_cold = 0.0
    if a.expert_cold and vis:
        sys.exit(f'--expert-cold is not measured with --vision (--mode {a.mode})')
    pinned_gib = m.get('gtt_pinned_gib'); pageable_gib = filebacked_gib = 0.0
    ph = COMMON['ple_host']; fbh = ph.get('filebacked') or {}
    if a.ple_host in ('pageable', 'filebacked'):
        key = 'gtt_pinned_gib_pageable' if a.ple_host == 'pageable' else 'ple_filebacked'
        if not m.get(key):
            ok = [k for k, v in MODES.items() if v.get(key)]
            sys.exit(f'--ple-host {a.ple_host} is not measured on --mode {a.mode}: '
                     + ('use --mode ' + ' or --mode '.join(ok) if ok else 'not measured on any mode yet'))
        pinned_gib = m['gtt_pinned_gib_pageable']
        if a.ple_host == 'pageable': pageable_gib = ph['pageable_gib_both_ranks']
        else: filebacked_gib = ph['pageable_gib_both_ranks']
    if a.ple_host == 'filebacked' and not any(ref in image for ref in fbh.get('images', [])):
        msg = (f'--ple-host filebacked needs an image with the file-backed PLE runtime ({", ".join(fbh.get("images", [])) or "none published yet"})')
        if a.dry_run: print('WARNING:', msg)
        else: sys.exit(msg)
    if a.ple_host == 'pageable' and not any(ref in image for ref in ph['images']):
        msg = (f'--ple-host pageable needs an image with the pageable PLE runtime ({", ".join(ph["images"]) or "none published yet"}); '
               f'{image} keeps the PLE tables pinned (about 96 GiB of pinned host memory)')
        if a.dry_run: print('WARNING:', msg)
        else: sys.exit(msg)
    if a.ple_host == 'filebacked':
        ple_raw_preflight(a.ple_raw_dir or os.path.expanduser(a.weights) + '-ple-raw', a.dry_run)
    cold_dir = (a.expert_cold_dir or os.path.expanduser(a.weights) + '-expert-cold') if a.expert_cold else None
    if cold_dir: cold_preflight(cold_dir, a.expert_cold, a.dry_run)
    missing = prerequisites(COMMON['tp'])
    tier = None; tier_gib = 0.0
    if m.get('expert_tier'):
        et = m['expert_tier']
        if a.host_pool == 'off' and tier_params(m['env'].get(et['env'], et['default']))[1].get('classes') == '1':
            sys.exit(f'--host-pool off does not work with --mode {a.mode}: its expert tier swaps experts through the host pool (classes=1 '
                     'needs pool=1); keep the default')
        vg = a.expert_vram_gib if a.expert_vram_gib is not None else (vis or {}).get('vram_gib')   # --vision: the mode's own (lower) tier
        if vg is not None and a.expert_vram_gib is None: print(f'image input: expert VRAM {vg:g} GiB per GPU (text-only {tier_params(m["env"].get(et["env"], et["default"]))[1].get("vram_gib")}): room for the vision tower')
        tier = tier_spec(m['env'].get(et['env'], et['default']), vg, a.host_pool, a.expert_cold or None)
        tier_gib = pinned_host_gib(tier, et['expert_bytes_gib_both_ranks'], COMMON['tp'])
        dflt_gib = pinned_host_gib(tier_spec(m['env'].get(et['env'], et['default'])), et['expert_bytes_gib_both_ranks'], COMMON['tp'])
        if pinned_gib and abs(tier_gib - dflt_gib) > 0.05:   # the measured pinned totals hold the default expert tier
            pinned_gib = round(pinned_gib - dflt_gib + tier_gib, 1)
    vx = vis['ram_extra_gib'] if vis else 0.0   # --vision: the vision tower's host memory on top of the measured figures
    if a.native_kv_tier == 'auto':
        res0 = resident_gib(m, a.ple_host, a.expert_cold) if m.get('expert_tier') else None
        need, marg = (res0, RESIDENT_MARGIN_GIB) if res0 is not None else \
            (ram_need_gib(tier_gib, pinned_gib, pageable_gib, filebacked_gib), 16.0 if pinned_gib else 8.0)
        a.native_kv_tier, a.native_kv_tier_why = kv_tier_auto(m['native']['kv_tier'], need + vx, marg)
        if not a.native_kv_tier: print(f'native RAM prefix tier: {a.native_kv_tier_why}; prefix caching stays on, in VRAM only')
    vx += 2 * a.native_kv_tier if a.native_kv_tier else 0.0   # --native-kv-tier: both GPUs' tier in host RAM (pageable; pinned also counts here)
    res = resident_gib(m, a.ple_host, a.expert_cold) if m.get('expert_tier') else None
    missing += ram_check(tier_gib, (16.0 if pinned_gib else 8.0) + vx, pinned_gib, pageable_gib, filebacked_gib,
                         resident=res + vx if res is not None else None)
    if m.get('expert_tier'):
        tot = meminfo_gib('MemTotal')
        if pageable_gib or filebacked_gib:
            print(f'host RAM: {"unknown" if tot is None else f"{tot:.0f} GiB"} total; --ple-host {a.ple_host} keeps about {pinned_gib:.0f} GiB of pinned host '
                  f'memory in one GPU\'s GTT (about {pinned_gib + 2:.0f} GiB of free GTT; the GTT limit is half the RAM by default: a 128 GB host) and '
                  f'the {pageable_gib or filebacked_gib:.0f} GiB of PLE tables in {"pageable memory" if pageable_gib else "the page cache (file-backed)"}; '
                  'see the host pools line below')
        else:
            print(f'host RAM: {"unknown" if tot is None else f"{tot:.0f} GiB"} total; the benchmarked 4-bit configuration keeps about 96 GiB of pinned '
                  'host memory in one GPU\'s GTT, so it needs about 98 GiB of free GTT (the GTT limit is half the RAM by default: a 256 GB host); '
                  'see the host pools line below' + ('; --ple-host pageable needs about 49 GiB less' if 'gtt_pinned_gib_pageable' in m else ''))
    if missing and not a.dry_run: sys.exit('fix the missing prerequisites above and run again')
    for rel in m.get('requires', []):
        if not (pathlib.Path(a.weights) / rel).exists() and not baked_target(pathlib.Path(a.weights) / rel) and not baked and not a.dry_run: sys.exit(f'mode {a.mode} needs {rel} in the checkpoint dir (not in this copy of the repo)')
    if len(gpus) < COMMON['tp'] and not a.dry_run: sys.exit(f'need {COMMON["tp"]} AMD GPUs, found {gpus}')
    gpus = gpus[:COMMON['tp']]
    p2p_check(gpus)
    via_baked = baked and drafter_check(a.weights, wspec, True, soft=True) is not True
    if via_baked:
        # the image carries the default drafter: the published checkpoint (downloaded if absent) + the image's drafter files
        bspec = COMMON[wspec['baked']['of']]
        weights = ensure_baked(ensure_weights(a.weights, a.dry_run, bspec['hf_repo'], bspec['hf_revision']), wspec, a.dry_run)
        if a.dry_run and not os.environ.get('FLASHNEXT_DRY_SKIP_IMAGE_VERIFY') and shutil.which('docker') and \
                subprocess.run(['docker', 'image', 'inspect', image], capture_output=True).returncode == 0:   # the env var: offline CI / tests
            baked_verify(image, wspec)
    else:
        weights = ensure_overlay(a.weights, wspec, a.dry_run) if (wspec or {}).get('overlay') and wspec.get('hf_revision') else \
            ensure_weights(a.weights, a.dry_run, *((wspec['hf_repo'], wspec['hf_revision']) if wspec else (None, None)))
    if drafters and a.drafter is None and 'stock' in drafters and not via_baked and drafter_check(weights, wspec, a.dry_run, soft=True) is False:
        # the default drafter's files are not in this checkpoint (e.g. the published one at --weights): its own (stock) drafter, with a note
        print(f'drafter: stock (the {wspec["drafter"].split(" (")[0]} drafter files are not in {weights}; --drafter distilled requires them)')
        a.drafter = 'stock'; wkey = drafters['stock']; wspec = COMMON.get(wkey)
    if not (via_baked and a.dry_run): drafter_check(weights, wspec, a.dry_run)
    want_rel = (wspec or COMMON)['hf_repo'].split('/')[-1]
    try: got_rel = json.loads((weights / 'config.json').read_text()).get('paiton_release')
    except Exception: got_rel = None
    # a release may name its revision: '<repo name> (YYYYMMDD, <recipe>)'; any other marker (e.g. an unpublished candidate) is refused
    if got_rel and not re.fullmatch(re.escape(want_rel) + r'( \(\d{8}, [A-Za-z0-9.-]+\))?', got_rel):
        msg = f'mode {a.mode} expects the {want_rel} weights, but {weights} holds {got_rel}; pass --weights with the matching download'
        if a.dry_run: print('WARNING:', msg)
        else: sys.exit(msg)
    base_cfg = ensure_base_cfg(a.base_cfg or str(weights) + '-base-cfg', a.dry_run); view = ensure_view(weights, a.view or str(weights) + ('-view-vision' if vis else '-view'), base_cfg, a.dry_run, tuple(m.get('requires', [])), int6=bool(m['env'].get('PAITON_FN_INT6_EXTRA')), vision=bool(vis))
    cache = pathlib.Path(a.cache or os.path.expanduser(f'~/.cache/paiton/flashnext/{a.mode}' + ('-vision' if vis else ''))); cache.mkdir(parents=True, exist_ok=True)
    ple_raw = ensure_ple_raw(weights, a.ple_raw_dir or str(weights) + '-ple-raw', a.dry_run) if a.ple_host == 'filebacked' else None
    env = dict(m['env']); env['VLLM_CACHE_ROOT'] = COMMON['mounts']['cache']; env['HF_HUB_OFFLINE'] = '1'
    if tier: env[m['expert_tier']['env']] = tier
    engine_args = list(m['engine_args'])
    if a.prefix_caching and '--enable-prefix-caching' not in engine_args and m.get('expert_tier'):
        sys.exit(f'--prefix-caching is not measured on --mode {a.mode}: use --mode 4bit-cache (prefix caching + RAM prefix tier)')
    if a.prefix_caching and '--enable-prefix-caching' in engine_args: print(f'mode {a.mode}: prefix caching is part of this mode (--prefix-caching not needed)')
    elif a.prefix_caching:
        pc = COMMON['prefix_caching']; engine_args = [x for x in engine_args if x not in pc['engine_args_drop']] + pc['engine_args_add']; env.update(pc['env'])
        if m.get('alloc_fraction_prefix_caching'):   # prefix caching has its own measured allocator cap (per mode)
            set_alloc_fraction(env, m['alloc_fraction_prefix_caching'])
    env['ROCR_VISIBLE_DEVICES'] = ','.join(gpus); env['HIP_VISIBLE_DEVICES'] = ','.join(str(i) for i in range(len(gpus))); env['CUDA_VISIBLE_DEVICES'] = env['HIP_VISIBLE_DEVICES']
    env['HOME'] = '/tmp'; env['USER'] = env['LOGNAME'] = 'paiton'
    for k in COMMON['host_pools']['env']: env.pop(k, None)   # a mode profile may carry it; the host-pools choice decides
    if vis:
        engine_args, m = vision_apply(a, m, vis, engine_args, env)
    if a.kv_cache_memory_bytes is not None:
        i = engine_args.index('--kv-cache-memory-bytes'); engine_args[i + 1] = str(a.kv_cache_memory_bytes)
    if a.prefix_cache_retention_interval is not None:
        engine_args += ['--prefix-cache-retention-interval', str(a.prefix_cache_retention_interval)]
        print(f'prefix cache: recurrent-state checkpoints every {a.prefix_cache_retention_interval} tokens' if a.prefix_cache_retention_interval
              else 'prefix cache: recurrent-state checkpoints at the latest replay boundary and shared-prefix junctions only')
    kv_pool_check(a.mode, m, engine_args)
    rtier = m.get('ram_tier') or {}
    hp_choice = m.get('host_pools_default', 'auto') if a.host_pools == 'auto' else a.host_pools
    if cold_dir: env.update(COMMON['expert_cold']['env'])
    if a.ple_host == 'filebacked' and m.get('expert_tier'):
        # measured: file-backed tables with user-pointer pools on 4bit / 4bit-cache / 4bit-long; with GTT pools on 4bit only
        if a.host_pools == 'auto': hp_choice = 'userptr'
        elif a.host_pools == 'gtt' and a.mode != '4bit':
            sys.exit(f'--ple-host filebacked with --host-pools gtt is not measured on --mode {a.mode} (measured: --mode 4bit); '
                     'drop --host-pools')
    if a.host_pools == 'auto' and hp_choice == 'userptr':
        print('host pools: userptr (the default of this mode, as published); --host-pools gtt removes the occasional memory pauses under load '
              'and costs ~1 GiB VRAM in decode')
    hp_env, rt = host_pool_env(hp_choice, gpus, tier_gib, pinned_gib=pinned_gib, ram_tier_gib=float(rtier.get('gib', 0.0)),
                               ram_tier_overhead_gib=float(rtier.get('gtt_overhead_gib', 0.0)))
    env.update(hp_env)
    if m.get('expert_tier'):
        print(f'memory layout: PLE tables {a.ple_host}, host pools {"gtt" if hp_env else "userptr"}'
              + (f', cold experts {a.expert_cold:.0%} from {cold_dir} (slower)' if a.expert_cold else '')
              + (f' ({a.layout_reason})' if a.layout_reason else ' (as benchmarked)' if hp_env
                 else ' (pinned tables, but the free GTT is below their pinned total: user-pointer pools, see the host pools line)'))
    for k in list(ph['env']) + list(fbh.get('env', {})): env.pop(k, None)
    if a.ple_host == 'pageable':
        env.update(ph['env']); print(f'PLE tables: pageable host memory ({pageable_gib:.1f} GiB, both GPUs; not pinned, not in GTT)')
    elif a.ple_host == 'filebacked':
        env.update(fbh['env']); print(f'PLE tables: file-backed ({filebacked_gib:.1f} GiB, both GPUs) from {ple_raw}, page cache; not pinned, not in GTT')
    if rtier:   # RAM prefix tier (cache mode): prefix-cache blocks evicted from VRAM are kept in pinned host memory (GTT) and loaded back on a hit
        rt = int(rt)
        if rt >= rtier.get('min_gib', 4.0):
            kvt = dict(rtier['kv_transfer_config']); kvt['kv_connector_extra_config'] = dict(kvt['kv_connector_extra_config'], cpu_bytes_to_use=rt << 30)
            engine_args += ['--kv-transfer-config', json.dumps(kvt)]
            print(f'RAM prefix tier: {rt} GiB of pinned host memory')
        else:
            for k in rtier['env']: env.pop(k, None)
            print(f'WARNING RAM prefix tier: off (it needs {rtier["gib"] + rtier.get("gtt_overhead_gib", 0.0):g} GiB of free GTT beyond the host pools); '
                  'prefix caching stays on, in VRAM only')
    env['PAITON_FN_PLACEMENT'] = f'counts:{COMMON["mounts"]["checkpoint"]}/common/routing-counts.json'   # the container runs as the host uid, which has no passwd entry: getpass.getuser() needs these
    cmd = ['docker', 'run', '--rm', '--name', a.name, '--user', f'{os.getuid()}:{os.getgid()}'] + COMMON['docker'] + (ph['docker_add'] if a.ple_host == 'pageable' else []) + (list(vis.get('docker_add', [])) if vis else [])
    gid = render_gid(); cmd += ['--group-add', gid] if gid else []
    if a.memory: cmd += ['--memory', f'{int(a.memory * 1024)}m', '--memory-swap', f'{int(a.memory * 1024)}m']
    if a.detach: cmd += ['-d']
    native_mounts = native_apply(a, m, env, engine_args, cache, weights) if m.get('native') else []
    for k in ENV_OVERRIDES:   # caller's environment wins over the mode default (e.g. PAITON_FN_ROPE_TRIM=0 to keep the full rotary cache)
        if k in os.environ: env[k] = os.environ[k]
    for kv in a.dev_env:   # last word on the environment: overrides mode values (e.g. the native R2 defaults' opt-out)
        k, sep, v = kv.partition('=')
        if not sep or not re.fullmatch(r'PAITON_FN_[A-Z0-9_]+', k): sys.exit(f'--dev-env {kv!r}: expected PAITON_FN_<NAME>=<value>')
        env[k] = v
    if env.get('PAITON_FN_MALL_PF', 'off') not in ('', '0', 'off') and (env.get('PAITON_FN_KV_TIER') == '1' or env.get('PAITON_FN_KVT_LAYOUT')):
        sys.exit(f'--mode {a.mode}: PAITON_FN_MALL_PF={env["PAITON_FN_MALL_PF"]} and the RAM prefix tier (--native-kv-tier) are not supported together '
                 '(the head4 executor refuses the pair); pass --native-env PAITON_FN_MALL_PF=off with the tier')
    if (m.get('native') or {}).get('all_vram') and (a.fn_worker or 'native') == 'native':   # no host experts: no dynamic tier to lend from
        bad = [f'{k}={env[k]}' for k in ('PAITON_FN_KV_LEASE', 'PAITON_FN_TIER_LEND_ACT') if env.get(k, 'off') not in ('', '0', 'off')]
        bad += [f'PAITON_FN_TIER_DYNAMIC={env["PAITON_FN_TIER_DYNAMIC"]}'] if env.get('PAITON_FN_TIER_DYNAMIC', '0') not in ('', '0') else []
        if bad: sys.exit(f'--mode {a.mode}: {", ".join(bad)} refused: every expert is in VRAM, so there is no dynamic tier '
                         '(the executor would fail its start with E_ARGS)')
    for k, v in sorted(env.items()): cmd += ['-e', f'{k}={v}']
    if a.api_key: cmd += ['-e', 'VLLM_API_KEY']   # by name: Docker reads the value from the launcher's environment (below)
    cmd += ['-v', f'{weights}:{COMMON["mounts"]["checkpoint"]}:ro', '-v', f'{view}:{COMMON["mounts"]["view"]}:ro', '-v', f'{cache}:{COMMON["mounts"]["cache"]}']
    if ple_raw: cmd += ['-v', f'{ple_raw}:{fbh["mount"]}:ro']
    for f in a.plugin_overlay:
        f = pathlib.Path(f).resolve()
        if not f.is_file() and not a.dry_run: sys.exit(f'--plugin-overlay {f}: no such file')
        cmd += ['-v', f'{f}:{SITE_PACKAGES}/paiton_vllm_plugin/flashnext/{f.name}:ro']
    for mnt in a.dev_mount:
        src, sep, rest = mnt.partition(':')
        if not sep or not rest.startswith('/') or rest.split(':')[1:] not in ([], ['ro'], ['rw']): sys.exit(f'--dev-mount {mnt!r}: expected SRC:/DST[:ro|rw]')
        src = pathlib.Path(src).resolve()
        if not src.exists() and not a.dry_run: sys.exit(f'--dev-mount {src}: no such file or directory')
        cmd += ['-v', f'{src}:{rest}']
    cmd += native_mounts
    if cold_dir:
        if not a.dry_run: pathlib.Path(cold_dir).mkdir(parents=True, exist_ok=True)
        cmd += ['-v', f'{cold_dir}:{COMMON["expert_cold"]["mount"]}']
    if env.get('PAITON_FN_KV_TIER') == '1':
        # RAM prefix tier: its boot hook must sit in site-packages at interpreter start (every engine and worker process), or stock
        # vLLM offloading rejects the route-1 scratch ring group at init. The container runs as the host uid and cannot write there,
        # so the launcher writes the one-line hook next to the cache dir (not inside it: an empty cache gets the seed) and mounts it.
        hook = cache.with_name(cache.name + '-kv-tier.pth')
        if not a.dry_run: hook.write_text(KV_TIER_PTH + '\n')
        cmd += ['-v', f'{hook}:{SITE_PACKAGES}/zz_paiton_fn_kv_tier.pth:ro']
    if env.get('PAITON_FN_ADMIT', 'off') != 'off' and not any('fnadmit' in m_ for m_ in a.dev_mount):
        # admission timing (the 4bit-mtp-native default; PAITON_FN_ADMIT=off through --native-env / --dev-env turns it off): its hook
        # must be in site-packages at interpreter start, written next to the cache dir like the KV-tier hook (a --dev-mount of a
        # development fnadmit replaces it)
        hook = cache.with_name(cache.name + '-admit.pth')
        if not a.dry_run: hook.write_text(ADMIT_PTH + '\n')
        cmd += ['-v', f'{hook}:{SITE_PACKAGES}/zzzzz_paiton_fn_admit.pth:ro']
    # first start of a mode: copy the baked compile-cache seed into the (empty) cache, then serve
    inner = (f'if [ -d /opt/paiton/cache-seed/{m.get("seed", a.mode)} ] && [ -z "$(ls -A {COMMON["mounts"]["cache"]} 2>/dev/null)" ]; then cp -a /opt/paiton/cache-seed/{m.get("seed", a.mode)}/. {COMMON["mounts"]["cache"]}/; fi; '
             f'exec python3 -m vllm.entrypoints.openai.api_server --model {COMMON["mounts"]["view"]} --tokenizer {COMMON["mounts"]["view"]} '
             + ' '.join(shlex.quote(x) for x in engine_args) + f' --served-model-name {shlex.quote(a.served_model_name)} --host {shlex.quote(a.host)} --port {a.port}')
    cmd += ['--entrypoint', 'bash', image, '-c', inner]
    print(f'mode {a.mode}: image {image}; GPUs {gpus}; weights {weights}; cache {cache}; port {a.port}' + (f'; expert tier {env.get(m["expert_tier"]["env"], tier)}' if tier else ''))   # as handed to the server (--vision may lower nospec_gib)
    if a.dry_run: print(' '.join(shlex.quote(x) for x in cmd)); return
    ready = dict(port=a.port, name=a.name, settle_s=a.settle_s, weights=weights, drop_cache=not a.keep_page_cache, host=a.host)
    run_env = dict(os.environ, VLLM_API_KEY=a.api_key) if a.api_key else None
    if a.detach:
        rc = subprocess.run(cmd, env=run_env).returncode
        if rc: sys.exit(rc)
        sys.exit(0 if wait_ready(proc=None, **ready) else 1)
    proc = subprocess.Popen(cmd, env=run_env)
    import signal
    for sig in (signal.SIGTERM, signal.SIGHUP): signal.signal(sig, lambda n, f: proc.send_signal(n))   # the docker client stops the container
    try:
        wait_ready(proc=proc, **ready); sys.exit(proc.wait())
    except KeyboardInterrupt:
        try: sys.exit(proc.wait())
        except KeyboardInterrupt: proc.terminate(); sys.exit(proc.wait())

if __name__ == '__main__':
    main()
