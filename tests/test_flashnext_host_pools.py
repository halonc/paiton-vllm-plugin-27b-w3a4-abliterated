"""Host-pool ladder of the Flash Next launcher (4-bit modes: GTT pools only when the free GTT holds the measured pinned memory)."""
import importlib.util
import pathlib
import sys

import pytest
from conftest import fixed_host_open

LAUNCHER = pathlib.Path(__file__).resolve().parents[1] / 'models' / 'Qwen3.8-Flash-Next' / 'launch-flashnext.py'


@pytest.fixture()
def lf():
    spec = importlib.util.spec_from_file_location('launch_flashnext', LAUNCHER)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    mod.open = fixed_host_open()   # fixed host RAM (conftest), not the live /proc/meminfo
    return mod


def run(lf, monkeypatch, free, choice='auto', pinned=95.5, ram_tier=0.0):
    monkeypatch.setattr(lf, 'gtt_limits_gib', lambda gpus: (None, None) if free is None else (free, free))
    return lf.host_pool_env(choice, ['GPU-0'], 40.8, pinned_gib=pinned, ram_tier_gib=ram_tier)


def test_gtt_when_pinned_fits(lf, monkeypatch):
    assert run(lf, monkeypatch, 125.8) == ({'HSA_USERPTR_FOR_PAGED_MEM': '0'}, 0.0)


def test_userptr_fallback_on_200gb_host(lf, monkeypatch, capsys):
    assert run(lf, monkeypatch, 93.0) == ({}, 0.0)
    assert 'WARNING host pools' in capsys.readouterr().out


def test_unknown_gtt_falls_back(lf, monkeypatch):
    assert run(lf, monkeypatch, None) == ({}, 0.0)


def test_ram_tier_full_shrunk_off(lf, monkeypatch):
    assert run(lf, monkeypatch, 125.8, ram_tier=12.0)[1] == 12.0
    assert run(lf, monkeypatch, 105.0, ram_tier=12.0)[1] == pytest.approx(7.5)
    assert run(lf, monkeypatch, 99.0, ram_tier=12.0) == ({'HSA_USERPTR_FOR_PAGED_MEM': '0'}, 0.0)
    assert run(lf, monkeypatch, 90.0, ram_tier=12.0) == ({}, 0.0)


def test_explicit_gtt_refuses_when_short(lf, monkeypatch):
    with pytest.raises(SystemExit):
        run(lf, monkeypatch, 93.0, choice='gtt')


def test_3bit_rule_unchanged(lf, monkeypatch):
    monkeypatch.setattr(lf, 'gtt_limits_gib', lambda gpus: (62.0, 62.0))
    assert lf.host_pool_env('auto', ['GPU-0'])[0] == {'HSA_USERPTR_FOR_PAGED_MEM': '0'}
    monkeypatch.setattr(lf, 'gtt_limits_gib', lambda gpus: (58.0, 58.0))
    assert lf.host_pool_env('auto', ['GPU-0'])[0] == {}


def _health_server():
    import http.server, threading
    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self): self.send_response(200 if self.path == '/health' else 404); self.end_headers()
        def log_message(self, *a): pass
    srv = http.server.HTTPServer(('127.0.0.1', 0), H); threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def test_wait_ready_settles_after_migration_burst(lf, monkeypatch, tmp_path, capsys):
    import subprocess as sp
    srv = _health_server(); proc = sp.Popen(['sleep', '30'])
    seq = iter([0] + [i * 20000 for i in range(1, 7)] + [120000 + i * 100 for i in range(1, 40)])   # 3 s of 40K pages/s, then quiet
    monkeypatch.setattr(lf, 'pgmigrate_success', lambda: next(seq))
    (tmp_path / 'a.safetensors').write_bytes(b'x' * 4096)
    try:
        assert lf.wait_ready(srv.server_address[1], proc, 'x', settle_s=12, quiet_s=2, weights=tmp_path)
    finally:
        proc.kill(); srv.shutdown()
    out = capsys.readouterr().out
    assert 'dropped the page cache of 1 checkpoint files' in out and 'READY on port' in out and 'settle wait' in out


def test_wait_ready_false_when_server_exits(lf):
    import subprocess as sp
    proc = sp.Popen(['true']); proc.wait()
    assert lf.wait_ready(1, proc, 'x', settle_s=0, weights=None) is False


def dry_cmd(lf, monkeypatch, tmp_path, capsys, kv_tier):
    """main() --dry-run for --mode 4bit with the host checks stubbed; returns the printed docker command."""
    import copy, sys
    modes = copy.deepcopy(lf.MODES)
    if kv_tier: modes['4bit']['env']['PAITON_FN_KV_TIER'] = '1'
    monkeypatch.setattr(lf, 'MODES', modes)
    monkeypatch.setattr(lf, 'p2p_check', lambda *a, **k: None); monkeypatch.setattr(lf, 'ram_check', lambda *a, **k: [])
    monkeypatch.setattr(lf, 'render_gid', lambda: None)
    monkeypatch.setattr(lf, 'gtt_limits_gib', lambda gpus: (200.0, 200.0))
    monkeypatch.setattr(lf, 'ensure_weights', lambda d, dry, *a: tmp_path / 'w')
    monkeypatch.setattr(lf, 'ensure_base_cfg', lambda d, dry: tmp_path / 'cfg')
    monkeypatch.setattr(lf, 'ensure_view', lambda *a, **k: tmp_path / 'view')
    monkeypatch.setattr(sys, 'argv', ['launch-flashnext.py', '--mode', '4bit', '--fn-worker', 'torch', '--dry-run', '--devices', 'GPU-0,GPU-1',
                                      '--cache', str(tmp_path / 'cache')])   # the torch worker's hook (--mode 4bit serves native by default)
    lf.main()
    return capsys.readouterr().out


def test_kv_tier_hook_mounted_only_with_kv_tier(lf, monkeypatch, tmp_path, capsys):
    target = '/usr/local/lib/python3.12/dist-packages/zz_paiton_fn_kv_tier.pth:ro'
    assert target not in dry_cmd(lf, monkeypatch, tmp_path, capsys, False)
    out = dry_cmd(lf, monkeypatch, tmp_path, capsys, True)
    assert f'{tmp_path}/cache-kv-tier.pth:{target}' in out
    assert "PAITON_FN_KV_TIER', '0') == '1'" in lf.KV_TIER_PTH   # the hook stays a no-op without the env


def test_int6_overlay_linked_only_for_int6_modes(lf, tmp_path):
    w = tmp_path / 'w'; (w / 'layers').mkdir(parents=True); (w / 'trunk-int6' / 'layers').mkdir(parents=True); (w / 'common').mkdir(); (w / 'mtp').mkdir()
    for n in ('L00', 'L01'):
        (w / 'layers' / f'{n}.safetensors').write_bytes(b'x'); (w / 'trunk-int6' / 'layers' / f'{n}.safetensors').write_bytes(b'x')
    for rel in ('model.safetensors', 'manifest.json', 'common/ple-e4m3', 'common/bf16-mtp-vision.safetensors', 'mtp/mtp-experts.safetensors',
                'mtp/draft-head-int2.safetensors', 'mtp/draft-head-int2.json'):
        (w / rel).write_bytes(b'x')
    cfg = tmp_path / 'cfg'; cfg.mkdir(); (cfg / 'config.json').write_text('{}')
    plain = lf.ensure_view(w, tmp_path / 'view', cfg, False)
    assert not [p for p in plain.iterdir() if p.name.startswith('i6-')]
    with_int6 = lf.ensure_view(w, tmp_path / 'view', cfg, False, int6=True)   # same dir: rebuilt with the overlay links
    assert sorted(p.name for p in with_int6.iterdir() if p.name.startswith('i6-')) == ['i6-L00.safetensors', 'i6-L01.safetensors']
    again = lf.ensure_view(w, tmp_path / 'view', cfg, False)                  # and back: stale overlay links removed
    assert not [p for p in again.iterdir() if p.name.startswith('i6-')]


def cache_cmd(lf, monkeypatch, capsys, tmp_path, free, extra=()):
    """main() --dry-run for --mode 4bit-cache with the host checks stubbed and `free` GiB of free GTT."""
    import sys
    monkeypatch.setattr(lf, 'p2p_check', lambda *a, **k: None); monkeypatch.setattr(lf, 'ram_check', lambda *a, **k: [])
    monkeypatch.setattr(lf, 'render_gid', lambda: None); monkeypatch.setattr(lf, 'gtt_limits_gib', lambda gpus: (free, free))
    monkeypatch.setattr(lf, 'ensure_weights', lambda d, dry, *a: tmp_path / 'w'); monkeypatch.setattr(lf, 'ensure_base_cfg', lambda d, dry: tmp_path / 'cfg')
    monkeypatch.setattr(lf, 'ensure_view', lambda *a, **k: tmp_path / 'view')
    monkeypatch.setattr(sys, 'argv', ['launch-flashnext.py', '--mode', '4bit-cache', '--dry-run', '--devices', 'GPU-0,GPU-1', '--cache', str(tmp_path / 'cache'), *extra])
    lf.main()
    return capsys.readouterr().out


def test_cache_mode_ram_tier_ladder(lf, monkeypatch, capsys, tmp_path):
    hook = 'zz_paiton_fn_kv_tier.pth:ro'
    full = cache_cmd(lf, monkeypatch, capsys, tmp_path, 125.8)
    assert '"cpu_bytes_to_use": 8589934592' in full and 'PAITON_FN_KV_TIER=1' in full and 'PAITON_HOST_KV_PRIVATE=1' in full and hook in full
    assert '--enable-prefix-caching --block-size 2048 --mamba-cache-mode align' in full and '--no-enable-prefix-caching' not in full
    assert 'PAITON_FN_KV_LEASE=off' in full and 'VLLM_USE_V2_MODEL_RUNNER' not in full
    part = cache_cmd(lf, monkeypatch, capsys, tmp_path, 106.0)          # 6 GiB beyond pinned + margin + tier overhead: 6 GiB tier
    assert '"cpu_bytes_to_use": 6442450944' in part
    off = cache_cmd(lf, monkeypatch, capsys, tmp_path, 103.0)           # GTT pools fit, the tier (with its overhead) does not
    assert 'WARNING RAM prefix tier: off' in off and 'kv-transfer-config' not in off and 'PAITON_FN_KV_TIER' not in off and hook not in off
    assert 'PAITON_HOST_KV_PRIVATE' not in off and '--enable-prefix-caching' in off
    up = cache_cmd(lf, monkeypatch, capsys, tmp_path, 90.0, ('--ple-host', 'pinned'))   # pinned tables, user-pointer pools: no tier
    assert 'kv-transfer-config' not in up and 'HSA_USERPTR_FOR_PAGED_MEM' not in up.split('docker run')[1]
    auto = cache_cmd(lf, monkeypatch, capsys, tmp_path, 90.0)           # auto on the pinned r2 image: pageable tables, GTT pools + full tier
    assert 'memory layout: PLE tables pageable, host pools gtt' in auto and '"cpu_bytes_to_use": 8589934592' in auto
    assert 'HSA_USERPTR_FOR_PAGED_MEM=0' in auto.split('docker run')[1]


def test_cache_mode_ignores_prefix_caching_flag(lf, monkeypatch, capsys, tmp_path):
    out = cache_cmd(lf, monkeypatch, capsys, tmp_path, 125.8, ('--prefix-caching',))
    assert 'prefix caching is part of this mode' in out and 'VLLM_USE_V2_MODEL_RUNNER' not in out
    assert out.count('--enable-prefix-caching') == 1 and '--mamba-cache-mode align' in out


def mode_cmd(lf, monkeypatch, capsys, tmp_path, mode, free, extra=()):
    import sys
    monkeypatch.setattr(lf, 'p2p_check', lambda *a, **k: None); monkeypatch.setattr(lf, 'ram_check', lambda *a, **k: [])
    monkeypatch.setattr(lf, 'render_gid', lambda: None); monkeypatch.setattr(lf, 'gtt_limits_gib', lambda gpus: (free, free))
    monkeypatch.setattr(lf, 'ensure_weights', lambda d, dry, *a: tmp_path / 'w'); monkeypatch.setattr(lf, 'ensure_base_cfg', lambda d, dry: tmp_path / 'cfg')
    monkeypatch.setattr(lf, 'ensure_view', lambda *a, **k: tmp_path / 'view')
    monkeypatch.setattr(sys, 'argv', ['launch-flashnext.py', '--mode', mode, '--dry-run', '--devices', 'GPU-0,GPU-1', '--cache', str(tmp_path / 'cache'), *extra])
    lf.main()
    return capsys.readouterr().out


@pytest.mark.parametrize('mode', ['decode', 'prefill-long', 'decode-nopf', 'prefill-long-nopf'])
def test_3bit_modes_default_userptr_gtt_opt_in(lf, monkeypatch, capsys, tmp_path, mode):
    tw = ('--fn-worker', 'torch') if mode == 'decode' else ()   # r6: --mode decode defaults to the native worker; this is the torch path
    out = mode_cmd(lf, monkeypatch, capsys, tmp_path, mode, 125.8, tw)
    assert 'HSA_USERPTR_FOR_PAGED_MEM' not in out.split('docker run')[1] and '--host-pools gtt removes' in out
    out = mode_cmd(lf, monkeypatch, capsys, tmp_path, mode, 125.8, tw + ('--host-pools', 'gtt'))
    assert 'HSA_USERPTR_FOR_PAGED_MEM=0' in out.split('docker run')[1]


@pytest.mark.parametrize('mode', ['decode', 'prefill-long', 'decode-nopf', 'prefill-long-nopf'])
def test_prefix_caching_sets_block_2048(lf, monkeypatch, capsys, tmp_path, mode):
    run_args = mode_cmd(lf, monkeypatch, capsys, tmp_path, mode, 125.8, ('--prefix-caching',)).split('docker run')[1]
    assert '--enable-prefix-caching --block-size 2048 --mamba-cache-mode align' in run_args     # hits resume on a block boundary
    assert run_args.count('--block-size') == 1 and '--no-enable-prefix-caching' not in run_args
    assert '--block-size' not in mode_cmd(lf, monkeypatch, capsys, tmp_path, mode, 125.8).split('docker run')[1]


def test_cache_mode_single_block_size(lf, monkeypatch, capsys, tmp_path):
    for extra in ((), ('--prefix-caching',)):
        run_args = mode_cmd(lf, monkeypatch, capsys, tmp_path, '4bit-cache', 125.8, extra).split('docker run')[1]
        assert run_args.count('--block-size') == 1 and '--block-size 2048' in run_args


def test_4bit_default_stays_gtt(lf, monkeypatch, capsys, tmp_path):
    assert 'HSA_USERPTR_FOR_PAGED_MEM=0' in mode_cmd(lf, monkeypatch, capsys, tmp_path, '4bit', 125.8).split('docker run')[1]


@pytest.mark.parametrize('mode', ['4bit', '4bit-200k', '4bit-long', '4bit-xl'])
def test_prefix_caching_refused_on_unmeasured_4bit_modes(lf, monkeypatch, capsys, tmp_path, mode):
    with pytest.raises(SystemExit, match='use --mode 4bit-cache'):
        mode_cmd(lf, monkeypatch, capsys, tmp_path, mode, 125.8, ('--prefix-caching',))


def test_kv_pool_warnings(lf, capsys):
    m = lf.MODES
    assert lf.kv_pool_check('decode', m['decode'], list(m['decode']['engine_args'])) == 112347
    assert 'WARNING' not in capsys.readouterr().out                      # no prefix caching, mode value: silent
    pc = [x for x in m['decode']['engine_args'] if x != '--no-enable-prefix-caching'] + ['--enable-prefix-caching']
    lf.kv_pool_check('decode', m['decode'], pc)
    out = capsys.readouterr().out
    assert 'WARNING prefix caching: the KV cache holds about 112K tokens < 2 x the 98K window' in out and '4bit-cache' in out
    lf.kv_pool_check('4bit-cache', m['4bit-cache'], list(m['4bit-cache']['engine_args']))
    assert 'WARNING' not in capsys.readouterr().out                      # 305K >= 2 x 98K
    low = list(m['4bit']['engine_args']); low[low.index('--kv-cache-memory-bytes') + 1] = str(2684354560 // 4)
    assert lf.kv_pool_check('4bit', m['4bit'], low) == 76334
    out = capsys.readouterr().out
    assert 'is below the 4bit value 2,684,354,560' in out and 'refuse to start' in out


@pytest.mark.parametrize('mode,kv_bytes,vram,tokens', [('4bit-long', '3758096384', '19.75', 461566), ('4bit-xl', '4831838208', '18.75', 595050)])
def test_4bit_long_modes(lf, monkeypatch, capsys, tmp_path, mode, kv_bytes, vram, tokens):
    m = lf.MODES[mode]
    assert m['kv_pool_tokens'] == tokens and 'PAITON_FN_EXPERT_DYN_LOG' not in m['env']
    run_args = mode_cmd(lf, monkeypatch, capsys, tmp_path, mode, 125.8).split('docker run')[1]
    assert '--max-model-len 262144' in run_args and f'--kv-cache-memory-bytes {kv_bytes}' in run_args
    assert f'vram_gib={vram},' in run_args and 'per_process_memory_fraction:0.925' in run_args
    assert '--no-enable-prefix-caching' in run_args and 'HSA_USERPTR_FOR_PAGED_MEM=0' in run_args


PARSER_ARGS = '--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml'


def test_every_mode_parses_reasoning_and_tools(lf):
    for name, m in lf.MODES.items():
        e = ' '.join(m['engine_args'])
        assert PARSER_ARGS in e and e.count('--reasoning-parser') == 1, name


@pytest.mark.parametrize('mode,extra', [('decode', ()), ('prefill-long', ()), ('decode', ('--prefix-caching',)), ('4bit', ()),
                                        ('4bit-200k', ()), ('4bit-cache', ()), ('4bit-long', ()), ('4bit-xl', ())])
def test_parser_args_reach_the_server(lf, monkeypatch, capsys, tmp_path, mode, extra):
    run_args = mode_cmd(lf, monkeypatch, capsys, tmp_path, mode, 125.8, extra).split('docker run')[1]
    assert PARSER_ARGS in run_args


# resolved allocator cap per 3-bit mode, without / with --prefix-caching (prefill-long keeps 0.95: 0.94 ran out of memory on a
# 131K-token prefill there)
CAP_TABLE = {'decode': ('0.95', '0.94'), 'decode-nopf': ('0.95', '0.94'), 'prefill-long': ('0.95', '0.95'), 'prefill-long-nopf': ('0.95', '0.95')}


@pytest.mark.parametrize('mode', sorted(CAP_TABLE))
def test_3bit_allocator_cap(lf, monkeypatch, capsys, tmp_path, mode):
    tw = ('--fn-worker', 'torch') if mode == 'decode' else ()   # r6: --mode decode defaults to the native worker; this is the torch path
    for extra, frac in zip(((), ('--prefix-caching',)), CAP_TABLE[mode]):
        run_args = mode_cmd(lf, monkeypatch, capsys, tmp_path, mode, 125.8, tw + extra).split('docker run')[1]
        assert run_args.count('per_process_memory_fraction') == 1
        assert f'PYTORCH_ALLOC_CONF=max_split_size_mb:64,per_process_memory_fraction:{frac}' in run_args


def test_set_alloc_fraction(lf):
    env = {'PYTORCH_ALLOC_CONF': 'max_split_size_mb:64,per_process_memory_fraction:0.95'}
    lf.set_alloc_fraction(env, 0.94)
    assert env['PYTORCH_ALLOC_CONF'] == 'max_split_size_mb:64,per_process_memory_fraction:0.94'
    env = {}; lf.set_alloc_fraction(env, 0.925)
    assert env['PYTORCH_ALLOC_CONF'] == 'per_process_memory_fraction:0.925'


def test_4bit_cache_keeps_its_cap(lf, monkeypatch, capsys, tmp_path):
    run_args = mode_cmd(lf, monkeypatch, capsys, tmp_path, '4bit-cache', 125.8, ('--prefix-caching',)).split('docker run')[1]
    assert 'per_process_memory_fraction:0.925' in run_args and run_args.count('per_process_memory_fraction') == 1


PAGEABLE = 'PAITON_FN_PLE_HOST=pageable'


def test_ple_host_pageable_measured_modes_only(lf):
    assert sorted(k for k, v in lf.MODES.items() if 'gtt_pinned_gib_pageable' in v) == ['4bit', '4bit-cache', '4bit-long']
    assert sorted(k for k, v in lf.MODES.items() if v.get('ple_filebacked')) == ['4bit', '4bit-cache', '4bit-long']
    for k in ('4bit', '4bit-cache', '4bit-long'):                                      # measured resident RAM per layout
        r = lf.MODES[k]['resident_gib']; assert r['pinned'] > r['filebacked'] and set(r) == {'pinned', 'pageable', 'filebacked'}
    assert all('PAITON_FN_PLE_HOST' not in v['env'] for v in lf.MODES.values())     # pinned unless the flag asks


@pytest.mark.parametrize('mode', ['4bit-200k', '4bit-xl', 'decode', 'prefill-long'])
def test_ple_host_pageable_refused_on_unmeasured_modes(lf, monkeypatch, capsys, tmp_path, mode):
    with pytest.raises(SystemExit, match='use --mode 4bit or --mode 4bit-cache or --mode 4bit-long'):
        mode_cmd(lf, monkeypatch, capsys, tmp_path, mode, 125.8, ('--ple-host', 'pageable'))


def test_ple_host_pageable_needs_a_supporting_image(lf, monkeypatch, capsys, tmp_path):
    other = ('--image', 'flashnext-test:without-pageable-runtime')
    out = mode_cmd(lf, monkeypatch, capsys, tmp_path, '4bit', 125.8, ('--ple-host', 'pageable') + other)
    assert 'WARNING: --ple-host pageable needs an image with the pageable PLE runtime' in out
    monkeypatch.setattr(lf, 'prerequisites', lambda tp: [])
    with pytest.raises(SystemExit, match='pageable PLE runtime'):                    # a real start refuses
        monkeypatch.setattr(sys, 'argv', ['launch-flashnext.py', '--mode', '4bit', '--devices', 'GPU-0,GPU-1', '--cache', str(tmp_path / 'cache'),
                                          '--ple-host', 'pageable', *other])
        lf.main()


@pytest.mark.parametrize('mode', ['4bit', '4bit-cache'])
def test_ple_host_pageable_fits_a_128gb_host(lf, monkeypatch, capsys, tmp_path, mode):
    monkeypatch.setitem(lf.COMMON['ple_host'], 'images', ['flashnext-test:pageable'])
    img = ('--image', 'flashnext-test:pageable')
    pinned = mode_cmd(lf, monkeypatch, capsys, tmp_path, mode, 60.0, img + ('--ple-host', 'pinned')).split('docker run')[1]   # GTT limit of a 128 GB host: ~62 GiB
    assert 'HSA_USERPTR_FOR_PAGED_MEM' not in pinned and PAGEABLE not in pinned and 'memlock' not in pinned
    out = mode_cmd(lf, monkeypatch, capsys, tmp_path, mode, 60.0, img + ('--ple-host', 'pageable'))
    run_args = out.split('docker run')[1]
    assert 'HSA_USERPTR_FOR_PAGED_MEM=0' in run_args and PAGEABLE in run_args and '--ulimit memlock=-1:-1' in run_args
    assert 'needed 48.6 GiB for GTT pools' in out and 'WARNING' not in out
    if mode == '4bit-cache':
        assert '"cpu_bytes_to_use": 8589934592' in run_args                             # 60 - 48.6 - 2.5 overhead >= the 8 GiB RAM tier
    assert 'per_process_memory_fraction:0.925' in run_args                             # the 4-bit allocator cap is unchanged


def test_ple_host_pageable_ram_check(lf, monkeypatch, capsys):
    import io
    def meminfo(gib):
        monkeypatch.setattr(lf, 'open', lambda *a, **k: io.StringIO(f'MemAvailable: {int(gib * (1 << 20))} kB\n'), raising=False)
    meminfo(120.0)                                                                     # a 128 GB host after boot
    assert lf.ram_check(40.8, 16.0, 46.6, 48.9) == []
    out = capsys.readouterr().out
    assert 'pinned host memory 46.9 GiB (host tables 6.1 + expert host tier 40.8, both GPUs) + pageable PLE tables 48.9 GiB' in out
    assert 'needed 111.8 GiB' in out                                                  # the same RAM as pinned (95.8 + 16)
    meminfo(100.0)                                                                     # pageable tables still need the RAM
    assert lf.ram_check(40.8, 16.0, 46.6, 48.9) and 'NOT ENOUGH' in capsys.readouterr().out


ALL_MODES = ['decode', '3bit-mtp-native', 'prefill-long', 'decode-nopf', 'prefill-long-nopf', '4bit', '4bit-200k', '4bit-cache', '4bit-long', '4bit-xl', '4bit-native', '4bit-long-native', '4bit-200k-native', '4bit-cache-native', '4bit-long-cache-native', '4bit-long-cache', '4bit-mtp-native', '4bit-cache-mtp-native', '3bit-cache-native', '3bit-long-cache-native', '4bit-long-cache-mtp-native', '3bit-long-cache-mtp-native']


def test_fix_knobs_in_modes(lf):
    assert sorted(lf.MODES) == sorted(ALL_MODES)
    for name, m in lf.MODES.items():
        assert m['env']['PAITON_FN_PLE_QSL_PAD'] == '1', name                       # n-gram tail refresh: every mode
        assert m['env'].get('PAITON_FN_STEP_PREP_ALIGN') == ('inplace' if name in ('4bit-cache', '4bit-cache-native', '4bit-long-cache-native', '4bit-long-cache', '4bit-cache-mtp-native', '4bit-long-cache-mtp-native', '3bit-long-cache-mtp-native', '3bit-cache-native') else None)   # 3bit-long-cache-native: gated 18 Oct without it, name
    assert lf.COMMON['prefix_caching']['env']['PAITON_FN_STEP_PREP_ALIGN'] == 'inplace'


@pytest.mark.parametrize('mode,extra,align', [('decode', (), False), ('decode', ('--prefix-caching',), True),
                                              ('prefill-long', ('--prefix-caching',), True), ('4bit', (), False),
                                              ('4bit-cache', (), True), ('4bit-long', (), False)])
def test_fix_knobs_reach_the_server(lf, monkeypatch, capsys, tmp_path, mode, extra, align):
    run_args = mode_cmd(lf, monkeypatch, capsys, tmp_path, mode, 125.8, extra).split('docker run')[1]
    assert run_args.count('PAITON_FN_PLE_QSL_PAD=1') == 1
    assert run_args.count('PAITON_FN_STEP_PREP_ALIGN=inplace') == (1 if align else 0)


def test_r2_image_pins(lf):
    assert lf.COMMON['image'] == {'tag': 'ghcr.io/eliovp/paiton-vllm-plugin:qwen38-flashnext-rocm10-vllm029-20261010-r2', 'digest': 'sha256:ea84a147c6dd44f47a41836ace6b39864fab940095c4c0ba13e82537dd9e2a4b'}
    assert (lf.COMMON['image_4bit_native']['tag'], lf.COMMON['image_4bit_native']['digest']) == ('ghcr.io/eliovp/paiton-vllm-plugin:qwen38-flashnext-rocm10-vllm029-20261026-pc-rc4', 'sha256:328be15c6f1bc3ec99b1b7a436c4a38eeb82ec36d0589b7bbef38f0e465e11cf')
    assert (lf.COMMON['image_4bit']['tag'], lf.COMMON['image_4bit']['digest']) == ('ghcr.io/eliovp/paiton-vllm-plugin:qwen38-flashnext-rocm10-vllm029-20261017-r1', 'sha256:b801bafd3f4cbc04b99153180514845894f98554f49fccdbfe0d6344b47034d9')   # 4-bit release 20261017 rc: one image for every 4-bit mode (pinned at publish)
    assert lf.COMMON['ple_host']['images'] == ['sha256:0862ca1c999f12834b6169e84bdf594eb5f8e4db4efbd79a909e32efdf16a8c0', 'sha256:13166ca5894684ca72f7f15584c9e81fd77c7fbf8297cfc1958ecf0c28c2cb69', 'sha256:b801bafd3f4cbc04b99153180514845894f98554f49fccdbfe0d6344b47034d9']
    assert lf.COMMON['ple_host']['filebacked']['images'] == ['sha256:0862ca1c999f12834b6169e84bdf594eb5f8e4db4efbd79a909e32efdf16a8c0', 'sha256:b801bafd3f4cbc04b99153180514845894f98554f49fccdbfe0d6344b47034d9']   # r3 + the 20261017 rc: file-backed PLE runtime


@pytest.mark.parametrize('mode', ALL_MODES)
def test_every_mode_runs_its_r2_image(lf, monkeypatch, capsys, tmp_path, mode):
    nat = lf.MODES[mode].get('native') or {}; key = lf.MODES[mode].get('image_key', 'image'); fm = lf.COMMON[key].get('fnexec_modes')
    if mode.endswith('-native') and fm is not None and nat['env_native']['PAITON_FN_MODE_NAME'] not in fm:   # a dev mode: --fnexec-lib
        with pytest.raises(SystemExit, match='carries the native executor files of'):
            mode_cmd(lf, monkeypatch, capsys, tmp_path, mode, 125.8)
        monkeypatch.setitem(lf.COMMON, key, dict(lf.COMMON[key], fnexec_modes=None))
    run_args = mode_cmd(lf, monkeypatch, capsys, tmp_path, mode, 125.8).split('docker run')[1]
    if mode.endswith('-native'):   # release candidate 20261023-pc: one local image for every native mode
        assert 'ghcr.io/eliovp/paiton-vllm-plugin:qwen38-flashnext-rocm10-vllm029-20261026-pc-rc4@sha256:328be15c6f1bc3ec99b1b7a436c4a38eeb82ec36d0589b7bbef38f0e465e11cf' in run_args
        return
    if mode.startswith('4bit'):   # release 20261017: the 4-bit torch modes keep the published image
        assert 'sha256:b801bafd3f4cbc04b99153180514845894f98554f49fccdbfe0d6344b47034d9' in run_args
        return   # (r6, decision B: --mode decode stays the torch worker on the published 3-bit image)
    assert ('sha256:0862ca1c999f12834b6169e84bdf594eb5f8e4db4efbd79a909e32efdf16a8c0' if mode.startswith('4bit') else 'sha256:ea84a147c6dd44f47a41836ace6b39864fab940095c4c0ba13e82537dd9e2a4b') in run_args


LOWRAM = 'memory layout: PLE tables p'   # pageable / filebacked chosen automatically: the line carries the reason


def lowram_cmd(lf, monkeypatch, capsys, tmp_path, mode, gtt, avail=None, extra=()):
    monkeypatch.setitem(lf.COMMON['ple_host'], 'images', ['flashnext-test:pageable'])
    monkeypatch.setattr(lf, 'meminfo_gib', lambda f: {'MemTotal': gtt * 2, 'MemAvailable': avail}[f])
    return mode_cmd(lf, monkeypatch, capsys, tmp_path, mode, gtt, ('--image', 'flashnext-test:pageable') + tuple(extra))


@pytest.mark.parametrize('mode', ['4bit', '4bit-cache'])
def test_low_ram_auto_picks_pageable_on_small_gtt(lf, monkeypatch, capsys, tmp_path, mode):
    out = lowram_cmd(lf, monkeypatch, capsys, tmp_path, mode, 60.0, 128.0)               # 128 GB host: GTT limit ~62 GiB
    assert out.count('memory layout:') == 1 and 'memory layout: PLE tables pageable, host pools gtt' in out and 'GTT limit 60 GiB < 98 GiB' in out
    run_args = out.split('docker run')[1]
    assert PAGEABLE in run_args and '--ulimit memlock=-1:-1' in run_args and 'HSA_USERPTR_FOR_PAGED_MEM=0' in run_args
    assert 'pool=1' in run_args                                                           # 118 GiB available: the full expert copy stays


@pytest.mark.parametrize('mode', ['4bit', '4bit-cache'])
def test_low_ram_auto_keeps_pinned_on_big_hosts_and_explicit_flags(lf, monkeypatch, capsys, tmp_path, mode):
    out = lowram_cmd(lf, monkeypatch, capsys, tmp_path, mode, 125.8, 240.0)              # 256 GB host
    assert 'memory layout: PLE tables pinned, host pools gtt (as benchmarked)' in out and PAGEABLE not in out.split('docker run')[1]
    out = lowram_cmd(lf, monkeypatch, capsys, tmp_path, mode, 60.0, 118.0, ('--ple-host', 'pinned'))
    assert 'PLE tables pinned' in out and '(as requested)' in out and PAGEABLE not in out.split('docker run')[1]   # explicit pinned wins


@pytest.mark.parametrize('mode', ['4bit-xl', '4bit-200k'])
def test_low_ram_auto_unmeasured_modes_stay_pinned(lf, monkeypatch, capsys, tmp_path, mode):
    out = lowram_cmd(lf, monkeypatch, capsys, tmp_path, mode, 60.0, 118.0)
    assert f'--mode {mode} has no measured low-RAM layout yet' in out and PAGEABLE not in out.split('docker run')[1]


def test_low_ram_auto_needs_the_pageable_image(lf, monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(lf, 'meminfo_gib', lambda f: {'MemTotal': 120.0, 'MemAvailable': 118.0}[f])
    out = mode_cmd(lf, monkeypatch, capsys, tmp_path, '4bit', 60.0, ('--image', 'flashnext-test:without-pageable-runtime'))
    assert 'has no low-RAM PLE runtime; keeping them pinned' in out and PAGEABLE not in out.split('docker run')[1]


@pytest.mark.parametrize('mode', ['decode', 'prefill-long', 'decode-nopf', 'prefill-long-nopf'])
def test_low_ram_auto_leaves_3bit_alone(lf, monkeypatch, capsys, tmp_path, mode):
    out = lowram_cmd(lf, monkeypatch, capsys, tmp_path, mode, 30.0, 60.0)
    assert LOWRAM not in out and PAGEABLE not in out.split('docker run')[1]


@pytest.mark.parametrize('mode', ['4bit', '4bit-200k', '4bit-cache', '4bit-long', '4bit-xl'])
def test_host_pool_off_refused_on_class_tiers(lf, monkeypatch, capsys, tmp_path, mode):
    assert 'classes=1' in lf.MODES[mode]['expert_tier']['default']                        # the runtime refuses classes=1 without pool=1
    with pytest.raises(SystemExit, match='--host-pool off does not work with --mode ' + mode):
        mode_cmd(lf, monkeypatch, capsys, tmp_path, mode, 125.8, ('--host-pool', 'off'))
    assert 'pool=1' in mode_cmd(lf, monkeypatch, capsys, tmp_path, mode, 125.8, ('--host-pool', 'on')).split('docker run')[1]


def _fake_ple_export(root, shards=((3, 4), (5, 4))):
    """a tiny E4M3 PLE export: shard_i.safetensors with row_scale [n] F32 + codes [n, dim] F8_E4M3 (row r = bytes r, scale r)."""
    import json as _j, struct
    d = root / 'common' / 'ple-e4m3'; d.mkdir(parents=True); r0 = 0
    for i, (n, dim) in enumerate(shards):
        sc = b''.join(struct.pack('<f', float(r0 + k)) for k in range(n)); co = bytes((r0 + k) % 256 for k in range(n) for _ in range(dim))
        h = _j.dumps({'__metadata__': {'format': 'e4m3'}, 'row_scale': {'dtype': 'F32', 'shape': [n], 'data_offsets': [0, len(sc)]},
                      'codes': {'dtype': 'F8_E4M3', 'shape': [n, dim], 'data_offsets': [len(sc), len(sc) + len(co)]}}).encode()
        (d / f'shard_{i:03d}.safetensors').write_bytes(struct.pack('<Q', len(h)) + h + sc + co); r0 += n
    return root


def test_ple_raw_files_built_once(lf, tmp_path, capsys):
    import json, struct
    w = _fake_ple_export(tmp_path / 'w'); out = tmp_path / 'raw'
    lf.ensure_ple_raw(w, out, False)
    codes, scale = (out / 'codes.e4m3').read_bytes(), (out / 'row_scale.f32').read_bytes()
    assert codes == bytes(r for r in range(8) for _ in range(4))                        # rows of both shards, in order
    assert [struct.unpack_from('<f', scale, 4 * r)[0] for r in range(8)] == [float(r) for r in range(8)]
    assert json.loads((out / 'meta.json').read_text())['rows_total'] == 8 and 'building' in capsys.readouterr().out
    lf.ensure_ple_raw(w, out, False)
    assert 'reused' in capsys.readouterr().out
    (w / 'common/ple-e4m3/shard_001.safetensors').write_bytes((w / 'common/ple-e4m3/shard_001.safetensors').read_bytes() + b'x')
    lf.ensure_ple_raw(w, out, False)                                                       # a changed export rebuilds
    assert 'building' in capsys.readouterr().out


def test_ple_filebacked_launch(lf, monkeypatch, capsys, tmp_path):
    monkeypatch.setitem(lf.COMMON['ple_host']['filebacked'], 'images', ['flashnext-test:filebacked'])
    monkeypatch.setitem(lf.MODES['4bit'], 'ple_filebacked', True)
    monkeypatch.setattr(lf, 'ensure_ple_raw', lambda w, out, dry: tmp_path / 'raw')
    out = mode_cmd(lf, monkeypatch, capsys, tmp_path, '4bit', 60.0, ('--image', 'flashnext-test:filebacked', '--ple-host', 'filebacked'))
    run_args = out.split('docker run')[1]
    assert 'PAITON_FN_PLE_HOST=filebacked' in run_args and 'PAITON_FN_PLE_FILE_DIR=/models/ple-raw' in run_args
    assert f'{tmp_path / "raw"}:/models/ple-raw:ro' in run_args and 'memlock' not in run_args
    assert 'PLE tables: file-backed' in out and 'memory layout: PLE tables filebacked, host pools userptr (as requested)' in out
    out = mode_cmd(lf, monkeypatch, capsys, tmp_path, '4bit', 60.0, ('--image', 'flashnext-test:filebacked', '--ple-host', 'filebacked',
                                                                      '--host-pools', 'gtt'))
    assert 'needed 48.6 GiB for GTT pools' in out and 'host pools gtt' in out                  # measured on 4bit (128 GB host)
    monkeypatch.setitem(lf.MODES['4bit-long'], 'ple_filebacked', True)
    with pytest.raises(SystemExit, match='with --host-pools gtt is not measured on --mode 4bit-long'):
        mode_cmd(lf, monkeypatch, capsys, tmp_path, '4bit-long', 60.0, ('--image', 'flashnext-test:filebacked', '--ple-host', 'filebacked',
                                                                         '--host-pools', 'gtt'))


def test_ple_filebacked_refused_unmeasured_or_wrong_image(lf, monkeypatch, capsys, tmp_path):
    with pytest.raises(SystemExit, match='--ple-host filebacked is not measured on --mode 4bit-xl'):
        mode_cmd(lf, monkeypatch, capsys, tmp_path, '4bit-xl', 60.0, ('--ple-host', 'filebacked'))
    monkeypatch.setitem(lf.MODES['4bit'], 'ple_filebacked', True)
    monkeypatch.setattr(lf, 'ensure_ple_raw', lambda w, out, dry: tmp_path / 'raw')
    monkeypatch.setitem(lf.COMMON['ple_host']['filebacked'], 'images', ['sha256:not-the-pinned-image'])   # whatever 4-bit image is pinned
    out = mode_cmd(lf, monkeypatch, capsys, tmp_path, '4bit', 60.0, ('--ple-host', 'filebacked'))   # dry run: warning, a start refuses
    assert 'WARNING: --ple-host filebacked needs an image with the file-backed PLE runtime' in out


def test_ple_filebacked_ram_check(lf, monkeypatch, capsys):
    import io
    monkeypatch.setattr(lf, 'open', lambda *a, **k: io.StringIO(f'MemAvailable: {int(80 * (1 << 20))} kB\n'), raising=False)
    assert lf.ram_check(40.8, 16.0, 46.6, 0.0, 48.9) == []                                 # 46.9 pinned + 16 < 80 GiB
    assert 'PLE tables 48.9 GiB file-backed' in capsys.readouterr().out
    assert lf.ram_check(40.8, 16.0, 46.6, 48.9) and 'NOT ENOUGH' in capsys.readouterr().out   # pageable needs the tables resident


def test_low_ram_auto_picks_filebacked(lf, monkeypatch, capsys, tmp_path):
    monkeypatch.setitem(lf.COMMON['ple_host']['filebacked'], 'images', ['flashnext-test:pageable'])
    monkeypatch.setattr(lf, 'ensure_ple_raw', lambda w, out, dry: tmp_path / 'raw')
    monkeypatch.setattr(lf, 'ple_raw_preflight', lambda out, dry: None)
    out = lowram_cmd(lf, monkeypatch, capsys, tmp_path, '4bit', 46.0, 88.0)                 # ~96 GB host
    assert out.count('memory layout:') == 1 and 'memory layout: PLE tables filebacked, host pools userptr' in out
    assert 'MemAvailable 88 GiB < 116 GiB with them resident' in out and 'PAITON_FN_PLE_HOST=filebacked' in out.split('docker run')[1]
    out = lowram_cmd(lf, monkeypatch, capsys, tmp_path, '4bit', 60.0, 120.0)                # 128 GB host: tables stay resident
    assert 'memory layout: PLE tables pageable' in out and 'PAITON_FN_PLE_HOST=pageable' in out.split('docker run')[1]
    monkeypatch.setitem(lf.MODES['4bit'], 'ple_filebacked', False)
    out = lowram_cmd(lf, monkeypatch, capsys, tmp_path, '4bit', 46.0, 88.0)                 # not marked: pageable (the RAM check refuses)
    assert 'PLE tables filebacked' not in out and 'memory layout: PLE tables pageable' in out


HOSTS = {256: (125.8, 240.0), 128: (61.0, 120.0), 96: (46.0, 88.0), 64: (30.0, 58.0)}   # GTT limit (half the RAM), MemAvailable


@pytest.mark.parametrize('mode', ['4bit', '4bit-cache', '4bit-long'])
@pytest.mark.parametrize('gb,layout,pools,ok', [(256, 'pinned', 'gtt', True), (128, 'pageable', 'gtt', True),
                                                (96, 'filebacked', 'userptr', True), (64, 'filebacked', 'userptr', False)])
def test_auto_layout_by_host_size(lf, monkeypatch, capsys, tmp_path, mode, gb, layout, pools, ok):
    """the auto decision on 256 / 128 / 96 / 64 GB hosts, with the RAM check of the measured resident RAM (not stubbed)."""
    gtt, avail = HOSTS[gb]
    monkeypatch.setitem(lf.COMMON['ple_host']['filebacked'], 'images', ['flashnext-test:pageable'])
    monkeypatch.setattr(lf, 'ensure_ple_raw', lambda w, out, dry: tmp_path / 'raw')
    monkeypatch.setattr(lf, 'ple_raw_preflight', lambda out, dry: None)
    real_ram_check = lf.ram_check
    seen = {}
    def ram_check(*a, **k):
        seen['missing'] = real_ram_check(*a, **k); return []
    import sys
    monkeypatch.setitem(lf.COMMON['ple_host'], 'images', ['flashnext-test:pageable'])
    monkeypatch.setattr(lf, 'meminfo_gib', lambda f: {'MemTotal': gtt * 2, 'MemAvailable': avail}[f])
    monkeypatch.setattr(lf, 'p2p_check', lambda *a, **k: None); monkeypatch.setattr(lf, 'ram_check', ram_check)
    monkeypatch.setattr(lf, 'render_gid', lambda: None); monkeypatch.setattr(lf, 'gtt_limits_gib', lambda gpus: (gtt, gtt))
    monkeypatch.setattr(lf, 'ensure_weights', lambda d, dry, *a: tmp_path / 'w'); monkeypatch.setattr(lf, 'ensure_base_cfg', lambda d, dry: tmp_path / 'cfg')
    monkeypatch.setattr(lf, 'ensure_view', lambda *a, **k: tmp_path / 'view')
    monkeypatch.setattr(sys, 'argv', ['launch-flashnext.py', '--mode', mode, '--dry-run', '--devices', 'GPU-0,GPU-1', '--cache',
                                      str(tmp_path / 'cache'), '--image', 'flashnext-test:pageable'])
    lf.main()
    out = capsys.readouterr().out
    if mode == '4bit-cache' and gb == 128:
        layout, pools = 'filebacked', 'userptr'                                               # 118 + 6 GiB does not fit 120 available
    assert f'memory layout: PLE tables {layout}, host pools {pools}' in out
    assert (seen['missing'] == []) == ok
    if not ok:
        assert '67 GiB of available RAM (have 58 GiB' in seen['missing'][0]


def test_filebacked_disk_preflight(lf, tmp_path, capsys, monkeypatch):
    import collections
    du = collections.namedtuple('du', 'total used free')
    monkeypatch.setattr(lf.shutil, 'disk_usage', lambda p: du(0, 0, 40 << 30))
    with pytest.raises(SystemExit, match='59 GiB of free disk.*has 40 GiB free.*--ple-raw-dir'):
        lf.ple_raw_preflight(tmp_path / 'w-ple-raw', False)                                   # 49 GiB raw files + 10 GiB headroom
    lf.ple_raw_preflight(tmp_path / 'w-ple-raw', True)
    assert 'WARNING: --ple-host filebacked' in capsys.readouterr().out                         # dry run: a warning
    monkeypatch.setattr(lf.shutil, 'disk_usage', lambda p: du(0, 0, 60 << 30))
    lf.ple_raw_preflight(tmp_path / 'w-ple-raw', False)                                       # enough
    raw = tmp_path / 'built'; raw.mkdir(); (raw / 'codes.e4m3').write_bytes(b'x' * 8)
    (raw / 'meta.json').write_text('{"rows_total": 2, "dim": 4}')
    monkeypatch.setattr(lf.shutil, 'disk_usage', lambda p: du(0, 0, 1 << 30))
    lf.ple_raw_preflight(raw, False)                                                          # already built: no disk needed


def test_filebacked_refused_with_vision(lf, monkeypatch, capsys, tmp_path):
    # --vision with --ple-host filebacked is refused (not measured with image input)
    with pytest.raises(SystemExit, match='not measured with --vision'):
        mode_cmd(lf, monkeypatch, capsys, tmp_path, '4bit', 60.0, ('--ple-host', 'filebacked', '--vision', '--image', 'local:vision'))


@pytest.mark.parametrize('mode,why', [('4bit-200k', 'not validated'), ('4bit-xl', 'not measured')])
def test_vision_refused_modes(lf, monkeypatch, capsys, tmp_path, mode, why):
    with pytest.raises(SystemExit, match=f'--vision is not available on --mode {mode}: image input is {why}'):
        mode_cmd(lf, monkeypatch, capsys, tmp_path, mode, 125.8, ('--vision', '--image', 'local:vision'))


@pytest.mark.parametrize('mode', ['decode', '4bit', '4bit-cache', '4bit-long'])
def test_vision_refused_on_images_without_the_vision_runtime(lf, monkeypatch, capsys, tmp_path, mode):
    # the published images before the vision runtime are refused (3-bit: the pinned r2; 4-bit: an explicit r3 --image); the 4-bit
    # 20261017 rc carries it (torch worker with --vision); an --image without a digest runs with a note
    old = () if mode == 'decode' else ('--image', 'ghcr.io/eliovp/paiton-vllm-plugin:qwen38-flashnext-rocm10-vllm029-20261011-r3@sha256:0862ca1c999f12834b6169e84bdf594eb5f8e4db4efbd79a909e32efdf16a8c0')
    with pytest.raises(SystemExit, match='--vision is not available with the image .*release 2026101'):
        mode_cmd(lf, monkeypatch, capsys, tmp_path, mode, 125.8, ('--vision',) + old)
    if mode != 'decode':
        out = mode_cmd(lf, monkeypatch, capsys, tmp_path, mode, 125.8, ('--vision',))
        assert 'image input: on (--vision)' in out and 'worker: native' not in out
    out = mode_cmd(lf, monkeypatch, capsys, tmp_path, mode, 125.8, ('--vision', '--image', 'local:vision'))
    assert 'is not checked for the vision runtime' in out and 'image input: on (--vision)' in out


@pytest.mark.parametrize('mode', ['4bit', '4bit-long'])
def test_expert_cold_auto_on_64gb_only(lf, monkeypatch, capsys, tmp_path, mode):
    """64 GB host: file-backed tables + the coldest 20 % of the experts from a file; 96 GB keeps every expert copy in RAM."""
    import sys
    monkeypatch.setitem(lf.COMMON['ple_host']['filebacked'], 'images', ['flashnext-test:cold'])
    monkeypatch.setitem(lf.COMMON['ple_host'], 'images', ['flashnext-test:cold'])
    monkeypatch.setitem(lf.COMMON['expert_cold'], 'images', ['flashnext-test:cold'])
    monkeypatch.setattr(lf, 'ensure_ple_raw', lambda w, out, dry: tmp_path / 'raw')
    monkeypatch.setattr(lf, 'ple_raw_preflight', lambda out, dry: None); monkeypatch.setattr(lf, 'cold_preflight', lambda out, s, dry: None)
    seen = {}
    real = lf.ram_check
    def ram_check(*a, **k):
        seen['missing'] = real(*a, **k); return []
    for gb, (gtt, avail) in ((96, (46.0, 88.0)), (64, (30.0, 58.0)), (48, (22.0, 44.0))):
        monkeypatch.setattr(lf, 'meminfo_gib', lambda f, g=gtt, v=avail: {'MemTotal': g * 2, 'MemAvailable': v}[f])
        monkeypatch.setattr(lf, 'p2p_check', lambda *a, **k: None); monkeypatch.setattr(lf, 'ram_check', ram_check)
        monkeypatch.setattr(lf, 'render_gid', lambda: None); monkeypatch.setattr(lf, 'gtt_limits_gib', lambda gpus, g=gtt: (g, g))
        monkeypatch.setattr(lf, 'ensure_weights', lambda d, dry, *a: tmp_path / 'w'); monkeypatch.setattr(lf, 'ensure_base_cfg', lambda d, dry: tmp_path / 'cfg')
        monkeypatch.setattr(lf, 'ensure_view', lambda *a, **k: tmp_path / 'view')
        monkeypatch.setattr(sys, 'argv', ['launch-flashnext.py', '--mode', mode, '--dry-run', '--devices', 'GPU-0,GPU-1', '--cache',
                                          str(tmp_path / 'cache'), '--image', 'flashnext-test:cold'])
        lf.main()
        out = capsys.readouterr().out; run_args = out.split('docker run')[1]
        if gb == 96:
            assert 'cold experts' not in out and 'cold=' not in run_args and seen['missing'] == []
        else:
            assert 'cold experts 20% from' in out and ',cold=0.2' in run_args and 'PAITON_FN_EXPERT_COLD_DIR=/models/expert-cold' in run_args
            assert '-expert-cold:/models/expert-cold' in run_args and 'PAITON_FN_EXPERT_COLD_CACHE=384' in run_args
            assert (seen['missing'] == []) == (gb == 64)                                       # 48 GB: refused (needs 56 GiB)


def test_expert_cold_explicit_and_refusals(lf, monkeypatch, capsys, tmp_path):
    monkeypatch.setitem(lf.COMMON['ple_host']['filebacked'], 'images', ['flashnext-test:cold'])
    monkeypatch.setitem(lf.COMMON['expert_cold'], 'images', ['flashnext-test:cold'])
    monkeypatch.setattr(lf, 'ensure_ple_raw', lambda w, out, dry: tmp_path / 'raw')
    monkeypatch.setattr(lf, 'ple_raw_preflight', lambda out, dry: None); monkeypatch.setattr(lf, 'cold_preflight', lambda out, s, dry: None)
    img = ('--image', 'flashnext-test:cold')
    out = mode_cmd(lf, monkeypatch, capsys, tmp_path, '4bit', 60.0, img + ('--ple-host', 'filebacked', '--expert-cold', '0.2'))
    assert ',cold=0.2' in out.split('docker run')[1]
    with pytest.raises(SystemExit, match='--expert-cold 0.3 is not measured on --mode 4bit'):
        mode_cmd(lf, monkeypatch, capsys, tmp_path, '4bit', 60.0, img + ('--ple-host', 'filebacked', '--expert-cold', '0.3'))
    with pytest.raises(SystemExit, match='needs the file-backed PLE tables'):
        mode_cmd(lf, monkeypatch, capsys, tmp_path, '4bit', 125.8, img + ('--ple-host', 'pinned', '--expert-cold', '0.2'))
    with pytest.raises(SystemExit, match='--expert-cold 0.2 is not measured on --mode 4bit-cache'):
        mode_cmd(lf, monkeypatch, capsys, tmp_path, '4bit-cache', 60.0, img + ('--ple-host', 'filebacked', '--expert-cold', '0.2'))
    out = mode_cmd(lf, monkeypatch, capsys, tmp_path, '4bit', 60.0, img + ('--ple-host', 'filebacked', '--expert-cold', 'off'))
    assert 'cold=' not in out.split('docker run')[1]


def test_expert_cold_pool_shrinks(lf):
    spec = 'static:vram_gib=20.75,pool=1,core=0.3,classes=1'
    full = lf.pinned_host_gib(spec, 58.23, 2); cold = lf.pinned_host_gib(lf.tier_spec(spec, cold=0.2), 58.23, 2)
    assert full - cold == pytest.approx(0.2 * 58.23)


def test_cold_preflight(lf, tmp_path, monkeypatch):
    import collections
    du = collections.namedtuple('du', 'total used free')
    monkeypatch.setattr(lf.shutil, 'disk_usage', lambda p: du(0, 0, 15 << 30))
    with pytest.raises(SystemExit, match='22 GiB of free disk'):
        lf.cold_preflight(tmp_path / 'c', 0.2, False)                                         # 0.2 x 58.2 + 10
    monkeypatch.setattr(lf.shutil, 'disk_usage', lambda p: du(0, 0, 30 << 30))
    lf.cold_preflight(tmp_path / 'c', 0.2, False)


def test_expert_cold_refused_with_vision(lf, monkeypatch, capsys, tmp_path):
    """--vision is not measured with cold experts (nor with the file-backed tables they need): refused either way."""
    monkeypatch.setitem(lf.COMMON['expert_cold'], 'images', ['local:vision'])
    with pytest.raises(SystemExit):
        mode_cmd(lf, monkeypatch, capsys, tmp_path, '4bit', 60.0, ('--vision', '--image', 'local:vision', '--expert-cold', '0.2'))
    with pytest.raises(SystemExit, match='not measured with --vision'):
        mode_cmd(lf, monkeypatch, capsys, tmp_path, '4bit', 60.0, ('--vision', '--image', 'local:vision', '--ple-host', 'filebacked',
                                                                  '--expert-cold', '0.2'))


def test_userptr_warning_only_when_gtt_is_short(lf, monkeypatch, capsys):
    # --ple-host filebacked picks user-pointer pools by design: no "host GTT 126 GiB < 49 GiB" warning when the GTT fits
    monkeypatch.setattr(lf, 'gtt_limits_gib', lambda gpus: (126.0, 126.0))
    lf.host_pool_env('userptr', ['GPU-0', 'GPU-1'], pinned_gib=47.0)
    assert 'WARNING host pools' not in capsys.readouterr().out
    monkeypatch.setattr(lf, 'gtt_limits_gib', lambda gpus: (40.0, 40.0))
    lf.host_pool_env('auto', ['GPU-0', 'GPU-1'], pinned_gib=47.0)
    assert 'WARNING host pools: host GTT 40 GiB < 49 GiB' in capsys.readouterr().out
