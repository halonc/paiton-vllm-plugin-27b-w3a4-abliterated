"""--mode 4bit-native (native Paiton worker, DESIGN 4.1 / 5): the r4.1 4-bit config with MTP off; the native worker gets the
worker class, the GPU mask (layer 1), the interpreter-start guard hook (layer 2) and its init args through the environment;
--fn-worker torch serves the same config through the torch worker (the reference); other modes refuse the native flags."""
import importlib.util
import json
import pathlib
import shlex
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


def dry(lf, monkeypatch, tmp_path, capsys, *extra, mode='4bit-native'):
    monkeypatch.setattr(lf, 'p2p_check', lambda *a, **k: None); monkeypatch.setattr(lf, 'ram_check', lambda *a, **k: [])
    monkeypatch.setattr(lf, 'render_gid', lambda: None); monkeypatch.setattr(lf, 'gtt_limits_gib', lambda gpus: (200.0, 200.0))
    monkeypatch.setattr(lf, 'ensure_weights', lambda d, dry, *a: tmp_path / 'w')
    monkeypatch.setattr(lf, 'ensure_base_cfg', lambda d, dry: tmp_path / 'cfg')
    monkeypatch.setattr(lf, 'ensure_view', lambda *a, **k: tmp_path / 'view')
    monkeypatch.setattr(sys, 'argv', ['launch-flashnext.py', '--mode', mode, '--dry-run', '--devices', 'GPU-0,GPU-1',
                                      '--cache', str(tmp_path / 'cache'), *extra])
    lf.main()
    out = capsys.readouterr().out
    line = next(l for l in out.splitlines() if l.startswith('docker run '))
    argv = shlex.split(line)
    env = dict(argv[i + 1].split('=', 1) for i, t in enumerate(argv) if t == '-e' and '=' in argv[i + 1])
    vols = [argv[i + 1] for i, t in enumerate(argv) if t == '-v']
    return out, argv, env, vols, argv[-1]


def test_mode_is_4bit_with_mtp_off(lf):
    n, b = lf.MODES['4bit-native'], lf.MODES['4bit']
    assert '--speculative-config' not in n['engine_args'] and '--speculative-config' in b['engine_args']
    # r6: --mode 4bit's torch text default is back on r3; the native modes keep the r4 configuration (= 4bit's --vision levers)
    skip = ('PAITON_FN_KV_LEASE', 'PAITON_FN_EXPERT_TIER', 'PAITON_FN_ROPE_TRIM')
    nb = {k: v for k, v in n['env'].items() if k not in skip}; bb = {k: v for k, v in {**b['env'], **b['vision']['env']}.items() if k not in skip}
    assert nb == bb   # the same configuration except the KV lease (native: off until a served lease A/B), the tier and the rope trim
    assert n['env']['PAITON_FN_KV_LEASE'] == 'off' and b['env']['PAITON_FN_KV_LEASE'] == 'on:stage=102,log=1024'
    assert [x for x in n['engine_args'] if x != '--compilation-config' and 'cudagraph' not in x] == \
        [x for i, x in enumerate(b['engine_args']) if x not in ('--speculative-config', '--compilation-config')
         and 'mtp' not in x and 'cudagraph' not in x]


def test_native_wiring(lf, monkeypatch, tmp_path, capsys):
    lib = tmp_path / 'libpaiton_fnexec.so'; lib.write_bytes(b'x')
    out, argv, env, vols, inner = dry(lf, monkeypatch, tmp_path, capsys, '--fnexec-lib', str(lib), '--card-budget-gib', '31.55')
    assert env['ROCR_VISIBLE_DEVICES'] == '99,98' and env['PAITON_FN_GPU_MASK'] == 'GPU-0,GPU-1'
    assert env['PAITON_FN_NATIVE'] == '1' and env['PAITON_FN_CARD_BUDGET_GIB'] == '31.55' and env['PAITON_FN_GUARD'] == 'enforce'
    assert env['PAITON_FN_FNEXEC_LIB'] == '/opt/paiton/runtime/fnexec/libpaiton_fnexec.so'
    assert f'{tmp_path}:/opt/paiton/runtime/fnexec:ro' in vols
    assert env['PAITON_FN_MODE_NAME'] == '4bit-native' and env['PAITON_FN_HOST_LAYOUT'] == 'pinned'
    assert any(v.endswith(':/usr/local/lib/python3.12/dist-packages/zzzz_paiton_fn_native_guard.pth:ro') for v in vols)
    assert '--worker-cls paiton_vllm_plugin.flashnext.fnworker.PaitonFNWorker' in inner and '--enforce-eager' in inner
    assert '--speculative-config' not in inner and 'ghcr.io/eliovp/paiton-vllm-plugin:qwen38-flashnext-rocm10-vllm029-20261026-pc-rc4@sha256:328be15c6f1bc3ec99b1b7a436c4a38eeb82ec36d0589b7bbef38f0e465e11cf' in argv


def test_budget_default_and_reserve_passthrough(lf, monkeypatch, tmp_path, capsys):
    lib = tmp_path / 'libpaiton_fnexec.so'; lib.write_bytes(b'x')
    monkeypatch.delenv('PAITON_FN_OUTSIDE_RESERVE_MIB', raising=False)
    out, argv, env, vols, inner = dry(lf, monkeypatch, tmp_path, capsys, '--fnexec-lib', str(lib))
    assert env['PAITON_FN_CARD_BUDGET_GIB'] == '31.55' and 'PAITON_FN_OUTSIDE_RESERVE_MIB' not in env   # unset: the executor default (256)
    monkeypatch.setenv('PAITON_FN_OUTSIDE_RESERVE_MIB', '384')
    out, argv, env, vols, inner = dry(lf, monkeypatch, tmp_path, capsys, '--fnexec-lib', str(lib))
    assert env['PAITON_FN_OUTSIDE_RESERVE_MIB'] == '384'


def test_native_env_switches(lf, monkeypatch, tmp_path, capsys):
    lib = tmp_path / 'libpaiton_fnexec.so'; lib.write_bytes(b'x')
    out, argv, env, vols, inner = dry(lf, monkeypatch, tmp_path, capsys, '--fnexec-lib', str(lib), '--native-env', 'PAITON_FN_PF_TRUE_MIXED=1')
    assert env['PAITON_FN_PF_TRUE_MIXED'] == '1'
    for bad in ('PF_TRUE_MIXED=1', 'PAITON_FN_PF_TRUE_MIXED', 'HSA_X=1'):
        with pytest.raises(SystemExit):
            dry(lf, monkeypatch, tmp_path, capsys, '--fnexec-lib', str(lib), '--native-env', bad)


def test_dev_mount(lf, monkeypatch, tmp_path, capsys):
    d = tmp_path / 'hook'; d.mkdir()
    out, argv, env, vols, inner = dry(lf, monkeypatch, tmp_path, capsys, '--fn-worker', 'torch', '--dev-mount', f'{d}:/nsv-sched-lib:ro',
                                      '--dev-mount', f'{d}:/nsv-sched')
    assert f'{d}:/nsv-sched-lib:ro' in vols and f'{d}:/nsv-sched' in vols
    for bad in (f'{d}', f'{d}:rel/path', f'{d}:/x:rx'):
        with pytest.raises(SystemExit):
            dry(lf, monkeypatch, tmp_path, capsys, '--fn-worker', 'torch', '--dev-mount', bad)


@pytest.mark.parametrize('mode,window,kv,ref', [('4bit-long-native', '262144', str(5 * 2 ** 30), '4bit-long'),
                                                 ('4bit-200k-native', '200000', str(int(1.75 * 2 ** 30)), '4bit-200k')])
def test_long_native_modes(lf, monkeypatch, tmp_path, capsys, mode, window, kv, ref):
    lib = tmp_path / 'libpaiton_fnexec.so'; lib.write_bytes(b'x')
    out, argv, env, vols, inner = dry(lf, monkeypatch, tmp_path, capsys, '--fnexec-lib', str(lib), mode=mode)
    assert env['PAITON_FN_MODE_NAME'] == mode and env['PAITON_FN_NATIVE'] == '1' and env['ROCR_VISIBLE_DEVICES'] == '99,98'
    assert f'--max-model-len {window}' in inner and f'--kv-cache-memory-bytes {kv}' in inner and '--max-num-batched-tokens 2048' in inner
    assert '--speculative-config' not in inner and '--enable-prefix-caching' not in inner and '--enforce-eager' in inner
    assert f'{tmp_path}:/opt/paiton/runtime/fnexec:ro' in vols
    with pytest.raises(SystemExit, match=ref):
        dry(lf, monkeypatch, tmp_path, capsys, '--fn-worker', 'torch', mode=mode)


def test_dev_env(lf, monkeypatch, tmp_path, capsys):
    out, argv, env, vols, inner = dry(lf, monkeypatch, tmp_path, capsys, '--dev-env', 'PAITON_FN_NLL_DIR=/nll', mode='4bit-long')
    assert env['PAITON_FN_NLL_DIR'] == '/nll'
    with pytest.raises(SystemExit):
        dry(lf, monkeypatch, tmp_path, capsys, '--dev-env', 'HSA_X=1', mode='4bit-long')


def test_cache_native_mode(lf, monkeypatch, tmp_path, capsys):
    lib = tmp_path / 'libpaiton_fnexec.so'; lib.write_bytes(b'x')
    out, argv, env, vols, inner = dry(lf, monkeypatch, tmp_path, capsys, '--fnexec-lib', str(lib), mode='4bit-cache-native')
    assert env['PAITON_FN_MODE_NAME'] == '4bit-cache-native' and env['PAITON_FN_RS_ALIGN'] == '1' and env['PAITON_FN_STEP_PREP_ALIGN'] == 'inplace'
    assert env['VLLM_USE_V2_MODEL_RUNNER'] == '0'   # native worker: V1 pinned (a forced V2 fails validation without Triton)
    assert '--enable-prefix-caching --block-size 2048 --mamba-cache-mode align' in inner and '--no-enable-prefix-caching' not in inner
    assert '--speculative-config' not in inner and '--enforce-eager' in inner
    with pytest.raises(SystemExit, match='4bit-cache'):
        dry(lf, monkeypatch, tmp_path, capsys, '--fn-worker', 'torch', mode='4bit-cache-native')


def test_long_cache_native_mode(lf, monkeypatch, tmp_path, capsys):
    lib = tmp_path / 'libpaiton_fnexec.so'; lib.write_bytes(b'x')
    out, argv, env, vols, inner = dry(lf, monkeypatch, tmp_path, capsys, '--fnexec-lib', str(lib), mode='4bit-long-cache-native')
    assert env['PAITON_FN_MODE_NAME'] == '4bit-long-cache-native' and env['PAITON_FN_RS_ALIGN'] == '1' and env['PAITON_FN_STEP_PREP_ALIGN'] == 'inplace'
    assert '--max-model-len 262144' in inner and '--kv-cache-memory-bytes 5368709120' in inner
    assert '--enable-prefix-caching --block-size 2048 --mamba-cache-mode align' in inner and '--no-enable-prefix-caching' not in inner
    assert '--speculative-config' not in inner
    with pytest.raises(SystemExit, match='4bit-long-cache'):
        dry(lf, monkeypatch, tmp_path, capsys, '--fn-worker', 'torch', mode='4bit-long-cache-native')
    out, argv, env, vols, inner = dry(lf, monkeypatch, tmp_path, capsys, '--fn-worker', 'torch', '--kv-cache-memory-bytes', '3758096384',
                                      mode='4bit-long-cache-native')   # the same contract at the torch reference's KV
    assert '--kv-cache-memory-bytes 3758096384' in inner and 'PAITON_FN_NATIVE' not in env and env['PAITON_FN_ROPE_TRIM'] == '1'
    with pytest.raises(SystemExit, match='<= 3758096384'):
        dry(lf, monkeypatch, tmp_path, capsys, '--fn-worker', 'torch', '--kv-cache-memory-bytes', '3758096385', mode='4bit-long-cache-native')
    out, argv, env, vols, inner = dry(lf, monkeypatch, tmp_path, capsys, mode='4bit-long-cache')   # its torch reference
    assert '--enable-prefix-caching --block-size 2048 --mamba-cache-mode align' in inner and '--speculative-config' in inner
    assert env['PAITON_FN_KV_LEASE'] == 'off' and 'PAITON_FN_NATIVE' not in env


def test_torch_reference(lf, monkeypatch, tmp_path, capsys):
    out, argv, env, vols, inner = dry(lf, monkeypatch, tmp_path, capsys, '--fn-worker', 'torch')
    assert env['ROCR_VISIBLE_DEVICES'] == 'GPU-0,GPU-1' and 'PAITON_FN_NATIVE' not in env and 'PAITON_FN_GPU_MASK' not in env
    assert '--worker-cls' not in inner and '--speculative-config' not in inner
    assert '"cudagraph_capture_sizes":[1,2,3,4,5,6,7,8]' in inner and not any('fnexec' in v for v in vols)


def test_guard_hook_text(lf):
    line = lf.NATIVE_GUARD_PTH.format(mod='paiton_vllm_plugin.flashnext.fnguard')
    assert line.startswith('import os; ') and "== '1' and __import__('paiton_vllm_plugin.flashnext.fnguard')" in line


def test_plugin_overlay_and_refusals(lf, monkeypatch, tmp_path, capsys):
    f = tmp_path / 'fnworker.py'; f.write_text('#')
    _, _, _, vols, _ = dry(lf, monkeypatch, tmp_path, capsys, '--fn-worker', 'torch', '--plugin-overlay', str(f))
    assert f'{f}:/usr/local/lib/python3.12/dist-packages/paiton_vllm_plugin/flashnext/fnworker.py:ro' in vols
    with pytest.raises(SystemExit, match='has no native worker'):
        dry(lf, monkeypatch, tmp_path, capsys, '--fn-worker', 'native', mode='4bit-long')


def test_native_r2_defaults(lf, monkeypatch, tmp_path, capsys):
    # every native mode runs prefill steps of <= 64 tokens through the bitwise R2 path; opt-out only through --dev-env
    lib = tmp_path / 'libpaiton_fnexec.so'; lib.write_bytes(b'x')
    native = [k for k, m in lf.MODES.items() if m.get('native')]
    assert native and all(k.endswith('-native') for k in native)
    for mode in native:
        out, argv, env, vols, inner = dry(lf, monkeypatch, tmp_path, capsys, '--fnexec-lib', str(lib), mode=mode)
        assert env['PAITON_FN_PF_R2'] == '1' and env['PAITON_FN_PF_R2_MIXED'] == '1', mode
        assert 'PAITON_FN_PF_R2' not in lf.MODES[mode]['env'], mode   # native only: the torch reference keeps its own path
    out, argv, env, vols, inner = dry(lf, monkeypatch, tmp_path, capsys, '--fnexec-lib', str(lib), '--dev-env', 'PAITON_FN_PF_R2=0',
                                      '--dev-env', 'PAITON_FN_PF_R2_MIXED=0', mode='4bit-native')
    assert env['PAITON_FN_PF_R2'] == '0' and env['PAITON_FN_PF_R2_MIXED'] == '0'
    assert sum(1 for x in argv if x.startswith('PAITON_FN_PF_R2=')) == 1


def test_native_cache_modes_v1_runner(lf, monkeypatch, tmp_path, capsys):
    # the native worker replaces the model runner, and its API server / engine see no GPU (layer 1), so a forced V2 runner fails
    # VllmConfig validation ('Model Runner V2 requires Triton'): native cache modes pin V1; the torch reference keeps V2
    lib = tmp_path / 'libpaiton_fnexec.so'; lib.write_bytes(b'x')
    for mode in [k for k, m in lf.MODES.items() if m.get('native')]:
        out, argv, env, vols, inner = dry(lf, monkeypatch, tmp_path, capsys, '--fnexec-lib', str(lib), mode=mode)
        assert env.get('VLLM_USE_V2_MODEL_RUNNER', '0') == '0', mode
    out, argv, env, vols, inner = dry(lf, monkeypatch, tmp_path, capsys, '--fn-worker', 'torch', '--kv-cache-memory-bytes', '3758096384',
                                      mode='4bit-long-cache-native')
    assert env['VLLM_USE_V2_MODEL_RUNNER'] == '1'


def test_mtp_native_mode(lf, monkeypatch, tmp_path, capsys):
    # native MTP-on (K 3): --mode 4bit's configuration through the native worker; executor mode name 4bit (manifest / kv files *-4bit.json)
    lib = tmp_path / 'libpaiton_fnexec.so'; lib.write_bytes(b'x')
    for args, mode in ((('--fnexec-lib', str(lib)), '4bit-mtp-native'), (('--fn-worker', 'native', '--fnexec-lib', str(lib)), '4bit')):
        out, argv, env, vols, inner = dry(lf, monkeypatch, tmp_path, capsys, *args, mode=mode)
        assert env['PAITON_FN_MODE_NAME'] == '4bit' and env['PAITON_FN_NATIVE'] == '1' and env['ROCR_VISIBLE_DEVICES'] == '99,98'
        assert env['PAITON_FN_PF_R2'] == '1' and env['PAITON_FN_PF_R2_MIXED'] == '1' and env['VLLM_USE_V2_MODEL_RUNNER'] == '0'
        assert '"num_speculative_tokens": 3' in inner and '--enforce-eager' in inner and '--compilation-config' not in inner
        assert '--max-model-len 98304' in inner and '--kv-cache-memory-bytes 2684354560' in inner and '--no-enable-prefix-caching' in inner   # r6: --kv-profile capacity default (17 Oct)
        assert env['PAITON_FN_TAIL_BYPASS'] == '1'   # submit timing only (an EngineCore patch), never what a step computes
    out, argv, env, vols, inner = dry(lf, monkeypatch, tmp_path, capsys, '--fn-worker', 'torch', mode='4bit')   # the r3 torch path, no native wiring
    assert 'PAITON_FN_NATIVE' not in env and '--worker-cls' not in inner and '"num_speculative_tokens": 3' in inner
    out, argv, env, vols, inner = dry(lf, monkeypatch, tmp_path, capsys, '--fn-worker', 'torch', mode='4bit-mtp-native')   # r4-contract torch reference
    assert 'PAITON_FN_NATIVE' not in env and '"num_speculative_tokens": 3' in inner and 'cudagraph_capture_sizes":[4,8,16,24,32]' in inner
    assert 'PAITON_FN_TAIL_BYPASS' not in env   # native only; a like-for-like torch arm sets it with --dev-env


def test_cache_mtp_native_mode(lf, monkeypatch, tmp_path, capsys):
    # served MTP 3 + align on the native worker: 4bit-cache-native + --mode 4bit's spec config; opt-in, no alias
    lib = tmp_path / 'libpaiton_fnexec.so'; lib.write_bytes(b'x')
    out, argv, env, vols, inner = dry(lf, monkeypatch, tmp_path, capsys, '--fnexec-lib', str(lib), mode='4bit-cache-mtp-native')
    assert env['PAITON_FN_MODE_NAME'] == '4bit-cache-mtp-native' and env['PAITON_FN_NATIVE'] == '1' and env['VLLM_USE_V2_MODEL_RUNNER'] == '0'
    assert env['PAITON_FN_PF_R2'] == '1' and env['PAITON_FN_PF_R2_MIXED'] == '1' and env['PAITON_FN_RS_ALIGN'] == '1'
    assert '"num_speculative_tokens": 3' in inner and '--enable-prefix-caching' in inner and '--mamba-cache-mode align' in inner
    assert '--block-size 2048' in inner and '--max-model-len 98304' in inner and '--kv-cache-memory-bytes 2684354560' in inner
    assert '--kv-transfer-config' not in inner and 'PAITON_FN_KV_TIER' not in env   # RAM tier stays opt-in
    with pytest.raises(SystemExit, match='has no native worker'):   # no alias from the shipped mode until a served gate passes
        dry(lf, monkeypatch, tmp_path, capsys, '--fn-worker', 'native', '--fnexec-lib', str(lib), mode='4bit-cache')


def test_kv_profiles_mtp_native(lf, monkeypatch, tmp_path, capsys):
    # user 17 Oct ~20:17 (r6): --kv-profile capacity (KV 2.5 GiB, the r3 capacity) is the default of --mode 4bit-mtp-native (= --mode 4bit);
    # speed (KV 1.5 GiB + the dynamic tier, the headline numbers) is the opt-in
    lib = tmp_path / 'libpaiton_fnexec.so'; lib.write_bytes(b'x')
    for args in ((), ('--kv-profile', 'capacity')):
        out, argv, env, vols, inner = dry(lf, monkeypatch, tmp_path, capsys, '--fnexec-lib', str(lib), *args, mode='4bit-mtp-native')
        assert '--kv-cache-memory-bytes 2684354560' in inner and env['PAITON_FN_TIER_DYNAMIC'] == '1' and 'WARNING KV cache' not in out
        assert ('KV profile capacity (the default)' in out) == (not args) and '305,338 tokens' in out and '--kv-profile speed: about 183,202 tokens' in out
    assert lf.MODES['4bit-mtp-native']['native']['kv_profile_default'] == 'capacity'
    out, argv, env, vols, inner = dry(lf, monkeypatch, tmp_path, capsys, '--fnexec-lib', str(lib), '--kv-profile', 'speed', mode='4bit-mtp-native')
    assert '--kv-cache-memory-bytes 1610612736' in inner and env['PAITON_FN_TIER_DYNAMIC'] == '1' and '183,202' in out
    assert '--kv-profile capacity: about 305,338 tokens' in out and 'WARNING KV cache' not in out
    out, argv, env, vols, inner = dry(lf, monkeypatch, tmp_path, capsys, '--fnexec-lib', str(lib), '--kv-profile', 'speed',
                                      '--native-env', 'PAITON_FN_TIER_DYNAMIC=0', mode='4bit-mtp-native')
    assert env['PAITON_FN_TIER_DYNAMIC'] == '0'   # an explicit switch wins
    out, argv, env, vols, inner = dry(lf, monkeypatch, tmp_path, capsys, '--fn-worker', 'native', '--fnexec-lib', str(lib), mode='4bit')
    assert '--kv-cache-memory-bytes 2684354560' in inner   # --mode 4bit --fn-worker native = 4bit-mtp-native: the same default
    out, argv, env, vols, inner = dry(lf, monkeypatch, tmp_path, capsys, '--fnexec-lib', str(lib), '--kv-cache-memory-bytes', '2147483648',
                                      mode='4bit-mtp-native')
    assert '--kv-cache-memory-bytes 2147483648' in inner and 'KV profile' not in out   # an explicit KV size: no profile
    out, argv, env, vols, inner = native_default_env(lf, monkeypatch, tmp_path, capsys, '4bit', '--fn-worker', 'torch')
    assert '--kv-cache-memory-bytes 2684354560' in inner and 'KV profile' not in out   # the torch worker: the r3 mode KV
    for mode in ('4bit-native', '4bit-long-native', '4bit-cache-native', '4bit-cache-mtp-native', '4bit-long-cache-native', '3bit-mtp-native'):
        out, argv, env, vols, inner = dry(lf, monkeypatch, tmp_path, capsys, '--fnexec-lib', str(lib), mode=mode)
        assert 'KV profile' not in out, mode   # other modes unchanged
    for args, mode, msg in (((), '4bit-native', 'no such KV profile'), (('--fn-worker', 'torch'), '4bit-mtp-native', 'no such KV profile'),
                            (('--kv-cache-memory-bytes', '2147483648'), '4bit-mtp-native', 'give one of the two')):
        with pytest.raises(SystemExit, match=msg):
            dry(lf, monkeypatch, tmp_path, capsys, '--fnexec-lib', str(lib), '--kv-profile', 'speed', *args, mode=mode)


def test_native_tier_dynamic_default(lf, monkeypatch, tmp_path, capsys):
    # no-flag launch = the measured configuration: every native gate served with the dynamic expert tier
    lib = tmp_path / 'libpaiton_fnexec.so'; lib.write_bytes(b'x')
    for mode in [k for k, v in lf.MODES.items() if v.get('native')]:
        out, argv, env, vols, inner = dry(lf, monkeypatch, tmp_path, capsys, '--fnexec-lib', str(lib), mode=mode)
        if mode in ('3bit-cache-native', '3bit-long-cache-native'): continue   # as gated 18 Oct: no tier / lease switch in the mode
        assert env['PAITON_FN_TIER_DYNAMIC'] == ('0' if lf.MODES[mode]['native'].get('all_vram') else '1'), mode   # 3-bit: no host experts
        assert sum(1 for x in argv if x.startswith('PAITON_FN_TIER_DYNAMIC=')) <= 1, mode
    out, argv, env, vols, inner = dry(lf, monkeypatch, tmp_path, capsys, '--fnexec-lib', str(lib), '--native-env', 'PAITON_FN_TIER_DYNAMIC=0', mode='4bit-mtp-native')
    assert env['PAITON_FN_TIER_DYNAMIC'] == '0'   # still switchable for an A/B
    out, argv, env, vols, inner = dry(lf, monkeypatch, tmp_path, capsys, '--fn-worker', 'torch', mode='4bit-mtp-native')
    assert 'PAITON_FN_TIER_DYNAMIC' not in env   # native only


def test_native_kv_lease_off(lf, monkeypatch, tmp_path, capsys):
    # the native modes do not inherit the torch-era KV lease string: off until a lease-wired lib passes a served A/B; torch modes unchanged
    lib = tmp_path / 'libpaiton_fnexec.so'; lib.write_bytes(b'x')
    for mode in [k for k, v in lf.MODES.items() if v.get('native')]:
        out, argv, env, vols, inner = dry(lf, monkeypatch, tmp_path, capsys, '--fnexec-lib', str(lib), mode=mode)
        if mode in ('3bit-cache-native', '3bit-long-cache-native'): continue   # as gated 18 Oct: no tier / lease switch in the mode
        assert env['PAITON_FN_KV_LEASE'] == ('on' if mode == '4bit-mtp-native' else 'off'), mode   # the benchmark MTP mode leases
    assert lf.MODES['4bit']['env']['PAITON_FN_KV_LEASE'] == 'on:stage=102,log=1024' and lf.MODES['4bit-cache']['env']['PAITON_FN_KV_LEASE'] == 'off'
    out, argv, env, vols, inner = dry(lf, monkeypatch, tmp_path, capsys, '--fnexec-lib', str(lib), '--native-env', 'PAITON_FN_KV_LEASE=on:log=512', mode='4bit-mtp-native')
    assert env['PAITON_FN_KV_LEASE'] == 'on:log=512'   # the served lease A/B's on arm


def test_native_mtp_async_defaults(lf, monkeypatch, tmp_path, capsys):
    # measured (async vs sync, identical tokens, 0 engine deaths): the native MTP modes run async + early tokens + dead-row drop;
    # needs an async-capable lib (prefill under an in-flight step) and the glue's async ticket guard
    lib = tmp_path / 'libpaiton_fnexec.so'; lib.write_bytes(b'x')
    for mode in ('4bit-mtp-native', '4bit-cache-mtp-native'):
        out, argv, env, vols, inner = dry(lf, monkeypatch, tmp_path, capsys, '--fnexec-lib', str(lib), mode=mode)
        assert (env['PAITON_FN_ASYNC'], env['PAITON_FN_EARLY_TOKENS'], env['PAITON_FN_DROP_DEAD_ROWS']) == ('1', '1', '1'), mode
    for mode in ('4bit-native', '4bit-long-native', '4bit-cache-native', '4bit-long-cache-native', '4bit-200k-native'):   # MTP-off modes unchanged
        out, argv, env, vols, inner = dry(lf, monkeypatch, tmp_path, capsys, '--fnexec-lib', str(lib), mode=mode)
        assert 'PAITON_FN_ASYNC' not in env and 'PAITON_FN_EARLY_TOKENS' not in env, mode
    out, argv, env, vols, inner = dry(lf, monkeypatch, tmp_path, capsys, '--fnexec-lib', str(lib), '--native-env', 'PAITON_FN_ASYNC=0', mode='4bit-mtp-native')
    assert env['PAITON_FN_ASYNC'] == '0'   # still switchable (a sync arm)


def test_native_mtp_requires_async_scheduling(lf, monkeypatch, tmp_path, capsys):
    # the native worker hands no drafts to a synchronous vLLM scheduler (take_draft_token_ids is None): MTP modes pass
    # --async-scheduling explicitly (vLLM then fails at start instead of falling back) and refuse anything that turns it off
    lib = tmp_path / 'libpaiton_fnexec.so'; lib.write_bytes(b'x')
    for mode in ('4bit-mtp-native', '4bit-cache-mtp-native'):
        out, argv, env, vols, inner = dry(lf, monkeypatch, tmp_path, capsys, '--fnexec-lib', str(lib), mode=mode)
        assert '--async-scheduling' in inner and '--no-async-scheduling' not in inner, mode
        out, argv, env, vols, inner = dry(lf, monkeypatch, tmp_path, capsys, '--fnexec-lib', str(lib), '--native-env', 'PAITON_FN_ASYNC=0', mode=mode)
        assert '--async-scheduling' in inner, mode   # a sync-step arm still schedules async
    for mode in ('4bit-native', '4bit-cache-native'):   # MTP off: vLLM's own default
        out, argv, env, vols, inner = dry(lf, monkeypatch, tmp_path, capsys, '--fnexec-lib', str(lib), mode=mode)
        assert 'async-scheduling' not in inner, mode
    out, argv, env, vols, inner = dry(lf, monkeypatch, tmp_path, capsys, '--fn-worker', 'torch', mode='4bit-mtp-native')
    assert '--async-scheduling' not in inner   # the torch worker: unchanged
    nat = lf.MODES['4bit-mtp-native']['native']; ea = list(nat.get('engine_args_native', []))
    spec = list(lf.MODES['4bit-mtp-native']['engine_args']); i = spec.index('--speculative-config')
    spec[i + 1] = json.dumps(dict(json.loads(spec[i + 1]), disable_padded_drafter_batch=True))
    for bad, patch in (('--no-async-scheduling', lambda mp: mp.setitem(nat, 'engine_args_native', ea + ['--no-async-scheduling'])),
                       ('disable_padded_drafter_batch', lambda mp: mp.setitem(lf.MODES['4bit-mtp-native'], 'engine_args', spec)),
                       ('--distributed-executor-backend ray', lambda mp: mp.setitem(nat, 'engine_args_native', ea + ['--distributed-executor-backend', 'ray']))):
        with monkeypatch.context() as mp:
            patch(mp)
            with pytest.raises(SystemExit, match='needs vLLM async scheduling; ' + bad):
                dry(lf, mp, tmp_path, capsys, '--fnexec-lib', str(lib), mode='4bit-mtp-native')


def test_native_kv_tier_opt_in(lf, monkeypatch, tmp_path, capsys):
    # native RAM prefix tier: off by default (lightest stable), opt-in on the native cache modes only
    lib = tmp_path / 'libpaiton_fnexec.so'; lib.write_bytes(b'x')
    for mode in ('4bit-cache-native', '4bit-long-cache-native'):
        out, argv, env, vols, inner = dry(lf, monkeypatch, tmp_path, capsys, '--fnexec-lib', str(lib), mode=mode)
        if mode == '4bit-cache-native':   # DEV mode: no sizing rule, the tier stays opt-in
            assert '--kv-transfer-config' not in inner and 'PAITON_FN_KV_TIER' not in env and 'PAITON_FN_KVT_LAYOUT' not in env
        else:   # release candidate: --native-kv-tier auto by default
            assert 'native RAM prefix tier:' in out and ('--kv-transfer-config' in inner) == ('auto (' in out)
        assert '--prefix-cache-retention-interval' not in inner
        out, argv, env, vols, inner = dry(lf, monkeypatch, tmp_path, capsys, '--fnexec-lib', str(lib), '--native-kv-tier', '16',
                                          '--prefix-cache-retention-interval', '32768', mode=mode)
        i = argv.index('-c'); eng = shlex.split(argv[i + 1].split('exec ', 1)[1])
        cfg = json.loads(eng[eng.index('--kv-transfer-config') + 1])
        assert cfg['kv_connector'] == 'OffloadingConnector' and cfg['kv_connector_extra_config']['spec_name'] == 'NativeCPUOffloadingSpec'
        assert cfg['kv_connector_extra_config']['cpu_bytes_to_use'] == 32 << 30            # both ranks
        assert env['PAITON_FN_KV_TIER'] == '1' and env['PAITON_FN_KVT_LAYOUT'] == 'pageable'
        assert any(v.endswith(':ro') and 'zz_paiton_fn_kv_tier.pth' in v for v in vols)   # the kv_tier_boot overlays' hook
        assert eng[eng.index('--prefix-cache-retention-interval') + 1] == '32768'
    out, argv, env, vols, inner = dry(lf, monkeypatch, tmp_path, capsys, '--fnexec-lib', str(lib), '--native-kv-tier', '8',
                                      '--native-kv-tier-layout', 'pinned', mode='4bit-cache-native')
    assert env['PAITON_FN_KVT_LAYOUT'] == 'pinned'
    for args, mode in (((), '4bit-native'), (('--fn-worker', 'torch'), '4bit-cache-native')):
        with pytest.raises(SystemExit, match='no native RAM prefix tier'):
            dry(lf, monkeypatch, tmp_path, capsys, '--fnexec-lib', str(lib), '--native-kv-tier', '8', *args, mode=mode)
    with pytest.raises(SystemExit, match='multiple of the 2048'):
        dry(lf, monkeypatch, tmp_path, capsys, '--fnexec-lib', str(lib), '--prefix-cache-retention-interval', '1000', mode='4bit-cache-native')
    with pytest.raises(SystemExit, match='no prefix caching'):
        dry(lf, monkeypatch, tmp_path, capsys, '--fnexec-lib', str(lib), '--prefix-cache-retention-interval', '32768', mode='4bit-native')


def test_fnexec_in_image(lf, monkeypatch, tmp_path, capsys):
    # an image that carries /opt/paiton/runtime/fnexec needs no --fnexec-lib (nothing mounted over it); --fnexec-lib stays the dev override
    assert lf.COMMON['image_4bit_native'].get('fnexec_in_image')   # the 20261017 rc carries the executor and the 4-bit release modes' files
    key = lf.MODES['4bit-native'].get('image_key', 'image')
    with pytest.raises(SystemExit, match='carries the native executor files of 4bit, 4bit-long-native, 3bit, 4bit-long-cache-native, 3bit-cache-native, 3bit-long-cache-native, 4bit-long-cache-mtp-native, 3bit-long-cache-mtp-native only'):
        dry(lf, monkeypatch, tmp_path, capsys, mode='4bit-native')   # a dev mode: no mode files in the image
    monkeypatch.setitem(lf.COMMON, key, dict(lf.COMMON[key], fnexec_modes=None))
    out, argv, env, vols, inner = dry(lf, monkeypatch, tmp_path, capsys, mode='4bit-native')
    assert env['PAITON_FN_FNEXEC_LIB'] == '/opt/paiton/runtime/fnexec/libpaiton_fnexec.so'
    assert not any(v.endswith(':/opt/paiton/runtime/fnexec:ro') for v in vols)
    lib = tmp_path / 'libpaiton_fnexec.so'; lib.write_bytes(b'x')
    out, argv, env, vols, inner = dry(lf, monkeypatch, tmp_path, capsys, '--fnexec-lib', str(lib), mode='4bit-native')
    assert any(v == f'{tmp_path}:/opt/paiton/runtime/fnexec:ro' for v in vols)


def test_native_mtp_drafter_default(lf, monkeypatch, tmp_path, capsys):
    # the distilled drafter (B4gh5-mtpft) is the default of the native MTP modes; --drafter stock = the published checkpoint
    lib = tmp_path / 'libpaiton_fnexec.so'; lib.write_bytes(b'x')
    want = lf.COMMON['weights_4bit_mtpft']['drafter_sha256']
    w = tmp_path / 'w'; w.mkdir()
    (w / 'SHA256SUMS').write_text(''.join(f'{h}  ./{f}\n' for f, h in want.items()))
    for mode in ('4bit-mtp-native', '4bit-cache-mtp-native'):
        out, argv, env, vols, inner = dry(lf, monkeypatch, tmp_path, capsys, '--fnexec-lib', str(lib), '--weights', str(w), mode=mode)
        assert 'drafter: distilled (weights_4bit_mtpft, the mode default)' in out and f'({len(want)} files match SHA256SUMS)' in out
        assert 'not the B4gh5-mtpft checkpoint' not in out, mode
        out, argv, env, vols, inner = dry(lf, monkeypatch, tmp_path, capsys, '--fnexec-lib', str(lib), '--weights', str(w), '--drafter', 'stock', mode=mode)
        assert 'drafter: stock (weights_4bit)' in out and 'match SHA256SUMS' not in out, mode
    (w / 'SHA256SUMS').write_text('00  ./mtp/mtp-draft-e4m3.safetensors\n')   # a stock (or other) checkpoint under the distilled default
    out, argv, env, vols, inner = dry(lf, monkeypatch, tmp_path, capsys, '--fnexec-lib', str(lib), '--weights', str(w), mode='4bit-mtp-native')
    assert 'drafter: distilled (weights_4bit_mtpft, the mode default)' in out and 'the image drafter b4gh5-mtpft' in out   # D1: baked
    assert 'WARNING: weights' in out and 'lacks' in out   # dry run: this toy checkpoint lacks the published drafter files (a refusal on a real start)
    key = lf.MODES['4bit-mtp-native'].get('image_key', 'image')   # an image without the baked drafter: the checkpoint-only path
    monkeypatch.setitem(lf.COMMON, key, {k: v for k, v in lf.COMMON[key].items() if k != 'drafter_in_image'})
    out, argv, env, vols, inner = dry(lf, monkeypatch, tmp_path, capsys, '--fnexec-lib', str(lib), '--weights', str(w), mode='4bit-mtp-native')
    assert 'drafter: stock (the B4gh5-mtpft drafter files are not in' in out   # the mode default without its files: the stock drafter
    out, argv, env, vols, inner = dry(lf, monkeypatch, tmp_path, capsys, '--fnexec-lib', str(lib), '--weights', str(w), '--drafter', 'distilled',
                                      mode='4bit-mtp-native')
    assert 'WARNING: weights' in out and 'not the B4gh5-mtpft checkpoint' in out   # explicit --drafter distilled, dry run: a warning
    with pytest.raises(SystemExit, match='not the B4gh5-mtpft checkpoint'):
        lf.drafter_check(w, lf.COMMON['weights_4bit_mtpft'], False)   # a real start: refused
    for mode in ('4bit-native', '4bit-cache-native'):   # no drafter choice without MTP
        with pytest.raises(SystemExit, match='have a drafter choice'):
            dry(lf, monkeypatch, tmp_path, capsys, '--fnexec-lib', str(lib), '--drafter', 'stock', mode=mode)
    monkeypatch.undo()   # dry() stubs ensure_weights
    with pytest.raises(SystemExit, match='not downloadable yet'):   # local-only: never falls back to another repo's revision
        lf.ensure_weights(tmp_path / 'absent', False, lf.COMMON['weights_4bit_mtpft']['hf_repo'], None)


def test_native_manifest_matches_checkpoint(lf, tmp_path):
    # the executor manifest next to the lib is hashed from one checkpoint: container.manifest_sha256 == sha256(<weights>/manifest.json)
    import hashlib, types
    libdir = tmp_path / 'lib'; libdir.mkdir(); lib = libdir / 'libpaiton_fnexec.so'; lib.write_bytes(b'x')
    w = tmp_path / 'w'; w.mkdir(); (w / 'manifest.json').write_text('{"ck": 1}')
    good = hashlib.sha256(b'{"ck": 1}').hexdigest()
    a = types.SimpleNamespace(mode='4bit-mtp-native', weights=str(w), dry_run=False); env = {'PAITON_FN_MODE_NAME': '4bit'}
    lf.native_manifest_check(a, env, lib)   # no manifest next to the lib: skipped
    (libdir / 'manifest-4bit.json').write_text(json.dumps({'container': {'manifest_sha256': good}}))
    lf.native_manifest_check(a, env, lib)   # matching
    (libdir / 'manifest-4bit.json').write_text(json.dumps({'container': {'manifest_sha256': '0996' + '0' * 60}}))
    with pytest.raises(SystemExit, match='was built from a checkpoint whose manifest.json is 09960000'):
        lf.native_manifest_check(a, env, lib)
    lf.native_manifest_check(a, env, lib, tmp_path / 'absent')   # the resolved checkpoint wins (image drafter: no manifest.json on the host)


def test_native_3bit_mtp_mode(lf, monkeypatch, tmp_path, capsys):
    # 16 Oct: --mode decode through the native worker; every expert in VRAM, so no dynamic tier / lease / ACT lend
    lib = tmp_path / 'libpaiton_fnexec.so'; lib.write_bytes(b'x')
    out, argv, env, vols, inner = dry(lf, monkeypatch, tmp_path, capsys, '--fnexec-lib', str(lib), mode='3bit-mtp-native')
    assert (env['PAITON_FN_MODE_NAME'], env['PAITON_FN_TIER_DYNAMIC'], env['PAITON_FN_KV_LEASE']) == ('3bit', '0', 'off')
    assert (env['PAITON_FN_ASYNC'], env['PAITON_FN_EARLY_TOKENS'], env['PAITON_FN_DROP_DEAD_ROWS']) == ('1', '1', '1')
    assert 'PAITON_FN_EXPERT_TIER' not in env and 'PAITON_FN_TIER_LEND_ACT' not in env
    four = lf.MODES['4bit-mtp-native']['env']   # the gated 4-bit native config's capsule / eviction / rope / subchunk keys
    assert all(env[k] == four[k] for k in ('PAITON_FN_ROPE_TRIM', 'PAITON_FN_SUBCHUNK'))
    # r6 (17 Oct): the W3 prefill down162 + gate-up capsule in its own bundle dir; 4-bit keeps moe-grouped-r1
    assert (env['PAITON_MOE_GROUPED_PATH'], env['PAITON_MOE_PF_W3']) == ('/opt/paiton/runtime/moe-grouped-w3', 'down162+gu')
    assert four['PAITON_MOE_GROUPED_PATH'] == '/opt/paiton/runtime/moe-grouped-r1' and 'PAITON_MOE_PF_W3' not in four
    assert env['HSA_USERPTR_FOR_PAGED_MEM'] == '0'   # host pools auto: GTT pools when the GTT holds the host PLE (dry(): 200 GiB free)
    assert '--kv-cache-memory-bytes 4831838208' in inner and '--async-scheduling' in inner and '--compilation-config' not in inner
    # 17 Oct: the distilled drafter is the 3-bit default (decode +9.0 %); its checkpoint is local only, so without it: stock
    assert 'drafter: stock (the B4gh5-mtpft checkpoint is not at' in out or 'drafter: distilled (weights_3bit_mtpft, the mode default)' in out
    assert (env['PAITON_FN_PF_RS_LL'], env['PAITON_FN_ADMIT'], env['PAITON_FN_ADMIT_MARGIN_MS'], env['PAITON_FN_PF_MIXED_MTP']) == ('1', 'finish', '5', '1')
    assert env['PAITON_FN_MALL_PF'] == 'd1:mlp:4:40,d2:mlp:4,d3-8:win:4' and env['PAITON_TP_COLLECTIVE_PATH'] == '/opt/paiton/runtime/tp-collective-rsll/tp_collective.so'
    for bad in (('--native-env', 'PAITON_FN_KV_LEASE=on'), ('--native-env', 'PAITON_FN_TIER_DYNAMIC=1'), ('--dev-env', 'PAITON_FN_TIER_LEND_ACT=1')):
        with pytest.raises(SystemExit, match='every expert is in VRAM'):
            dry(lf, monkeypatch, tmp_path, capsys, '--fnexec-lib', str(lib), *bad, mode='3bit-mtp-native')
    with pytest.raises(SystemExit, match='native only'):
        dry(lf, monkeypatch, tmp_path, capsys, '--fn-worker', 'torch', mode='3bit-mtp-native')
    with pytest.raises(SystemExit, match='has no such KV profile'):
        dry(lf, monkeypatch, tmp_path, capsys, '--fnexec-lib', str(lib), '--kv-profile', 'capacity', mode='3bit-mtp-native')
    out, argv, env, vols, inner = dry(lf, monkeypatch, tmp_path, capsys, '--fnexec-lib', str(lib), '--fn-worker', 'native', mode='decode')
    assert 'mode decode --fn-worker native: --mode 3bit-mtp-native' in out and env['PAITON_FN_MODE_NAME'] == '3bit'
    out, argv, env, vols, inner = dry(lf, monkeypatch, tmp_path, capsys, '--fn-worker', 'torch', mode='decode')   # the torch 3-bit mode itself: unchanged
    assert 'PAITON_FN_MODE_NAME' not in env and '--compilation-config' in inner


def test_native_spec_tokens_dev_switch(lf, monkeypatch, tmp_path, capsys):
    # SPECN SPEC2 (16 Oct): opt-in num_speculative_tokens 4 / 5 on the native MTP modes; the lib's kv files must match that K
    lib = tmp_path / 'libpaiton_fnexec.so'; lib.write_bytes(b'x')
    for mode in ('4bit-mtp-native', '3bit-mtp-native'):
        out, argv, env, vols, inner = dry(lf, monkeypatch, tmp_path, capsys, '--fnexec-lib', str(lib), mode=mode)
        assert '"num_speculative_tokens": 3' in inner, mode   # default: the mode
        out, argv, env, vols, inner = dry(lf, monkeypatch, tmp_path, capsys, '--fnexec-lib', str(lib), '--spec-tokens', '5', mode=mode)
        assert '"num_speculative_tokens": 5' in inner and '"method": "mtp"' in inner and '--async-scheduling' in inner, mode
    with pytest.raises(SystemExit, match='serves prefix caching'):
        dry(lf, monkeypatch, tmp_path, capsys, '--fnexec-lib', str(lib), '--spec-tokens', '5', mode='4bit-cache-mtp-native')
    out, argv, env, vols, inner = dry(lf, monkeypatch, tmp_path, capsys, '--fnexec-lib', str(lib), '--spec-tokens', '3', mode='4bit-cache-mtp-native')
    assert '"num_speculative_tokens": 3' in inner
    with pytest.raises(SystemExit, match='has no MTP'):
        dry(lf, monkeypatch, tmp_path, capsys, '--fnexec-lib', str(lib), '--spec-tokens', '5', mode='4bit-native')
    with pytest.raises(SystemExit, match='has no native worker'):
        dry(lf, monkeypatch, tmp_path, capsys, '--fn-worker', 'torch', '--spec-tokens', '5', mode='4bit')


def test_native_mtp_admit_default(lf, monkeypatch, tmp_path, capsys):
    # measured: 4bit-mtp-native runs the admission timing (PAITON_FN_ADMIT=finish, margin 5 ms) by default; off by env;
    # other native modes unchanged; a development fnadmit mount replaces the launcher's hook
    import os, subprocess
    lib = tmp_path / 'libpaiton_fnexec.so'; lib.write_bytes(b'x')
    hook = lambda vols: [v for v in vols if v.endswith(':ro') and 'zzzzz_paiton_fn_admit.pth' in v]
    out, argv, env, vols, inner = dry(lf, monkeypatch, tmp_path, capsys, '--fnexec-lib', str(lib), mode='4bit-mtp-native')
    assert (env['PAITON_FN_ADMIT'], env['PAITON_FN_ADMIT_MARGIN_MS']) == ('finish', '5') and len(hook(vols)) == 1
    assert hook(vols)[0].startswith(str(tmp_path / 'cache-admit.pth') + ':')
    for off in ('--native-env', '--dev-env'):
        out, argv, env, vols, inner = dry(lf, monkeypatch, tmp_path, capsys, '--fnexec-lib', str(lib), off, 'PAITON_FN_ADMIT=off', mode='4bit-mtp-native')
        assert env['PAITON_FN_ADMIT'] == 'off' and not hook(vols), off
    dev = tmp_path / 'fnadmit.pth'; dev.write_text('')
    out, argv, env, vols, inner = dry(lf, monkeypatch, tmp_path, capsys, '--fnexec-lib', str(lib), '--dev-mount',
                                      f'{dev}:/usr/local/lib/python3.12/dist-packages/zzzzz_fnadmit.pth:ro', mode='4bit-mtp-native')
    assert not hook(vols)
    for mode in ('4bit-cache-mtp-native', '4bit-native'):   # 3bit-mtp-native: fnadmit on (test_native_3bit_mtp_mode)
        out, argv, env, vols, inner = dry(lf, monkeypatch, tmp_path, capsys, '--fnexec-lib', str(lib), mode=mode)
        assert 'PAITON_FN_ADMIT' not in env and not hook(vols), mode
    run = lambda v: subprocess.run([sys.executable, '-c', lf.ADMIT_PTH], env={**os.environ, 'PAITON_FN_ADMIT': v}, capture_output=True, text=True)
    r = run('finish')   # this host has no image site-packages: the hook says so and does nothing else
    assert r.returncode == 0 and 'admission timing off' in r.stderr
    r = run('off')
    assert r.returncode == 0 and r.stderr == ''


def test_native_mtp_head4_defaults(lf, monkeypatch, tmp_path, capsys):
    # measured (libs c96e2190 + 0b6a61f1): 4bit-mtp-native runs the idle-KV lease + the per-bucket S1 MALL lookahead by default; both switchable off
    lib = tmp_path / 'libpaiton_fnexec.so'; lib.write_bytes(b'x')
    out, argv, env, vols, inner = dry(lf, monkeypatch, tmp_path, capsys, '--fnexec-lib', str(lib), mode='4bit-mtp-native')
    assert (env['PAITON_FN_KV_LEASE'], env['PAITON_FN_MALL_PF'], env['PAITON_FN_TIER_DYNAMIC']) == ('on', 'd1-2:mlp:4,d3-8:win:4', '1')
    assert env['PAITON_FN_PF_MIXED_MTP'] == '1'   # measured (MIXED-MTP lib 3a616b29): one-launch MIXED steps by default
    assert sum(1 for x in argv if x.startswith('PAITON_FN_KV_LEASE=')) == 1
    out, argv, env, vols, inner = dry(lf, monkeypatch, tmp_path, capsys, '--fnexec-lib', str(lib), '--native-env', 'PAITON_FN_KV_LEASE=off',
                                      '--native-env', 'PAITON_FN_MALL_PF=off', mode='4bit-mtp-native')
    assert (env['PAITON_FN_KV_LEASE'], env['PAITON_FN_MALL_PF']) == ('off', 'off')
    for mode in ('4bit-cache-mtp-native', '4bit-native'):   # only the benchmark MTP modes flip (3bit-mtp-native: MALL_PF on, lease off)
        out, argv, env, vols, inner = dry(lf, monkeypatch, tmp_path, capsys, '--fnexec-lib', str(lib), mode=mode)
        assert env['PAITON_FN_KV_LEASE'] == 'off' and 'PAITON_FN_MALL_PF' not in env, mode
    with pytest.raises(SystemExit, match='not supported together'):   # MALL lookahead + RAM prefix tier: refused at launch
        dry(lf, monkeypatch, tmp_path, capsys, '--fnexec-lib', str(lib), '--native-kv-tier', '8', '--native-env', 'PAITON_FN_MALL_PF=mlp:4',
            mode='4bit-cache-native')


def native_default_env(lf, monkeypatch, tmp_path, capsys, mode, *extra, gtt=200.0, avail=500.0, native_default=True):
    # r6 ships with native.is_default false (user 17 Oct ~21:02: torch default everywhere); the switch stays one modes.json line, so the
    # native-default machinery is still tested with it flipped on here
    for k in ('3bit-mtp-native', '4bit-mtp-native'):
        monkeypatch.setitem(lf.MODES, k, dict(lf.MODES[k], native=dict(lf.MODES[k]['native'], is_default=native_default)))
    monkeypatch.setattr(lf, 'meminfo_gib', lambda k: avail)
    real = lf.gtt_limits_gib
    out, argv, env, vols, inner = dry(lf, monkeypatch, tmp_path, capsys, *extra, mode=mode) if gtt == 200.0 else \
        dry_gtt(lf, monkeypatch, tmp_path, capsys, gtt, mode, *extra)
    return out, argv, env, vols, inner


def dry_gtt(lf, monkeypatch, tmp_path, capsys, gtt, mode, *extra):
    orig = lf.main
    def main():
        monkeypatch.setattr(lf, 'gtt_limits_gib', lambda gpus: (gtt, gtt)); return orig()
    monkeypatch.setattr(lf, 'main', main)
    try:
        return dry(lf, monkeypatch, tmp_path, capsys, *extra, mode=mode)
    finally:
        monkeypatch.setattr(lf, 'main', orig)


def test_release_native_default(lf, monkeypatch, tmp_path, capsys):
    # 4-bit release 20261017: --mode 4bit serves through 4bit-mtp-native from the image's executor (no --fnexec-lib), RS-LL on with the
    # RS-LL TP collective capsule, the distilled drafter's manifest (the executor default manifest-4bit.json)
    dd = tmp_path / 'mtpft'; dd.mkdir(); (dd / 'SHA256SUMS').write_text('')   # the distilled checkpoint present at its default dir
    w = tmp_path / 'w'; w.mkdir(exist_ok=True)   # dry() returns tmp_path / 'w' as the weights: give it the distilled drafter files
    (w / 'SHA256SUMS').write_text(''.join(f'{h}  ./{f}\n' for f, h in lf.COMMON['weights_4bit_mtpft']['drafter_sha256'].items()))
    monkeypatch.setitem(lf.COMMON, 'weights_4bit_mtpft', dict(lf.COMMON['weights_4bit_mtpft'], default_dir=str(dd)))
    out, argv, env, vols, inner = native_default_env(lf, monkeypatch, tmp_path, capsys, '4bit')
    assert 'mode 4bit: --mode 4bit-mtp-native (the native worker' in out and 'the default, --fn-worker torch for the torch worker' in out
    assert env['PAITON_FN_NATIVE'] == '1' and env['PAITON_FN_MODE_NAME'] == '4bit' and 'PAITON_FN_MANIFEST' not in env
    assert env['PAITON_FN_PF_RS_LL'] == '1' and env['PAITON_TP_COLLECTIVE_PATH'] == '/opt/paiton/runtime/tp-collective-rsll/tp_collective.so'
    assert env['PAITON_FN_ADMIT'] == 'finish' and env['PAITON_FN_FNEXEC_LIB'] == '/opt/paiton/runtime/fnexec/libpaiton_fnexec.so'
    assert not any(v.endswith(':/opt/paiton/runtime/fnexec:ro') for v in vols) and 'ghcr.io/eliovp/paiton-vllm-plugin:qwen38-flashnext-rocm10-vllm029-20261026-pc-rc4@sha256:328be15c6f1bc3ec99b1b7a436c4a38eeb82ec36d0589b7bbef38f0e465e11cf' in argv
    # D1 switch: the stock drafter selects the image's stock manifest
    out, argv, env, vols, inner = native_default_env(lf, monkeypatch, tmp_path, capsys, '4bit', '--drafter', 'stock')
    assert env['PAITON_FN_MANIFEST'] == '/opt/paiton/runtime/fnexec/manifest-4bit-stock.json'
    # a dev --fnexec-lib: its own dir's manifest (no PAITON_FN_MANIFEST)
    lib = tmp_path / 'libpaiton_fnexec.so'; lib.write_bytes(b'x')
    out, argv, env, vols, inner = native_default_env(lf, monkeypatch, tmp_path, capsys, '4bit', '--drafter', 'stock', '--fnexec-lib', str(lib))
    assert 'PAITON_FN_MANIFEST' not in env and f'{tmp_path}:/opt/paiton/runtime/fnexec:ro' in vols


@pytest.mark.parametrize('extra, why', [(('--fn-worker', 'torch'), None), (('--vision',), '--vision: the torch worker'),
                                        (('--ple-host', 'pageable'), 'the torch worker (--ple-host pageable')])
def test_release_native_default_keeps_torch(lf, monkeypatch, tmp_path, capsys, extra, why):
    out, argv, env, vols, inner = native_default_env(lf, monkeypatch, tmp_path, capsys, '4bit', *extra)
    assert 'PAITON_FN_NATIVE' not in env and '4bit-mtp-native' not in ' '.join(l for l in out.splitlines() if not l.startswith('mode 4bit')), out[-600:]
    assert why is None or why in out


def test_release_native_default_low_ram_host(lf, monkeypatch, tmp_path, capsys):
    # the native worker is measured with the PLE tables pinned only: a host whose GTT limit cannot hold them keeps the torch worker
    out, argv, env, vols, inner = native_default_env(lf, monkeypatch, tmp_path, capsys, '4bit', gtt=40.0)
    assert 'mode 4bit: the torch worker (GTT limit 40 GiB <' in out and 'PAITON_FN_NATIVE' not in env


def test_release_native_default_3bit_decode(lf, monkeypatch, tmp_path, capsys):
    # r6 (17 Oct): --mode decode serves through 3bit-mtp-native by default (the image carries the 3-bit executor files)
    out, argv, env, vols, inner = native_default_env(lf, monkeypatch, tmp_path, capsys, 'decode')
    assert env['PAITON_FN_NATIVE'] == '1' and env['PAITON_FN_MODE_NAME'] == '3bit' and 'mode decode: --mode 3bit-mtp-native (the native worker' in out


@pytest.mark.parametrize('mode', ['4bit-long', '4bit-cache'])
def test_release_native_default_other_modes_unchanged(lf, monkeypatch, tmp_path, capsys, mode):
    # 4bit-long (stays torch, MTP on; 4bit-long-native is opt-in), 4bit-cache (prefix caching: torch)
    out, argv, env, vols, inner = native_default_env(lf, monkeypatch, tmp_path, capsys, mode)
    assert 'PAITON_FN_NATIVE' not in env and '(the native worker' not in out


def test_d1_default_drafter_absent_falls_back_to_stock(lf, monkeypatch, tmp_path, capsys):
    # D1 open: the distilled drafter is local only; without it (and no --weights / --drafter) the native MTP default serves the published
    # checkpoint's drafter with its manifest, with a note, instead of refusing to start
    monkeypatch.setitem(lf.COMMON, 'weights_4bit_mtpft', dict(lf.COMMON['weights_4bit_mtpft'], default_dir=str(tmp_path / 'absent')))
    key = lf.MODES['4bit-mtp-native'].get('image_key', 'image')   # D1 bake: only an image without the drafter falls back (checkpoint-only path)
    monkeypatch.setitem(lf.COMMON, key, {k: v for k, v in lf.COMMON[key].items() if k != 'drafter_in_image'})
    out, argv, env, vols, inner = native_default_env(lf, monkeypatch, tmp_path, capsys, '4bit')
    assert 'drafter: stock (the B4gh5-mtpft checkpoint is not at' in out and 'drafter: stock (weights_4bit)' in out
    assert env['PAITON_FN_MANIFEST'] == '/opt/paiton/runtime/fnexec/manifest-4bit-stock.json' and env['PAITON_FN_NATIVE'] == '1'
    w = tmp_path / 'w'; w.mkdir(); (w / 'SHA256SUMS').write_text('00  ./mtp/mtp-draft-e4m3.safetensors\n')   # the published checkpoint
    out, argv, env, vols, inner = native_default_env(lf, monkeypatch, tmp_path, capsys, '4bit', '--weights', str(w))
    assert 'drafter: stock (the B4gh5-mtpft drafter files are not in' in out and env['PAITON_FN_MANIFEST'].endswith('/manifest-4bit-stock.json')
    out, argv, env, vols, inner = native_default_env(lf, monkeypatch, tmp_path, capsys, '4bit', '--weights', str(w), '--drafter', 'distilled')
    assert 'not the B4gh5-mtpft checkpoint' in out   # an explicit --drafter distilled keeps the check (a refusal on a real start)


def test_d1_overlay_assembly(lf, monkeypatch, tmp_path):
    # D1 (b): base checkpoint + the repo folder mtp-ft/ at the overlay revision -> hard-linked view, overridden files, merged SHA256SUMS
    import hashlib
    h = lambda b: hashlib.sha256(b).hexdigest()
    base = tmp_path / 'base'; (base / 'mtp').mkdir(parents=True)
    files = {'model.safetensors': b'trunk', 'mtp/mtp-draft-e4m3.safetensors': b'stock-draft', 'manifest.json': b'{"m": 1}'}
    for f, b in files.items(): (base / f).write_bytes(b)
    (base / 'SHA256SUMS').write_text(''.join(f'{h(b)}  ./{f}\n' for f, b in files.items()))
    over = {'mtp/mtp-draft-e4m3.safetensors': b'distilled-draft', 'manifest.json': b'{"m": 2}'}
    def fake_dl(repo, rev, dest, patterns=None, dry=False):
        assert rev == 'rev-ft' and patterns[0] == 'mtp-ft/*'
        src = pathlib.Path(dest) / 'mtp-ft'; (src / 'mtp').mkdir(parents=True, exist_ok=True)
        for f, b in over.items(): (src / f).write_bytes(b)
        (src / 'SHA256SUMS').write_text(''.join(f'{h(b)}  ./{f}\n' for f, b in over.items()))
    monkeypatch.setattr(lf, 'hf_download', fake_dl)
    monkeypatch.setitem(lf.COMMON, 'weights_4bit', dict(lf.COMMON['weights_4bit'], default_dir=str(base)))
    spec = dict(lf.COMMON['weights_4bit_mtpft'], hf_revision='rev-ft', local_only=False, overlay={'of': 'weights_4bit', 'subdir': 'mtp-ft'})
    d = lf.ensure_overlay(tmp_path / 'view', spec, False)
    assert (d / 'mtp/mtp-draft-e4m3.safetensors').read_bytes() == b'distilled-draft' and (d / 'model.safetensors').read_bytes() == b'trunk'
    assert (d / 'model.safetensors').stat().st_ino == (base / 'model.safetensors').stat().st_ino   # hard link, no copy
    assert (base / 'mtp/mtp-draft-e4m3.safetensors').read_bytes() == b'stock-draft'   # the base dir is untouched
    lf.verify_sums(d)
    sums = (d / 'SHA256SUMS').read_text()
    assert h(b'distilled-draft') in sums and h(b'stock-draft') not in sums


def test_d1_baked_assembly(lf, monkeypatch, tmp_path):
    # D1 bake (17 Oct): published checkpoint + the image's drafter files -> <base>-b4gh5-mtpft (hard links + links into the image)
    import hashlib, os, types
    h = lambda b: hashlib.sha256(b).hexdigest()
    for key in ('weights_4bit_mtpft', 'weights_3bit_mtpft'):
        wspec = lf.COMMON[key]; bk = wspec['baked']
        assert set(bk['files']) == set(wspec['drafter_sha256']) and bk['id'] == 'b4gh5-mtpft'
        base = tmp_path / key / 'base'
        files = {'model-00001.safetensors': b'trunk', **{f: b'stock ' + f.encode() for f in bk['files']}}
        for f, b in files.items(): (base / f).parent.mkdir(parents=True, exist_ok=True); (base / f).write_bytes(b)
        (base / 'config.json').write_text('{}')
        (base / 'SHA256SUMS').write_text(''.join(f'{h(b)}  ./{f}\n' for f, b in files.items()))
        before = (base / 'SHA256SUMS').read_text()
        d = lf.ensure_baked(base, wspec, False)
        assert d == base.parent / 'base-b4gh5-mtpft' and (base / 'SHA256SUMS').read_text() == before   # the base dir is untouched
        assert os.stat(d / 'model-00001.safetensors').st_ino == os.stat(base / 'model-00001.safetensors').st_ino and (d / 'config.json').exists()
        for rel, p in bk['files'].items():
            assert lf.baked_target(d / rel) and os.readlink(d / rel) == f'{lf.DRAFTER_ROOT}/b4gh5-mtpft/{p}'
        sums = dict(l.split('  ./')[::-1] for l in (d / 'SHA256SUMS').read_text().split('\n') if l)
        assert sums == {'model-00001.safetensors': h(b'trunk'), **wspec['drafter_sha256']}
        assert lf.drafter_check(d, wspec, False) is not False   # the assembled dir passes the pinned-drafter check
        assert lf.ensure_baked(base, wspec, False) == d   # idempotent
        (base / 'SHA256SUMS').write_text(f'{h(b"trunk")}  ./model-00001.safetensors\n')   # not the published checkpoint
        with pytest.raises(SystemExit, match='lacks'):
            lf.ensure_baked(base, wspec, False)
    assert lf.COMMON['weights_3bit_mtpft']['baked']['files']['manifest.json'] == 'w3a8/manifest.json'
    assert lf.COMMON['weights_4bit_mtpft']['baked']['files']['manifest.json'] == 'w4a8/manifest.json'
    seen = {}
    def run(argv, input=None, **kw):
        seen['argv'], seen['input'] = argv, input; return types.SimpleNamespace(returncode=0, stdout='', stderr='')
    monkeypatch.setattr(lf.subprocess, 'run', run)
    assert lf.baked_verify('img:tag', lf.COMMON['weights_4bit_mtpft'])
    assert seen['argv'][:6] == ['docker', 'run', '--rm', '-i', '--network', 'none'] and seen['argv'][-3:] == ['img:tag', '-c', '-']
    assert seen['input'].count(f'  {lf.DRAFTER_ROOT}/b4gh5-mtpft/') == len(lf.COMMON['weights_4bit_mtpft']['drafter_sha256'])


@pytest.mark.parametrize('mode, name', [('4bit', '4bit'), ('decode', '3bit')])
def test_d1_baked_is_the_default(lf, monkeypatch, tmp_path, capsys, mode, name):
    # no distilled dir, no --weights: the image's drafter (distilled) on the published checkpoint, the executor default manifest
    for key in ('weights_4bit_mtpft', 'weights_3bit_mtpft'):
        monkeypatch.setitem(lf.COMMON, key, dict(lf.COMMON[key], default_dir=str(tmp_path / 'absent')))
    out, argv, env, vols, inner = native_default_env(lf, monkeypatch, tmp_path, capsys, mode)
    assert 'drafter: distilled' in out and 'the image drafter b4gh5-mtpft' in out and 'drafter: stock' not in out, out[-800:]
    assert env['PAITON_FN_MODE_NAME'] == name and 'PAITON_FN_MANIFEST' not in env
    out, argv, env, vols, inner = native_default_env(lf, monkeypatch, tmp_path, capsys, mode, '--drafter', 'stock')
    assert 'image drafter' not in out and env['PAITON_FN_MANIFEST'].endswith(f'/manifest-{name}-stock.json')


def test_release_prefix_caching_keeps_torch_3bit(lf, monkeypatch, tmp_path, capsys):
    # r6: --mode decode --prefix-caching is the published torch align-mode prefix caching (real hits), never the native worker
    out, argv, env, vols, inner = native_default_env(lf, monkeypatch, tmp_path, capsys, 'decode', '--prefix-caching')
    assert 'PAITON_FN_NATIVE' not in env and 'mode decode --prefix-caching: the torch worker' in out
    assert '--enable-prefix-caching' in inner and '3bit-mtp-native' not in ' '.join(l for l in out.splitlines() if 'the torch worker' not in l)
    with pytest.raises(SystemExit, match='use --mode 4bit-cache'):   # unchanged vs r3: --mode 4bit --prefix-caching is refused
        native_default_env(lf, monkeypatch, tmp_path, capsys, '4bit', '--prefix-caching')


@pytest.mark.parametrize('gtt, extra, native', [(200.0, (), True), (47.0, (), False), (62.0, (), True), (200.0, ('--host-pools', 'userptr'), False)])
def test_release_3bit_native_needs_gtt_pools(lf, monkeypatch, tmp_path, capsys, gtt, extra, native):
    # host parity (r6): --mode decode keeps the torch worker (user-pointer pools, as published) where the host tables do not fit the free GTT
    out, argv, env, vols, inner = native_default_env(lf, monkeypatch, tmp_path, capsys, 'decode', *extra, gtt=gtt)
    assert (env.get('PAITON_FN_NATIVE') == '1') == native, out[-600:]
    assert native or 'mode decode: the torch worker (' in out


def test_release_4bit_native_userptr_keeps_torch(lf, monkeypatch, tmp_path, capsys):
    out, argv, env, vols, inner = native_default_env(lf, monkeypatch, tmp_path, capsys, '4bit', '--host-pools', 'userptr')
    assert 'PAITON_FN_NATIVE' not in env and 'mode 4bit: the torch worker (--host-pools userptr' in out


@pytest.mark.parametrize('mode, alias, name', [('4bit', '4bit-mtp-native', '4bit'), ('decode', '3bit-mtp-native', '3bit')])
def test_r6_default_is_torch(lf, monkeypatch, tmp_path, capsys, mode, alias, name):
    # user 17 Oct ~21:02 (decision B): every mode defaults to the torch worker (the r3 behaviour); native = opt-in --fn-worker native
    assert not lf.MODES[alias]['native']['is_default']
    out, argv, env, vols, inner = native_default_env(lf, monkeypatch, tmp_path, capsys, mode, native_default=False)
    assert 'PAITON_FN_NATIVE' not in env and f'mode {mode}: the torch worker (the default); --fn-worker native = --mode {alias}' in out
    assert 'KV profile' not in out and 'image drafter' not in out
    out, argv, env, vols, inner = native_default_env(lf, monkeypatch, tmp_path, capsys, mode, '--fn-worker', 'native', native_default=False)
    assert env['PAITON_FN_NATIVE'] == '1' and env['PAITON_FN_MODE_NAME'] == name and 'the image drafter b4gh5-mtpft' in out
    assert (name == '3bit') or 'KV profile capacity (the default)' in out


@pytest.mark.parametrize('mode, gtt, extra, why', [('4bit', 62.0, (), 'GTT limit 62 GiB'), ('decode', 47.0, (), 'free GTT 47 GiB'),
                                                   ('decode', 200.0, ('--host-pools', 'userptr'), '--host-pools userptr')])
def test_explicit_native_unmeasured_host_refused(lf, monkeypatch, tmp_path, capsys, mode, gtt, extra, why):
    # r6 (decision B): --fn-worker native is an explicit opt-in; on a host / layout it is not measured on, the dry run warns and a real
    # start is refused with the reason (never a failure later at load)
    out, argv, env, vols, inner = native_default_env(lf, monkeypatch, tmp_path, capsys, mode, '--fn-worker', 'native', *extra, gtt=gtt,
                                                     native_default=False)
    assert f'WARNING: --mode {mode} --fn-worker native: {why}' in out and 'drop --fn-worker native' in out
    out, argv, env, vols, inner = native_default_env(lf, monkeypatch, tmp_path, capsys, mode, '--fn-worker', 'native', native_default=False)
    assert 'WARNING: --mode' not in out and env['PAITON_FN_NATIVE'] == '1'   # a measured host: no warning


def test_native_kv_tier_auto_from_host(lf, monkeypatch, tmp_path, capsys):
    # release candidate 20261023: the RAM prefix tier is sized from the host (never a fixed size); --memory emulates a smaller host
    kvt = lf.MODES['4bit-long-cache-native']['native']['kv_tier']
    monkeypatch.setattr(lf, 'MEM_CAP_GIB', 64.0)   # --memory 64: every RAM figure is capped
    assert lf.meminfo_gib('MemTotal') == 64.0
    monkeypatch.setattr(lf, 'MEM_CAP_GIB', None)
    for avail, want in ((242.6, 16.0), (128.0, 2.0), (120.0, 0.0)):   # 4-bit: resident 108.5 + margin 6 + reserve 8
        monkeypatch.setattr(lf, 'meminfo_gib', lambda f, v=avail: v)
        assert lf.kv_tier_auto(kvt, 108.5, 6.0)[0] == want, avail
    monkeypatch.setattr(lf, 'meminfo_gib', lambda f: 100.0)
    out, argv, env, vols, inner = dry(lf, monkeypatch, tmp_path, capsys, '--memory', '100', mode='3bit-cache-native')
    assert '--memory 102400m --memory-swap 102400m' in ' '.join(argv)
    i = argv.index('-c'); eng = shlex.split(argv[i + 1].split('exec ', 1)[1])
    assert json.loads(eng[eng.index('--kv-transfer-config') + 1])['kv_connector_extra_config']['cpu_bytes_to_use'] == 2 * (14 << 30)   # (100-55-8-8)/2
    assert env['PAITON_FN_FNEXEC_LIB'] == '/opt/paiton/runtime/fnexec-pc/libpaiton_fnexec.so'
    out, argv, env, vols, inner = dry(lf, monkeypatch, tmp_path, capsys, '--native-kv-tier', 'off', mode='3bit-cache-native')
    assert '--kv-transfer-config' not in inner


def test_native_prefix_caching_routes_to_cache_mode(lf, monkeypatch, tmp_path, capsys):
    # 2a (25 Oct): 3-bit --prefix-caching keeps MTP (3bit-long-cache-mtp-native) where the free GTT holds the GTT host pools
    out, argv, env, vols, inner = dry(lf, monkeypatch, tmp_path, capsys, '--fn-worker', 'native', '--prefix-caching', mode='decode')
    assert '--mode 3bit-long-cache-mtp-native' in out and env['PAITON_FN_MODE_NAME'] == '3bit-long-cache-mtp-native'
    assert '"num_speculative_tokens": 3' in inner and '--max-model-len 262144' in inner
    for gtt, extra in ((59.0, ()), (200.0, ('--host-pools', 'userptr'))):   # a smaller host (or user-pointer pools): MTP off, not refused
        out, argv, env, vols, inner = dry_gtt(lf, monkeypatch, tmp_path, capsys, gtt, 'decode', '--fn-worker', 'native', '--prefix-caching', *extra)
        assert 'this host gets --mode 3bit-cache-native (speculative decoding off)' in out and 'WARNING: --mode decode' not in out
        assert env['PAITON_FN_MODE_NAME'] == '3bit-cache-native' and 'num_speculative_tokens' not in inner
    monkeypatch.setattr(lf, 'gtt_limits_gib', lambda gpus: (64.0, 64.0))   # a 128 GB host: 4-bit native refused, the 3-bit mode named
    monkeypatch.setattr(sys, 'argv', ['launch-flashnext.py', '--mode', '4bit-long-native', '--prefix-caching', '--devices', 'GPU-0,GPU-1',
                                      '--cache', str(tmp_path / 'cache')])
    monkeypatch.setattr(lf, 'meminfo_gib', lambda f: 120.0)
    with pytest.raises(SystemExit, match='this host fits --mode 3bit-cache-native or --mode 3bit-long-cache-native'):
        lf.main()


def test_native_prefix_caching_4bit_mtp(lf, monkeypatch, tmp_path, capsys):
    # rc2: --prefix-caching on the 4-bit native worker serves the 262K MTP cache mode; the RAM tier (auto) swaps in the
    # executor's tier-compatible settings, --native-kv-tier off keeps the faster ones
    out, argv, env, vols, inner = dry(lf, monkeypatch, tmp_path, capsys, '--fn-worker', 'native', '--prefix-caching', mode='4bit')
    assert '--mode 4bit-long-cache-mtp-native' in out and env['PAITON_FN_MODE_NAME'] == '4bit-long-cache-mtp-native'
    assert '"num_speculative_tokens": 3' in inner and '--mamba-cache-mode align' in inner and '--max-model-len 262144' in inner
    assert 'native RAM prefix tier: 16 GiB per GPU' in out and '--kv-transfer-config' in inner and '207.3' in out
    assert '(slots - 14 x sessions) x 2,048 tokens' in out and 'prefilled cold' in out
    assert (env['PAITON_FN_EARLY_TOKENS'], env['PAITON_FN_MALL_PF']) == ('0', 'off')
    out, argv, env, vols, inner = dry(lf, monkeypatch, tmp_path, capsys, '--fn-worker', 'native', '--prefix-caching',
                                      '--native-kv-tier', 'off', mode='4bit')
    assert '--kv-transfer-config' not in inner and env['PAITON_FN_EARLY_TOKENS'] == '1' and env['PAITON_FN_MALL_PF'] not in ('', '0', 'off')
