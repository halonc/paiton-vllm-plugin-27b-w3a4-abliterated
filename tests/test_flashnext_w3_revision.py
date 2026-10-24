"""3-bit weights pinned by one revision (modes.json common.hf_revision): the default dir is keyed by it, so a new revision
downloads into a fresh dir; an older un-keyed dir gets a delete note; an explicit --weights dir with another SHA256SUMS a warning."""
import hashlib
import importlib.util
import pathlib
import sys

import pytest
from conftest import fixed_host_open

LAUNCHER = pathlib.Path(__file__).resolve().parents[1] / 'models' / 'Qwen3.8-Flash-Next' / 'launch-flashnext.py'


@pytest.fixture()
def lf(monkeypatch, tmp_path):
    monkeypatch.setenv('HOME', str(tmp_path / 'home'))
    spec = importlib.util.spec_from_file_location('launch_flashnext', LAUNCHER)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    mod.open = fixed_host_open()
    return mod


def dry(lf, monkeypatch, tmp_path, capsys, *extra, mode='decode'):
    monkeypatch.setattr(lf, 'p2p_check', lambda *a, **k: None); monkeypatch.setattr(lf, 'ram_check', lambda *a, **k: [])
    monkeypatch.setattr(lf, 'render_gid', lambda: None); monkeypatch.setattr(lf, 'gtt_limits_gib', lambda gpus: (200.0, 200.0))
    monkeypatch.setattr(lf, 'ensure_base_cfg', lambda d, dry: tmp_path / 'cfg')
    monkeypatch.setattr(lf, 'ensure_view', lambda *a, **k: tmp_path / 'view')
    monkeypatch.setattr(sys, 'argv', ['launch-flashnext.py', '--mode', mode, '--dry-run', '--devices', 'GPU-0,GPU-1',
                                      '--cache', str(tmp_path / 'cache'), *extra])
    lf.main()
    return capsys.readouterr().out


def checkpoint(d, sums=b'abc  ./model.safetensors\n'):
    d.mkdir(parents=True); (d / 'SHA256SUMS').write_bytes(sums); (d / 'model.safetensors').write_bytes(b'x' * 1024)
    return d


def test_one_revision_keys_the_default_dir(lf):
    rev = lf.COMMON['hf_revision']
    assert lf.W3['hf_revision'] == rev and lf.W3['hf_repo'] == lf.COMMON['hf_repo']
    assert lf.W3['default_dir'] == f"{lf.W3['previous_default_dir']}-{rev[:8]}"


@pytest.mark.parametrize('mode', ['decode', 'prefill-long'])
def test_fresh_install_downloads_into_the_keyed_dir(lf, monkeypatch, tmp_path, capsys, mode):
    out = dry(lf, monkeypatch, tmp_path, capsys, mode=mode)
    keyed = tmp_path / 'home' / 'paiton-models' / f"Qwen3.8-Flash-Next-W3A8-{lf.COMMON['hf_revision'][:8]}"
    assert f"download (dry run): {lf.COMMON['hf_repo']}@{lf.COMMON['hf_revision'][:12]} (all files) -> {keyed}" in out
    assert 'can be deleted' not in out and 'WARNING weights' not in out


def test_upgrade_from_the_old_dir(lf, monkeypatch, tmp_path, capsys):
    old = checkpoint(tmp_path / 'home' / 'paiton-models' / 'Qwen3.8-Flash-Next-W3A8')
    out = dry(lf, monkeypatch, tmp_path, capsys)
    keyed = f"{old}-{lf.COMMON['hf_revision'][:8]}"
    assert f'-> {keyed}' in out   # the new revision goes to a fresh dir; the old files are never mixed in
    note = next(l for l in out.splitlines() if 'can be deleted' in l)
    assert str(old) in note and keyed in note and '0 GiB' in note
    assert (old / 'SHA256SUMS').exists()   # never deleted by the launcher


def test_upgrade_native_mode_uses_the_keyed_base(lf, monkeypatch, tmp_path, capsys):
    checkpoint(tmp_path / 'home' / 'paiton-models' / 'Qwen3.8-Flash-Next-W3A8')
    out = dry(lf, monkeypatch, tmp_path, capsys, '--fn-worker', 'native')
    keyed = f"Qwen3.8-Flash-Next-W3A8-{lf.COMMON['hf_revision'][:8]}"
    assert f'{keyed}-b4gh5-mtpft' in out and 'can be deleted' in out


def test_explicit_weights_with_other_sums_warns(lf, monkeypatch, tmp_path, capsys):
    w = checkpoint(tmp_path / 'mine')
    out = dry(lf, monkeypatch, tmp_path, capsys, '--weights', str(w))
    assert 'WARNING weights' in out and 'serving its files as they are' in out and 'can be deleted' not in out


def test_explicit_weights_of_the_pinned_revision_is_quiet(lf, monkeypatch, tmp_path, capsys):
    sums = b'abc  ./model.safetensors\n'; w = checkpoint(tmp_path / 'mine', sums)
    monkeypatch.setitem(lf.COMMON, 'hf_sha256sums_sha256', hashlib.sha256(sums).hexdigest())
    out = dry(lf, monkeypatch, tmp_path, capsys, '--weights', str(w))
    assert 'WARNING weights' not in out


def test_explicit_4bit_weights_are_not_checked(lf, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(lf, 'ensure_weights', lambda d, dry, *a: tmp_path / 'w')
    w = checkpoint(tmp_path / 'w4')
    out = dry(lf, monkeypatch, tmp_path, capsys, '--weights', str(w), mode='4bit')
    assert 'WARNING weights' not in out


@pytest.mark.parametrize('rel,ok', [
    ('Qwen3.8-Flash-Next-W3A8-Paiton-RDNA4', True),                                    # the first release (829b089b)
    ('Qwen3.8-Flash-Next-W3A8-Paiton-RDNA4 (20261024, B3h-dnld4m)', True),             # a release that names itself
    ('Qwen3.8-Flash-Next-W3A8-Paiton-RDNA4 (candidate FN3-B3h-dnld4m, not published)', False),
    ('Qwen3.8-Flash-Next-W4A8-Paiton-RDNA4 (20261024, B3h-dnld4m)', False),
])
def test_release_name_check(lf, monkeypatch, tmp_path, capsys, rel, ok):
    w = checkpoint(tmp_path / 'mine'); (w / 'config.json').write_text('{"paiton_release": "%s"}' % rel)
    out = dry(lf, monkeypatch, tmp_path, capsys, '--weights', str(w))
    assert ('expects the Qwen3.8-Flash-Next-W3A8-Paiton-RDNA4 weights' in out) is not ok


ALIASED = [(['--mode', 'decode', '--fn-worker', 'native', '--drafter', 'stock'], 'fnexec/manifest-3bit-stock'),
           (['--mode', 'decode', '--fn-worker', 'native', '--prefix-caching'], 'fnexec-pc/manifest-3bit-cache-native'),
           (['--mode', '3bit-long-cache-native'], 'fnexec-pc/manifest-3bit-long-cache-native'),
           (['--mode', '3bit-long-cache-mtp-native', '--drafter', 'stock'], 'fnexec-pc/manifest-3bit-long-cache-mtp-native-stock')]


def aliased_checkpoint(lf, monkeypatch, tmp_path, body=b'{"name": "new"}'):
    w = checkpoint(tmp_path / 'new'); (w / 'manifest.json').write_bytes(body)
    img = lf.COMMON['image_4bit_native']; sha = hashlib.sha256(body).hexdigest()
    monkeypatch.setitem(img, 'fnexec_manifest_alias', {sha: dict(next(iter(img['fnexec_manifest_alias'].values())))})
    return w


@pytest.mark.parametrize('args,stem', ALIASED)
def test_drop_in_checkpoint_gets_the_images_own_executor_manifest(lf, monkeypatch, tmp_path, capsys, args, stem):
    w = aliased_checkpoint(lf, monkeypatch, tmp_path)
    out = dry(lf, monkeypatch, tmp_path, capsys, '--weights', str(w), *args[2:], mode=args[1])
    assert f'PAITON_FN_MANIFEST=/opt/paiton/runtime/{stem}-w3a8-71faf406.json' in out


@pytest.mark.parametrize('args,stem', ALIASED[:2])
def test_other_checkpoints_keep_the_mode_manifest(lf, monkeypatch, tmp_path, capsys, args, stem):
    aliased_checkpoint(lf, monkeypatch, tmp_path)
    w = checkpoint(tmp_path / 'old'); (w / 'manifest.json').write_bytes(b'{"name": "RX-B3h-r2"}')
    out = dry(lf, monkeypatch, tmp_path, capsys, '--weights', str(w), *args[2:], mode=args[1])
    assert 'w3a8-71faf406' not in out


def test_the_shipped_alias_names_image_files_of_the_new_revision(lf):
    (sha, spec), = lf.COMMON['image_4bit_native']['fnexec_manifest_alias'].items()
    # the plan files in the rc3 image are named after the weights commit (71faf406); later card-only commits reuse them
    assert spec['suffix'] == 'w3a8-71faf406' and len(spec['files']) == 4 and len(sha) == 64


@pytest.mark.parametrize('mode', ['decode', '4bit'])
def test_ipc_legacy_not_forced_but_caller_env_wins(lf, monkeypatch, tmp_path, capsys, mode):
    # issue 27: a forced HSA_ENABLE_IPC_MODE_LEGACY=1 broke RCCL init on in-tree amdgpu hosts
    monkeypatch.delenv('HSA_ENABLE_IPC_MODE_LEGACY', raising=False)
    assert 'HSA_ENABLE_IPC_MODE_LEGACY' not in dry(lf, monkeypatch, tmp_path, capsys, mode=mode)
    monkeypatch.setenv('HSA_ENABLE_IPC_MODE_LEGACY', '1')
    assert '-e HSA_ENABLE_IPC_MODE_LEGACY=1' in dry(lf, monkeypatch, tmp_path, capsys, mode=mode)
