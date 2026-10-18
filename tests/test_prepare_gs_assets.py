import hashlib
import importlib.util
from pathlib import Path
import shutil

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('prepare_gs_assets', ROOT / 'scripts/prepare_gs_assets.py')
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


def write_ply(path, values):
    header = ('ply\nformat binary_little_endian 1.0\nelement vertex ' + str(len(values)) + '\n'
              + ''.join(f'property float {name}\n' for name in values.dtype.names) + 'end_header\n')
    payload = header.encode() + values.tobytes()
    path.write_bytes(payload)
    return payload


@pytest.fixture
def assets(tmp_path, monkeypatch):
    names = (['x', 'y', 'z', 'nx', 'ny', 'nz'] + [f'f_dc_{i}' for i in range(3)]
             + [f'f_rest_{i}' for i in range(45)] + ['opacity']
             + [f'scale_{i}' for i in range(3)] + [f'rot_{i}' for i in range(4)])
    values = np.zeros(2, dtype=[(name, '<f4') for name in names])
    values['x'], values['y'], values['z'] = [1, 2], [2, 4], [3, 6]
    values['rot_0'], values['rot_1'] = [1, 0], [0, 1]
    values['f_dc_0'], values['opacity'], values['scale_0'] = 0.3, 0.5, -2
    values['f_rest_0'] = 7
    source = tmp_path / 'compressed.ply'
    write_ply(source, values)
    expected = values.copy()
    expected['y'], expected['z'] = [3, 6], [-2, -4]
    expected['f_rest_0'] = 0
    c = np.float32(1 / np.sqrt(2) / (1 + 1e-8))
    expected['rot_0'], expected['rot_1'] = c, [-c, c]
    expected_path = tmp_path / 'expected.ply'
    payload = write_ply(expected_path, expected)

    def unpack(command, **kwargs):
        assert command[1] == str(source) and len(command) == 3
        shutil.copyfile(source, command[2])

    monkeypatch.setattr(prepare.subprocess, 'run', unpack)
    return source, expected, hashlib.sha256(payload).hexdigest(), len(payload)


def test_reconstruction_preserves_benchmark_geometry_and_dc_color(assets, tmp_path):
    source, expected, checksum, size = assets
    destination = tmp_path / 'scene.gs.ply'
    result = prepare.reconstruct_runtime_gs(source, destination, expected_sha256=checksum,
                                           expected_bytes=size, converter='converter')
    assert result == {'sha256': checksum, 'bytes': size, 'gaussians': 2}
    np.testing.assert_array_equal(prepare._read_vertices(destination), expected)
    assert not list(tmp_path.glob('.gs-convert-*'))


@pytest.mark.parametrize('wrong_size', [False, True])
def test_failed_asset_validation_does_not_publish_partial_files(assets, tmp_path, wrong_size):
    source, _, checksum, size = assets
    destination = tmp_path / 'scene.gs.ply'
    with pytest.raises(ValueError, match='does not match the benchmark'):
        prepare.reconstruct_runtime_gs(source, destination,
                                       expected_sha256=checksum if wrong_size else '0' * 64,
                                       expected_bytes=size + 1 if wrong_size else size)
    assert not destination.exists()
    assert not list(tmp_path.glob('.gs-convert-*'))


def test_existing_asset_is_never_overwritten(assets, tmp_path):
    source, _, checksum, size = assets
    destination = tmp_path / 'scene.gs.ply'
    destination.write_bytes(b'keep existing asset')
    with pytest.raises(FileExistsError):
        prepare.reconstruct_runtime_gs(source, destination, expected_sha256=checksum, expected_bytes=size)
    assert destination.read_bytes() == b'keep existing asset'


def test_truncated_converter_output_is_rejected(assets, tmp_path, monkeypatch):
    source, _, checksum, size = assets
    monkeypatch.setattr(prepare.subprocess, 'run',
                        lambda command, **kwargs: Path(command[2]).write_bytes(source.read_bytes()[:-1]))
    destination = tmp_path / 'scene.gs.ply'
    with pytest.raises(ValueError, match='Incomplete PLY vertex payload'):
        prepare.reconstruct_runtime_gs(source, destination, expected_sha256=checksum, expected_bytes=size)
    assert not destination.exists()
