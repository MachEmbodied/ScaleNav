"""Reconstruct the benchmark's visual PLY from InteriorGS compressed splats."""
import hashlib
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import numpy as np
from scipy.spatial.transform import Rotation

FIELDS = (['x', 'y', 'z', 'nx', 'ny', 'nz']
          + [f'f_dc_{i}' for i in range(3)] + [f'f_rest_{i}' for i in range(45)]
          + ['opacity'] + [f'scale_{i}' for i in range(3)] + [f'rot_{i}' for i in range(4)])
COPIED_FIELDS = ['opacity'] + [f'f_dc_{i}' for i in range(3)] + [f'scale_{i}' for i in range(3)]


def _read_vertices(path):
    fields = []
    count = None
    with open(path, 'rb') as stream:
        if stream.readline() != b'ply\n' or stream.readline() != b'format binary_little_endian 1.0\n':
            raise ValueError('Expected a binary little-endian PLY')
        while True:
            line = stream.readline()
            if not line:
                raise ValueError('Incomplete PLY header')
            parts = line.decode('ascii').split()
            if not parts or parts[0] == 'comment':
                continue
            if parts[0] == 'element':
                if len(parts) != 3 or parts[1] != 'vertex' or count is not None:
                    raise ValueError('Expected a single PLY vertex element')
                count = int(parts[2])
            elif parts[0] == 'property':
                if len(parts) != 3 or parts[1] != 'float':
                    raise ValueError('Expected float32 PLY vertex properties')
                fields.append((parts[2], '<f4'))
            elif parts[0] == 'end_header':
                break
            else:
                raise ValueError('Unsupported PLY header')
        offset = stream.tell()
    required = set(COPIED_FIELDS + ['x', 'y', 'z'] + [f'rot_{i}' for i in range(4)])
    if not count or not required.issubset(name for name, _ in fields):
        raise ValueError('Missing Gaussian vertex properties')
    dtype = np.dtype(fields)
    if Path(path).stat().st_size != offset + count * dtype.itemsize:
        raise ValueError('Incomplete PLY vertex payload')
    return np.memmap(path, dtype=dtype, mode='r', offset=offset, shape=(count,))


def _write_runtime_ply(unpacked, output):
    values = _read_vertices(unpacked)
    matrix = Rotation.from_euler('xyz', [-90, 0, 0], degrees=True).as_matrix()
    xyzw = Rotation.from_matrix(matrix).as_quat()
    w1, x1, y1, z1 = xyzw[[3, 0, 1, 2]]
    header = ('ply\nformat binary_little_endian 1.0\nelement vertex ' + str(len(values)) + '\n'
              + ''.join('property float ' + name + '\n' for name in FIELDS) + 'end_header\n').encode()
    digest = hashlib.sha256()
    with open(output, 'wb') as stream:
        stream.write(header)
        digest.update(header)
        for start in range(0, len(values), 65536):
            source = values[start:start + 65536]
            block = np.zeros(len(source), dtype=[(name, '<f4') for name in FIELDS])
            for name in COPIED_FIELDS:
                block[name] = source[name]
            positions = np.column_stack([source[name] for name in ('x', 'y', 'z')]) @ matrix.T
            for i, name in enumerate(('x', 'y', 'z')):
                block[name] = positions[:, i]
            w2, x2, y2, z2 = [source[f'rot_{i}'].astype(np.float64) for i in range(4)]
            quaternions = np.stack([
                w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2,
            ], axis=-1)
            quaternions /= np.linalg.norm(quaternions, axis=1, keepdims=True) + 1e-8
            for i in range(4):
                block[f'rot_{i}'] = quaternions[:, i]
            payload = block.tobytes()
            stream.write(payload)
            digest.update(payload)
    return {'sha256': digest.hexdigest(), 'bytes': Path(output).stat().st_size,
            'gaussians': len(values)}


def reconstruct_runtime_gs(source, destination, *, expected_sha256, expected_bytes,
                           converter='splat-transform'):
    """Publish only a byte-exact reconstruction; navmeshes are obtained separately."""
    source, destination = Path(source), Path(destination)
    if len(expected_sha256) != 64 or any(c not in '0123456789abcdef' for c in expected_sha256):
        raise ValueError('A published SHA256 is required')
    if expected_bytes <= 0:
        raise ValueError('A positive published byte count is required')
    if not source.is_file():
        raise FileNotFoundError(source)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.gs-convert-', dir=destination.parent) as work:
        unpacked, output = Path(work) / 'unpacked.ply', Path(work) / 'runtime.ply'
        subprocess.run([str(converter), str(source), str(unpacked)], check=True,
                       capture_output=True, text=True, timeout=300)
        result = _write_runtime_ply(unpacked, output)
        if result['bytes'] != expected_bytes or result['sha256'] != expected_sha256:
            raise ValueError(f'Converted visual asset does not match the benchmark: {source.name}')
        if destination.exists():
            raise FileExistsError(destination)
        os.replace(output, destination)
    return result
