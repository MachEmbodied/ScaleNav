"""Bind latent-cache preparation to its encoder and selected source records."""
import hashlib
import json
from pathlib import Path


def encoder_fingerprint(base_model, config, dtype):
    vae_config = config.get('vae_kwargs', {})
    path = Path(base_model) / vae_config.get('vae_subpath', 'Wan2.2_VAE.pth')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return {'vae_sha256': digest.hexdigest(), 'vae_config': vae_config, 'dtype': dtype}


def records_fingerprint(records):
    payload = json.dumps(records, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(payload).hexdigest()
