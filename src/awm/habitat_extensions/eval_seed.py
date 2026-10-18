"""Stable seed schedules for Habitat-GS evaluation."""

from __future__ import annotations

import hashlib


EVAL_SEED_MODES = frozenset({"episode", "fixed", "index"})
_NUMPY_SEED_MODULUS = 1 << 32


def resolve_eval_seed(
    base_seed: int,
    mode: str,
    *,
    scene_id: str,
    episode_id: str,
    local_index: int,
) -> int:
    """Resolve a NumPy/PyTorch-compatible seed for one episode.

    ``episode`` is the production schedule: the same episode receives the
    same seed regardless of ordering, filtering, sharding, or worker count.
    ``fixed`` and ``index`` are retained for exact reproduction of older runs.
    """
    if mode not in EVAL_SEED_MODES:
        raise ValueError(f"unsupported eval_seed_mode: {mode!r}")

    seed = int(base_seed)
    if mode == "index":
        seed += int(local_index)
    elif mode == "episode":
        episode_key = f"{scene_id}\x1f{episode_id}".encode("utf-8")
        offset = int.from_bytes(
            hashlib.blake2s(
                episode_key,
                digest_size=4,
                person=b"AWMEVAL",
            ).digest(),
            byteorder="little",
            signed=False,
        )
        seed += offset

    return seed % _NUMPY_SEED_MODULUS
