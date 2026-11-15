"""High-entropy keystream for whitening the embedded payload.

Whitening XORs the payload bytes with a pseudo-random stream so the embedded
signal looks like natural sensor noise rather than structured data -- it raises
the bar for *steganalysis* (detecting that something is hidden). It is symmetric
and does NOT provide authenticity: unforgeability comes solely from the Ed25519
signature in authtoken.py. Keep these two concerns separate.

This is where QryptGen+ plugs in. QryptGen+ (Ahn & Hong, Quantum Inf. Process.
2026) is a quantum-GAN generator of high-entropy 28x28 key images. Point
`--qryptgen-key <file>` at such a key image (any grayscale image works as an
entropy source) and it is folded into the keystream seed. With no key image we
fall back to a classical SHA-256 counter-mode CSPRNG, so the pipeline runs
end-to-end without the quantum generator present.
"""
from __future__ import annotations

import hashlib
from pathlib import Path


def _seed_from_qryptgen(key_image_path: str | Path) -> bytes:
    """Derive seed entropy from a QryptGen+ key image (or any image file)."""
    import numpy as np
    from PIL import Image

    arr = np.asarray(Image.open(key_image_path).convert("L"), dtype=np.uint8)
    return hashlib.sha256(arr.tobytes()).digest()


def make_keystream(length: int, shared_key: bytes, qryptgen_key=None) -> bytes:
    """Deterministic keystream of `length` bytes from a shared secret.

    shared_key   : symmetric secret known to embedder and verifier.
    qryptgen_key : optional path to a QryptGen+ key image for extra entropy.
    """
    seed = hashlib.sha256(b"hips-whiten-v1" + shared_key).digest()
    if qryptgen_key is not None:
        seed = hashlib.sha256(seed + _seed_from_qryptgen(qryptgen_key)).digest()
    out = bytearray()
    counter = 0
    while len(out) < length:
        out.extend(hashlib.sha256(seed + counter.to_bytes(8, "big")).digest())
        counter += 1
    return bytes(out[:length])


def whiten(data: bytes, shared_key: bytes, qryptgen_key=None) -> bytes:
    ks = make_keystream(len(data), shared_key, qryptgen_key)
    return bytes(a ^ b for a, b in zip(data, ks))


# Whitening is an involution: applying it twice with the same key restores data.
dewhiten = whiten
