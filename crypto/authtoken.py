"""Unforgeable, image-bound authenticated instruction tokens.

A token certifies: "the holder of the private key authorized THIS instruction
for THIS image." Properties:

  * Unforgeable  -- an Ed25519 signature; without the private key you cannot mint
                    a token that verifies.
  * Image-bound  -- the signed payload includes a perceptual hash (pHash) of the
                    image, so a captured token will not verify if transplanted
                    onto a different image. The pHash is tolerant to the tiny
                    embedding perturbation and mild transforms (that is the
                    point), so binding is approximate -- a deliberate tradeoff
                    documented below.
  * Replay-aware -- a random nonce + optional expiry let the verifier reject
                    stale/re-used tokens.

SECURITY NOTE: pHash binding tolerates small image changes, so it is not a
collision-resistant commitment to exact pixels -- an adversary who finds a
different image with a near-identical pHash could reuse a token. For this
research prototype that is acceptable and measured; a production design would
bind to a robust-but-stronger image fingerprint. The signature itself is over
the EXACT payload (including the stored pHash), so authenticity is exact; only
the image-match check is fuzzy.

Token layout (bytes), then base64 for embedding as text:
  version(1) | key_id(2) | nonce(8) | expiry(8) | phash(8) | ilen(2) | instr(ilen)
  followed by signature(64) over all of the above.
"""
from __future__ import annotations

import base64
import os
import struct
import time

import numpy as np
from cryptography.exceptions import InvalidSignature

from crypto.keys import Ed25519PrivateKey, Ed25519PublicKey  # re-export types

VERSION = 1
PHASH_BITS = 64  # 8 bytes


def perceptual_hash(image: np.ndarray) -> int:
    """64-bit DCT perceptual hash (pHash) of an RGB/gray uint8 image."""
    from PIL import Image
    from scipy.fft import dctn

    if image.ndim == 3:
        gray = np.asarray(Image.fromarray(image).convert("L"), dtype=np.float64)
    else:
        gray = image.astype(np.float64)
    # 32x32 -> DCT -> top-left 8x8 low frequencies (excluding DC) vs their median.
    small = np.asarray(Image.fromarray(gray.astype(np.uint8)).resize((32, 32)), dtype=np.float64)
    d = dctn(small, norm="ortho")[:8, :8]
    flat = d.flatten()[1:]  # drop DC
    med = np.median(flat)
    bits = (flat > med).astype(np.uint8)
    h = 0
    for b in bits[:PHASH_BITS]:
        h = (h << 1) | int(b)
    return h


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def _pack_payload(key_id: int, nonce: bytes, expiry: int, phash: int, instruction: str) -> bytes:
    instr = instruction.encode("utf-8")
    return (
        struct.pack(">B", VERSION)
        + struct.pack(">H", key_id)
        + nonce
        + struct.pack(">q", expiry)
        + struct.pack(">Q", phash)
        + struct.pack(">H", len(instr))
        + instr
    )


def _unpack_payload(payload: bytes):
    off = 0
    (version,) = struct.unpack_from(">B", payload, off); off += 1
    (key_id,) = struct.unpack_from(">H", payload, off); off += 2
    nonce = payload[off:off + 8]; off += 8
    (expiry,) = struct.unpack_from(">q", payload, off); off += 8
    (phash,) = struct.unpack_from(">Q", payload, off); off += 8
    (ilen,) = struct.unpack_from(">H", payload, off); off += 2
    instruction = payload[off:off + ilen].decode("utf-8"); off += ilen
    return dict(version=version, key_id=key_id, nonce=nonce, expiry=expiry,
                phash=phash, instruction=instruction), off


def mint_token(instruction: str, image: np.ndarray, priv: Ed25519PrivateKey,
               key_id: int = 1, ttl_seconds: int | None = None) -> str:
    """Create a base64 authenticated token binding `instruction` to `image`."""
    nonce = os.urandom(8)
    expiry = int(time.time()) + ttl_seconds if ttl_seconds else -1
    phash = perceptual_hash(image)
    payload = _pack_payload(key_id, nonce, expiry, phash, instruction)
    signature = priv.sign(payload)
    return base64.b64encode(payload + signature).decode("ascii")


def verify_token(token: str, image: np.ndarray, pub: Ed25519PublicKey,
                 phash_tolerance: int = 10) -> dict:
    """Verify a token against an image. Returns a dict with 'valid' and 'reason'.

    phash_tolerance: max Hamming distance (of 64 bits) allowed between the token's
    stored pHash and the image's pHash. ~10 tolerates embedding + mild transforms
    while still rejecting transplants onto unrelated images.
    """
    try:
        raw = base64.b64decode(token)
    except Exception:
        return {"valid": False, "reason": "not base64", "instruction": None}
    if len(raw) < 64:
        return {"valid": False, "reason": "too short", "instruction": None}
    payload, sig = raw[:-64], raw[-64:]
    try:
        fields, _ = _unpack_payload(payload)
    except Exception as e:
        return {"valid": False, "reason": f"bad payload: {e}", "instruction": None}

    # 1. signature authenticity (exact)
    try:
        pub.verify(sig, payload)
    except InvalidSignature:
        return {"valid": False, "reason": "bad signature", "instruction": None,
                "key_id": fields["key_id"]}

    # 2. expiry / replay window
    if fields["expiry"] != -1 and time.time() > fields["expiry"]:
        return {"valid": False, "reason": "expired", "instruction": None,
                "key_id": fields["key_id"]}

    # 3. image binding (fuzzy)
    dist = _hamming(fields["phash"], perceptual_hash(image))
    if dist > phash_tolerance:
        return {"valid": False, "reason": f"image mismatch (phash dist {dist})",
                "instruction": None, "key_id": fields["key_id"]}

    return {"valid": True, "reason": "ok", "instruction": fields["instruction"],
            "key_id": fields["key_id"], "nonce": fields["nonce"].hex(),
            "phash_distance": dist}
