"""Asymmetric keys for the authenticated visual-trigger channel.

We use Ed25519: 32-byte keys, deterministic 64-byte signatures -- compact enough
to embed in an image's steganographic payload, and fast to verify. This provides
the RSA-like property the project needs: only the holder of the PRIVATE key can
produce a valid trigger, but anyone with the PUBLIC key can verify one. That is
what makes a trigger unforgeable.

(For a post-quantum variant, swap Ed25519 for Falcon-512; the embed/verify flow
is identical, only the signature size changes. See README note in authtoken.py.)

Keys are stored as raw bytes (hex) so they are easy to version or distribute:
  - private key: keep secret; used by the survey author to stamp images.
  - public key: ship with the verifier/agent harness to check triggers.
"""
from __future__ import annotations

from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)


def generate_keypair() -> tuple[Ed25519PrivateKey, Ed25519PublicKey]:
    priv = Ed25519PrivateKey.generate()
    return priv, priv.public_key()


def save_keypair(priv: Ed25519PrivateKey, directory: str | Path, name: str = "trigger") -> dict:
    """Write <name>.sk (private) and <name>.pk (public) as hex files."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    from cryptography.hazmat.primitives import serialization

    sk_raw = priv.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    pk_raw = priv.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    sk_path = directory / f"{name}.sk"
    pk_path = directory / f"{name}.pk"
    sk_path.write_text(sk_raw.hex())
    pk_path.write_text(pk_raw.hex())
    try:  # best-effort: keep the private key owner-only
        sk_path.chmod(0o600)
    except OSError:
        pass
    return {"private": str(sk_path), "public": str(pk_path)}


def load_private(path: str | Path) -> Ed25519PrivateKey:
    raw = bytes.fromhex(Path(path).read_text().strip())
    return Ed25519PrivateKey.from_private_bytes(raw)


def load_public(path: str | Path) -> Ed25519PublicKey:
    raw = bytes.fromhex(Path(path).read_text().strip())
    return Ed25519PublicKey.from_public_bytes(raw)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Generate an Ed25519 trigger keypair.")
    p.add_argument("--out-dir", default="crypto/keys")
    p.add_argument("--name", default="trigger")
    args = p.parse_args()
    priv, _ = generate_keypair()
    paths = save_keypair(priv, args.out_dir, args.name)
    print("wrote", paths)
