import pytest
import numpy as np

try:
    import cupy as cp
except Exception as e:
    pytest.skip(f"CuPy is not available: {e}", allow_module_level=True)

try:
    if cp.cuda.runtime.getDeviceCount() < 1:
        pytest.skip("No CUDA GPU available.", allow_module_level=True)
except Exception as e:
    pytest.skip(f"CUDA runtime is not available: {e}", allow_module_level=True)

from ciphers.skinny_cpu import SkinnyCPU
from ciphers.skinny_gpu import SkinnyGPU


SKINNY_64_64_VECTOR = {
    "key":        bytes.fromhex("f5269826fc681238"),
    "plaintext":  bytes.fromhex("06034f957724d19d"),
    "ciphertext": bytes.fromhex("bb39dfb2429b8ac7"),
}

SKINNY_64_128_VECTOR = {
    "key":        bytes.fromhex("9eb93640d088da6376a39d1c8bea71e1"),
    "plaintext":  bytes.fromhex("cf16cfe8fd0f98aa"),
    "ciphertext": bytes.fromhex("6ceda1f43de92b9e"),
}

SKINNY_64_192_VECTOR = {
    "key":        bytes.fromhex("ed00c85b120d68618753e24bfd908f60b2dbb41b422dfcd0"),
    "plaintext":  bytes.fromhex("530c61d35e8663c3"),
    "ciphertext": bytes.fromhex("dd2cf1a8f330303c"),
}


class TestSkinnyGPU64_64:
    """
    Validate SKINNY-64/64 GPU implementation against the original publication's
    test vector.
    """

    def test_encrypt(self):
        cipher = SkinnyGPU(SKINNY_64_64_VECTOR["key"])
        ct = cipher.encrypt(SKINNY_64_64_VECTOR["plaintext"])
        assert ct == SKINNY_64_64_VECTOR["ciphertext"], (
            f"GPU encrypt mismatch.\n"
            f"  expected: {SKINNY_64_64_VECTOR['ciphertext'].hex()}\n"
            f"  got:      {ct.hex()}"
        )

    def test_decrypt(self):
        cipher = SkinnyGPU(SKINNY_64_64_VECTOR["key"])
        pt = cipher.decrypt(SKINNY_64_64_VECTOR["ciphertext"])
        assert pt == SKINNY_64_64_VECTOR["plaintext"], (
            f"GPU decrypt mismatch.\n"
            f"  expected: {SKINNY_64_64_VECTOR['plaintext'].hex()}\n"
            f"  got:      {pt.hex()}"
        )


class TestSkinnyGPU64_128:
    """
    Validate SKINNY-64/128 GPU implementation against the original publication's
    test vector.
    """

    def test_encrypt(self):
        cipher = SkinnyGPU(SKINNY_64_128_VECTOR["key"])
        ct = cipher.encrypt(SKINNY_64_128_VECTOR["plaintext"])
        assert ct == SKINNY_64_128_VECTOR["ciphertext"], (
            f"GPU encrypt mismatch.\n"
            f"  expected: {SKINNY_64_128_VECTOR['ciphertext'].hex()}\n"
            f"  got:      {ct.hex()}"
        )

    def test_decrypt(self):
        cipher = SkinnyGPU(SKINNY_64_128_VECTOR["key"])
        pt = cipher.decrypt(SKINNY_64_128_VECTOR["ciphertext"])
        assert pt == SKINNY_64_128_VECTOR["plaintext"], (
            f"GPU decrypt mismatch.\n"
            f"  expected: {SKINNY_64_128_VECTOR['plaintext'].hex()}\n"
            f"  got:      {pt.hex()}"
        )


class TestSkinnyGPU64_192:
    """
    Validate SKINNY-64/192 GPU implementation against the original publication's
    test vector.
    """

    def test_encrypt(self):
        cipher = SkinnyGPU(SKINNY_64_192_VECTOR["key"])
        ct = cipher.encrypt(SKINNY_64_192_VECTOR["plaintext"])
        assert ct == SKINNY_64_192_VECTOR["ciphertext"], (
            f"GPU encrypt mismatch.\n"
            f"  expected: {SKINNY_64_192_VECTOR['ciphertext'].hex()}\n"
            f"  got:      {ct.hex()}"
        )

    def test_decrypt(self):
        cipher = SkinnyGPU(SKINNY_64_192_VECTOR["key"])
        pt = cipher.decrypt(SKINNY_64_192_VECTOR["ciphertext"])
        assert pt == SKINNY_64_192_VECTOR["plaintext"], (
            f"GPU decrypt mismatch.\n"
            f"  expected: {SKINNY_64_192_VECTOR['plaintext'].hex()}\n"
            f"  got:      {pt.hex()}"
        )


class TestSkinnyGPUBatchMatchesCPU:
    """
    Verify GPU batch encryption produces byte-for-byte identical results to the
    reference SkinnyCPU implementation across many random blocks.
    """

    def test_batch_encrypt_matches_cpu_64_128(self):
        rng = np.random.default_rng(seed=0)
        num_blocks = 1024
        plaintext = rng.integers(low=0, high=256, size=num_blocks * 8, dtype=np.uint8).tobytes()
        key = SKINNY_64_128_VECTOR["key"]

        cpu_cipher = SkinnyCPU(key)
        gpu_cipher = SkinnyGPU(key)

        expected = b"".join(
            cpu_cipher.encrypt(plaintext[i : i + 8])
            for i in range(0, len(plaintext), 8)
        )
        got = gpu_cipher.encrypt_blocks(plaintext)

        assert got == expected, "GPU batch encrypt does not match CPU reference."

    def test_batch_roundtrip_64_128(self):
        rng = np.random.default_rng(seed=1)
        num_blocks = 1024
        plaintext = rng.integers(low=0, high=256, size=num_blocks * 8, dtype=np.uint8).tobytes()
        key = SKINNY_64_128_VECTOR["key"]

        cipher = SkinnyGPU(key)
        ciphertext = cipher.encrypt_blocks(plaintext)
        recovered = cipher.decrypt_blocks(ciphertext)

        assert recovered == plaintext, "GPU encrypt/decrypt roundtrip lost data."
