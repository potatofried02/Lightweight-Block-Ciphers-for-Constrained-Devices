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

from ciphers.speck_cpu import SpeckCPU
from ciphers.speck_gpu import SpeckGPU


SPECK_64_128_VECTOR = {
    "key": bytes.fromhex("1b1a1918131211100b0a090803020100"),
    "plaintext": bytes.fromhex("3b7265747475432d"),
    "ciphertext": bytes.fromhex("8c6fa548454e028b"),
}


SPECK_64_96_VECTOR = {
    "key": bytes.fromhex("131211100b0a090803020100"),
    "plaintext": bytes.fromhex("74614620736e6165"),
    "ciphertext": bytes.fromhex("9f7952ec4175946c"),
}


class TestSpeckGPU64_128:
    """
    Validate SPECK-64/128 GPU implementation.
    """

    def test_encrypt(self):
        cipher = SpeckGPU(SPECK_64_128_VECTOR["key"])
        ct = cipher.encrypt(SPECK_64_128_VECTOR["plaintext"])

        assert ct == SPECK_64_128_VECTOR["ciphertext"], (
            f"GPU encrypt mismatch.\n"
            f"  expected: {SPECK_64_128_VECTOR['ciphertext'].hex()}\n"
            f"  got:      {ct.hex()}"
        )

    def test_decrypt(self):
        cipher = SpeckGPU(SPECK_64_128_VECTOR["key"])
        pt = cipher.decrypt(SPECK_64_128_VECTOR["ciphertext"])

        assert pt == SPECK_64_128_VECTOR["plaintext"], (
            f"GPU decrypt mismatch.\n"
            f"  expected: {SPECK_64_128_VECTOR['plaintext'].hex()}\n"
            f"  got:      {pt.hex()}"
        )


class TestSpeckGPU64_96:
    """
    Validate SPECK-64/96 GPU implementation.
    """

    def test_encrypt(self):
        cipher = SpeckGPU(SPECK_64_96_VECTOR["key"])
        ct = cipher.encrypt(SPECK_64_96_VECTOR["plaintext"])

        assert ct == SPECK_64_96_VECTOR["ciphertext"], (
            f"GPU encrypt mismatch.\n"
            f"  expected: {SPECK_64_96_VECTOR['ciphertext'].hex()}\n"
            f"  got:      {ct.hex()}"
        )

    def test_decrypt(self):
        cipher = SpeckGPU(SPECK_64_96_VECTOR["key"])
        pt = cipher.decrypt(SPECK_64_96_VECTOR["ciphertext"])

        assert pt == SPECK_64_96_VECTOR["plaintext"], (
            f"GPU decrypt mismatch.\n"
            f"  expected: {SPECK_64_96_VECTOR['plaintext'].hex()}\n"
            f"  got:      {pt.hex()}"
        )


# def test_speck_gpu_matches_cpu_random_blocks_64_128():
#     rng = np.random.default_rng(seed=0)

#     num_blocks = 1024
#     plaintext = rng.integers(
#         low=0,
#         high=256,
#         size=num_blocks * 8,
#         dtype=np.uint8,
#     ).tobytes()

#     key = SPECK_64_128_VECTOR["key"]

#     cpu_cipher = SpeckCPU(key)
#     gpu_cipher = SpeckGPU(key)

#     expected = b"".join(
#         cpu_cipher.encrypt(plaintext[i : i + 8])
#         for i in range(0, len(plaintext), 8)
#     )

#     got = gpu_cipher.encrypt_blocks(plaintext)

#     assert got == expected


# # def test_speck_gpu_roundtrip_random_blocks_64_128():
#     rng = np.random.default_rng(seed=1)

#     num_blocks = 1024
#     plaintext = rng.integers(
#         low=0,
#         high=256,
#         size=num_blocks * 8,
#         dtype=np.uint8,
#     ).tobytes()

#     key = SPECK_64_128_VECTOR["key"]

#     cipher = SpeckGPU(key)
#     ciphertext = cipher.encrypt_blocks(plaintext)
#     recovered = cipher.decrypt_blocks(ciphertext)

#     assert recovered == plaintext
