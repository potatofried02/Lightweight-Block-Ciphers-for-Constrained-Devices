import numpy as np
import cupy as cp


class SpeckGPU:
    """
    GPU-based implementation of SPECK-64/128 and SPECK-64/96.

    This implementation follows the same byte ordering and round function
    as SpeckCPU. Single-block encrypt/decrypt wrappers are provided, but
    the main GPU benefit comes from encrypt_blocks/decrypt_blocks.
    """

    BLOCK_SIZE = 64
    WORD_SIZE = 32
    ALPHA = 8
    BETA = 3
    _MASK = 0xFFFFFFFF

    def __init__(self, key: bytes):
        """
        Initialize the cipher with a 96-bit or 128-bit key.

        Arguments:
            key -- 12-byte key for SPECK-64/96 or 16-byte key for SPECK-64/128.
        """
        if len(key) == 12:
            self.key_size = 96
            self.num_key_words = 3
            self.num_rounds = 26
        elif len(key) == 16:
            self.key_size = 128
            self.num_key_words = 4
            self.num_rounds = 27
        else:
            raise ValueError(
                f"Key length must be 96 or 128 bits, given key is {len(key) * 8} bits"
            )

        self._round_keys_host = self._key_schedule(key)
        self._round_keys = cp.asarray(self._round_keys_host, dtype=cp.uint32)
        self._mask_gpu = cp.uint32(self._MASK)

    def _rotr32_host(self, x: int, n: int) -> int:
        """Apply a host-side 32-bit right rotation."""
        return ((x >> n) | (x << (32 - n))) & self._MASK

    def _rotl32_host(self, x: int, n: int) -> int:
        """Apply a host-side 32-bit left rotation."""
        return ((x << n) | (x >> (32 - n))) & self._MASK

    def _rotr32(self, x, n: int):
        """Apply a GPU-side 32-bit right rotation."""
        return ((x >> n) | (x << (32 - n))) & self._mask_gpu

    def _rotl32(self, x, n: int):
        """Apply a GPU-side 32-bit left rotation."""
        return ((x << n) | (x >> (32 - n))) & self._mask_gpu

    def _key_schedule(self, key: bytes) -> list[int]:
        """
        Expand the master key into SPECK round keys.

        This matches the SpeckCPU key schedule exactly.
        """
        m = self.num_key_words

        k_0 = int.from_bytes(key[-4:], "big")
        ell = [
            int.from_bytes(key[(m - 2 - i) * 4 : (m - 1 - i) * 4], "big")
            for i in range(m - 1)
        ]

        k = [k_0]

        for i in range(self.num_rounds - 1):
            new_ell = ((k[i] + self._rotr32_host(ell[i], self.ALPHA)) & self._MASK) ^ i
            ell.append(new_ell)
            k.append(self._rotl32_host(k[i], self.BETA) ^ new_ell)

        return k

    def _bytes_to_words_gpu(self, data: bytes):
        """
        Convert byte string into two CuPy uint32 arrays.

        Byte packing is done on CPU with NumPy to avoid triggering unnecessary
        CuPy JIT compilation during input conversion.
        """
        if len(data) % 8 != 0:
            raise ValueError("Input length must be a multiple of 8 bytes.")

        # Interpret every 4 bytes as one big-endian uint32.
        # Each 64-bit block becomes two 32-bit words: x and y.
        host_words = np.frombuffer(data, dtype=">u4").astype(np.uint32).reshape(-1, 2)

        x = cp.asarray(host_words[:, 0], dtype=cp.uint32)
        y = cp.asarray(host_words[:, 1], dtype=cp.uint32)

        return x, y


    def _words_gpu_to_bytes(self, x, y) -> bytes:
        """
        Convert two CuPy uint32 arrays back into a byte string.

        Byte unpacking is done on CPU with NumPy after copying GPU results back.
        """
        cp.cuda.Stream.null.synchronize()

        x_host = cp.asnumpy(x).astype(np.uint32)
        y_host = cp.asnumpy(y).astype(np.uint32)

        host_words = np.empty((x_host.size, 2), dtype=">u4")
        host_words[:, 0] = x_host
        host_words[:, 1] = y_host

        return host_words.reshape(-1).tobytes()

    def encrypt_blocks(self, plaintext: bytes) -> bytes:
        """
        Encrypt multiple 64-bit blocks.

        Arguments:
            plaintext -- byte string with length multiple of 8.

        Returns:
            ciphertext byte string.
        """
        x, y = self._bytes_to_words_gpu(plaintext)

        for i in range(self.num_rounds):
            x = ((self._rotr32(x, self.ALPHA) + y) & self._mask_gpu) ^ self._round_keys[i]
            y = self._rotl32(y, self.BETA) ^ x

        return self._words_gpu_to_bytes(x, y)

    def decrypt_blocks(self, ciphertext: bytes) -> bytes:
        """
        Decrypt multiple 64-bit blocks.

        Arguments:
            ciphertext -- byte string with length multiple of 8.

        Returns:
            plaintext byte string.
        """
        x, y = self._bytes_to_words_gpu(ciphertext)

        for i in reversed(range(self.num_rounds)):
            y = self._rotr32(y ^ x, self.BETA)
            x = self._rotl32(((x ^ self._round_keys[i]) - y) & self._mask_gpu, self.ALPHA)

        return self._words_gpu_to_bytes(x, y)

    def encrypt(self, plaintext: bytes) -> bytes:
        """
        Encrypt a single 64-bit block.
        """
        if len(plaintext) != 8:
            raise ValueError("encrypt() expects exactly one 8-byte block.")

        return self.encrypt_blocks(plaintext)

    def decrypt(self, ciphertext: bytes) -> bytes:
        """
        Decrypt a single 64-bit block.
        """
        if len(ciphertext) != 8:
            raise ValueError("decrypt() expects exactly one 8-byte block.")

        return self.decrypt_blocks(ciphertext)
