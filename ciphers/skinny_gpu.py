import numpy as np
import cupy as cp


class SkinnyGPU:
    """
    GPU-based implementation of SKINNY-64/64, SKINNY-64/128, and SKINNY-64/192,
    a lightweight tweakable block cipher that encrypts a 64-bit block.

    The implementation mirrors the round structure of SkinnyCPU. Single-block
    encrypt/decrypt wrappers are provided, but the main GPU benefit comes from
    encrypt_blocks/decrypt_blocks, which process many 64-bit blocks in parallel
    as a (N, 16) CuPy array of nibbles.
    """

    BLOCK_SIZE = 64  # bits
    CELL_SIZE = 4    # bits

    S_BOX = [0xc, 0x6, 0x9, 0x0, 0x1, 0xa, 0x2, 0xb, 0x3, 0x8, 0x5, 0xd, 0x4, 0xe, 0x7, 0xf]
    S_BOX_INV = [0x3, 0x4, 0x6, 0x8, 0xc, 0xa, 0x1, 0xe, 0x9, 0x2, 0x5, 0x7, 0x0, 0xb, 0xd, 0xf]

    P_T = [9, 15, 8, 13, 10, 14, 12, 11, 0, 1, 2, 3, 4, 5, 6, 7]
    P_SR = [0, 1, 2, 3, 7, 4, 5, 6, 10, 11, 8, 9, 13, 14, 15, 12]
    P_SR_INV = [0, 1, 2, 3, 5, 6, 7, 4, 10, 11, 8, 9, 15, 12, 13, 14]

    ROUND_CONSTANTS = [0x01, 0x03, 0x07, 0x0F, 0x1F, 0x3E, 0x3D, 0x3B, 0x37, 0x2F, 0x1E, 0x3C, 0x39, 0x33, 0x27, 0x0E,
                       0x1D, 0x3A, 0x35, 0x2B, 0x16, 0x2C, 0x18, 0x30, 0x21, 0x02, 0x05, 0x0B, 0x17, 0x2E, 0x1C, 0x38,
                       0x31, 0x23, 0x06, 0x0D, 0x1B, 0x36, 0x2D, 0x1A, 0x34, 0x29, 0x12, 0x24, 0x08, 0x11, 0x22, 0x04,
                       0x09, 0x13, 0x26, 0x0C, 0x19, 0x32, 0x25, 0x0A, 0x15, 0x2A, 0x14, 0x28, 0x10, 0x20]

    def __init__(self, key: bytes):
        """
        Initialize the cipher with a 64-bit, 128-bit, or 192-bit key. The key
        schedule runs once on the host (mirroring SkinnyCPU) and the resulting
        round subtweakeys are uploaded to the GPU together with the lookup
        tables used by the round function.

        Arguments:
            key -- 8-byte, 16-byte, or 24-byte key.
        """
        if len(key) == 8:
            self.key_size = 64
            self.num_tweakey = 1
            self.num_rounds = 32
        elif len(key) == 16:
            self.key_size = 128
            self.num_tweakey = 2
            self.num_rounds = 36
        elif len(key) == 24:
            self.key_size = 192
            self.num_tweakey = 3
            self.num_rounds = 40
        else:
            raise ValueError(f"Key length must be 64, 128, or 192 bits, given key is {len(key) * 8} bits")

        self.key = key
        nibbles = self._bytes_to_nibbles(key)
        initial_tweakey = [nibbles[16 * i : 16 * (i + 1)] for i in range(self.num_tweakey)]
        self.round_subtweakeys = self._key_schedule(initial_tweakey)

        # Upload all read-only state used by the round function to the GPU once.
        self._round_subtweakeys_gpu = cp.asarray(
            np.array(self.round_subtweakeys, dtype=np.uint8)
        )
        self._sbox_gpu = cp.asarray(np.array(self.S_BOX, dtype=np.uint8))
        self._sbox_inv_gpu = cp.asarray(np.array(self.S_BOX_INV, dtype=np.uint8))
        self._p_sr_gpu = cp.asarray(np.array(self.P_SR, dtype=np.int32))
        self._p_sr_inv_gpu = cp.asarray(np.array(self.P_SR_INV, dtype=np.int32))

    def _bytes_to_nibbles(self, byte: bytes) -> list[int]:
        """
        Unpacks given bytes to nibbles, high nibble first (4 bits/nibble).

        Arguments:
            byte -- Input bytes

        Returns:
            nibbles -- List of nibbles
        """
        nibbles = []
        for b in byte:
            nibbles.append(b >> self.CELL_SIZE)
            nibbles.append(b & 0xF)
        return nibbles

    def _lfsr_tk2(self, x: int) -> int:
        """
        Linear feedback shift register (LFSR) for TK2.
        (x3 x2 x1 x0) -> (x2 x1 x0  x3^x2)
        """
        return ((x << 1) & 0xF) | (((x >> 3) ^ (x >> 2)) & 1)

    def _lfsr_tk3(self, x: int) -> int:
        """
        Linear feedback shift register (LFSR) for TK3.
        (x3 x2 x1 x0) -> (x0^x3  x3 x2 x1)
        """
        return (x >> 1) | (((x & 1) ^ ((x >> 3) & 1)) << 3)

    def _key_schedule(self, tweakey: list[list[int]]) -> list[list[int]]:
        """
        Precompute the round subtweakeys for all rounds. This is identical to
        the host-side schedule used by SkinnyCPU; it intentionally runs on the
        CPU because the cost (a few hundred XOR/permutations) is negligible
        compared to GPU-side encryption of many blocks.

        Arguments:
            tweakey -- List of `num_tweakey` elements, each a list of 16 nibbles
                       representing one tweakey word (TK1, TK2, TK3).

        Returns:
            round_subtweakeys -- List of `num_rounds` elements, each a list of 8 nibbles.
        """
        tk_words = [word.copy() for word in tweakey]
        lfsrs_by_word_idx = {1: self._lfsr_tk2, 2: self._lfsr_tk3}

        round_subtweakeys = []
        for _ in range(self.num_rounds):
            subtweakey = [0] * 8
            for i in range(8):
                for word in tk_words:
                    subtweakey[i] ^= word[i]
            round_subtweakeys.append(subtweakey)

            tk_words = [[word[self.P_T[i]] for i in range(16)] for word in tk_words]

            for word_idx, lfsr in lfsrs_by_word_idx.items():
                if word_idx < len(tk_words):
                    for i in range(8):
                        tk_words[word_idx][i] = lfsr(tk_words[word_idx][i])

        return round_subtweakeys

    def _bytes_to_state_gpu(self, data: bytes):
        """
        Pack a byte string containing N 64-bit blocks into a (N, 16) uint8
        CuPy array of nibbles (high nibble first, matching SkinnyCPU).
        """
        if len(data) % 8 != 0:
            raise ValueError("Input length must be a multiple of 8 bytes.")

        arr = np.frombuffer(data, dtype=np.uint8).reshape(-1, 8)
        nibbles = np.empty((arr.shape[0], 16), dtype=np.uint8)
        nibbles[:, 0::2] = (arr >> 4) & 0xF
        nibbles[:, 1::2] = arr & 0xF

        return cp.asarray(nibbles)

    def _state_gpu_to_bytes(self, state) -> bytes:
        """
        Convert a (N, 16) uint8 CuPy array of nibbles back into a byte string,
        synchronising the device first so the host sees the final values.
        """
        cp.cuda.Stream.null.synchronize()
        nibbles = cp.asnumpy(state)
        bytes_arr = (nibbles[:, 0::2].astype(np.uint8) << 4) | nibbles[:, 1::2].astype(np.uint8)
        return bytes_arr.tobytes()

    def _sub_cells(self, state) -> None:
        """
        Apply the 4-bit Sbox to every cell, broadcast over all blocks.
        """
        state[...] = self._sbox_gpu[state]

    def _inv_sub_cells(self, state) -> None:
        """
        Apply the inverse 4-bit Sbox to every cell, broadcast over all blocks.
        """
        state[...] = self._sbox_inv_gpu[state]

    def _add_constants(self, state, round_idx: int) -> None:
        """
        XOR the round constant into cells 0, 4, and 8 of every block.
        """
        rc = int(self.ROUND_CONSTANTS[round_idx])
        state[:, 0] ^= cp.uint8(rc & 0xF)
        state[:, 4] ^= cp.uint8((rc >> 4) & 0x3)
        state[:, 8] ^= cp.uint8(0x2)

    def _add_round_tweakey(self, state, round_idx: int) -> None:
        """
        XOR the precomputed subtweakey for `round_idx` into cells 0..7 of every block.
        """
        state[:, 0:8] ^= self._round_subtweakeys_gpu[round_idx]

    def _shift_rows(self, state) -> None:
        """
        Right-rotate row i by i cells, applied identically to every block.
        """
        state[...] = state[:, self._p_sr_gpu]

    def _inv_shift_rows(self, state) -> None:
        """
        Invert the shift row operation, applied identically to every block.
        """
        state[...] = state[:, self._p_sr_inv_gpu]

    def _mix_columns(self, state) -> None:
        """
        Multiply each column by the binary matrix M, vectorised over all blocks.
        Each column-slice is copied first to avoid read-after-write hazards.
        """
        c0 = state[:, 0:4].copy()
        c1 = state[:, 4:8].copy()
        c2 = state[:, 8:12].copy()
        c3 = state[:, 12:16].copy()
        state[:, 0:4]   = c0 ^ c2 ^ c3
        state[:, 4:8]   = c0
        state[:, 8:12]  = c1 ^ c2
        state[:, 12:16] = c0 ^ c2

    def _inv_mix_columns(self, state) -> None:
        """
        Multiply each column by the inverse of the binary matrix M, vectorised over all blocks.
        """
        c0 = state[:, 0:4].copy()
        c1 = state[:, 4:8].copy()
        c2 = state[:, 8:12].copy()
        c3 = state[:, 12:16].copy()
        state[:, 0:4]   = c1
        state[:, 4:8]   = c1 ^ c2 ^ c3
        state[:, 8:12]  = c1 ^ c3
        state[:, 12:16] = c0 ^ c3

    def encrypt_blocks(self, plaintext: bytes) -> bytes:
        """
        Encrypt N 64-bit blocks in parallel on the GPU.

        Arguments:
            plaintext -- byte string with length multiple of 8.

        Returns:
            ciphertext byte string of the same length.
        """
        state = self._bytes_to_state_gpu(plaintext)
        for r in range(self.num_rounds):
            self._sub_cells(state)
            self._add_constants(state, r)
            self._add_round_tweakey(state, r)
            self._shift_rows(state)
            self._mix_columns(state)
        return self._state_gpu_to_bytes(state)

    def decrypt_blocks(self, ciphertext: bytes) -> bytes:
        """
        Decrypt N 64-bit blocks in parallel on the GPU.

        Arguments:
            ciphertext -- byte string with length multiple of 8.

        Returns:
            plaintext byte string of the same length.
        """
        state = self._bytes_to_state_gpu(ciphertext)
        for r in reversed(range(self.num_rounds)):
            self._inv_mix_columns(state)
            self._inv_shift_rows(state)
            self._add_round_tweakey(state, r)
            self._add_constants(state, r)
            self._inv_sub_cells(state)
        return self._state_gpu_to_bytes(state)

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
