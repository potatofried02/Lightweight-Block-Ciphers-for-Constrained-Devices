class SkinnyCPU:
    '''
    CPU-based implementation of SKINNY-64/128, a lightweight tweakable
    block cipher that encrypts a 64-bit block with a 64-bit, 128-bit, or 192-bit key.
    '''
    BLOCK_SIZE = 64  # bits
    CELL_SIZE = 4    # bits

    S_BOX = [0xc, 0x6, 0x9, 0x0, 0x1, 0xa, 0x2, 0xb, 0x3, 0x8, 0x5, 0xd, 0x4, 0xe, 0x7, 0xf]
    S_BOX_INV = [0x3, 0x4, 0x6, 0x8, 0xc, 0xa, 0x1, 0xe, 0x9, 0x2, 0x5, 0x7, 0x0, 0xb, 0xd, 0xf]

    P_T = [9, 15, 8, 13, 10, 14, 12, 11, 0, 1, 2, 3, 4, 5, 6, 7]
    P_SR = [0, 1, 2, 3, 7, 4, 5, 6, 10, 11, 8, 9, 13, 14, 15, 12]
    P_SR_INV = [0, 1, 2, 3, 5, 6, 7, 4, 10, 11, 8, 9, 15, 12, 13, 14]

    ROUND_CONSTANTS = [ 0x01, 0x03, 0x07, 0x0F, 0x1F, 0x3E, 0x3D, 0x3B, 0x37, 0x2F, 0x1E, 0x3C, 0x39, 0x33, 0x27, 0x0E, 
                        0x1D, 0x3A, 0x35, 0x2B, 0x16, 0x2C, 0x18, 0x30, 0x21, 0x02, 0x05, 0x0B, 0x17, 0x2E, 0x1C, 0x38, 
                        0x31, 0x23, 0x06, 0x0D, 0x1B, 0x36, 0x2D, 0x1A, 0x34, 0x29, 0x12, 0x24, 0x08, 0x11, 0x22, 0x04, 
                        0x09, 0x13, 0x26, 0x0C, 0x19, 0x32, 0x25, 0x0A, 0x15, 0x2A, 0x14, 0x28, 0x10, 0x20,
                    ]

    def __init__(self, key: bytes):
        '''
        Initialize the cipher with a 64-bit, 128-bit, or 192-bit key. The key schedule runs once
        and subsequent encrypt/decrypt calls reuse the precomputed round keys. Internal state is 
        viewed as a 4x4 square array of cells, where each cell is a nibble in original paper. A 
        flattened version is used here to improve implementation efficiency.

        Arguments:
            key -- 64-bit, 128-bit, or 192-bit key
        '''
        if (len(key) == 8):
            self.key_size = 64      # bits
            self.num_tweakey = 1
            self.num_rounds = 32
        elif (len(key) == 16):
            self.key_size = 128     # bits
            self.num_tweakey = 2
            self.num_rounds = 36
        elif (len(key) == 24):
            self.key_size = 192     # bits
            self.num_tweakey = 3
            self.num_rounds = 40
        else:
            raise ValueError(f"Key length must be 64, 128, or 192 bits, given key is {len(key) * 8} bits")
    
        self.key = key
        nibbles = self._bytes_to_nibbles(key)

        initial_tweakey = [nibbles[16*i : 16*(i+1)] for i in range(self.num_tweakey)]
        self.round_subtweakeys = self._key_schedule(initial_tweakey)

    def _bytes_to_nibbles(self, byte: bytes) -> list[int]:
        '''
        Unpacks given bytes to nibbles, high nibble first (4 bits/nibble).

        Arguments:
            byte -- Input bytes

        Returns:
            nibbles -- List of nibbles
        '''
        nibbles = []
        for b in byte:
            nibbles.append(b >> self.CELL_SIZE)
            nibbles.append(b & 0xF)

        return nibbles
    
    def _nibbles_to_bytes(self, nibbles: list[int]) -> bytes:
        '''
        Pack a list of nibbles into bytes, high nibble first (2 nibbles/byte).
        
        Arguments:
            nibbles -- List of nibbles

        Returns:
            bytes -- Nibbles converted back into bytes
        '''
        return bytes((nibbles[2*i] << 4) | nibbles[2*i + 1] for i in range(len(nibbles) // 2))
    
    def _lfsr_tk2(self, x: int) -> int:
        '''
        Linear feedback shift register (LFSR) for TK2 (2 tweakey words in the cipher). 
        (x3 x2 x1 x0) -> (x2 x1 x0  x3^x2)

        Arguments:
            x -- Input to be shifted (x3 x2 x1 x0)

        Returns:
            Shifted input (x2 x1 x0  x3^x2)
        '''
        return ((x << 1) & 0xF) | (((x >> 3) ^ (x >> 2)) & 1)

    def _lfsr_tk3(self, x: int) -> int:
        '''
        Linear feedback shift register (LFSR) for TK3 (3 tweakey words in the cipher). 
        (x3 x2 x1 x0) -> (x0^x3  x3 x2 x1)

        Arguments:
            x -- Input to be shifted (x3 x2 x1 x0)

        Returns:
            Shifted input (x0^x3  x3 x2 x1)
        '''
        return (x >> 1) | (((x & 1) ^ ((x >> 3) & 1)) << 3)

    def _key_schedule(self, tweakey: list[list[int]]) -> list[list[int]]:
        '''
        Precompute the round subtweakeys for all rounds.

        Arguments:
            tweakey -- List of `num_tweakey` elements, each a list of 16 nibbles 
                        representing one tweakey word (TK1, TK2, TK3).

        Returns:
            round_subtweakeys -- List of `num_rounds` elements, each a list of 8 nibbles.
        '''
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

    def _sub_cells(self, state: list[int]) -> None:
        ''' 
        Apply a 4-bit Sbox (substitution table) to every cell of the cipher internal state.

        Arguments:
            state -- List of cells (each a nibble) that describe the internal state of the cipher.
        '''
        for i in range(16):
            state[i] = self.S_BOX[state[i]]

    def _add_constants(self, state: list[int], round_idx: int) -> None:
        ''' 
        XOR the round constant into cells 0, 4, and 8.

        Arguments:
            state -- List of cells (each a nibble) that describe the internal state of the cipher.
            round_idx -- Current round index (used to retrieve the corresponding constant for that round).
        '''
        rc = self.ROUND_CONSTANTS[round_idx]
        state[0] ^= rc & 0xF
        state[4] ^= (rc >> 4) & 0x3
        state[8] ^= 0x2

    def _add_round_tweakey(self, state: list[int], round_idx: int) -> None:
        ''' 
        XOR the precomputed subtweakey into the first two rows (cells 0-7).

        Arguments:
            state -- List of cells (each a nibble) that describe the internal state of the cipher.
            round_idx -- Current round index (used to retrieve the corresponding subtweakey for that round).
        '''
        subtweakey = self.round_subtweakeys[round_idx]
        for i in range(8):
            state[i] ^= subtweakey[i]

    def _shift_rows(self, state: list[int]) -> None:
        ''' 
        Right-rotate row i by i cells.

        Arguments:
            state -- List of cells (each a nibble) that describe the internal state of the cipher.
        '''
        state[:] = [state[self.P_SR[i]] for i in range(16)]

    def _mix_columns(self, state: list[int]) -> None:
        ''' 
        Multiply each column by the binary matrix M.

        Arguments:
            state -- List of cells (each a nibble) that describe the internal state of the cipher.
        '''
        for j in range(4):
            c0, c1, c2, c3 = state[j], state[j+4], state[j+8], state[j+12]
            state[j]    = c0 ^ c2 ^ c3
            state[j+4]  = c0
            state[j+8]  = c1 ^ c2
            state[j+12] = c0 ^ c2

    def _inv_sub_cells(self, state: list[int]) -> None:
        '''
        Apply the inverse 4-bit Sbox (substitution table) to every cell of the cipher internal state. 

        Arguments:
            state -- List of cells (each a nibble) that describe the internal state of the cipher.
        '''
        for i in range(16):
            state[i] = self.S_BOX_INV[state[i]]

    def _inv_shift_rows(self, state: list[int]) -> None:
        '''
        Invert the shift row operation.

        Arguments:
            state -- List of cells (each a nibble) that describe the internal state of the cipher.
        '''
        state[:] = [state[self.P_SR_INV[i]] for i in range(16)]

    def _inv_mix_columns(self, state: list[int]) -> None:
        '''
        Multiply each column by the inverse of the binary matrix M.

        Arguments:
            state -- List of cells (each a nibble) that describe the internal state of the cipher.
        '''
        for j in range(4):
            c0, c1, c2, c3 = state[j], state[j+4], state[j+8], state[j+12]
            state[j]    = c1
            state[j+4]  = c1 ^ c2 ^ c3
            state[j+8]  = c1 ^ c3
            state[j+12] = c0 ^ c3

    def encrypt(self, plaintext: bytes) -> bytes:
        ''' 
        Encrypt a single 64-bit block.

        Arguments:
            plaintext -- 8 bytes (64 bits) of plaintext.

        Returns:
            8 bytes of encrypted plaintext.
        '''
        state = self._bytes_to_nibbles(plaintext)
        for r in range(self.num_rounds):
            self._sub_cells(state)
            self._add_constants(state, r)
            self._add_round_tweakey(state, r)
            self._shift_rows(state)
            self._mix_columns(state)

        return self._nibbles_to_bytes(state)

    def decrypt(self, ciphertext: bytes) -> bytes:
        ''' 
        Decrypt a single 64-bit block.

        Arguments:
            ciphertext -- 8 bytes (64 bits) of ciphertext.

        Returns:
            8 bytes of recovered plaintext.
        '''
        state = self._bytes_to_nibbles(ciphertext)
        for r in reversed(range(self.num_rounds)):
            self._inv_mix_columns(state)
            self._inv_shift_rows(state)
            self._add_round_tweakey(state, r)
            self._add_constants(state, r)
            self._inv_sub_cells(state)

        return self._nibbles_to_bytes(state)