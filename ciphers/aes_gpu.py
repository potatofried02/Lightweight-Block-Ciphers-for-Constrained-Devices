import cupy as cp
import numpy as np

class AesGPU:
    '''
    GPU-based implementation of AES-128
    '''
    NK = 4  # key length (number of 32-bit words)
    NB = 4  # block size (number of 32-bit words)
    NR = 10   # num rounds

    S_BOX = [
        0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
        0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
        0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
        0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
        0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
        0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
        0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
        0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
        0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
        0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
        0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
        0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
        0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
        0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
        0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
        0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16,
    ]

    INV_SBOX = [
        0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb,
        0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87, 0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb,
        0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d, 0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e,
        0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2, 0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25,
        0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92,
        0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda, 0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84,
        0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a, 0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06,
        0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02, 0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b,
        0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea, 0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73,
        0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85, 0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e,
        0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89, 0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b,
        0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20, 0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4,
        0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31, 0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f,
        0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d, 0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef,
        0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0, 0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61,
        0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26, 0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d,
    ]

    ROUND_CONSTANTS = [0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36]

    def __init__(self, key: bytes):
        '''
        Initialize the cipher with a 128-bit key. The key schedule runs once
        and subsequent encrypt/decrypt calls reuse the precomputed round keys. 
        Internal state is viewed as a 4x4 square matrix. A flattened version 
        is used here to improve implementation efficiency. Each byte in the 
        state array is interpreted as one of the 256 elements of a finite field, 
        known as a Galois Field, denoted by GF(2^8).
        
        Arguments:
            key -- 128-bit key
        '''
        self.key = key

        # load sboxes onto GPU
        self._sbox = cp.asarray(self.S_BOX, dtype=cp.uint8)
        self._inv_sbox = cp.asarray(self.INV_SBOX, dtype=cp.uint8)

        # load round keys onto GPU
        w = self._key_schedule(key)
        rk = np.empty((self.NR + 1, 16), dtype=np.uint8)
        for rnd in range(self.NR + 1):
            for c in range(4):
                word = w[4 * rnd + c]
                for r in range(4):
                    rk[rnd, r + 4 * c] = word[r]
        self.round_keys = cp.asarray(rk)

    def _xtime(self, b: cp.ndarray) -> cp.ndarray:
        '''
        Multiply a byte b by c = {02} in GF(2^8).
        
        Arguments:
            b -- Input byte

        Returns:
            Input byte multiplied by c = {02}
        '''
        b = b.astype(cp.uint16) << 1
        b = cp.where(b & 0x100, b ^ 0x11b, b)

        return (b & 0xff).astype(cp.uint8)
    
    def _mul(self, a: int, b: cp.ndarray) -> cp.ndarray:
        '''
        Multiply two bytes in GF(2^8).
        
        Arguments:
            a -- 1st input byte
            b -- 2nd input byte(s)
        
        Returns:
            Product of a and b in in GF(2^8)
        '''
        p = cp.zeros_like(b)
        for _ in range(8):
            if a & 1:
                p ^= b
            a >>= 1
            b = self._xtime(b)

        return (p & 0xff).astype(cp.uint8)
    
    def _key_schedule(self, key: bytes) -> list[list[int]]:
        '''
        Expand the master key into w = NB*(NR+1) = 44 words.

        Arguments:
            key -- 128-bit key

        Returns:
            List with 44 entries (each entry is a 4-byte list)
        '''
        w = [list(key[4 * i: 4 * i + 4]) for i in range(self.NK)]
        for i in range(self.NK, self.NB * (self.NR + 1)):
            temp = list(w[i - 1])
            if i % self.NK == 0:
                temp = temp[1:] + temp[:1]                     
                temp = [self.S_BOX[b] for b in temp]           
                temp[0] ^= self.ROUND_CONSTANTS[i // self.NK]       
            w.append([a ^ b for a, b in zip(w[i - self.NK], temp)])

        return w

    def _sub_bytes(self, state: cp.ndarray) -> cp.ndarray:
        '''
        Apply S-box independently to each byte in the state.
        
        Arguments:
            state -- Flattened 4x4 matrix that represented the internal state of the cipher

        Returns:
            Input state modified by the S-box
        '''
        return self._sbox[state]
    
    def _inv_sub_bytes(self, state: cp.ndarray) -> cp.ndarray:
        '''
        Apply inverse S-box independently to each byte in the state.
        
        Arguments:
            state -- Flattened 4x4 matrix that represented the internal state of the cipher

        Returns:
            Input state modified by the inverse S-box
        '''
        return self._inv_sbox[state]

    def _shift_rows(self, state: cp.ndarray) -> cp.ndarray:
        '''
        Cyclically shift the last three rows of the state.

        Arguments:
            state -- Flattened 4x4 matrix that represented the internal state of the cipher

        Returns:
            Input state with last three rows cyclically shifted
        '''
        out = cp.empty_like(state)
        for r in range(4):
            for c in range(4):
                out[:, r + 4 * c] = state[:, r + 4 * ((c + r) % 4)]

        return out
    
    def _inv_shift_rows(self, state: cp.ndarray) -> cp.ndarray:
        '''
        Apply the inverse cyclical shift to the last three rows of the state.

        Arguments:
            state -- Flattened 4x4 matrix that represented the internal state of the cipher

        Returns:
            Input state with inverse cyclical shift applied to the last three rows
        '''
        out = cp.empty_like(state)
        for r in range(4):
            for c in range(4):
                out[:, r + 4 * c] = state[:, r + 4 * ((c - r) % 4)]

        return out

    def _mix_columns(self, state: cp.ndarray) -> cp.ndarray:
        '''
        Multiples each of the four columns of the state by a single fixed matrix.

        Arguments:
            state -- Flattened 4x4 matrix that represented the internal state of the cipher

        Returns:
            Input state's columns transformed by the fixed matrix
        '''
        out = cp.empty_like(state)
        for c in range(4):
            s0, s1, s2, s3 = (state[:, r + 4 * c] for r in range(4))
            out[:, 0 + 4 * c] = self._mul(2, s0) ^ self._mul(3, s1) ^ s2 ^ s3
            out[:, 1 + 4 * c] = s0 ^ self._mul(2, s1) ^ self._mul(3, s2) ^ s3
            out[:, 2 + 4 * c] = s0 ^ s1 ^ self._mul(2, s2) ^ self._mul(3, s3)
            out[:, 3 + 4 * c] = self._mul(3, s0) ^ s1 ^ s2 ^ self._mul(2, s3)

        return out
    
    def _inv_mix_columns(self, state: cp.ndarray) -> cp.ndarray:
        '''
        Multiples each of the four columns of the state by a single fixed matrix.

        Arguments:
            state -- Flattened 4x4 matrix that represented the internal state of the cipher

        Returns:
            Input state's columns transformed by the fixed matrix
        '''
        out = cp.empty_like(state)
        for c in range(4):
            s0, s1, s2, s3 = (state[:, r + 4 * c] for r in range(4))
            out[:, 0 + 4 * c] = self._mul(0x0e, s0) ^ self._mul(0x0b, s1) ^ self._mul(0x0d, s2) ^ self._mul(0x09, s3)
            out[:, 1 + 4 * c] = self._mul(0x09, s0) ^ self._mul(0x0e, s1) ^ self._mul(0x0b, s2) ^ self._mul(0x0d, s3)
            out[:, 2 + 4 * c] = self._mul(0x0d, s0) ^ self._mul(0x09, s1) ^ self._mul(0x0e, s2) ^ self._mul(0x0b, s3)
            out[:, 3 + 4 * c] = self._mul(0x0b, s0) ^ self._mul(0x0d, s1) ^ self._mul(0x09, s2) ^ self._mul(0x0e, s3)

        return out

    def _add_round_key(self, state: cp.ndarray, rnd: int) -> cp.ndarray:
        '''
        Round key is combined with the state by applying the bitwise XOR operation.

        Arguments:
            state -- Flattened 4x4 matrix that represented the internal state of the cipher
            rnd -- current round

        Returns:
            Input state transformed by the round key
        '''
        return state ^ self.round_keys[rnd]

    def encrypt(self, plaintext: bytes) -> bytes:
        ''' 
        Encrypt a single 128-bit block.

        Arguments:
            plaintext -- 16 bytes (128-bits) of plaintext

        Returns:
            16 bytes of encrypted plaintext
        '''
        state = cp.asarray(np.frombuffer(plaintext, dtype=np.uint8)).reshape(-1, 16)              
        state = self._add_round_key(state, 0)
        for rnd in range(1, self.NR):
            state = self._sub_bytes(state)
            state = self._shift_rows(state)
            state = self._mix_columns(state)
            state = self._add_round_key(state, rnd)
        state = self._sub_bytes(state)
        state = self._shift_rows(state)
        state = self._add_round_key(state, self.NR)  

        return cp.asnumpy(state).reshape(-1).tobytes()

    def decrypt(self, ciphertext: bytes) -> bytes:
        ''' 
        Encrypt a single 128-bit block.

        Arguments:
            ciphertext -- 16 bytes (128-bits) of ciphertext

        Returns:
            16 bytes of recovered plaintext
        '''
        state = cp.asarray(np.frombuffer(ciphertext, dtype=np.uint8)).reshape(-1, 16)
        state = self._add_round_key(state, self.NR)
        for rnd in range(self.NR - 1, 0, -1):
            state = self._inv_shift_rows(state)
            state = self._inv_sub_bytes(state)
            state = self._add_round_key(state, rnd)
            state = self._inv_mix_columns(state)
        state = self._inv_shift_rows(state)
        state = self._inv_sub_bytes(state)
        state = self._add_round_key(state, 0)

        return cp.asnumpy(state).reshape(-1).tobytes()