class SpeckCPU:
    '''
    CPU based implementation of SPECK-64/128, a lightweight block cipher, 
    that encrypts a 64-bit block with a 128-bit key
    '''
    BLOCK_SIZE = 64    # bits
    WORD_SIZE = 32     # bits
    ALPHA = 8
    BETA = 3
    # Mask used to force python to trim to 32-bits
    _MASK = 0xFFFFFFFF  # 2**WORD_SIZE - 1 (hexadecimal)

    def __init__(self, key: bytes):
        '''
        Initialize the cipher with a 128-bit key. The key schedule runs once
        and subsequent encrypt/decrypt calls reuse the precomputed round keys.

        Arguments:
            key -- 16 byte (128 bit) or 12 byte (96 bit) key.
        '''
        if (len(key) == 12):
            self.key_size = 96      # bits
            self.num_key_words = 3  
            self.num_rounds = 26
        elif (len(key) == 16):
            self.key_size = 128     # bits
            self.num_key_words = 4
            self.num_rounds = 27
        else:
            raise ValueError(f"Key length must be 96 or 128 bits, given key is {len(key) * 8} bits")

        self._round_keys = self._key_expansion(key)

    def _rotr32(self, x: int, n: int) -> int:
        '''
        Apply a right circular shift to the 32-bit word input x by n bits

        Arguments:
        x -- 32-bit word input
        n -- Number of bits to rotate input word by

        Returns:
        Input right circular shifted by 32-bits
        '''
        return ((x >> n) | (x << (32 - n))) & self._MASK

    def _rotl32(self, x: int, n: int) -> int:
        '''
        Apply a left circular shift the 32-bit word input x by n bits

        Arguments:
        x -- 32-bit word input
        n -- Number of bits to rotate input word by

        Returns:
        Input left circular shifted by 32-bits
        '''
        return ((x << n) | (x >> (32 - n))) & self._MASK
    
    def _key_expansion(self, key: bytes) -> list[int]:
        '''
        Expand the master key into 26 or 27 32-bit round keys.

        Arguments:
            key -- 16 byte (128 bit) or 12 byte (96 bit) key in hexadecimal format.

        Returns:
            k -- List of 26 or 27 integers, each a 32-bit round key.
        '''
        m = self.num_key_words
        k_0 = int.from_bytes(key[-4:], 'big')
        ell = [int.from_bytes(key[(m - 2 - i) * 4 : (m - 1 - i) * 4], 'big') for i in range(m - 1)]
        k = [k_0]

        for i in range(self.num_rounds - 1):
            new_ell = ((k[i] + self._rotr32(ell[i], self.ALPHA)) & self._MASK) ^ i
            ell.append(new_ell)
            k.append(self._rotl32(k[i], self.BETA) ^ new_ell)

        return k

    def encrypt(self, plaintext: bytes) -> bytes:
        ''' 
        Encrypt a single 64-bit block.

        Arguments:
            plaintext -- 8 bytes (64 bits) of plaintext.

        Returns:
            8 bytes of encrypted plaintext.
        '''
        x = int.from_bytes(plaintext[0:4], 'big')
        y = int.from_bytes(plaintext[4:8], 'big')

        for i in range(self.num_rounds):
            x = ((self._rotr32(x, self.ALPHA) + y) & self._MASK) ^ self._round_keys[i]
            y = self._rotl32(y, self.BETA) ^ x
        
        return x.to_bytes(4, 'big') + y.to_bytes(4, 'big')

    def decrypt(self, ciphertext: bytes) -> bytes:
        ''' 
        Decrypt a single 64-bit block.

        Arguments:
            ciphertext -- 8 bytes (64 bits) of ciphertext.

        Returns:
            8 bytes of recovered plaintext.
        '''
        x = int.from_bytes(ciphertext[0:4], 'big')
        y = int.from_bytes(ciphertext[4:8], 'big')

        for i in reversed(range(self.num_rounds)):
            y = self._rotr32(y ^ x, self.BETA)
            x = self._rotl32(((x ^ self._round_keys[i]) - y) & self._MASK, self.ALPHA)
        
        return x.to_bytes(4, 'big') + y.to_bytes(4, 'big')