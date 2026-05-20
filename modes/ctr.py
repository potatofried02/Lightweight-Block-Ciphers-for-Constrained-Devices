class CTR:
    def __init__(self, cipher, counter: bytes, block_size: int = 8):
        '''
        Counter mode (CTR). Uses a block cipher as a stream cipher where 
        the key stream is computed in a blockwise fashion. Input to the 
        block cipher is a counter which assumes a different value each time 
        block cipher computes a new key stream block.
        '''
        self.cipher = cipher
        self.counter = counter
        self.block_size = block_size

    def _xor(self, a: bytes, b: bytes) -> bytes:
        '''
        Xor a string of bytes together byte by byte.

        Arguments:
            a -- 1st byte string
            b -- 2nd byte string

        Returns:
            Xor of a and b
        '''
        return bytes(x ^ y for x, y in zip(a, b))

    def encrypt(self, plaintext: bytes) -> bytes:
        '''
        Encryption using CTR mode with the provided cipher.

        Arguments:
            plaintext -- Unencrypted data in bytes

        Returns:
            ciphertext -- Encrypted plaintext
        '''
        ciphertext = b""
        counter_int = int.from_bytes(self.counter, byteorder="big")
        max_counter = 1 << (self.block_size * 8)

        for i in range(0, len(plaintext), self.block_size):
            block = plaintext[i:i + self.block_size]
            counter_bytes = counter_int.to_bytes(self.block_size, byteorder="big")
            keystream = self.cipher.encrypt(counter_bytes)
            ciphertext += self._xor(block, keystream[:len(block)])
            counter_int = (counter_int + 1) % max_counter

        return ciphertext

    def decrypt(self, ciphertext: bytes) -> bytes:
        ''' 
        Decryption using CTR mode with the provided cipher. Since CTR 
        is a stream cipher, decryption is identical to encryption (XOR 
        the ciphertext with the same keystream to recover the plaintext).

        Arguments:
            ciphertext -- Encrypted data in bytes

        Returns:
            Recovered plaintext
        '''
        return self.encrypt(ciphertext)