class CBC:
    def __init__(self, cipher, iv: bytes, block_size: int = 8):
        '''
        Cipher Block Chaining mode (CBC). Encryption of all blocks are 
        chained together and randomized using a provided initialization vector (IV).

        Arguments:
            cipher -- Instance of a cipher object
            iv -- Initialization vector
        '''
        self.cipher = cipher
        self.iv = iv
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
    
    def _pad(self, data: bytes) -> bytes:
        '''
        Pad a given string of bytes with N bytes of value N (PKCS#7) such that 
        its length becomes a multiple of the block size. If it is already 
        a multiple of the block size then a full block of padding is appended.

        Arguments:
            data -- Input bytes to be padded
        
        Returns:
            Input data with padding appended
        '''
        n = self.block_size - (len(data) % self.block_size)
        return data + bytes([n]) * n
    
    def _unpad(self, data: bytes) -> bytes:
        ''''
        Remove the padding appended to a given string of bytes.

        Arguments:
            data -- Input bytes with padding to be removed
        
        Returns:
            Input data with padding removed

        '''
        n = data[-1]
        return data[:-n]
    
    def encrypt(self, plaintext: bytes) -> bytes:
        '''
        Encryption using CBC mode with the provided cipher.

        Arguments:
            plaintext -- Unencrypted data in bytes

        Returns:
            ciphertext -- Encrypted plaintext
        '''
        padded = self._pad(plaintext)
        ciphertext = b""
        prev = self.iv
        for i in range(0, len(padded), self.block_size):
            block = padded[i:i + self.block_size]
            c = self.cipher.encrypt(self._xor(block, prev))
            ciphertext += c
            prev = c
        return ciphertext
        
    def decrypt(self, ciphertext: bytes) -> bytes:
        ''' 
        Decryption using CBC mode with the provided cipher.

        Arguments:
            ciphertext -- Encrypted data in bytes

        Returns:
            Recovered plaintext
        '''
        plaintext = b""
        prev = self.iv
        for i in range(0, len(ciphertext), self.block_size):
            block = ciphertext[i:i + self.block_size]
            p = self._xor(self.cipher.decrypt(block), prev)
            plaintext += p
            prev = block
        return self._unpad(plaintext)