import cupy as cp

class SpeckGPU:
    '''
    GPU based implementation of SPECK-64/128, a lightweight block cipher, 
    that encrypts a 64-bit block with a 128-bit key
    '''
    def __int__(self, key: bytes):
        '''
        Initialize the cipher with a 128-bit key. The key schedule runs once
        and subsequent encrypt/decrypt calls reuse the precomputed round keys.

        Arguments:
            key -- 16 byte (128 bit) key.
        '''
        pass

    def encrypt(self, plaintext: bytes) -> bytes:
        ''' 
        Encrypt a single 64-bit block.

        Arguments:
            plaintext -- 8 bytes (64 bits) of plaintext.

        Returns:
            8 bytes of encrypted plaintext.
        '''
        pass
    
    def decrypt(self, plaintext: bytes) -> bytes:
        ''' 
        Decrypt a single 64-bit block.

        Arguments:
            ciphertext -- 8 bytes (64 bits) of ciphertext.

        Returns:
            8 bytes of recovered plaintext.
        '''
        pass