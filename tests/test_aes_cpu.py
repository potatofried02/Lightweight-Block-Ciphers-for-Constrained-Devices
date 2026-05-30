from ciphers.aes_cpu import AesCPU

# AES-128 test vector from FIPS 197 Appendix B 
# https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.197-upd1.pdf
AES_128_VECTOR = {
    'key':        bytes.fromhex("2b7e151628aed2a6abf7158809cf4f3c"),
    'plaintext':  bytes.fromhex("3243f6a8885a308d313198a2e0370734"),
    'ciphertext': bytes.fromhex("3925841d02dc09fbdc118597196a0b32"),
}

class Test_Aes128:
    '''
    Validates encrypt and decrypt against AES-128 test vector from FIPS 197 Appendix B.
    '''
    def test_encrypt(self):
        '''
        Validates encryption.
        '''
        cipher = AesCPU(AES_128_VECTOR['key'])
        ct = cipher.encrypt(AES_128_VECTOR['plaintext'])
        assert ct == AES_128_VECTOR['ciphertext'], (
            f"Encrypt mismatch.\n"
            f"  expected: {AES_128_VECTOR['ciphertext'].hex()}\n"
            f"  got:      {ct.hex()}"
        )

    def test_decrypt(self):
        '''
        Validates decryption.
        '''
        cipher = AesCPU(AES_128_VECTOR['key'])
        pt = cipher.decrypt(AES_128_VECTOR['ciphertext'])
        assert pt == AES_128_VECTOR['plaintext'], (
            f"Decrypt mismatch.\n"
            f"  expected: {AES_128_VECTOR['plaintext'].hex()}\n"
            f"  got:      {pt.hex()}"
        )