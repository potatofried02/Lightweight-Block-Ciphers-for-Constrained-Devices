from ciphers.speck_cpu import SpeckCPU 

SPECK_64_128_VECTOR = {
    'key':        bytes.fromhex("1b1a1918131211100b0a090803020100"),
    'plaintext':  bytes.fromhex("3b7265747475432d"),
    'ciphertext': bytes.fromhex("8c6fa548454e028b"),
}

class Test_Speck64_128:
    '''
    Validates encrypt and decrypt against SPECK-64/128 test vector from original publication.
    '''
    def test_encrypt(self):
        '''
        Validates encryption.
        '''
        cipher = SpeckCPU(SPECK_64_128_VECTOR['key'])
        ct = cipher.encrypt(SPECK_64_128_VECTOR['plaintext'])
        assert ct == SPECK_64_128_VECTOR['ciphertext'], (
            f"Encrypt mismatch.\n"
            f"  expected: {SPECK_64_128_VECTOR['ciphertext'].hex()}\n"
            f"  got:      {ct.hex()}"
        )

    def test_decrypt(self):
        '''
        Validates decryption.
        '''
        cipher = SpeckCPU(SPECK_64_128_VECTOR['key'])
        pt = cipher.decrypt(SPECK_64_128_VECTOR['ciphertext'])
        assert pt == SPECK_64_128_VECTOR['plaintext'], (
            f"Decrypt mismatch.\n"
            f"  expected: {SPECK_64_128_VECTOR['plaintext'].hex()}\n"
            f"  got:      {pt.hex()}"
        )


SPECK_64_96_VECTOR = {
    'key':        bytes.fromhex("131211100b0a090803020100"),
    'plaintext':  bytes.fromhex("74614620736e6165"),
    'ciphertext': bytes.fromhex("9f7952ec4175946c"),
}

class Test_Speck64_96:
    '''
    Validates encrypt and decrypt against SPECK-64/96 test vector from original publication.
    '''
    def test_encrypt(self):
        '''
        Validates encryption.
        '''
        cipher = SpeckCPU(SPECK_64_96_VECTOR['key'])
        ct = cipher.encrypt(SPECK_64_96_VECTOR['plaintext'])
        assert ct == SPECK_64_96_VECTOR['ciphertext'], (
            f"Encrypt mismatch.\n"
            f"  expected: {SPECK_64_96_VECTOR['ciphertext'].hex()}\n"
            f"  got:      {ct.hex()}"
        )

    def test_decrypt(self):
        '''
        Validates decryption.
        '''
        cipher = SpeckCPU(SPECK_64_96_VECTOR['key'])
        pt = cipher.decrypt(SPECK_64_96_VECTOR['ciphertext'])
        assert pt == SPECK_64_96_VECTOR['plaintext'], (
            f"Decrypt mismatch.\n"
            f"  expected: {SPECK_64_96_VECTOR['plaintext'].hex()}\n"
            f"  got:      {pt.hex()}"
        )