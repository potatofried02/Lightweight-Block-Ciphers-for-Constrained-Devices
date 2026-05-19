from ciphers.skinny_cpu import SkinnyCPU

SKINNY_64_64_VECTOR = {
    'key':        bytes.fromhex("f5269826fc681238"),
    'plaintext':  bytes.fromhex("06034f957724d19d"),
    'ciphertext': bytes.fromhex("bb39dfb2429b8ac7"),
}

class Test_Skinny64_64:
    '''
    Validates encrypt and decrypt against SKINNY-64/64 test vector from original publication.
    '''
    def test_encrypt(self):
        '''
        Validates encryption.
        '''
        cipher = SkinnyCPU(SKINNY_64_64_VECTOR['key'])
        ct = cipher.encrypt(SKINNY_64_64_VECTOR['plaintext'])
        assert ct == SKINNY_64_64_VECTOR['ciphertext'], (
            f"Encrypt mismatch.\n"
            f"  expected: {SKINNY_64_64_VECTOR['ciphertext'].hex()}\n"
            f"  got:      {ct.hex()}"
        )

    def test_decrypt(self):
        '''
        Validates decryption.
        '''
        cipher = SkinnyCPU(SKINNY_64_64_VECTOR['key'])
        pt = cipher.decrypt(SKINNY_64_64_VECTOR['ciphertext'])
        assert pt == SKINNY_64_64_VECTOR['plaintext'], (
            f"Decrypt mismatch.\n"
            f"  expected: {SKINNY_64_64_VECTOR['plaintext'].hex()}\n"
            f"  got:      {pt.hex()}"
        )


SKINNY_64_128_VECTOR = {
    'key':        bytes.fromhex("9eb93640d088da6376a39d1c8bea71e1"),
    'plaintext':  bytes.fromhex("cf16cfe8fd0f98aa"),
    'ciphertext': bytes.fromhex("6ceda1f43de92b9e"),
}

class Test_Skinny64_128:
    '''
    Validates encrypt and decrypt against SKINNY-64/128 test vector from original publication.
    '''
    def test_encrypt(self):
        '''
        Validates encryption.
        '''
        cipher = SkinnyCPU(SKINNY_64_128_VECTOR['key'])
        ct = cipher.encrypt(SKINNY_64_128_VECTOR['plaintext'])
        assert ct == SKINNY_64_128_VECTOR['ciphertext'], (
            f"Encrypt mismatch.\n"
            f"  expected: {SKINNY_64_128_VECTOR['ciphertext'].hex()}\n"
            f"  got:      {ct.hex()}"
        )

    def test_decrypt(self):
        '''
        Validates decryption.
        '''
        cipher = SkinnyCPU(SKINNY_64_128_VECTOR['key'])
        pt = cipher.decrypt(SKINNY_64_128_VECTOR['ciphertext'])
        assert pt == SKINNY_64_128_VECTOR['plaintext'], (
            f"Decrypt mismatch.\n"
            f"  expected: {SKINNY_64_128_VECTOR['plaintext'].hex()}\n"
            f"  got:      {pt.hex()}"
        )


SKINNY_64_192_VECTOR = {
    'key':        bytes.fromhex("ed00c85b120d68618753e24bfd908f60b2dbb41b422dfcd0"),
    'plaintext':  bytes.fromhex("530c61d35e8663c3"),
    'ciphertext': bytes.fromhex("dd2cf1a8f330303c"),
}

class Test_Skinny64_192:
    '''
    Validates encrypt and decrypt against SKINNY-64/192 test vector from original publication.
    '''
    def test_encrypt(self):
        '''
        Validates encryption.
        '''
        cipher = SkinnyCPU(SKINNY_64_192_VECTOR['key'])
        ct = cipher.encrypt(SKINNY_64_192_VECTOR['plaintext'])
        assert ct == SKINNY_64_192_VECTOR['ciphertext'], (
            f"Encrypt mismatch.\n"
            f"  expected: {SKINNY_64_192_VECTOR['ciphertext'].hex()}\n"
            f"  got:      {ct.hex()}"
        )

    def test_decrypt(self):
        '''
        Validates decryption.
        '''
        cipher = SkinnyCPU(SKINNY_64_192_VECTOR['key'])
        pt = cipher.decrypt(SKINNY_64_192_VECTOR['ciphertext'])
        assert pt == SKINNY_64_192_VECTOR['plaintext'], (
            f"Decrypt mismatch.\n"
            f"  expected: {SKINNY_64_192_VECTOR['plaintext'].hex()}\n"
            f"  got:      {pt.hex()}"
        )