from ciphers.aes_cpu import AesCPU
from modes.cbc import CBC
from ciphers.skinny_cpu import SkinnyCPU
from ciphers.speck_cpu import SpeckCPU 
from modes.ctr import CTR

# ---------------- CBC mode tests ----------------

# CBC-AES128 test vectors from NIST SP 800-38A Appendix F.2.1
# https://nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication800-38a.pdf
NIST_CBC_AES128_VECTOR = {
    'key':        bytes.fromhex("2b7e151628aed2a6abf7158809cf4f3c"),
    'iv':         bytes.fromhex("000102030405060708090a0b0c0d0e0f"),
    'plaintext':  bytes.fromhex(
        "6bc1bee22e409f96e93d7e117393172a"
        "ae2d8a571e03ac9c9eb76fac45af8e51"
        "30c81c46a35ce411e5fbc1191a0a52ef"
        "f69f2445df4f9b17ad2b417be66c3710"
    ),
    'ciphertext': bytes.fromhex(
        "7649abac8119b246cee98e9b12e9197d"
        "5086cb9b507219ee95db113a917678b2"
        "73bed6b8e3c1743b7116e69e22229516"
        "3ff1caa1681fac09120eca307586e1a7"
    ),
}

class Test_CBC_NIST_AES128:
    '''
    Validates CBC against NIST CBC-AES128 test vectors. The NIST plaintext is 
    a multiple of the block size and uses no padding in its CBC implementation, 
    but the CBC class applies PKCS#7, so the encryption produces a ciphertext
    with one extra block (padding). Meaning that post encryption, the padding 
    will be removed from the ciphertext for comparison with the NIST test vectors 
    (CBC class takes care of padding for decryption).
    '''
    def test_encrypt(self):
        '''
        Validates that CBC encryption with AES.MODE_ECB produces ciphertext 
        that matches NIST CBC-AES128 ciphertext.
        '''
        v = NIST_CBC_AES128_VECTOR
        cbc = CBC(AesCPU(v['key']), v['iv'], block_size=16)
        ct = cbc.encrypt(v['plaintext'])

        assert ct[:len(v['ciphertext'])] == v['ciphertext'], (
            f"Encrypt mismatch.\n"
            f"  expected: {v['ciphertext'].hex()}\n"
            f"  got:      {ct[:len(v['ciphertext'])].hex()}"
        )

    def test_decrypt(self):
        '''
        Encrypt -> decrypt to verify that CBC with AES-128 recovers the NIST plaintext.
        '''
        v = NIST_CBC_AES128_VECTOR
        cbc = CBC(AesCPU(v['key']), v['iv'], block_size=16)
        ct = cbc.encrypt(v['plaintext'])
        recovered = cbc.decrypt(ct)

        assert recovered == v['plaintext'], (
            f"Decrypt mismatch.\n"
            f"  expected: {v['plaintext'].hex()}\n"
            f"  got:      {recovered.hex()}"
        )

# NIST key (16 bytes) and NIST IV (16 bytes) truncated to 8 bytes for SPECK-64 
# and SKINNY-64 roundtrip validation (encrypt -> decrypt -> compare results to input)
ROUNDTRIP_VECTOR = {
    'key':        bytes.fromhex("2b7e151628aed2a6abf7158809cf4f3c"),
    'iv':         bytes.fromhex("000102030405060708090a0b0c0d0e0f")[:8],
    'plaintext':  b"We are group 19 and this is the ECE 268 final project."
}

class Test_CBC_Skinny64_128_Roundtrip:
    '''
    Validates end-to-end CBC + SKINNY-64/128 round-trip with NIST key/IV
    and a plain English plaintext.
    '''
    def test_roundtrip(self):
        '''
        Encrypt -> decrypt to verify that CBC with SKINNY-64 recovers the given plaintext.
        '''
        cbc = CBC(SkinnyCPU(ROUNDTRIP_VECTOR["key"]), ROUNDTRIP_VECTOR["iv"])
        ct = cbc.encrypt(ROUNDTRIP_VECTOR["plaintext"])
        recovered = cbc.decrypt(ct)

        assert recovered == ROUNDTRIP_VECTOR["plaintext"], (
            f"Round-trip mismatch.\n"
            f"  plaintext:  {ROUNDTRIP_VECTOR["plaintext"].hex()}\n"
            f"  ciphertext: {ct.hex()}\n"
            f"  recovered:  {recovered.hex()}"
        )

class Test_CBC_Speck64_128_Roundtrip:
    '''
    Validates end-to-end CBC + Speck-64/128 round-trip with NIST key/IV
    and a plain English plaintext.
    '''
    def test_roundtrip(self):
        '''
        Encrypt -> decrypt to verify that CBC with SPECK-64 recovers the given plaintext.
        '''
        cbc = CBC(SpeckCPU(ROUNDTRIP_VECTOR["key"]), ROUNDTRIP_VECTOR["iv"])
        ct = cbc.encrypt(ROUNDTRIP_VECTOR["plaintext"])
        recovered = cbc.decrypt(ct)

        assert recovered == ROUNDTRIP_VECTOR["plaintext"], (
            f"Round-trip mismatch.\n"
            f"  plaintext:  {ROUNDTRIP_VECTOR["plaintext"].hex()}\n"
            f"  ciphertext: {ct.hex()}\n"
            f"  recovered:  {recovered.hex()}"
        )

# ---------------- CTR mode tests ---------------- 

# CTR-AES128 test vectors from NIST SP 800-38A Appendix F.5.1
# https://nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication800-38a.pdf
NIST_CTR_AES128_VECTOR = {
    'key':        bytes.fromhex("2b7e151628aed2a6abf7158809cf4f3c"),
    'counter':    bytes.fromhex("f0f1f2f3f4f5f6f7f8f9fafbfcfdfeff"),
    'plaintext':  bytes.fromhex(
        "6bc1bee22e409f96e93d7e117393172a"
        "ae2d8a571e03ac9c9eb76fac45af8e51"
        "30c81c46a35ce411e5fbc1191a0a52ef"
        "f69f2445df4f9b17ad2b417be66c3710"
    ),
    'ciphertext': bytes.fromhex(
        "874d6191b620e3261bef6864990db6ce"
        "9806f66b7970fdff8617187bb9fffdff"
        "5ae4df3edbd5d35e5b4f09020db03eab"
        "1e031dda2fbe03d1792170a0f3009cee"
    ),
}


class Test_CTR_NIST_AES128:
    '''
    Validates CTR against NIST CTR-AES128 test vectors.
    '''
    def test_encrypt(self):
        '''
        Validates that CTR encryption with AES.MODE_ECB produces ciphertext 
        that matches NIST CTR-AES128 ciphertext.
        '''
        v = NIST_CTR_AES128_VECTOR
        ctr = CTR(AesCPU(v['key']), v['counter'], block_size=16)
        ct = ctr.encrypt(v['plaintext'])

        assert ct == v['ciphertext'], (
            f"Encrypt mismatch.\n"
            f"  expected: {v['ciphertext'].hex()}\n"
            f"  got:      {ct.hex()}"
        )

    def test_decrypt(self):
        '''
        Validates that CTR decryption with AES.MODE_ECB produces plaintext 
        that matches NIST CBC-AES128 plaintext.
        '''
        v = NIST_CTR_AES128_VECTOR
        ctr = CTR(AesCPU(v['key']), v['counter'], block_size=16)
        recovered = ctr.decrypt(v['ciphertext'])

        assert recovered == v['plaintext'], (
            f"Decrypt mismatch.\n"
            f"  expected: {v['plaintext'].hex()}\n"
            f"  got:      {recovered.hex()}"
        )

class Test_CTR_Skinny64_128_Roundtrip:
    '''
    Validates end-to-end CTR + SKINNY-64/128 round-trip with NIST key/counter
    and a plain English plaintext.
    '''
    def test_roundtrip(self):
        '''
        Encrypt -> decrypt to verify that CTR with SKINNY-64/128 recovers the given plaintext.
        '''
        ctr = CTR(SkinnyCPU(ROUNDTRIP_VECTOR["key"]), ROUNDTRIP_VECTOR["iv"])
        ct = ctr.encrypt(ROUNDTRIP_VECTOR["plaintext"])
        recovered = ctr.decrypt(ct)

        assert recovered == ROUNDTRIP_VECTOR["plaintext"], (
            f"Round-trip mismatch.\n"
            f"  plaintext:  {ROUNDTRIP_VECTOR["plaintext"].hex()}\n"
            f"  ciphertext: {ct.hex()}\n"
            f"  recovered:  {recovered.hex()}"
        )

class Test_CTR_Speck64_128_Roundtrip:
    '''
    Validates end-to-end CTR + Speck-64/128 round-trip with NIST key/counter
    and a plain English plaintext.
    '''
    def test_roundtrip(self):
        '''
        Encrypt -> decrypt to verify that CTR with SPECK-64/128 recovers the given plaintext.
        '''
        ctr = CTR(SpeckCPU(ROUNDTRIP_VECTOR["key"]), ROUNDTRIP_VECTOR["iv"])
        ct = ctr.encrypt(ROUNDTRIP_VECTOR["plaintext"])
        recovered = ctr.decrypt(ct)

        assert recovered == ROUNDTRIP_VECTOR["plaintext"], (
            f"Round-trip mismatch.\n"
            f"  plaintext:  {ROUNDTRIP_VECTOR["plaintext"].hex()}\n"
            f"  ciphertext: {ct.hex()}\n"
            f"  recovered:  {recovered.hex()}"
        )