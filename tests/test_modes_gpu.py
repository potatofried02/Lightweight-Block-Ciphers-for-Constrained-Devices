from modes.cbc import CBC
from ciphers.skinny_gpu import SkinnyGPU
from ciphers.speck_gpu import SpeckGPU
from modes.ctr import CTR
from ciphers.skinny_cpu import SkinnyCPU
from ciphers.speck_cpu import SpeckCPU

# ---------------- CBC mode tests ----------------

# NIST key (16 bytes) and NIST IV (16 bytes) truncated to 8 bytes for SPECK-64 
# and SKINNY-64 roundtrip validation (encrypt -> decrypt -> compare results to input)
# https://nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication800-38a.pdf
ROUNDTRIP_VECTOR = {
    'key':        bytes.fromhex("2b7e151628aed2a6abf7158809cf4f3c"),
    'iv':         bytes.fromhex("000102030405060708090a0b0c0d0e0f")[:8],
    'plaintext':  b"We are group 19 and this is the ECE 268 final project."
}

class Test_CBC_Skinny64_128_Roundtrip_GPU:
    '''
    Validates end-to-end CBC + SKINNY-64/128 round-trip with NIST key/IV
    and a plain English plaintext.
    '''
    def test_roundtrip(self):
        '''
        Encrypt -> decrypt to verify that CBC with SKINNY-64 recovers the given plaintext.
        '''
        cbc = CBC(SkinnyGPU(ROUNDTRIP_VECTOR["key"]), ROUNDTRIP_VECTOR["iv"])
        ct = cbc.encrypt(ROUNDTRIP_VECTOR["plaintext"])
        recovered = cbc.decrypt(ct)

        assert recovered == ROUNDTRIP_VECTOR["plaintext"], (
            f"Round-trip mismatch.\n"
            f"  plaintext:  {ROUNDTRIP_VECTOR["plaintext"].hex()}\n"
            f"  ciphertext: {ct.hex()}\n"
            f"  recovered:  {recovered.hex()}"
        )

class Test_CBC_Speck64_128_Roundtrip_GPU:
    '''
    Validates end-to-end CBC + Speck-64/128 round-trip with NIST key/IV
    and a plain English plaintext.
    '''
    def test_roundtrip(self):
        '''
        Encrypt -> decrypt to verify that CBC with SPECK-64 recovers the given plaintext.
        '''
        cbc = CBC(SpeckGPU(ROUNDTRIP_VECTOR["key"]), ROUNDTRIP_VECTOR["iv"])
        ct = cbc.encrypt(ROUNDTRIP_VECTOR["plaintext"])
        recovered = cbc.decrypt(ct)

        assert recovered == ROUNDTRIP_VECTOR["plaintext"], (
            f"Round-trip mismatch.\n"
            f"  plaintext:  {ROUNDTRIP_VECTOR["plaintext"].hex()}\n"
            f"  ciphertext: {ct.hex()}\n"
            f"  recovered:  {recovered.hex()}"
        )

# ---------------- CTR mode tests ---------------- 

class Test_CTR_Skinny64_128_Roundtrip_GPU:
    '''
    Validates end-to-end CTR + SKINNY-64/128 round-trip with NIST key/counter
    and a plain English plaintext.
    '''
    def test_roundtrip(self):
        '''
        Encrypt -> decrypt to verify that CTR with SKINNY-64/128 recovers the given plaintext.
        '''
        ctr = CTR(SkinnyGPU(ROUNDTRIP_VECTOR["key"]), ROUNDTRIP_VECTOR["iv"])
        ct = ctr.encrypt(ROUNDTRIP_VECTOR["plaintext"])
        recovered = ctr.decrypt(ct)

        assert recovered == ROUNDTRIP_VECTOR["plaintext"], (
            f"Round-trip mismatch.\n"
            f"  plaintext:  {ROUNDTRIP_VECTOR["plaintext"].hex()}\n"
            f"  ciphertext: {ct.hex()}\n"
            f"  recovered:  {recovered.hex()}"
        )

class Test_CTR_Speck64_128_Roundtrip_GPU:
    '''
    Validates end-to-end CTR + Speck-64/128 round-trip with NIST key/counter
    and a plain English plaintext.
    '''
    def test_roundtrip(self):
        '''
        Encrypt -> decrypt to verify that CTR with SPECK-64/128 recovers the given plaintext.
        '''
        ctr = CTR(SpeckGPU(ROUNDTRIP_VECTOR["key"]), ROUNDTRIP_VECTOR["iv"])
        ct = ctr.encrypt(ROUNDTRIP_VECTOR["plaintext"])
        recovered = ctr.decrypt(ct)

        assert recovered == ROUNDTRIP_VECTOR["plaintext"], (
            f"Round-trip mismatch.\n"
            f"  plaintext:  {ROUNDTRIP_VECTOR["plaintext"].hex()}\n"
            f"  ciphertext: {ct.hex()}\n"
            f"  recovered:  {recovered.hex()}"
        )