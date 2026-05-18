# ECE 268 Final Project
- Implementation of SPECK-64 and SKINNY-64, including:
  - Encryption, decryption, and key schedule from the specification.
- Each cipher is wrapped using CTR and CBC modes and validated against test vectors.
- Quantitative comparison vs. AES baseline:
  - Code size, gate count, cycles/byte, key-schedule cost
