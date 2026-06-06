import csv
import statistics
import sys
import time
from pathlib import Path
import numpy as np
import cupy as cp
sys.path.insert(0, "..")
from ciphers.speck_cpu import SpeckCPU
from ciphers.skinny_cpu import SkinnyCPU
from ciphers.aes_cpu import AesCPU
from ciphers.speck_gpu import SpeckGPU
from ciphers.skinny_gpu import SkinnyGPU
from ciphers.aes_gpu import AesGPU
from modes.ctr import CTR
from modes.cbc import CBC

SEED = 1234

# Keys/IVs from test vectors in ./test
SPECK_KEY = bytes.fromhex("1b1a1918131211100b0a090803020100")
SKINNY_KEY = bytes.fromhex("000102030405060708090a0b0c0d0e0f")
AES_KEY = bytes.fromhex("000102030405060708090a0b0c0d0e0f")
IV8 = bytes.fromhex("0001020304050607")
IV16 = bytes.fromhex("000102030405060708090a0b0c0d0e0f")

# 3 points per octave, 1 KB -> 16 MB
SIZES = [int(x) for x in np.geomspace(1024, 16 * 1024 * 1024, 43)] 

# GPU CBC encryption is crazy slow since it can't be parallelized, so reduced data is collected (16 KB max)
GPU_CBC_MAX = 16 * 1024
GPU_CBC_TRIALS = 2
GPU_CBC_WARMUPS = 1

# key schedule benchmarking configs
KEYSCHED_TRIALS = 100
KEYSCHED_WARMUPS = 10


def measure(fn, device: str, trials: int, warmups: int) -> dict:
    '''
    Measure a given mode/cipher and return its stats.

    Arguments:
        fn -- mode/cipher to be measured
        device -- "CPU" or "GPU"
        trials -- Number of measured iterations
        warmups -- Number of warm-up iterations

    Returns:
        Dict with median/min/mean/stdev/cv of the per-trial seconds
    '''
    for _ in range(warmups):
        fn()
        if device == "GPU":
            cp.cuda.Stream.null.synchronize()

    times = []
    for _ in range(trials):
        start = time.perf_counter()
        fn()
        if device == "GPU":
            cp.cuda.Stream.null.synchronize()
        times.append(time.perf_counter() - start)

    median = statistics.median(times)
    stdev = statistics.pstdev(times)

    return {
        "median_s": median,
        "min_s": min(times),
        "mean_s": statistics.mean(times),
        "stdev_s": stdev,
        "cv": stdev / median if median else 0.0,
    }


def make_data(num_bytes: int, block_size: int) -> bytes:
    '''
    Generate a reproducible random string of bytes (whole number of blocks).

    Arguments:
        num_bytes -- Size in bytes
        block_size -- Cipher block size in bytes

    Returns:
        Random byte string (length is a multiple of block_size)
    '''
    n = (num_bytes // block_size) * block_size
    rng = np.random.default_rng(SEED + n + block_size)

    return rng.integers(0, 256, size=n, dtype=np.uint8).tobytes()


def xor_bytes(a: bytes, b: bytes) -> bytes:
    '''
    XOR two byte strings using NumPy.

    Arguments:
        a -- First byte string
        b -- Second byte string (same length as a)

    Returns:
        Byte string of the element-wise XOR
    '''
    return (np.frombuffer(a, dtype=np.uint8) ^ np.frombuffer(b, dtype=np.uint8)).tobytes()


def ctr_input_blocks(counter: bytes, block_size: int, num_blocks: int) -> bytes:
    '''
    Build the concatenated counter blocks for CTR mode.

    Arguments:
        counter -- Initial counter value
        block_size -- Cipher block size in bytes
        num_blocks -- Number of counter blocks to produce

    Returns:
        Byte string of num_blocks * block_size counter bytes
    '''
    start = int.from_bytes(counter, "big")
    idx = np.arange(num_blocks, dtype=np.uint64)
    mask64 = (1 << 64) - 1

    if block_size == 8:
        return (np.uint64(start & mask64) + idx).astype(">u8").tobytes()

    start_low = start & mask64
    start_high = (start >> 64) & mask64
    low = np.uint64(start_low) + idx
    carry = (idx > np.uint64(mask64 - start_low)).astype(np.uint64)
    high = np.uint64(start_high) + carry
    out = np.empty((num_blocks, 2), dtype=">u8")
    out[:, 0] = high
    out[:, 1] = low

    return out.tobytes()


def gpu_cbc_encrypt(cipher, iv: bytes, block_size: int, plaintext: bytes) -> bytes:
    '''
    Run CBC encryption on the GPU.

    Arguments:
        cipher -- GPU cipher instance
        iv -- Initialization vector
        block_size -- Cipher block size in bytes
        plaintext -- Input (length is a multiple of block_size)

    Returns:
        Ciphertext byte string
    '''
    out = bytearray()
    prev = iv
    for i in range(0, len(plaintext), block_size):
        prev = cipher.encrypt(xor_bytes(plaintext[i:i + block_size], prev))
        out += prev

    return bytes(out)


def cpu_ecb(cipher, data: bytes, block_size: int, op: str) -> bytes:
    '''
    Run ECB encryption on the CPU.

    Arguments:
        cipher -- CPU cipher instance
        data -- Input (length is a multiple of block_size)
        block_size -- Cipher block size in bytes
        op -- "encrypt" or "decrypt"

    Returns:
        Concatenated per-block output byte string
    '''
    fn = cipher.encrypt if op == "encrypt" else cipher.decrypt

    return b"".join(fn(data[i:i + block_size]) for i in range(0, len(data), block_size))


def collect_throughput(targets: list) -> list:
    '''
    Measure throughput and runtime for every cipher/device/mode/op/size.

    Arguments:
        targets -- List of (name, device, cls, key, block_size, iv)

    Returns:
        List of row dicts ready for throughput.csv
    '''
    rows = []
    for name, device, cls, key, bs, iv in targets:
        cipher = cls(key)
        counter = bytes(bs)
        transfer_cache = {}

        for size in SIZES:
            data = make_data(size, bs)
            nbytes = len(data)

            if nbytes <= 64 * 1024:
                trials, warmups = 5, 2
            elif nbytes <= 1024 * 1024:
                trials, warmups = 3, 1
            else:
                trials, warmups = 1, 1

            if device == "GPU":
                if nbytes not in transfer_cache:
                    buf = np.zeros(nbytes, dtype=np.uint8)
                    transfer_cache[nbytes] = measure(lambda b=buf: cp.asnumpy(cp.asarray(b)), "GPU", 5, 2)["median_s"]
                transfer_s = transfer_cache[nbytes]
            else:
                transfer_s = 0.0

            for mode in ["ecb", "ctr", "cbc"]:
                for op in ["encrypt", "decrypt"]:
                    if mode == "ecb" and device == "GPU":
                        meth = cipher.encrypt_blocks if op == "encrypt" else cipher.decrypt_blocks
                        fn = lambda: meth(data)
                    elif mode == "ecb":
                        fn = lambda: cpu_ecb(cipher, data, bs, op)
                    elif mode == "ctr" and device == "GPU":
                        n = len(data) // bs
                        fn = lambda: xor_bytes(data, cipher.encrypt_blocks(ctr_input_blocks(counter, bs, n)))
                    elif mode == "ctr":
                        ctr = CTR(cipher, counter, bs)
                        meth = ctr.encrypt if op == "encrypt" else ctr.decrypt
                        fn = lambda: meth(data)
                    elif device == "GPU" and op == "encrypt":
                        if len(data) > GPU_CBC_MAX:
                            continue
                        fn = lambda: gpu_cbc_encrypt(cipher, iv, bs, data)
                    elif device == "GPU":
                        fn = lambda: xor_bytes(cipher.decrypt_blocks(data), iv + data[:-bs])
                    else:
                        cbc = CBC(cipher, iv, bs)
                        meth = cbc.encrypt if op == "encrypt" else cbc.decrypt
                        fn = lambda: meth(data)
                    t, w = trials, warmups
                    if device == "GPU" and mode == "cbc" and op == "encrypt":
                        t, w = GPU_CBC_TRIALS, GPU_CBC_WARMUPS
                    print(f"  throughput {name:14s} {device} {mode:3s} {op:7s} {nbytes:>9d} B")
                    stats = measure(fn, device, t, w)
                    median = stats["median_s"]
                    compute_est = max(median - transfer_s, 0.0) if device == "GPU" else median
                    rows.append({
                        "cipher": name,
                        "device": device,
                        "mode": mode,
                        "op": op,
                        "bytes": nbytes,
                        "num_blocks": nbytes // bs,
                        "trials": t,
                        "warmups": w,
                        "median_s": median,
                        "min_s": stats["min_s"],
                        "mean_s": stats["mean_s"],
                        "stdev_s": stats["stdev_s"],
                        "cv": stats["cv"],
                        "mb_per_s": nbytes / median / 1e6,
                        "ns_per_byte": median * 1e9 / nbytes,
                        "end_to_end_s": median,
                        "transfer_s": transfer_s,
                        "compute_est_s": compute_est,
                    })

    return rows


def obj_bytes(x) -> int:
    '''
    Estimate the static storage size of a object in bytes.

    Arguments:
        x -- A NumPy/CuPy array, list/tuple, or an int

    Returns:
        Size in bytes
    '''
    if hasattr(x, "nbytes"):
        return int(x.nbytes)
    if isinstance(x, (list, tuple)):
        return sum(obj_bytes(e) for e in x)
    if isinstance(x, int):
        if x < 256:
            return 1
        if x < (1 << 32):
            return 4
        return 8
    
    return 0


def collect_footprint() -> list:
    '''
    Compute the static memory footprint of each cipher.

    Arguments:
        N/A

    Returns:
        List of row dicts (one per cipher)
    '''
    rows = []
    ciphers = [
        ("SPECK-64/128", SpeckCPU, SpeckGPU, SPECK_KEY, 8, IV8),
        ("SKINNY-64/128", SkinnyCPU, SkinnyGPU, SKINNY_KEY, 8, IV8),
        ("AES-128", AesCPU, AesGPU, AES_KEY, 16, IV16),
    ]
    
    for name, cpu_cls, _, key, _, _ in ciphers:
        inst = cpu_cls(key)
        sbox = sum(obj_bytes(getattr(cpu_cls, a)) for a in ("S_BOX", "S_BOX_INV", "INV_SBOX") if hasattr(cpu_cls, a))
        const = obj_bytes(getattr(cpu_cls, "ROUND_CONSTANTS", []))
        perm = sum(obj_bytes(getattr(cpu_cls, a)) for a in ("P_T", "P_SR", "P_SR_INV") if hasattr(cpu_cls, a))
        ks = 0
        for attr in ["_round_keys", "round_subtweakeys", "round_keys", "w"]:
            if hasattr(inst, attr):
                ks = obj_bytes(getattr(inst, attr))
                break
        rows.append({
            "cipher": name,
            "sbox_bytes": sbox,
            "const_bytes": const,
            "perm_bytes": perm,
            "keyschedule_bytes": ks,
            "total_bytes": sbox + const + perm + ks,
        })

    return rows


def count_loc(path: Path) -> int:
    '''
    Count lines of code (excluding blanks, comments, and docstrings).

    Arguments:
        path -- Path to a Python file

    Returns:
        Lines of code
    '''
    loc = 0
    in_doc = False
    delim = ""
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if in_doc:
            if delim in line:
                in_doc = False
            continue
        if not line or line.startswith("#"):
            continue
        if line.startswith("'''") or line.startswith('"""'):
            delim = line[:3]
            if line[3:].count(delim) == 0:
                in_doc = True
            continue
        loc += 1

    return loc


def collect_codesize() -> list:
    '''
    Kernel lines of code per cipher and device.

    Arguments:
        N/A

    Returns:
        List of row dicts
    '''
    code_files = [
        ("SPECK-64/128", "CPU", "../ciphers/speck_cpu.py"),
        ("SPECK-64/128", "GPU", "../ciphers/speck_gpu.py"),
        ("SKINNY-64/128", "CPU", "../ciphers/skinny_cpu.py"),
        ("SKINNY-64/128", "GPU", "../ciphers/skinny_gpu.py"),
        ("AES-128", "CPU", "../ciphers/aes_cpu.py"),
        ("AES-128", "GPU", "../ciphers/aes_gpu.py"),
    ]
    rows = []
    for name, device, rel in code_files:
        rows.append({
            "cipher": name,
            "device": device,
            "file": rel,
            "loc": count_loc(rel),
            "ptx_bytes": "N/A",
        })

    return rows


def collect_keyschedule(targets: list) -> list:
    '''
    Measure setup time for each key and expanded key schedule storage size.

    Arguments:
        targets -- List of (name, device, cls, key, block_size, iv)

    Returns:
        List of row dicts
    '''
    rows = []
    for name, device, cls, key, _, _ in targets:
        stats = measure(lambda: cls(key), device, KEYSCHED_TRIALS, KEYSCHED_WARMUPS)

        inst = cls(key)
        ks_bytes = 0
        for attr in ["_round_keys", "round_subtweakeys", "round_keys", "w"]:
            if hasattr(inst, attr):
                ks_bytes = obj_bytes(getattr(inst, attr))
                break

        if device == "GPU":
            buf = np.zeros(ks_bytes or 1, dtype=np.uint8)
            upload = measure(lambda b=buf: cp.asnumpy(cp.asarray(b)), "GPU", 5, 2)["median_s"] / 2.0
            host = max(stats["median_s"] - upload, 0.0)
        else:
            upload = 0.0
            host = stats["median_s"]

        rows.append({
            "cipher": name,
            "device": device,
            "setup_s_median": stats["median_s"],
            "setup_s_stdev": stats["stdev_s"],
            "host_setup_s": host,
            "device_upload_s": upload,
            "keyschedule_bytes": ks_bytes,
        })
    return rows


def write_csv(path: Path, rows: list) -> None:
    '''
    Write a list of dict rows to a CSV file.

    Arguments:
        path -- CSV path
        rows -- List of dicts with identical keys

    Returns:
        N/A
    '''
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {path}")


def main():
    ciphers = [
        ("SPECK-64/128", SpeckCPU, SpeckGPU, SPECK_KEY, 8, IV8),
        ("SKINNY-64/128", SkinnyCPU, SkinnyGPU, SKINNY_KEY, 8, IV8),
        ("AES-128", AesCPU, AesGPU, AES_KEY, 16, IV16),
    ]
    targets = []
    for name, cpu_cls, gpu_cls, key, bs, iv in ciphers:
        targets.append((name, "CPU", cpu_cls, key, bs, iv))
        targets.append((name, "GPU", gpu_cls, key, bs, iv))

    throughput = collect_throughput(targets)
    footprint = collect_footprint()
    codesize = collect_codesize()
    keyschedule = collect_keyschedule(targets)

    write_csv(Path("benchmark_results") / "throughput.csv", throughput)
    write_csv(Path("benchmark_results") / "footprint.csv", footprint)
    write_csv(Path("benchmark_results") / "codesize.csv", codesize)
    write_csv(Path("benchmark_results") / "keyschedule.csv", keyschedule)

if __name__ == "__main__":
    main()
