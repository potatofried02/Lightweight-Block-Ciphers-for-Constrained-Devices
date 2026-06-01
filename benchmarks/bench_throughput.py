import cupy as cp
import numpy as np

# benchmarks/bench_throughput.py

import csv
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ciphers.speck_cpu import SpeckCPU
from ciphers.skinny_cpu import SkinnyCPU
from ciphers.aes_cpu import AesCPU

try:
    import cupy as cp
    from ciphers.speck_gpu import SpeckGPU
    from ciphers.skinny_gpu import SkinnyGPU
except Exception:
    cp = None
    SpeckGPU = None
    SkinnyGPU = None

try:
    from ciphers.aes_gpu import AesGPU
except Exception:
    AesGPU = None


RESULT_DIR = ROOT / "benchmarks" / "results"
PLOT_DIR = ROOT / "benchmarks" / "plots"

SPECK_KEY = bytes.fromhex("1b1a1918131211100b0a090803020100")
SKINNY_KEY = bytes.fromhex("000102030405060708090a0b0c0d0e0f")
AES_KEY = bytes.fromhex("000102030405060708090a0b0c0d0e0f")

MESSAGE_SIZES = [
    1024,
    4 * 1024,
    16 * 1024,
    64 * 1024,
    256 * 1024,
    1024 * 1024,
]


def sync_gpu():
    if cp is not None:
        cp.cuda.Stream.null.synchronize()


def encrypt_blocks_cpu(cipher, data: bytes, block_size: int) -> bytes:
    return b"".join(
        cipher.encrypt(data[i : i + block_size])
        for i in range(0, len(data), block_size)
    )


def encrypt_blocks_auto(cipher, data: bytes, block_size: int, device: str) -> bytes:
    if hasattr(cipher, "encrypt_blocks"):
        return cipher.encrypt_blocks(data)

    return encrypt_blocks_cpu(cipher, data, block_size)


def measure_throughput(fn, device: str, repeats: int = 5, warmups: int = 2):
    for _ in range(warmups):
        fn()
        if device == "GPU":
            sync_gpu()

    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        if device == "GPU":
            sync_gpu()
        end = time.perf_counter()
        times.append(end - start)

    return {
        "median_s": statistics.median(times),
        "min_s": min(times),
        "mean_s": statistics.mean(times),
    }


def make_random_data(num_bytes: int, block_size: int, seed: int) -> bytes:
    if num_bytes % block_size != 0:
        num_bytes = (num_bytes // block_size) * block_size

    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=num_bytes, dtype=np.uint8).tobytes()


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    targets = [
        ("SPECK-64/128", "CPU", SpeckCPU, SPECK_KEY, 8),
        ("SKINNY-64/128", "CPU", SkinnyCPU, SKINNY_KEY, 8),
        ("AES-128", "CPU", AesCPU, AES_KEY, 16),
    ]

    if SpeckGPU is not None:
        targets.append(("SPECK-64/128", "GPU", SpeckGPU, SPECK_KEY, 8))
    if SkinnyGPU is not None:
        targets.append(("SKINNY-64/128", "GPU", SkinnyGPU, SKINNY_KEY, 8))
    if AesGPU is not None:
        targets.append(("AES-128", "GPU", AesGPU, AES_KEY, 16))

    rows = []

    for cipher_name, device, cls, key, block_size in targets:
        print(f"Preparing cipher: {cipher_name} {device}")
        cipher = cls(key)

        for size in MESSAGE_SIZES:
            data = make_random_data(size, block_size, seed=size + block_size)

            print(f"Measuring throughput: {cipher_name} {device}, {len(data)} bytes")

            def encrypt_message():
                return encrypt_blocks_auto(cipher, data, block_size, device)

            stats = measure_throughput(encrypt_message, device)

            median_s = stats["median_s"]
            mb_per_s = len(data) / median_s / 1e6
            ns_per_byte = median_s * 1e9 / len(data)

            rows.append({
                "cipher": cipher_name,
                "device": device,
                "bytes": len(data),
                "block_size": block_size,
                "num_blocks": len(data) // block_size,
                "median_s": median_s,
                "min_s": stats["min_s"],
                "mean_s": stats["mean_s"],
                "mb_per_s": mb_per_s,
                "ns_per_byte": ns_per_byte,
            })

    out_csv = RESULT_DIR / "throughput.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "cipher",
                "device",
                "bytes",
                "block_size",
                "num_blocks",
                "median_s",
                "min_s",
                "mean_s",
                "mb_per_s",
                "ns_per_byte",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {out_csv}")

    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 5))

        labels = sorted(set((r["cipher"], r["device"]) for r in rows))
        for cipher_name, device in labels:
            sub = [
                r for r in rows
                if r["cipher"] == cipher_name and r["device"] == device
            ]
            sub = sorted(sub, key=lambda r: r["bytes"])
            x = [r["bytes"] for r in sub]
            y = [r["mb_per_s"] for r in sub]
            plt.plot(x, y, marker="o", label=f"{cipher_name} {device}")

        plt.xscale("log", base=2)
        plt.xlabel("Message size (bytes)")
        plt.ylabel("Throughput (MB/s)")
        plt.title("Encryption Throughput vs. Message Size")
        plt.legend(fontsize=8)
        plt.tight_layout()

        out_png = PLOT_DIR / "throughput.png"
        plt.savefig(out_png, dpi=200)
        print(f"Saved {out_png}")
    except Exception as e:
        print(f"Plot skipped: {e}")


if __name__ == "__main__":
    main()
