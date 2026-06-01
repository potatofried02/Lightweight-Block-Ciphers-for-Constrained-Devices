import cupy as cp
import numpy as np

# benchmarks/bench_latency.py

import csv
import statistics
import sys
import time
from pathlib import Path

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

SPECK_BLOCK = bytes.fromhex("3b7265747475432d")
SKINNY_BLOCK = bytes.fromhex("0001020304050607")
AES_BLOCK = bytes.fromhex("00112233445566778899aabbccddeeff")


def sync_gpu():
    if cp is not None:
        cp.cuda.Stream.null.synchronize()


def measure_latency(fn, device: str, repeats: int = 1000, warmups: int = 20):
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


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    targets = [
        ("SPECK-64/128", "CPU", SpeckCPU, SPECK_KEY, SPECK_BLOCK, 8),
        ("SKINNY-64/128", "CPU", SkinnyCPU, SKINNY_KEY, SKINNY_BLOCK, 8),
        ("AES-128", "CPU", AesCPU, AES_KEY, AES_BLOCK, 16),
    ]

    if SpeckGPU is not None:
        targets.append(("SPECK-64/128", "GPU", SpeckGPU, SPECK_KEY, SPECK_BLOCK, 8))
    if SkinnyGPU is not None:
        targets.append(("SKINNY-64/128", "GPU", SkinnyGPU, SKINNY_KEY, SKINNY_BLOCK, 8))
    if AesGPU is not None:
        targets.append(("AES-128", "GPU", AesGPU, AES_KEY, AES_BLOCK, 16))

    rows = []

    for cipher_name, device, cls, key, block, block_size in targets:
        print(f"Measuring single-block latency: {cipher_name} {device}")

        cipher = cls(key)

        def encrypt_once():
            return cipher.encrypt(block)

        stats_no_key = measure_latency(encrypt_once, device)

        def encrypt_with_key_schedule():
            c = cls(key)
            return c.encrypt(block)

        stats_with_key = measure_latency(encrypt_with_key_schedule, device)

        for include_key_schedule, stats in [
            (False, stats_no_key),
            (True, stats_with_key),
        ]:
            rows.append({
                "cipher": cipher_name,
                "device": device,
                "block_bytes": block_size,
                "include_key_schedule": include_key_schedule,
                "median_s": stats["median_s"],
                "min_s": stats["min_s"],
                "mean_s": stats["mean_s"],
                "median_us": stats["median_s"] * 1e6,
                "median_ns_per_byte": stats["median_s"] * 1e9 / block_size,
            })

    out_csv = RESULT_DIR / "latency.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "cipher",
                "device",
                "block_bytes",
                "include_key_schedule",
                "median_s",
                "min_s",
                "mean_s",
                "median_us",
                "median_ns_per_byte",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {out_csv}")

    try:
        import matplotlib.pyplot as plt

        plot_rows = [r for r in rows if r["include_key_schedule"] is False]
        labels = [f"{r['cipher']}\n{r['device']}" for r in plot_rows]
        values = [r["median_ns_per_byte"] for r in plot_rows]

        plt.figure(figsize=(8, 4))
        plt.bar(labels, values)
        plt.ylabel("Median latency (ns/byte)")
        plt.title("Single-Block Encryption Latency without Key Schedule")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()

        out_png = PLOT_DIR / "latency.png"
        plt.savefig(out_png, dpi=200)
        print(f"Saved {out_png}")
    except Exception as e:
        print(f"Plot skipped: {e}")


if __name__ == "__main__":
    main()
