import cupy as cp
import numpy as np

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


def sync_gpu():
    if cp is not None:
        cp.cuda.Stream.null.synchronize()


def measure_constructor(cls, key: bytes, device: str, repeats: int = 200, warmups: int = 10):
    for _ in range(warmups):
        cls(key)
        if device == "GPU":
            sync_gpu()

    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        cls(key)
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
        ("SPECK-64/128", "CPU", SpeckCPU, SPECK_KEY),
        ("SKINNY-64/128", "CPU", SkinnyCPU, SKINNY_KEY),
        ("AES-128", "CPU", AesCPU, AES_KEY),
    ]

    if SpeckGPU is not None:
        targets.append(("SPECK-64/128", "GPU", SpeckGPU, SPECK_KEY))
    if SkinnyGPU is not None:
        targets.append(("SKINNY-64/128", "GPU", SkinnyGPU, SKINNY_KEY))
    if AesGPU is not None:
        targets.append(("AES-128", "GPU", AesGPU, AES_KEY))

    rows = []
    for cipher_name, device, cls, key in targets:
        print(f"Measuring key schedule: {cipher_name} {device}")
        stats = measure_constructor(cls, key, device)
        rows.append({
            "cipher": cipher_name,
            "device": device,
            "median_s": stats["median_s"],
            "min_s": stats["min_s"],
            "mean_s": stats["mean_s"],
            "median_us": stats["median_s"] * 1e6,
        })

    out_csv = RESULT_DIR / "keyschedule.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["cipher", "device", "median_s", "min_s", "mean_s", "median_us"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {out_csv}")

    try:
        import matplotlib.pyplot as plt

        labels = [f"{r['cipher']}\n{r['device']}" for r in rows]
        values = [r["median_us"] for r in rows]

        plt.figure(figsize=(8, 4))
        plt.bar(labels, values)
        plt.ylabel("Median key schedule time (us)")
        plt.title("Key Schedule Cost")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()

        out_png = PLOT_DIR / "keyschedule.png"
        plt.savefig(out_png, dpi=200)
        print(f"Saved {out_png}")
    except Exception as e:
        print(f"Plot skipped: {e}")


if __name__ == "__main__":
    main()
