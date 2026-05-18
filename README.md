# ECE 268 Final Project
- Implementation of SPECK-64 and SKINNY-64, including:
  - Encryption, decryption, and key schedule from the specification.
- Each cipher is wrapped using CTR and CBC modes and validated against test vectors.
- Quantitative comparison vs. AES baseline:
  - Code size, gate count, cycles/byte, key-schedule cost

# Environment Setup
## Prerequisites

- **NVIDIA GPU** with CUDA 12.x compatible driver (verify with `nvidia-smi`)
- **Python 3.14** 
- **Git**

> **Note:** CuPy does not support Apple Silicon or Intel Mac GPUs. On macOS, you can develop and run the CPU portions of the project (SPECK/SKINNY/AES reference implementations, test vector validation), but the GPU benchmarks must run on a machine with an NVIDIA GPU.

---

## Windows (PowerShell)

### 1. Install Python

Download Python 3.14 from [python.org](https://www.python.org/downloads/windows/). During installation, check **"Add Python to PATH"**.

Verify the install:

```powershell
python --version
```

### 2. Clone the repo and create a virtual environment

```powershell
git clone https://github.com/BJKin/Lightweight-Block-Ciphers-for-Constrained-Devices.git
cd Lightweight-Block-Ciphers-for-Constrained-Devices
python -m venv .venv
```

If you have multiple Python versions installed, use the `py` launcher to pick one:

```powershell
py -3.11 -m venv .venv
```

### 3. Activate the virtual environment

```powershell
.venv\Scripts\Activate.ps1
```

When activation succeeds, your prompt will show `(.venv)` at the start.

### 4. Install dependencies

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Verify GPU access

```powershell
python -c "import cupy; print('GPUs detected:', cupy.cuda.runtime.getDeviceCount())"
```

Expected output: `GPUs detected: 1` (or higher).

### Deactivating

```powershell
deactivate
```

---

## macOS (zsh / bash)

### 1. Install Python

Using [Homebrew](https://brew.sh/):

```bash
brew install python@3.14
```

Verify the install:

```bash
python3.14 --version
```

### 2. Clone the repo and create a virtual environment

```bash
git clone https://github.com/BJKin/Lightweight-Block-Ciphers-for-Constrained-Devices.git
cd Lightweight-Block-Ciphers-for-Constrained-Devices
python3.14 -m venv .venv
```

### 3. Activate the virtual environment

```bash
source .venv/bin/activate
```

When activation succeeds, your prompt will show `(.venv)` at the start.

### 4. Install dependencies

**Important:** macOS cannot run the GPU code. Install only the CPU-side packages:

```bash
pip install --upgrade pip
pip install pycryptodome>=3.20 numpy>=1.26 pytest>=8.0
```

Trying to install `cupy-cuda12x` on macOS will fail — there are no wheels and no CUDA support on Apple hardware.

### 5. Verify the CPU install

```bash
python -c "from Crypto.Cipher import AES; print('AES baseline ready')"
```

### Deactivating

```bash
deactivate
```

---
