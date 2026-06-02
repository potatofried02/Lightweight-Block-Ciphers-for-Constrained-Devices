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

## Windows (PowerShell)

### 1. Clone the repo and create a virtual environment

```powershell
git clone https://github.com/potatofried02/Lightweight-Block-Ciphers-for-Constrained-Devices.git
cd Lightweight-Block-Ciphers-for-Constrained-Devices
py -3.14 -m venv .venv
```

### 3. Activate the virtual environment

```powershell
.venv\Scripts\Activate.ps1
```

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


---

## macOS (zsh / bash)

### 1. Clone the repo and create a virtual environment

```bash
git clone https://github.com/potatofried02/Lightweight-Block-Ciphers-for-Constrained-Devices.git
cd Lightweight-Block-Ciphers-for-Constrained-Devices
python3.14 -m venv .venv
```

### 2. Activate the virtual environment

```bash
source .venv/bin/activate
```

### 3. Install dependencies

**Important:** macOS cannot run the GPU code. Install only the CPU-side packages:

```bash
pip install --upgrade pip
pip install pycryptodome>=3.20 numpy>=1.26 pytest>=8.0
```
## Linux (terminal) (Do not do on conda env! Do on base!, other wise conda env collides with venv!)

### 
```
python -m venv venv
source venv/bin/activate   
pip install -r requirements.txt
pytest
```
### separate tests.
```
pytest -v tests/test_speck_gpu.py
```
