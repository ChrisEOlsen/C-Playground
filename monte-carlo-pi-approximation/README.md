# Monte Carlo Pi Approximation Performance Tuning

This project compares **SIMD (CPU)** vs. **SIMT (GPU)** performance for a Monte Carlo Pi simulation. It uses highly optimized implementations to push the limits of modern hardware.

## Results: CPU vs. GPU Benchmarks

| Implementation | Hardware | Iterations | Time (s) | Throughput (Billion/s) | Relative Speed |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **CPU (AVX2)** | 16-Thread (AVX2) | 1,000,000,000 | 0.0316 | **31.5** | 1x (Baseline) |
| **GPU (CUDA)** | NVIDIA RTX 4070 | 10,485,760,000 | 0.0174 | **601.2** | **~19x Faster** |

### Insights
* **CPU (AVX2):** Utilizes 256-bit SIMD registers to process 8 iterations per thread simultaneously. With 16 threads, it achieves a massive 31.5 Billion calculations per second.
* **GPU (CUDA):** Leverages the massive parallelism of the RTX 4070 (Ada Lovelace). With over 1 million concurrent threads, it reaches over 600 Billion calculations per second, outperforming the optimized CPU version by nearly 20 times.

---

## Technical Implementations

### CPU (SIMD/AVX2)
* **File:** `main.cpp`
* **Approach:** 
    - Uses **AVX2 intrinsics** (`immintrin.h`).
    - Implements a parallel **Xorshift32** RNG.
    - Uses a floating-point bit-manipulation trick to generate random numbers in `[0, 1)` without expensive division.
    - Parallelized across all available cores using `std::thread`.

### GPU (SIMT/CUDA)
* **File:** `main.cu`
* **Approach:**
    - Uses **cuRAND** for high-quality, high-speed parallel random number generation.
    - Each CUDA thread handles its own RNG state and local hit count.
    - Final reduction is performed using `atomicAdd` to global memory (once per thread).
    - Compiled with architecture-specific optimizations (`sm_89` for RTX 4070).

---

## How to Build and Run

### Prerequisites
* **Visual Studio 2022** (with C++ Desktop Development workload)
* **CUDA Toolkit 13.1**
* **CMake**

### Build Configuration
The project uses CMake to manage builds. You can choose to build the CPU version, the CUDA version, or both (default).

#### 1. Configure the project
Generate the build files using CMake. You can use the `-DBUILD_CPU` and `-DBUILD_CUDA` flags to control what gets built.

**Option A: Build Everything (Default)**
Requires both C++ compiler and CUDA Toolkit.
```powershell
cmake -B build
```

**Option B: Build CPU Version Only**
Use this if you don't have the CUDA Toolkit installed.
```powershell
cmake -B build -DBUILD_CUDA=OFF
```

**Option C: Build CUDA Version Only**
```powershell
cmake -B build -DBUILD_CPU=OFF
```

#### 2. Build the targets
Compile the configured targets in Release mode.
```powershell
cmake --build build --config Release
```

### Running the Benchmarks
After building, the executables will be located in the `build/Release` (on Windows) or `build/` (on Linux) folder, depending on which ones you enabled.

```powershell
# Run CPU Benchmark (if built)
./build/pi_cpu

# Run GPU Benchmark (if built)
./build/pi_cuda
```
