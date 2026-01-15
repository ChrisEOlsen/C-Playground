#include <iostream>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <chrono>

#define CHECK_CUDA(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "Error: " << __FILE__ << ":" << __LINE__ << ", " \
                  << cudaGetErrorString(error) << std::endl; \
        exit(1); \
    } \
}

// CUDA Kernel
__global__ void monte_carlo_pi(long long iterations_per_thread, unsigned long long *d_total_hits, long long seed_offset) {
    // Calculate global thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize RNG state per thread
    // Each thread gets a unique seed based on its ID
    curandState state;
    curand_init(seed_offset + tid, 0, 0, &state);

    long long local_hits = 0;

    // Main Loop
    for (long long i = 0; i < iterations_per_thread; i++) {
        // curand_uniform returns float in (0.0, 1.0]
        float x = curand_uniform(&state);
        float y = curand_uniform(&state);

        if (x * x + y * y <= 1.0f) {
            local_hits++;
        }
    }

    // Atomic Add to global memory
    // We only do this ONCE per thread to minimize memory contention
    atomicAdd(d_total_hits, (unsigned long long)local_hits);
}

int main() {
    // Configuration
    const int BLOCKS = 1024 * 4;       // Number of blocks
    const int THREADS_PER_BLOCK = 256; // Standard block size
    const long long ITERATIONS_PER_THREAD = 10000; 
    
    long long total_threads = (long long)BLOCKS * THREADS_PER_BLOCK;
    long long total_iterations = total_threads * ITERATIONS_PER_THREAD;

    std::cout << "CUDA Monte Carlo Pi Approximation" << std::endl;
    std::cout << "---------------------------------" << std::endl;
    std::cout << "Blocks: " << BLOCKS << std::endl;
    std::cout << "Threads per block: " << THREADS_PER_BLOCK << std::endl;
    std::cout << "Total GPU Threads: " << total_threads << std::endl;
    std::cout << "Iterations per thread: " << ITERATIONS_PER_THREAD << std::endl;
    std::cout << "Total Iterations: " << total_iterations << std::endl;

    // Allocate memory on GPU
    unsigned long long *d_total_hits;
    CHECK_CUDA(cudaMalloc(&d_total_hits, sizeof(unsigned long long)));
    CHECK_CUDA(cudaMemset(d_total_hits, 0, sizeof(unsigned long long)));

    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    std::cout << "Launching kernel..." << std::endl;
    
    cudaEventRecord(start);

    // Launch Kernel
    monte_carlo_pi<<<BLOCKS, THREADS_PER_BLOCK>>>(ITERATIONS_PER_THREAD, d_total_hits, time(NULL));
    
    cudaEventRecord(stop);
    
    // Copy result back to Host (CPU)
    unsigned long long h_total_hits = 0;
    CHECK_CUDA(cudaMemcpy(&h_total_hits, d_total_hits, sizeof(unsigned long long), cudaMemcpyDeviceToHost));

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Calculate Pi
    double pi = 4.0 * (double)h_total_hits / (double)total_iterations;

    std::cout << "---------------------------------" << std::endl;
    std::cout << "Estimated Pi: " << pi << std::endl;
    std::cout << "Time Taken: " << milliseconds / 1000.0f << " seconds" << std::endl;
    
    // Performance metrics
    double seconds = milliseconds / 1000.0;
    double ops_per_sec = (double)total_iterations / seconds;
    std::cout << "Calculations per second: " << ops_per_sec / 1e9 << " Billion" << std::endl;

    // Cleanup
    cudaFree(d_total_hits);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

