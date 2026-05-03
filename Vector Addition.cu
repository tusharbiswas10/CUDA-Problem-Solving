#include <cuda_runtime.h>

// CUDA Kernel: Each thread calculates one element of the sum
__global__ void vectorAddKernel(const float* a, const float* b, float* c, size_t n) {
    // Calculate the unique index for this thread
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check to prevent illegal memory access
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

extern "C" void solution(const float* d_input1, const float* d_input2, float* d_output, size_t n) {
    // 1. Define the number of threads per block 
    int threadsPerBlock = 256;

    // Calculate the number of blocks needed to cover 'n' elements
    // We use (n + threadsPerBlock - 1) / threadsPerBlock to handle cases where n 
    // is not a perfect multiple of the block size.
    size_t blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel
    // Since n can be as large as 2^30 (~1 billion), we ensure blocksPerGrid 
    // is calculated using size_t.
    vectorAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input1, d_input2, d_output, n);
    
    // Optional: Synchronize to catch launch errors
    cudaDeviceSynchronize();
}
