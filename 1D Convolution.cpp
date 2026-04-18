#include <cuda_runtime.h>

__global__ void convolve1d_kernel(const float* A, const float* B, float* C, int N, int K) {
    // 1. Determine which index this thread is responsible for
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // 2. Boundary check: ensure we don't process beyond the array size N
    if (i < N) {
        float sum = 0.0f;
        int r = (K - 1) / 2; // The radius/offset

        // 3. Slide the kernel B over the window in A
        for (int j = 0; j < K; ++j) {
            int signal_idx = i + j - r;

            // 4. Zero Padding check: only multiply/add if within signal bounds
            if (signal_idx >= 0 && signal_idx < N) {
                sum += A[signal_idx] * B[j];
            }
        }
        
        // 5. Write the final result to device memory
        C[i] = sum;
    }
}

extern "C" void solution(const float* A, const float* B, float* C, size_t N, size_t K) {
    if (N == 0) return;

    // Define execution configuration
    // 256 is a standard, efficient thread block size for most GPUs
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel
    convolve1d_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, (int)N, (int)K);

    // Wait for the GPU to finish (optional depending on your framework's needs)
    cudaDeviceSynchronize();
}
