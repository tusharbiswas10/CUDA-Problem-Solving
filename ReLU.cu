#include <cuda_runtime.h>

__global__ void relu_kernel(const float* input, float* output, size_t size) {
  
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size) {
        float val = input[i];
       
        output[i] = (val > 0.0f) ? val : 0.0f;
    }
}

extern "C" void solution(const float* input, float* output, size_t n, size_t m) {
    size_t size = n * m; 

    int threadsPerBlock = 256;

    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, size);
}
