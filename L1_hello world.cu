#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void hello_cuda() // __global__: CUDA kernel functin prefix, CPU launch hello_cuda function, GPU execute hello_cuda function
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; // blockDim.x: number of thread in block; blockIdx.x: block id; threadIdx.x: thread id in block
    printf("[%d] hello cuda\n", idx);
}

int main()
{
    hello_cuda<<< 1, 1 >>>(); // the first "1" indicates the allocated the num of block, the second "1" means the num of threadldx per block;
    cudaDeviceSynchronize(); // forces the CPU to wait for the execution of CUDA kernel function hello_cuda in GPU;
    return 0;
}