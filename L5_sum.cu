#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>

// cuda kernel function
__global__ void sum(float *x)
{
    unsigned int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int block_id = blockIdx.x;
    unsigned int local_tid = threadIdx.x;

    printf("current block=%d, thread %d in current block, global thread id=%d\n", block_id, local_tid, global_tid);
    x[global_tid] += 1;
}

int main()
{
    int N = 32;
    int nbytes = N * sizeof(float);
    float *dx, *hx; // dx: device, hx: host;

    cudaMalloc((void **)&dx, nbytes); // allocate GPU mem
    hx = (float*)malloc(nbytes); //allocate  CPU mem

    // initialize host data
    for(int i = 0; i < N; i++)
    {
        hx[i] = i;
        printf("%g\n", hx[i]);
    }

    // copy data to GPU
    cudaMemcpy(dx, hx, nbytes, cudaMemcpyHostToDevice);

    // launch GPU kernel
    sum<<<1, N>>>(dx);

    // copy data to CPU, i.e. synchronize COU and GPU
    cudaMemcpy(hx, dx, nbytes, cudaMemcpyDeviceToHost);
    for(int i = 0; i < N; i++)
    {
        printf("%g\n", hx[i]);
    }

    cudaFree(dx);
    free(hx);

    return 0;
}