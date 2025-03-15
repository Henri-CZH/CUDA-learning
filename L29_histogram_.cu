#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>

// use shared memory
__global__ void histogram(const int* d_histogram_data, int* d_bin_data, int n)
{   
    int gtid = threadIdx.x + blockIdx.x * blockDim.x; // global thread idx
    int tid = threadIdx.x;
    __shared__ int cache[256]; // shared memory
    cache[tid] = 0; // initialize shared memory for all threads
    __syncthreads();

    // one thread load multiple data
    for(int i = gtid; i < n; i+= gridDim.x * blockDim.x)
    {
        int val = d_histogram_data[i];
        atomicAdd(&cache[val], 1); // here is bank conflict
    }
    __syncthreads(); // synchronize all threads, cache store histogram of each block

    atomicAdd(&d_bin_data[tid], cache[tid]); // forces threads to serially execute, and not make sure the execution sequence
}

bool checkResult(int* out, int groudtruth, int n)
{   
    float res = 0
    for(int i = 0; i < n; i++)
        res += out[i];

    if(*out != groudtruth)
        return false;
    
    return true;
}

int main()
{
    float milliseconds = 0;
    const int N = 25600000;
    cudaSetDevice(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    const int blockSize = 256; // number of threads in a block
    int gridSize = std::min((N + blockSize - 1) / blockSize, deviceProp.maxGridSize[0]); // N = 255->gridSize = 1; N = 257->gridSize = 2

    // allocate CPU and GPU memory
    int *h_histogram_data = (int*)malloc(N * sizeof(int));
    int *d_histogram_data;
    cudaMalloc((void**)&d_histogram_data, N * sizeof(int));

    int *h_bin_data = (int*)malloc(256 * sizeof(int));
    int *d_bin_data;
    cudaMalloc((void**)&d_bin_data, 256 * sizeof(int));


    // initialize data
    for(int i = 0; i < N; i++)
    {
        h_histogram_data[i] = i % 256;
    }

    int *groundTruth = (int*)malloc(256 * sizeof(int));
    for(int i = 0; i < 256; i++)
    {
        groundTruth[i] = 100000;
    }

    cudaMemcpy(d_histogram_data, h_histogram_data, N * sizeof(int), cudaMemcpyHostToDevice);

    // allocate block and thread size
    dim3 grid(gridSize);
    dim3 block(blockSize); // thrad size

    // record GPU execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    // allocate 1 block, 1 thread
    histogram<<<grid, block>>>(d_histogram_data, d_bin_data, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    cudaMemcpy(h_bin_data, d_bin_data, 256 * sizeof(int), cudaMemcpyDeviceToHost);
    printf("allocated %d blocks, data counts are %d\n", gridSize, N);

    bool is_right = checkResult(h_out, groundTruth, 256);
    if(is_right)
    {
        printf("the ans is right\n");
    }
    else
    {
        printf("the ans is wrong\n");
        for(int i = 0; i < gridSize; i++)
        {
            printf("res per block: %1f\n", d_bin_data[i]);
        }
        printf("groundTruth is %f \n", groundTruth);
    }

    printf("histogram latency = %f ms\n", milliseconds);

    cudaFree(d_histogram_data);
    cudaFree(d_bin_data);
    free(h_histogram_data);
    free(h_bin_data);

    return 0;

}