#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>

// baseline: use single threads

__global__ void reduce_baseline(const int* input, int* output, size_t n)
{
    int sum = 0;
    for(int i = 0; i < n; i++)
    {
        sum+= input[i];
    }
    *output = sum;
}

bool checkResult(int* out, int groudtruth, int n)
{
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

    const int blockSize = 1;
    int gridSize = 1;

    // allocate CPU and GPU memory
    int *h_in = (int*)malloc(N * sizeof(int));
    int *d_in;
    cudaMalloc((void**)&d_in, N * sizeof(int));

    int *h_out = (int*)malloc(N * sizeof(int));
    int *d_out;
    cudaMalloc((void**)&d_out, N * sizeof(int));

    // initialize data
    for(int i = 0; i < N; i++)
    {
        h_in[i] = 1;
    }

    int groundTruth = N * 1;

    cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice);

    // allocate block and thread
    dim3 grid(gridSize);
    dim3 block(blockSize);

    // record GPU execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    // allocate 1 block, 1 thread
    reduce_baseline<<<grid, block>>>(d_in, d_out, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);
    printf("allocated %d blocks, data counts are %d\n", gridSize, N);

    bool is_right = checkResult(h_out, groundTruth, gridSize);
    if(is_right)
    {
        printf("the ans is right\n");
    }
    else
    {
        printf("the ans is wrong\n");
        for(int i = 0; i < gridSize; i++)
        {
            printf("res per block: %1f\n", h_out[i]);
        }
        printf("groundTruth is %f \n", groundTruth);
    }

    printf("reduce_baseline latency = %f ms\n", milliseconds);

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);

    return 0;

}