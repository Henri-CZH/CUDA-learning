#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>

// warp level reduce
#define WarpSize 32;

__device__ int warp_shuffle(int sum)
{
    // __shfl_down_sync: sum[tid+0] += sum[tid+16]
    sum += __shfl_down_sync(0xffffffff, sum, 16); // 0-16, 1-17, 2-18, ...
    sum += __shfl_down_sync(0xffffffff, sum, 8); // 0-8, 1-9, 2-10, ...
    sum += __shfl_down_sync(0xffffffff, sum, 4); // 0-4, 1-5, 2-6, ...
    sum += __shfl_down_sync(0xffffffff, sum, 2); // 0-2, 1-3, 2-4, ...
    sum += __shfl_down_sync(0xffffffff, sum, 1); // 0-1, 1-2, 2-3, ...
    return sum;
}

template<int blockSize> // blockSize as template arg use for static shared memory size apply during compile phase
__global__ void warp_level_reduce(const int* d_in, int* d_out, size_t n)
{   
    int tid = threadIdx.x;
    int gtid = threadIdx.x + blockIdx.x * blockDim.x; // global thread idx, here can use blockSize or blockDim.x 
    unsigned int total_thread_num = gridDim.x * blockDim.x;
    int sum = 0; // each thread own a private register
    for(int i = gtid; i < n; i += total_thread_num)
    {
        if(i > n)
            break;
        sum += d_in[i];
    }

    __shared__ float warp_sum[blockSize / WarpSize]; // declare shared memory, here can't use blockDim.x, because shared memory size apply is in compile phase
    unsigned int laneId = tid % WarpSize; // thread idx in a warp
    unsigned int warpId = tid / WarpSize; // warp id in a block
    sum = warp_shuffle(sum); // sum for each warp of a block
    __syncthreads(); // synchronize all threads in a block

    if(laneId == 0)
    {
        warp_sum[warpId] = sum; // store sum of a warp into shared memory
    }

    sum = (tid < blockSize / WarpSize) ? warp_sum[laneId] : 0; // push each warp sum of a block into sum register of the first warp
    if(warpId == 0)
    {
        sum = warp_shuffle(sum); // sum for all warp of a block
    }
    
    // all (N / blockSize) block's threads have finished
    if(tid == 0)
        d_out[blockIdx.x] = sum; // sum: sum of all threads in a block
}

bool checkResult(int* out, int groudtruth, int n)
{   
    float res = 0;
    for(int i = 0; i < n; i++)
        res += out[i];

    if(*out != groudtruth)
        return false;
    
    return true;
}

int main()
{
    float milliseconds = 0;
    constexpr int N = 25600000;
    cudaSetDevice(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    const int blockSize = 256; // number of threads in a block
    int gridSize = std::min((N + blockSize - 1) / blockSize, deviceProp.maxGridSize[0]); // N = 255->gridSize = 1; N = 257->gridSize = 2

    // allocate CPU and GPU memory
    int *h_in = (int*)malloc(N * sizeof(int));
    int *d_in;
    cudaMalloc((void**)&d_in, N * sizeof(int));

    int *h_out = (int*)malloc(gridSize * sizeof(int));
    int *d_out, *part_out;
    cudaMalloc((void**)&d_out, 1 * sizeof(int));
    cudaMalloc((void**)&part_out, gridSize * sizeof(int));

    // initialize data
    for(int i = 0; i < N; i++)
    {
        h_in[i] = 1;
    }

    int groundTruth = N * 1;

    cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice);

    // allocate block and thread size
    dim3 grid(gridSize);
    dim3 block(blockSize); // thrad size

    // record GPU execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    // allocate 1 block, 1 thread
    warp_level_reduce<blockSize><<<grid, block>>>(d_in, part_out, N);
    warp_level_reduce<blockSize><<<1, block>>>(d_in, d_out, gridSize);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);
    printf("allocated %d blocks, data counts are %d\n", gridSize, N);

    bool is_right = checkResult(h_out, groundTruth, 1);
    if(is_right)
    {
        printf("the ans is right\n");
    }
    else
    {
        printf("the ans is wrong\n");
        // for(int i = 0; i < gridSize; i++)
        // {
        //     printf("res per block: %1f\n", h_out[i]);
        // }
        printf("groundTruth is %f \n", groundTruth);
    }

    printf("warp_level_reduce latency = %f ms\n", milliseconds);

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);

    return 0;

}