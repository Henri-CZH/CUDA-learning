#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>

// naive filter k
__global__ void naive_filter_k(const int* d_in, int* d_out, int *num_res, int n)
{   
    int gtid = threadIdx.x + blockIdx.x * blockDim.x; // global thread idx
    for(int i = gtid; i < n; i++)
    {  
        if(i < n && d_in[i] > 0)
        {
            d_out[atomicAdd(num_res, 1)] = src[i];
        }
    }
}

// block level filter k
__global__ void block_filter_k(const int* d_in, int* d_out, int *num_res, int n)
{
    __shared__ int ln; // number of d_in[i] > 0 for current block
    int gtid = threadIdx.x + blockIdx.x * blockDim.x
    int total_thread_num = gridDim.x * blockDim.x;

    for(int i = gtid; i < n, i+= total_thread_num)
    {
        // initialize in first thread 
        if(threadIdx.x == 0)
        {
            ln = 0;
        }
        __syncthreads();

        int pos;
        if(i < n && d_in[i] > 0)
        {
            pos = atomicAdd(&ln, 1); // pos = ln, ln = ln + 1;
        }
        __syncthreads();

        if(threadIdx.x == 0)
        {
            ln = atomicAdd(num_res, ln);  // ln = num_res, num_res = ln + 1;
        }
        __syncthreads();

        if(i < n && d_in[i] > 0)
        {
            pos += ln; // global offset for d_in[i] > 0
            d_out[pos] = d_in[i];
        }
        __syncthreads();
    }
}

// warp level filter k
__device__ int atomicAggInc(int *count)
{
    unsigned int active_mask = __activemask(); // mask of active thread in a warp: d_in[tid0] > 0, d_in[tid2]->active_mask = 0101
    unsigned int leader_idx = __ffs(active_mask) - 1; // return first active thread idex, active_mask = 0010->ffs(active_mask) = 2;
    unsigned int change = __popc(active_mask); // number of active thread in a warp
    unsigned int lane_mask; 
    asm("mov.u32 %0, %%lanemask_lt;" : "=r"(lane_mask)); // mask of thread ID is less than current thread ID: current threadID = 3->lane_mask = 0111;
    unsigned int rank = __popc(ative_mask & lane_mask); // number of active thread whose thread ID is less than current thread;
    unsigned int warp_res;
    if(rank == 0)
    {
        warp_res = atomicAdd(count, change); // compute global offset of warp
    }
    __shfl_sync(active_mask, res, leader_idx); // broadcast warp_res of leader thread to each active thread in a warp
    return warp_res + rank; // global offset + local offset
}

// warp level filter k
__global__ void warp_filter_k(const int* d_in, int* d_out, int *num_res, int n)
{
    int gtid = threadIdx.x + blockIdx.x * blockDim.x; // global thread idx

    if(gtid > n)
    {
        return;
    }
    if(d_in[i] > 0)
    {
        // thread which d_in[i] > 0
        d_out[atomicAggInc(num_res)] = src[i];
    }
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
    warp_filter_k<<<grid, block>>>(d_histogram_data, d_bin_data, N);

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