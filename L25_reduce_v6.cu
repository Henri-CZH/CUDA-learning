#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>

// v6 one tid operate multiple data per time

__device__ void warpSharedMemReduce(volatile float* smem, int tid)
{
    int x = smem[tid];
    if(blockDim.x >= 64)
    {
        x += smem[tid + 32];
        __syncwarp(); // synchronize all threads in a warp
        smem[tid] = x;
        __syncwarp();
    }

        x += smem[tid + 16];
        __syncwarp();
        smem[tid] = x;
        __syncwarp();
        x += smem[tid + 8];
        __syncwarp();
        smem[tid] = x;
        __syncwarp();
        x += smem[tid + 4];
        __syncwarp();
        smem[tid] = x;
        __syncwarp();
        x += smem[tid + 2];
        __syncwarp();
        smem[tid] = x;
        __syncwarp();
        x += smem[tid + 1];
        __syncwarp();
}

template<int blockSize> // blockSize as template arg use for static shared memory size apply during compile phase
__global__ void reduce_v6(const int* d_in, int* d_out, size_t n)
{   
    int tid = threadIdx.x;
    int gtid = threadIdx.x + blockIdx.x * blockDim.x; // global thread idx, here can use blockSize or blockDim.x 
    __shared__ float smem[blockSize]; // declare shared memory, here can't use blockDim.x, because shared memory size apply is in compile phase
    unsigned int total_thread_num = gridDim.x * blockDim.x;
    int sum = 0;
    for(int i = gtid; i < n; i += total_thread_num)
    {
        if(i > n)
            break;
        sum += d_in[i];
    }

    smem[tid] = sum; // load multiple data into shared memory
    __syncthreads(); // synchronize all threads in a block

    // for(int idx = blockDim.x / 2; idx > 32; idx >>=1)
    // {
    //     // method 1:
    //     // here is no warp divergent, because no use threads is idle
    //     // if(tid % (2 * idx) == 0)
    //     // if(tid & (2 * idx - 1) == 0)
    //     //     smem[tid]+= d_in[tid + idx];

    //     // method 2: 
    //     // here is bank conflict, because there is 32 bank memory, tid0 operate sdata[0]:bank0 and sdata[1]:bank1, tid16 operate sdata[32]:bank0 and sdata[33]:bank1;
    //     // unsigned s = 2 * idx * tid;
    //     // if(s < blockDim.x)
    //     //     smem[s] += smem[s + idx];
    //     if(tid > idx)
    //         smem[tid] += smem[tid + idx]; // tid0 operate smem[0]<-smem[0] + smem[128], smem[0]<-d_in[0] + d_in[256] + d_in[128] + d_in[384] all tid > 128 in a block are idle
        
    //     __syncthreads();
    // }

    // loop unrolling
    if(blockSize >= 1024)
    {
        if(tid < 512)
        {
            smem[tid] += smem[tid + 512];
        }
        __syncthreads();
    }

    if(blockSize >= 512)
    {
        if(tid < 256)
        {
            smem[tid] += smem[tid + 256];
        }
        __syncthreads();
    }

    if(blockSize >= 256)
    {
        if(tid < 128)
        {
            smem[tid] += smem[tid + 128];
        }
        __syncthreads();
    }

    if(blockSize >= 128)
    {
        if(tid < 64)
        {
            smem[tid] += smem[tid + 64];
        }
        __syncthreads();
    }


    // last warp in a block independently calc reduce 
    if(tid < 32)
        warpSharedMemReduce(smem, tid);
    
    // all (N / blockSize) block's threads have finished
    if(tid == 0)
        d_out[blockIdx.x] = smem[tid]; // smem[0]: sum of all threads in a block
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
    const int N = 25600000;
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
    reduce_v6<blockSize><<<grid, block>>>(d_in, part_out, N);
    reduce_v6<blockSize><<<1, block>>>(d_in, d_out, gridSize);

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

    printf("reduce_baseline latency = %f ms\n", milliseconds);

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);

    return 0;

}