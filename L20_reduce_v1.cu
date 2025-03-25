#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>

// v1: use position operator replace % 

template<int blockSize> // blockSize as template arg use for static shared memory size apply during compile phase
__global__ void reduce_v1(const int* d_in, int* d_out, size_t n)
{   
    int tid = threadIdx.x;
    int gtid = threadIdx.x + blockIdx.x * blockSize; // global thread idx
    __shared__ float smem[blockSize]; // declare shared memory 
    smem[tid] = d_in[gtid]; // load data into shared memory corresponding to thread gtid
    __syncthreads(); // synchronize all threads in a block

    for(int idx = 1; idx < blockIdx.x; idx*=2)
    {
        // method 1
        // here is no warp divergent, because no use threads is idle 
        // if(tid % (2 * idx) == 0)
        if(tid & (2 * idx - 1) == 0)
            smem[tid]+= d_in[tid + idx];
        
        // method 2:
        // unsigned s = 2 * idx * tid;
        // if(s < blockDim.x)
        //     smem[s] += smem[s + idx];

        __syncthreads();
    }

    // all (N / blockSize) block's threads have finished
    if(tid == 0)
        d_out[blockIdx.x] = smem[tid]; // sum of all threads in a block
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
    int *d_out;
    cudaMalloc((void**)&d_out, gridSize * sizeof(int));

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
    reduce_v1<blockSize><<<grid, block>>>(d_in, d_out, N);

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