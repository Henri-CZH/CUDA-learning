#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>

#define ARRAY_SIZE 100000000
#define MEMORY_OFFSET 100000000
#define BENCH_ITER 10
#define THREADS_NUM 256

// use vector add operator

// cuda kernel function
__global__ void mem_bw(float* A, float* B, float* C)
{

    int idx = blockDim.x * blockIdx.x +  + threadIdx.x;

    for(int i = idx; i < MEMORY_OFFSET / 4; i+= blockDim.x * gridDim.x) // avoid the umber of thread less than amount of data
    {
        float4 a1 = reinterpret_cast<float4*>(A)[i]; // read 4 data per time by a thread
        float4 b1 = reinterpret_cast<float4*>(B)[i];
        float4 c1;

        c1.x = a1.x + b1.x;
        c1.y = a1.y + b1.y;
        c1.z = a1.z + b1.z;
        c1.w = a1.w + b1.w;

        reinterpret_cast<float4*>(C)[i] = c1;
    }

}

void vec_add_cpu(float* hx, float* hy, float* hz, int N)
{
    for(int i = 0; i < 20; i++)
    {
        hz[i] = hx[i] + hy[i];
    }
}

int main()
{
    // allocate CPU memory to A/B/C
    float *A = (float*)malloc(ARRAY_SIZE * sizeof(float));
    float *B = (float*)malloc(ARRAY_SIZE * sizeof(float));
    float *C = (float*)malloc(ARRAY_SIZE * sizeof(float));

    float *A_g;
    float *B_g;
    float *C_g;

    float milliseconds = 0; // store GPU execution time slot

    //initilize data
    for(uint32_t i = 0; i < ARRAY_SIZE; i++)
    {
        A[i] = (float)i;
        B[i] = (float)i;
    }

    // allocate GPU memory to A_g/B_g/C_g
    cudaMalloc((void**)&A_g, ARRAY_SIZE * sizeof(float));
    cudaMalloc((void**)&B_g, ARRAY_SIZE * sizeof(float));
    cudaMalloc((void**)&C_g, ARRAY_SIZE * sizeof(float));

    // copy data from CPU to GPU
    cudaMemcpy(A_g, A, ARRAY_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_g, B, ARRAY_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    
    int blockNums = MEMORY_OFFSET / THREADS_NUM;

    // warm up to occpy L2 cache
    printf("warm up start\n");
    mem_bw<<<blockNums / 4, THREADS_NUM>>>(A_g, B_g, C_g); // one thread handle 4 data per time
    printf("warm up end\n");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // start record GPU execution time slot
    cudaEventRecord(start);
    for(int i = BENCH_ITER; i >= 0; i--)
    {
        mem_bw<<<blockNums / 4, THREADS_NUM>>>(A_g + i * MEMORY_OFFSET, B_g + i * MEMORY_OFFSET, C_g + i * MEMORY_OFFSET);
    }

    //stop record fot GPU
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    // copy data from GPU to CPU
    cudaMemcpy(C, C_g, ARRAY_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    // CPU compute
    float* C_cpu_res = (float *)malloc(20 * sizeof(float));
    vec_add_cpu(A, B, C_cpu_res, ARRAY_SIZE);

    // check CPU result with GPU
    for(int i = 0; i < 20; i++)
    {
        if(fabs(C_cpu_res[i] - C_g[i]) > 1e-6)
        {
            printf("Result verification failed at element index %d!\n", i);
        }
    }

    printf("res right!\n");

    // check GPU memory Bandwidth
    unsigned int N = ARRAY_SIZE * 4;
    printf("Mem BW= %f (GB/sec)\n", 3.0F * (float)N / milliseconds / 1e6);

    cudaFree(A_g);
    cudaFree(B_g);
    cudaFree(C_g);

    free(A);
    free(B);
    free(C);
    free(C_cpu_res);

    return 0;
}