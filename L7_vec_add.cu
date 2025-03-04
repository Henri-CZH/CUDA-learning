#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>

// cuda kernel function
__global__ void vec_add(float* dx, float* dy, float* dz, int N)
{
    // 2D grid
    int idx = (blockDim.x * (blockIdx.x + blockIdx.y * gridDim.x) + threadIdx.x);
    // 1D grid
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if(idx < N)
    {
        hz[idx] = hy[idx] + hx[idx];
    }
}

void vec_add_cpu(float* hx, float* hy, float* hz, int N)
{
    for(int i = 0; i < N; i++)
    {
        hz[i] = hx[i] + hy[i];
    }
}

int main()
{
    int N = 10000; // num of data
    int nbytes = N * sizeof(float);

    // 1D block
    int bs = 256; // num of thread

    // 2D grid
    int s = ceil(sqrt((N + bs - 1.) / bs))
    dim3 grid(s, s);

    // 1D grid
    // int s = ceil((N + bs - 1.) / bs)
    // dim3 grid(s)

    float *dx, *hx;
    float *dy, *hy;
    float *dz, *hz;

    // allocate GPU mem
    cudaMalloc((void **)&dx, nbytes);
    cudaMalloc((void **)&dy, nbytes);
    cudaMalloc((void **)&dz, nbytes);

    // init time
    float milliseconds = 0;

    // allocate CPU mem
    hx = (float *)malloc(nbytes);
    hy = (float *)malloc(nbytes);
    hz = (float *)malloc(nbytes);

    // init 
    for(int i = 0; i < N; i++)
    {
        hx[i] = 1.0F;
        hy[i] = 1.0F;
    }

    // copy data to GPU
    cudaMemcpy(dx, hx, nbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dy, hy, nbytes, cudaMemcpyHostToDevice);

    //record GPU runtime
    cudaEvent_t start, stop;
    cudaEventCreate(&start); // create start event
    cudaEventCreate(&stop);
    cudaEventRecord(start); // start record
    
    // launch GPU kernel
    vec_add<<<grid, bs>>>(dx, dy, dz, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop); // forces CPU wait for the execution CUDA kernel function stop in GPU
    cudaEventElapsedTime(&milliseconds, start, stop); // callback time slot to milliseconds [stop - stop]

    //copy GPU result to CPU
    cudaMemcpy(hz, dz, nbytes, cudaMemcpyDeviceToHost);

    // CPU compute
    float* hz_cpu_res = (float *)malloc(nbytes);
    vec_add_cpu(hx, hy, hz_cpu_res, N);

    // check CPU res with GPU
    for(int i = 0; i < N; i++)
    {
        if(fabs(hz_cpu_res[i] - hz[i]) > 1e-6)
        {
            printf("result verification failed at element idx %d!\n", i);
        }
    }

    printf("result right");
    printf("mem BW= %f (GB/sec)\n", (float)N*4/milliseconds/1e6);
    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dz);
    free(hx);
    free(hy);
    free(hz);
    free(hz_cpu_res);

    return 0;
}