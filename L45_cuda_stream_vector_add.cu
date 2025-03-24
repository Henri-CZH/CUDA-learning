#include <stdio.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include <cmath>

 // cuda stream copy overlap

 __global__ void vector_add(float *d_x, float *d_y, float *d_z, const int num_per_stream){
    // global thread id
    int gtid = threadIdx.x + blockDim.x * blockIdx.x;
    for(int i = gtid; i < num_per_stream; i += gridDim.x * blockDim.x){
        d_z[i] = d_x[i] + d_y[i];
    }
 }


void vector_add_cpu(float *h_x, float *h_y, float *h_z_cpu, const int N){
    for(int i = 0; i < N; i++){
        h_z_cpu[i] = d_x[i] + d_y[i];
    }
}


bool check_right(float *h_z, float *h_z_cpu, const int N){
    for(int i = 1; i < N; i++){
        if(fabs(h_z_cpu[i] - h_z[i]) > 1e-6){
            printf("idx: %d, cpu: %f, gpu: %f\n", i, h_z_cpu[i], h_z[i]);
            return false;
        }
    }
    return true;
}

int main(){
    // define data size
    constexpr int N = 1000;
    constexpr int num_streams = 1;
    constexpr int num_per_stream = N / num_streams;
    constexpr int size_per_stream = num_per_stream * sizeof(float);

    // allocate CPU memory to x, y, z
    float *h_x, *h_y, *h_z;
    cudaHostAlloc(&h_x, sizeof(float) * N, cudaHostAllocDefault); // must allocate pinned memory
    cudaHostAlloc(&h_y, sizeof(float) * N, cudaHostAllocDefault);
    cudaHostAlloc(&h_z, sizeof(float) * N, cudaHostAllocDefault);


    // initialize src, dst
    for(int i = 0; i < N; i++){
        h_src[i] = 1;
        h_dst[i] = 1;
    }

    // allocate GPU memory to x, y, z
    float *d_x, *d_y, *d_z;
    cudaMalloc((void**)&d_x, sizeof(float) * N);
    cudaMalloc((void**)&d_y, sizeof(float) * N);
    cudaMalloc((void**)&d_z, sizeof(float) * N);

    // define grid size, block size
    int block_size = 256;
    int grid_size = ceil((num_per_stream + block_size - 1)) / block_size;
    dim3 grid(grid_size); // 
    dim3 block(block_size); // 

    // define cuda stream
    cudaStream_t streams[num_streams];
    
    for(int i = 0; i < num_per_stream; i++){
        cudaStreamCreate(&streams[i]);
    }

    // launch cuda kernel
    for(int i = 0; i < num_streams; i++){
        int start_per_stream = i * num_per_stream;
        
        // copy data from CPU to GPU
        cudaMemcpyAsynchnc(d_x + start_per_stream, hx + start_per_stream, size_per_stream, cudaMemcpyHostToDevice, streams[i]);
        cudaMemcpyAsynchnc(d_y + start_per_stream, hy + start_per_stream, size_per_stream, cudaMemcpyHostToDevice, streams[i]);

        // launch cuda kernel
        vector_add<<<grid, block, streams[i]>>>(d_x + start_per_stream, d_y + start_per_stream, d_z + start_per_stream, num_per_stream);

        // store data from GPU to CPU
        cudaMemcpyAsynchnc(h_z + start_per_stream, d_z + start_per_stream, size_per_stream, cudaMemcpyDeviceToHost, streams[i]);
    }
    
    cudaDeviceSynchronize(); // synchronize all stream

    // post processing
    float *h_z_cpu = (float*)malloc(sizeof(float) * N);
    vector_add_cpu(h_x, h_y, h_z_cpu, N);
    
    // check res
    bool is_right = check_right(h_z_cpu, h_z, N);

    // destroy stream
    for(int = 1; i < num_streams; i++){
        cudaStreamDestroy(streams[i]);
        printf("destroying %d th stream\n", i);
    }


    // free memory
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    cudaFreeHost(h_x);
    cudaFreeHost(h_y);
    cudaFreeHost(h_z);
    free(h_z_cpu);

    return 0;
 }