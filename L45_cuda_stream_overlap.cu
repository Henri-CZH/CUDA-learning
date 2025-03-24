#include <stdio.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include <cmath>

 // cuda stream overlap

 __global__ void add(float *d_buffer, const int N){
    // global thread id
    int gtid = threadIdx.x + blockDim.x * blockIdx.x;
    for(int i = gtid; i < N; i += gridDim.x * blockDim.x){
        d_buffer[i] += 1.0F;
    }
 }


void vector_add_cpu(float *h_z_cpu, const int N){
    for(int i = 0; i < N; i++){
        h_z_cpu[i] += 1.0F;
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
    constexpr size_t N = 1024 * 10240;
    constexpr size_t num_streams = 5;

    // allocate GPU memory to d_buffers
    std::vector<float*> d_buffers(num_streams);
    for(auto d_buffer : d_buffers){
        cudaMalloc(&d_buffer, N * sizeof(float));
    }

    // define grid size, block size
    int block_size = 1024;
    int grid_size = 32; // 1, 2, 4, 8, 16, 32, 1024
    dim3 grid(grid_size); // 
    dim3 block(block_size); // 

    // define cuda stream
    vector<cudaStream_t> streams(num_streams);
    
    for(int i = 0; i < num_per_stream; i++){
        cudaStreamCreate(&streams[i]);
    }

    // launch cuda kernel
    for(int i = 0; i < num_streams; i++){

        // launch each independent cuda kernel on each stream
        add<<<grid, block, streams[i]>>>(d_buffers[i], N);
    }

    for(auto stream : streams){
        cudaStreamSynchronize(stream); // synchronize each stream
    }
    

    // post processing
    float *h_z_cpu = (float*)malloc(sizeof(float) * N);
    vector_add_cpu(h_z_cpu, N);
    
    // check res
    bool is_right = check_right(h_z_cpu, d_buffers[0], N);

    // destroy stream
    for(int = 1; i < num_streams; i++){
        cudaStreamDestroy(streams[i]);
        printf("destroying %d th stream\n", i);
    }


    // free memory
    for(auto d_buffer : d_buffers){
        cudaFree(d_buffer);
    }
    
    free(h_z_cpu);

    return 0;
 }