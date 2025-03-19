#include <stdio.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include <cmath>

 #define LOOP_NUM 1000

 // GPU peak performance measurement

 __global__ void gpu_peak_measure(float *start, float *stop, float *d_x, float *d_y, float *result){
    // global thread id
    int gtid = threadIdx.x + blockDim.x * blockIdx.x;
    
    // allocate res register
    float res = 0.0F;

    // allocate a register
    float a = d_x[gtid];

    // allocate b register
    float b = d_y[gtid];
    
    int start_clock = 0;
    // start to record FMA execution 
    asm volatile("mov.u32 %0, %%clock;" :"=r(start_clock) :: memory");

    // execute FMA->2 operation
    for(int i = 0; i < LOOP_NUM; i++){
        res = a + b;
        res = a + b;
        res = a + b;
        res = a + b;
    }

    // synchronize all threads of SM
    asm volatile("bar.sync 0;");

    // stop to record FMA execution
    int stop_clock = 0
    asm volatile("mov.u32 %0, %%clock;" : "=r(stop_clock) :: memory"); // move data of register "clock" to register "stop_clock", and inform compiler do not modify this command

    // store result to GPU memory
    result[gtid] = res;
    start[gtid] = start_clock;
    stop[gtid] = stop_clock;
 }

 int main(){
    // allocate CPU memory to src, dst, res
    constexpr int N = 1000;
    float *h_src = (float*)malloc(sizeof(float) * N);
    float *h_dst = (float*)malloc(sizeof(float) * N);
    float *h_res = (float*)malloc(sizeof(float) * N);

    // initialize src, dst
    for(int i = 0; i < N; i++){
        h_src[i] = (float)i;
        h_dst[i] = (float)i;
    }

    // allocate CPU memory to start clock, stop clock
    int *h_start = (int*)malloc(sizeof(int) * N);
    int *h_stop = (int*)malloc(sizeof(int) * N);

    // allocate GPU memory to device src, device dst, device result, device start clock, device stop clock
    float *d_src, *d_dst, *d_res;
    cudaMalloc((void**)&d_src, sizeof(float) * N);
    cudaMalloc((void**)&d_dst, sizeof(float) * N);
    cudaMalloc((void**)&d_res, sizeof(float) * N);

    int *d_start, *d_stop
    cudaMalloc((void**)&d_start, sizeof(int) * N);
    cudaMalloc((void**)&d_stop, sizeof(int) * N);


    // copy src from CPU to GPU
    cudaMemcpy(d_src, h_src, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dst, h_dst, sizeof(float) * N, cudaMemcpyHostToDevice);

    // define grid size, block size
    dim3 grid(1); // 1 block
    dim3 block(1024); // 1024 threads

    // launch cuda kernel
    gpu_peak_measure<<<grid, block>>>(float *d_start, float *d_stop, float *d_src, float *d_dst, float *d_res);

    // store data from GPU to CPU
    cudaMemcpy(h_start, d_start, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_stop, d_dtop, sizeof(int), cudaMemcpyDeviceToHost);

    // postprocessing GPU peak performance
    cudaDevice props;
    cudaGetDeviceProperties(&props, 0);
    float flops = LOOP_NUM * 4 * 2 * N / (h_start[0] - h_stop[0]); // FLOPS per SM
    printf("FPLOS: .2f\n", flops);
    printf("GPU clock rate: .2f\n", props.clockRate * 1e-6f);
    printf("GPU peak FLOPS: .2f TFLOPS \n", flops * props.clockRate * 1e-9f * props.multiProcessorCount); // GPU TFLOPS 

    // free memory
    cudaFree(d_src);
    cudaFree(d_dst);
    cudaFree(d_res);
    cudaFree(d_start);
    cudaFree(d_stop);
    free(h_src);
    free(h_dst);
    free(h_start);
    free(h_stop);
    return 0;
 }