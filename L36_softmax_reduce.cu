#include <stdio.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include "cuda_runtime.h"
#include <cmath>

template<typename T, int vec_size>
struct alignas(sizeof(T) * vec_size) VectorType{
    T val[vec_size];
}

// warp reduce operator
template<template<typename> class ReduceOp, typename T, const int warp_width>
__inline__ __device__ T reduce_op(T val){
    for(int lane_id = warp_width/2, lane_id > 0; lane_id/2){
        val = ReduceOp<T>()(val, __shfl_xor_sync(0xffffffff, val, lane_id)); // ReduceOp: SumOp/MaxOp; (): operator; (a, b): arg for ReduceOp; SumOp<float>(a, b)
    }
}

// max op
template<typename T>
struct MaxOp{
    __device__ __forceinline__ T operator()(const T& a, const T& b){
        return max(a, b);
    }
}

// sum op
template<typename T>
struct SumOp{
    __device__ __forceinline__ T operator()(const T& a, const T& b){
        return a + b;
    }
}

template<typename T>
__inline__ __device__ T exp_operator(T val){
    return exp(val);
}


template<typename T>
__inline__ __device__ T Inf(){
    return 10000000;
}

// load op
template<int vec_size>
__device__ void load(const float *src, float *dst, const int rows, const int cols, const int col){
    using vec_type = VectorType<float, vec_size>;
    // vectorization load
    *reinterpret_cast<*vec_type>(dst) = *reinterpret_cast<*vec_type>((const_cast<float*>(src) + row * cols + col) / vec_size);
}

// store op
template<int vec_size>
__device__ void store(const float *src, float *dst, const int rows, const int cols, const int col){
    using vec_type = VectorType<float, vec_size>;
    // vectorization load
    *reinterpret_cast<*vec_type>(dst) = *reinterpret_cast<*vec_type>((const_cast<float*>(src) + row * cols + col) / vec_size);
}

// softmax kernel
template<const int row_per_thread, const int pack_size, const int col_per_thread, const int warp_width>
__global__ void softmax(const float *d_src, float *d_dst, const int rows, const int cols){
    constexpr int num_pack = col_per_thread / pack_size;

    // define thread id;
    tid = threadIdx.x;

    // define global thread id (warp id)
    int gtid = blockIdx.y * blockDim.y + threadIdx.y;

    // define step
    int step = gridDim.y * blockDim.y * row_per_thread; // total number of row what all threads handles

    // define buffer for a thread
    float thread_buffer[row_per_thread][col_per_thread];

    // load data in warp level
    for(int row = gtid * row_per_thread; row < rows; row += step){
        // load data and compute max in thread level
        float thread_max[row_per_thread];
        for(int row_id = 0; row_id < row_per_thread; row_id++){
            // load data and compute max reduce in pack level
            thread_max[row_id] = -Inf<float>();
            // define ptr to load local pack data
            float *thread_row_buf = thread_buffer[row_id];
            for(int pack_id = 0; pack_id < num_pack; pack_id++){
                int pack_offset = pack_id * pack_size;
                load<pack_size>(d_src, thread_row_buf + pack_offset, row + row_id, cols, tid * col_per_thread + pack_offset);
                for(int i = 0; i < pack_size; i++){
                    thread_max[row_id] = max(thread_max[row_id], thread_row_buf[pack_offset + i]); // max value in thread level
                }
            }
        }

        // compute max reduce in warp level
        float warp_max[row_per_thread];
        for(row_id = 0; row_id < row_per_thread; row_id++){
            warp_max[row_id] = reduce_op<MaxOp, float, warp_width>(thread_max[row_id]);
        }
        
        // compute sum reduce in thread level
        float warp_sum[row_per_thread];
        float thread_sum[row_per_thread];
        for(int row_id = 0; row_id < row_per_thread; row_id++){
             // compute sum reduce in thread level
            thread_sum[row_id] = 0.0F;
            float *thread_rwo_buf = thread_buffer[row_id];
            for(int col_id = 0; col_id < col_per_thread; col_id++){
                thread_sum[row_id] += exp_operator(thread_rwo_buf[col_id] - warp_max[row_id]);
            }
            warp_sum[row_id] = reduce_op<SumOp, float, warp_width>(thread_sum[row_id]);
        }

        // compute softmax
        for(row_id = 0; row_id < row_per_thread; row_id++){
            float *thread_row_buf = thread_buffer[row_id];
            for(int col_id = 0; col_id < col_per_thread; col_id++){
                thread_row_buf[col_id] = thread_row_buf[col_id] / warp_sum[row_id];
            }
            // store result
            for(int pack_id = 0; pack_id < num_pack; pack_id++){
                int pack_offset = pack_id * pack_size;
                store<vec_size>(thread_row_buf + pack_offset, d_dst, row + row_id, cols, tid * col_per_thread + pack_offset);
            }
        }
    }
}


void softmax_CPU(float *src, float *dst, int rows, int cols){
    // compute max value of each row
    for(int row = 0; row < rows; row++){
        int max_val = -10000000;
        float row_buf[cols];
        for(int col = 0; col < cols; col++){
            row_buf[col] = src[row * cols + col];
            max_val = max(row_buf[col], max);
        }

        float row_sum = 0.0F;
        for(int col = 0; col < cols; col++){
            // compute exp sum of each row
            row_sum += exp(row_buf - max_val);
        }

        for(int col = 0; col < cols; col++){
            // store res
            dst[row * cols + col] = row_buf[col] / row_sum;
        }
    }
}


bool check_right(float *y, float *groundTrue, const int N){
    for(int i = 0; i < N; i++){
        if(y[i] != groundTrue[i]){
            printf("y[%d]: %f \n", i, y[i]);
            printf("groundTrue[%d]: %f \n", i, groundTrue[i]);
            return false;
        }
    }
}

int main(){
    // define variable for GPU execution time
    float milliseconds = 0.0F;

    // define and allocate CPU memory for src, dst
    constexpr int N = 1000 * 1024;
    float *h_src = new float[N];
    float *h_dst = new float[N];

    // initialize src
    for(int i = 0; i < N; i++){
        h_src[i] = 1.0F;
    }

    // define groundTrue and initialize it
    float *groundTrue = new float[N];

    // compute groundTrue by CPU
    softmaxCPU(src, groundTrue, 1000, 1024);

    // define and allocate GPU memory for src, dst
    float *d_src, *d_dst;
    cudaMalloc((void**)&d_src, sizeof(float) * N);
    cudaMalloc((void**)&d_dst, sizeof(float) * N);

    // copy src from CPU memory to GPU memory
    cudaMemcpy(d_src, h_src, N * sizeof(float), cudaMemcpyHostToDevice);

    // define cuda grid size, block size
    dim3 grid(1, 125); // x axis: 1; y axis: 125
    dim3 block(32, 8); // x axis: 32->1 warp(32 thread) handle 1 row(1024 element); y axis: 8->8*125 = 1000 row

    // define variable for cuda event record
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // start to record GPU execution time
    cudaEventRecord(start);

    // launch cuda kernel
    softmax_kernel<<<grid, block>>>(d_src, d_dst, 1000, 1024);

    // stop to record GPU execution time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    // copy d_dst from GPU memory to CPU memory
    cudaMemcpy(h_dst, d_dst, cudaMemcpyDeviceToHost);

    // check result
    bool is_right = check_right(d_dst, groundTruth, N);
    if(is_right){
        printf("result is correct! \n");
    }
    else{
        printf("result is wrong! \n");
    }

    // print GPU execution time
    printf("soft max latency = %f ms \n", milliseconds);

    // free CPU and GPU memory
    cudaFree(d_src);
    cudaFree(d_dst);
    delete h_src;
    h_src = nullptr;
    delete h_dst
    h_dst = nullptr;
    delete groundTrue;
    groundTrue = nullptr;

    return 0;
}