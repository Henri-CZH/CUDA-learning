#pragma once 

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>

namespace gemv{
    struct half8{
        __half2 h1 = __float2half2_rn(0.0f);
        __half2 h2 = __float2half2_rn(0.0f);
        __half2 h3 = __float2half2_rn(0.0f);
        __half2 h4 = __float2half2_rn(0.0f);

        __device__ half8& operator = (half8 h8){
            h1 = h8.h1;
            h2 = h8.h2;
            h3 = h8.h3;
            h4 = h8.h4;
            return *this;
        }
    };

    inline __device__ __half fma(__half a, __half b, __half c)
    {
        // here we need transit from __half to float due to some compiler do not identify _hadd()
        // return __hadd(__hmul(a, b), c);
        return __float2half((float)a * (float)b + (float)c);
    }


    inline __device__ half2 fma(__half a, __half2 b, __half2 c)
    {
        half2 res;
        res.x = gemv::fma(a, b.x, c.x);
        res.y = gemv::fma(a, b.y, c.y);
        return res;
    }

    inline __device__ half8 fma(__half a, half8 b, half8 c)
    {
        half8 d;
        d.h1 = gemv::fma(a, b.h1, c.h1);
        d.h2 = gemv::fma(a, b.h2, c.h2);
        d.h3 = gemv::fma(a, b.h3, c.h3);
        d.h4 = gemv::fma(a, b.h4, c.h4);
        return d;
    }

    inline __device__ float fma(float a, float b, float c)
    {
        return a * b + c;
    }

    inline __device__ float4 fma(float a, float4 b, float4 c)
    {
        float4 d;
        d.x = gemv::fma(a, b.x, c.x);
        d.y = gemv::fma(a, b.y, c.y);
        d.z = gemv::fma(a, b.z, c.z);
        d.w = gemv::fma(a, b.w, c.w);
        return d;
    }

    inline __device__ float add(float a, float b)
    {
        return a + b;
    }

    inline __device__ float4 add(float4 a, float4 b)
    {
        float4 c;
        c.x = gemv::add(a.x, b.x);
        c.y = gemv::add(a.y, b.y);
        c.z = gemv::add(a.z, b.z);
        c.w = gemv::add(a.w, b.w);
        return c;
    }
    inline __device__ __half add(__half a, __half b)
    {
        //if use L216, half+half is not really adding, its so weird, which  cause our result is 32, not 256
        return (__half)((float)a+(float)b);
    }

    inline __device__ __half2 add(__half2 a, __half2 b)
    {
        __half2 res;
        res.x = gemv::add(a.x, b.x);
        res.y = gemv::add(a.y, b.y);
        return res;
    }

    inline __device__ half8 add(half8 a, half8 b)
    {
        half8 c;
        c.h1 = gemv::add(a.h1, b.h1);
        c.h2 = gemv::add(a.h2, b.h2);
        c.h3 = gemv::add(a.h3, b.h3);
        c.h4 = gemv::add(a.h4, b.h4);
        return c;
    }
}

template<typename T>
struct Vec{
    static constexpr int vec_size = 4;
    using dtype = float4;
}; 

// template specialization
template<>
struct Vec<float>{
    static constexpr int vec_size = 4;
    
    using dtype = float4;
};

template<>
struct Vec<__half>{
    static constexpr int vec_size = 8;
    
    using dtype = gemv::half8;
};


template<int ROWS, typename T>
struct get_threads_per_mat_row {
    static const int value = ROWS * sizeof(T) / 16;
};


// sum reduce
template<typename T>
struct SumOp{
    __device__ __forceinline__ T operator()(const T &a, const T &b) const {
        return a + b;
    }
};

// template specialization
template<>
struct SumOp<__half>{
    __device__ __forceinline__ __half operator()(const __half &a, const __half &b) const {
        return __hadd(a, b);
    }
};

// max op
template<typename T>
struct MaxOp{
    __device__ __forceinline__ T operator()(const T& a, const T& b){
        return fmaxf(a, b); // CUDA math function
    }
};

// warp reduce
template<template<typename> class ReduceOp, typename T>
__device__ __forceinline__ T warp_reduce(T val){
    for(int lane_id = 16; lane_id > 0; lane_id >>= 1){
        val = ReduceOp<T>()(val, __shfl_xor_sync(0xffffffff, val, lane_id));
    }
    return val;
}

// block reduce
template<template<typename> class ReduceOp, typename T>
__device__ __forceinline__ T block_reduce(T val){
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int warp_num = (blockDim.x + 31) / 32;

    __shared__ T warp_smem[64];
    
    // intra warp reduce
    val = warp_reduce<ReduceOp, T>(val);
    // store wap reduce result
    if(lane_id == 0){
        warp_smem[warp_id] = val;
    }
    __syncthreads();

    // inter warp reduce
    T warp_res = tid < warp_num? warp_smem[tid] : (T)(0);
    return warp_reduce<ReduceOp, T>(warp_res);
}