#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <cuda_fp16.h>

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

static __global__ void row_rmsnorm_f32(float* in, float* weight, float* out, const int size, float eps) {
    const int tid = threadIdx.x;
    constexpr int pack_size = 4;
    const int pack_num = size / pack_size;
    const int pack_offset = pack_size * pack_num;

    // vectorizaltion load data->one thread process 4 data per time
    float sum = 0.0f;
    float4* in_pack = reinterpret_cast<float4*>(in);
    for (int i = tid; i < pack_num; i += blockDim.x) {
        float4 in_float4 = *(in_pack + i);
        sum += in_float4.x * in_float4.x;
        sum += in_float4.y * in_float4.y;
        sum += in_float4.z * in_float4.z;
        sum += in_float4.w * in_float4.w;
    }

    // remaining part
    for (int i = tid + pack_offset; i < size; i += blockDim.x) {
        sum += in[i] * in[i];
    }

    // block reduce
    __shared__ float shared_val;
    sum = block_reduce<SumOp, float>(sum);
    if (tid == 0) {
        shared_val = sum;
    }
    __syncthreads();
    sum = shared_val;

    // compute rms scale
    const float scale = rsqrtf(sum / static_cast<const float>(size) + eps);
    // vectorization store data
    float4* out_pack = reinterpret_cast<float4*>(out);
    float4* weight_pack = reinterpret_cast<float4*>(weight);
    for (int i = tid; i < pack_num; i += blockDim.x) {
        float4 in_float4 = *(in_pack + i);
        float4 weight_float4 = *(weight_pack + i);
        *(out_pack + i) = make_float4(in_float4.x * weight_float4.x * scale, in_float4.y * weight_float4.y * scale,
                                      in_float4.z * weight_float4.z * scale, in_float4.w * weight_float4.w * scale);
    }

    for (int i = tid + pack_offset; i < size; i += blockDim.x) {
        out[i] = in[i] * weight[i] * scale;
    }
}
