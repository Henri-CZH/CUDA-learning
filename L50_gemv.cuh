#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <stdio.h>

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

// FP32 version: (M,N)x(N,1)->(M,1), 1 block compute one col data, intra warp reduce->inter warp reduce
template<int VEC_SIZE, int VEC_PER_THREAD>
__global__ void gemv_kernel(const float* d_vec, const float* d_mat, float* d_res, const int rows, const int cols){
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    float thread_res = 0.0f;

    // load float4 per time
    for(int i = 0; i < VEC_PER_THREAD; i++){
        const float4 mat4 = reinterpret_cast<const float4*>(d_mat)[bid * (cols / VEC_SIZE) + i * blockDim.x + tid];
        const float4 vec4 = reinterpret_cast<const float4*>(d_vec)[i * blockDim.x + tid];
        thread_res += mat4.x * vec4.x;
        thread_res += mat4.y * vec4.y;
        thread_res += mat4.z * vec4.z;
        thread_res += mat4.w * vec4.w;
    }

    // block reduce sum: first intra warp reduce, then inter warp reduce
    float block_res = block_reduce<SumOp, float>(thread_res);

    // get res from 1st thread
    if(tid == 0){
        d_res[bid] = block_res;
    }
    __syncthreads();
}

// FP16 version: (M,N)x(N,1)->(M,1), 1 block compute one col data, intra warp reduce->inter warp reduce
template<int VEC_SIZE, int VEC_PER_THREAD>
__global__ void gemv_kernel(__half* d_vec, __half* d_mat, __half* d_res, const int rows, const int cols){
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    __half thread_res = (__half)(0);

    // load float4 per time
    for(int i = 0; i < VEC_PER_THREAD; i++){
        float4 mat4 = reinterpret_cast<float4*>(d_mat)[bid * (cols / VEC_SIZE) + i * blockDim.x + tid];
        float4 vec4 = reinterpret_cast<float4*>(d_vec)[i * blockDim.x + tid];
        __half2 *mat_h1 = (__half2*)&mat4.x;
        __half2 *mat_h2 = (__half2*)&mat4.y;
        __half2 *mat_h3 = (__half2*)&mat4.z;
        __half2 *mat_h4 = (__half2*)&mat4.w;
        __half2 *vec_h1 = (__half2*)&vec4.x;
        __half2 *vec_h2 = (__half2*)&vec4.y;
        __half2 *vec_h3 = (__half2*)&vec4.z;
        __half2 *vec_h4 = (__half2*)&vec4.w;
        __half2 res1 = __hmul2(*mat_h1, *vec_h1);
        __half2 res2 = __hmul2(*mat_h2, *vec_h2);
        __half2 res3 = __hmul2(*mat_h3, *vec_h3);
        __half2 res4 = __hmul2(*mat_h4, *vec_h4);
        __half2 res = __hadd2(res1, __hadd2(res2, __hadd2(res3, res4)));
        thread_res = __hadd(res.x, res.y);
    }

    // block reduce sum: first intra warp reduce, then inter warp reduce
    __half block_res = block_reduce<SumOp, __half>(thread_res);

    // get res from 1st thread
    if(tid == 0){
        d_res[bid] = block_res;
    }
    __syncthreads();
}


// FP16 version: (1,M)x(M,N)->(1,N), THREAD_PER_ROW compute cols data, one block compute row_per_iter data, iternatively compute all data
template<int THREAD_PER_ROW, int THREAD_PER_BLOCK, int VEC_SIZE>
__global__ void gevm_kernel(__half *d_vec, __half *d_mat, __half *d_res, const int rows, const int cols){
    // row id
    int row_id = threadIdx.x / THREAD_PER_ROW;

    // col id
    int col_id = threadIdx.x % THREAD_PER_ROW * VEC_SIZE;

    // row per iter/block
    int row_per_iter = THREAD_PER_BLOCK / THREAD_PER_ROW;

    gemv::half8 local_res;
    // compute inter block reduce
    for(int i = row_id; i < rows; i += row_per_iter){
        gemv::half8 mat4 = *reinterpret_cast<gemv::half8*>(&d_mat[i * cols + col_id]);
        __half vec = d_vec[i];
        local_res = gemv::fma(vec, mat4, local_res);
        //printf("mat4= %f %f %f %f \n", mat4.x, mat4.y, mat4.z, mat4.w);
    }
    // float2 tmp_local_res = __half22float2(local_res.h1);
    // printf("local_res %f \n", tmp_local_res.x);

    // compute intra block reduce and store res in 1st row's register
    // allocate number of half block handling matrize size;
    __shared__ __half block_res[512];

    // using binary method to compute inter block result, first half part + second half part-> row1 + row3, row2 + row4; row1 + row2
    for (int row_num = row_per_iter; row_num >= 2; row_num /= 2) {
        int mid_row_id = row_num / 2;
        if (row_id >= mid_row_id && row_id < row_num) {
            *reinterpret_cast<gemv::half8*>(&block_res[(row_id - mid_row_id) * cols + col_id]) = local_res;
        }
        __syncthreads();

        if (row_id < mid_row_id) {
            // first half part + second half part
            local_res = gemv::add(*reinterpret_cast<gemv::half8*>(&block_res[row_id * cols + col_id]), local_res);
        }
        __syncthreads();
    }

    // get final res from 1st row and store it to GPU memory
    if(row_id == 0){
        *reinterpret_cast<gemv::half8*>(&d_res[col_id]) = local_res;
        // printf("d_res %f \n", __half2float(d_res[col_id]));
    }
}

// FP32 version: (1,M)x(M,N)->(1,N), THREAD_PER_ROW compute cols data, one block compute row_per_iter data, iternatively compute all data
template<int THREAD_PER_ROW, int THREAD_PER_BLOCK, int VEC_SIZE>
__global__ void gevm_kernel(float *d_vec, float *d_mat, float *d_res, const int rows, const int cols){
    // row id
    int row_id = threadIdx.x / THREAD_PER_ROW;

    // col id
    int col_id = threadIdx.x % THREAD_PER_ROW * VEC_SIZE;

    // row per iter/block
    int row_per_iter = THREAD_PER_BLOCK / THREAD_PER_ROW;

    float4 local_res = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    // compute inter block reduce
    for(int i = row_id; i < rows; i += row_per_iter){
        float4 mat4 = *reinterpret_cast<float4*>(&d_mat[i * cols + col_id]);
        float vec = d_vec[i];
        local_res = gemv::fma(vec, mat4, local_res);
    }

    // compute intra block reduce and store res in 1st row's register
    // allocate number of half block handling matrize size;
    __shared__ float block_res[512];

    // using binary method to compute inter block result, first half part + second half part-> row1 + row3, row2 + row4; row1 + row2
    for (int row_num = row_per_iter; row_num >= 2; row_num /= 2) {
        int mid_row_id = row_num / 2;
        if (row_id >= mid_row_id && row_id < row_num) {
            *reinterpret_cast<float4*>(&block_res[(row_id - mid_row_id) * cols + col_id]) = local_res;
        }
        __syncthreads();

        if (row_id < mid_row_id) {
            // first half part + second half part
            local_res = gemv::add(*reinterpret_cast<float4*>(&block_res[row_id * cols + col_id]), local_res);
        }
        __syncthreads();
    }

    // get final res from 1st row and store it to GPU memory
    if(row_id == 0){
        *reinterpret_cast<float4*>(&d_res[col_id]) = local_res;
    }
}


// gemv w/. not aglined version
template<int PACK_SIZE, int ROWS_PER_BLOCK>
__global__ void fp32_gemv(float* d_src, float* d_weight, float* d_dst, const int rows, const int cols) {
    // define tid
    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;

    // define start row, end row
    unsigned int start_row = blockIdx.x * ROWS_PER_BLOCK;
    unsigned int end_row = start_row + ROWS_PER_BLOCK;
    if (start_row >= rows) {
        return;
    }

    // define pack_num, pack_offset
    const int pack_num = cols / PACK_SIZE;
    const int pack_offset = pack_num * PACK_SIZE;

    // block level
    float thread_sum = 0.0f;
    for (int row = start_row; row < end_row; ++row) {
        // vectorization load data
        int row_offset = row * cols;

        // thread level
        for (int i = tid; i < pack_num; i += blockDim.x) {
            float4 src_f4 = reinterpret_cast<float4*>(d_src)[i];
            float4 weight_f4 = reinterpret_cast<float4*>(d_weight + row_offset)[i];
            // compute sum
            thread_sum += 4 * src_f4.x * weight_f4.x;
            // thread_sum += src_f4.y * weight_f4.y;
            // thread_sum += src_f4.z * weight_f4.z;
            // thread_sum += src_f4.w * weight_f4.w;
        }

        // not aligned part
        for (int i = pack_offset + tid; i < cols; i += blockDim.x) {
            thread_sum += d_src[i] * d_weight[row_offset + i];
        }

        // block reduce
        float block_sum = block_reduce<SumOp, float>(thread_sum);
        __syncthreads();

        // store res
        if (tid == 0) {
            d_dst[row] = block_sum;
        }
    }
}


// not aglined quantization version w/ int8
template<int PACK_SIZE, int ROWS_PER_BLOCK>
__global__ void fp322int8_gemv(float* d_src, uint8_t* d_weight, float* d_dst, float* d_scale, const int group_size, const int rows, const int cols) {
    // define tid
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    // define start row, end row
    const int start_row = blockIdx.x * ROWS_PER_BLOCK;
    const int end_row = start_row + ROWS_PER_BLOCK;
    if (start_row >= rows) {
        return;
    }

    // define pack_num, pack_offset
    int pack_num = cols / PACK_SIZE;
    int pack_offset = pack_num * PACK_SIZE;

    // block level
    float thread_sum = 0.0f;
    for (int row = start_row; row < end_row; ++row) {
        // thread level
        for (int i = tid; i < cols; i += blockDim.x){
            const int weight_id = row * cols + i;
            const int group_id = weight_id / group_size;
            //compute sum
            thread_sum += d_src[i] * static_cast<float>(d_weight[weight_id]) * d_scale[group_id];
        }

        // block reduce
        float block_sum = block_reduce<SumOp, float>(thread_sum);
        __syncthreads();

        // store res
        if (tid == 0) {
            d_dst[row] = block_sum;
        }
    }
}
