#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>

// FP16 version
// y = 0.5 * x * (1 + tanh(alpha * (x + beta * x^3)))

template<typename T, int vec_size>
struct alignas(sizeof(T) * vec_size) AlignedVector{
    T val[vec_size];
    __host__ __device__ inline const T& operator[](int i) const { return val[i]};
    __host__ __device__ inline T& operator[](int i) { return val[i]}; // return the reference of val[i.th]
}

template<typename T>
struct GeluFunctor{
    static constexpr T alpha = static_cast<T>(0.79788456080);
    static constexpr T beta = static_cast<T>(0.0447149984);

    __device__ GeluFunctor() {};

    __device__ T operator()(T x) const{
        const T half = static_cast<T>(0.5);
        const T one = static_cast<T>(1);
        const T tanh_in = alpha * (x + beta * x * x * x);
        return half * x * (one + tanh(tanh_in));
    }

    // vectorization intrinsic
    __device__ void half_intrinsic(__half* x, const __half* y){
        static constexpr float alpha_ = static_cast<T>(0.79788456080);
        static constexpr float beta_ = static_cast<T>(0.0447149984);

        const __half2 x2 = *(reinterpret_cast<const __half2*>(x)); // define half precision variant with 2 element
        const float2 tanh_in = __half2float2(
            __hmul2(__float2half2_rn(alpha_), __hadd2(x2, __hmul2(__hmul2(__hmul2(__float2half2_rn(beta_), x2), x2), x2))));
        float2 tanh_out;
        tanh_out.x = tanhf(tanh_in.x); // one element for tanh
        tanh_out.y = tanhf(tanh_in.y);

        const __half2 y2 = __hmul2(__hmul2(__float2half2_rn(0.5F), x2),
                            __hadd2(__float2half2_rn(1.0F), __float22half2_rn(tanh_out)));
        
        // return to GPU memory
        reinterpret_cast<__half2*>(y) = y2;
    }
}

template<int vec_size, bool apply_intrinsic>
__global__ void fused_gelu_cuda_kernel(const __half *d_in, __half *d_out, int N){
    int offset = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) * vec_size;
    int stirde = static_cast<int>(gridDim.x * blockDim.x) * vec_size;

    using ArrT = AlignedVector<__half, vec_size>; // declare ArrT
    __half y_reg[vec_size];
    GeluFunctor<half> gelu_fwd; // declare gelu_fwd

    for(; offset < N; offset +=stride){
        const __half *in = d_in + offset; // define thread register
        if(vec_size == 1){
            y_reg[0] = gelu_fwd(in[0]);
        }
        else{
            if(apply_intrinsic){
                // vectorization intrinsic
                for(int i = 0; i < vec_size; i += 2){
                    gelu_fwd.half_intrinsic(in + i, d_out + offset + i);
                }
            }
            else{
                // scalar compute
                for(int i = 0; i < vec_size; i++){
                    y_reg[i] = gelu_fwd(in[i]);
                }
            }
        }
    }

    if(!apply_intrinsic){
        *reinterpret_cast<ArrT*>(d_out + offset) = *reinterpret_cast<ArrT*>(y_reg); // return to GPU mem
    }
}


int main()
{
    constexpr int N = 1000;
    float milliseconds = 0;
    constexpr int vec_size_ = 1;
    cudaSetDevice(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    // allocate CPU and GPU memory
    __half *h_x = new __half[N];
    __half *d_x;
    cudaMalloc((void**)&d_x, N * sizeof(half));

    int *h_y = new __half[N];
    int *d_y;
    cudaMalloc((void**)&d_y, N * sizeof(half));
    
    // initialize data
    for(int i = 0; i < N; i++){
        h_x[i] = i % 10;
    }
    cudaMemcpy(d_histogram_data, h_histogram_data, N * sizeof(int), cudaMemcpyHostToDevice);

    auto is_aligned = [](const void* p, int alignment){
        return reinterpret_cast<uintptr_t>(p) % alignment == 0;
    };

    constexpr auto kAlignment = alignof(AlignedVector<__half, 8>);

    if(N % 8 == 0 && is_aligned(d_x, kAlignment) && is_aligned(d_y, kAlignment)){
        int block_size = 256;
        int grid_size = (N / vec_size_ + block_size - 1) / block_size;
        dim3 grid(grid_size);
        dim3 block(block_size); // thrad size
        block = std::min(block, cudaDeviceProp.maxThreadPerBlock);

        // record GPU execution time
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        fused_gelu_cuda_kernel<vec_size_, false><<<grid, block>>>(dx, dy, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        cudaMemcpy(h_y, d_y, N * sizeof(__half), cudaMemcpyDeviceToHost);
        printf("FP16 gelu cuda kernel latency = %f ms\n", milliseconds);
    }

    cudaFree(d_x);
    cudaFree(d_y);
    delete h_x;
    h_x = nullptr;
    delete h_y;
    h_y = nullptr;

    return 0;

}