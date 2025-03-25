#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>

// FP16 verion
// y = (x + bias) * mask * scale + add

template<typename T>
struct MaskScaleElementwiseAddFunctor{
    MaskScaleElementwiseAddFunctor(const uint8_t *mask_tensor, const T *add_val, const float scale)
    :_mask_tensor(mask_tensor), _add_val(add_val), _scale(scale)
    {}

    // overload function
    __device__ T operator()(T x, int i) const{
        return x * (static_cast<float>(static_cast<bool>(_mask_tensor[i]) * scale) + _add_val[i]);
    }

    // vectorization load for FP16
    __device__ __half2 compute_half2(half2 x, int i){
        const char2 *_mask_tensor_h2 = reinterpret_cast<const char2*>_mask_tensor;
        const __half2 *_add_val_h2 = reinterpret_cast<const __half2*>_add_val;
        __half2 scale_h2= float2half2_rn(scale);
        char2 _mask_tensor_h2_i = _mask_tensor_h2[i];
        __half2 _mask
        _mask.x = _mask_tensor_h2_i.x;
        _mask.y = _mask_tensor_h2_i.y;
        return __hadd2(__hmul2(x, __hmul2(_mask, scale_h2)), [_add_val_h2[i]]);
    }

    const uint8_t *_mask_tensor;
    const T *_add_val;
    const float _scale;
};


template<int bias_size, typename FUNCTOR, typename T>
__global__ void fused_bias_add<bias_size>(FUNCTOR functor, T *d_x, T *d_y, T d_bias, const int bias_size, const int N){
    int gtid = blockDim.x * blockIdx.x + threadIdx.x;
    const int total_num_thread = gridDim.x * blockDim.x;
    const auto *d_x_h2 = reinterpret_cast<__half2*>(d_x);
    // auto *d_y_h2 = reinterpret_cast<__half2*>(d_y);
    const auto *d_bias_h2 = reinterpret_cast<__half2*>(d_bias);
    const auto bias_size_h2 = bias_size / 2;

    for(int i = gtid; i < N / 2; i += total_num_thread){
        T tmp_x = d_x_h2[i] + d_bias_h2[i % bias_size_h2]; // x + bias
        // d_y_h2[i] = functor.compute_half2(tmp_x, i); // (x + bias) * mask * scale + add
        reinterpret_cast<__half2 *>(d_y)[i] = functor.compute_half2(tmp_x, i);
    }
}


bool check_right(__half *y, __half *groundTrue, const int N){
    for(int i = 0; i < N; i++){
        if(y[i] != groundTrue[i]){
            printf("y[%d]: %f \n", i, y[i]);
            printf("groundTrue[%d]: %f \n", i, groundTrue[i]);
            return false;
        }
    }
}


int main(){
    // y = (x + bias) * mask * scale + add
    constexpr int N = 100000;
    constexpr int bias_size = 10;
    constexpr float scale = 0.5F;
    uint8_t *h_mask_tensor = new uint8_t[N];
    __half *h_add_val = new __half[N];
    __half *h_x = new __half[N];
    __half *h_y = new __half[N];
    __half *h_bias = new __half[bias_size];
    __half *groundTrue = new __half[N];

    // initialize
    for(int i = 0; i < bias_size; i++){
        h_bias[i] = i;
    }

    for(int i = 0; i < N; i++){
        h_x[i] = (__half)(i);
        h_y[i] = 0.0F;
        h_add_val[i] = (__half)(i);
        h_mask_tensor[i] = (uint8_t)(i);
        groundTrue[i] = (h_x[i] + h_bias[i % bias_size]) * static_cast<__half>(static_cast<bool>(h_mask_tensor[i]) * scale) + h_add_val[i];
    }

    float *d_x, *d_y, *d_bias, *d_add_val, *d_mask_tensor;
    cudaMalloc((void **)&d_x, N * sizeof(__half));
    cudaMalloc((void **)&d_y, N * sizeof(__half));
    cudaMalloc((void **)&d_bias, bias_size * sizeof(__half));
    cudaMalloc((void **)&d_add_val, N * sizeof(__half));
    cudaMalloc((void **)&d_mask_tensor, N * sizeof(uint8_t));

    cudaMemcpy(d_x, h_x, N * sizeof(__half), cudaMemcpycpyHostToDevice);
    cudaMemcpy(d_y, h_y, N * sizeof(__half), cudaMemcpycpyHostToDevice);
    cudaMemcpy(d_bias, h_bias, bias_size * sizeof(__half), cudaMemcpycpyHostToDevice);
    cudaMemcpy(d_add_val, h_add_val, N * sizeof(__half), cudaMemcpycpyHostToDevice);
    cudaMemcpy(d_mask_tensor, h_mask_tensor, N * sizeof(uint8_t), cudaMemcpycpyHostToDevice);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    int block_size = 512;
    int grid_size = std::min((N + block_size - 1) / block_size, deviceProp.maxGridSize[0]);

    // declare mask scale elementwise add functor
    MaskScaleElementwiseAddFunctor<__half>(d_mask_tensor, d_add_val, scale); // mask * scale + add

    dim3 block(block_size);
    dim3 grid(grid_size);

    float milliseconds = 0.0F;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for(int i = 0; i < 1000; i++){
        fused_bias_add<bias_size><<<grid, block>>>(d_x, d_y, d_bias, bias_size, N); // (x + bias) * mask * scale + add
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaMemcpy(h_y, d_y, sizeof(__half) * N, cudaMemcpyDeviceToHost);


    bool is_right = check_right(h_y, groundTrue, N);
    if(is_right){
        printf("result is right\n");
    }
    else{
        printf("result is wrong\n");
    }

    printf("it cost %f s \n", milliseconds/1000);

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_add_val);
    cudaFree(d_mask_tensor);
    cudaFree(d_bias);
    delete h_x;
    h_x = nullptr;
    delete h_y;
    h_y = nullptr;
    delete h_add_val;
    h_add_val = nullptr;
    delete h_bias;
    h_bias = nullptr;
    delete h_mask_tensor;
    h_mask_tensor = nullptr;
    delete groundTrue;
    groundTrue = nullptr;

    return 0;
}