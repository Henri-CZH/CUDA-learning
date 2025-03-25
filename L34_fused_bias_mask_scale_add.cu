#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>

// FP32 verion
// y = (x + bias) * mask * scale + add

template<typename T>
struct MaskScaleElementwiseAddFunctor{
    MaskScaleElementwiseAddFunctor(const uint8_t *mask_tensor, const T *add_val, float scale)
    :_mask_tensor(mask_tensor), _add_val(add_val), _scale(scale)
    {}

    // overload function
    __device__ T operator()(T x, int i) const{
        return x * (static_cast<float>(static_cast<bool>(_mask_tensor[i]) * scale) + _add_val[i]);
    }

    const uint8_t *_mask_tensor;
    const T *_add_val;
    float _scale;
};


template<int bias_size, typename FUNCTOR, typename T>
__global__ void fused_bias_add<bias_size>(FUNCTOR functor, T *d_x, T *d_y, T d_bias, const int bias_size, const int N){
    int gtid = blockDim.x * blockIdx.x + threadIdx.x;
    const int total_num_thread = gridDim.x * blockDim.x;
    for(int i = gtid; i < N; i += total_num_thread){
        T tmp_x = d_x[i] + d_bias[i % bias_size]; // x + bias
        d_y[i] = functor(tmp_x, i); // (x + bias) * mask * scale + add
    }
}


// vectorization load data
template<int bias_size, typename FUNCTOR, typename T>
__global__ void fused_bias_add_vectorization<bias_size>(FUNCTOR functor, T *d_x, T *d_y, T d_bias, const int bias_size, const int N){
    int gtid = blockDim.x * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;
    const int total_num_thread = gridDim.x * blockDim.x;
    __share__ T smem[bias_size];

    if(tid < bias_size){
        smem[tid] = d_bias[tid]; // store d_bias into shared memory
    }

    for(int i = gtid; i < N / 4; i += total_num_thread){
        float4 a = reinterpret_cast<float4 *>(d_x)[i];
        float4 b;
        b.x = functor(a.x + smem[(i * 4) % bias_size], i * 4);
        b.y = functor(a.y + smem[(i * 4 + 1) % bias_size], i * 4 + 1);
        b.z = functor(a.z + smem[(i * 4 + 2) % bias_size], i * 4 + 2);
        b.w = functor(a.w + smem[(i * 4 + 3) % bias_size], i * 4 + 3);
        reinterpret_cast<float4 *>(d_y)[i] = b; // vectorization store res into d_y
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
    // y = (x + bias) * mask * scale + add
    constexpr int N = 100000;
    constexpr int bias_size = 10;
    float scale = 0.5F;
    uint8_t *h_mask_tensor = new uint8_t[N];
    float *h_add_val = new float[N];
    float *h_x = new float[N];
    float *h_y = new float[N];
    float *h_bias = new float[bias_size];
    float *groundTrue = new float[N];

    // initialize
    for(int i = 0; i < bias_size; i++){
        h_bias[i] = i;
    }

    for(int i = 0; i < N; i++){
        h_x[i] = (float)(i);
        h_y[i] = 0.0F;
        h_add_val[i] = (float)(i);
        h_mask_tensor[i] = (uint8_t)(i);
        groundTrue[i] = (h_x[i] + h_bias[i % bias_size]) * static_cast<float>(static_cast<bool>(h_mask_tensor[i]) * scale) + h_add_val[i];
    }

    float *d_x, *d_y, *d_bias, *d_add_val, *d_mask_tensor;
    cudaMalloc((void **)&d_x, N * sizeof(float));
    cudaMalloc((void **)&d_y, N * sizeof(float));
    cudaMalloc((void **)&d_bias, bias_size * sizeof(float));
    cudaMalloc((void **)&d_add_val, N * sizeof(float));
    cudaMalloc((void **)&d_mask_tensor, N * sizeof(uint8_t));

    cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpycpyHostToDevice);
    cudaMemcpy(d_y, h_y, N * sizeof(float), cudaMemcpycpyHostToDevice);
    cudaMemcpy(d_bias, h_bias, bias_size * sizeof(float), cudaMemcpycpyHostToDevice);
    cudaMemcpy(d_add_val, h_add_val, N * sizeof(float), cudaMemcpycpyHostToDevice);
    cudaMemcpy(d_mask_tensor, h_mask_tensor, N * sizeof(uint8_t), cudaMemcpycpyHostToDevice);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    int block_size = 512;
    int grid_size = std::min((N + block_size - 1) / block_size, deviceProp.maxGridSize[0]);

    // declare mask scale elementwise add functor
    MaskScaleElementwiseAddFunctor<float>(d_mask_tensor, d_add_val, scale); // mask * scale + add

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
    cudaMemcpy(h_y, d_y, sizeof(float) * N, cudaMemcpyDeviceToHost);


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