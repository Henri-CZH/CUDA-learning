#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <curand.h>
#include<stdio.h>

// memory alignas
template<typename T, int VEC_SIZE>
struct alignas(sizeof(T) * VEC_SIZE) Vec{
    T val[VEC_SIZE];
};

template<typename T>
struct UniformDistribution_{
    __device__ T operator() (curandStatePhilox4_32_10_t *state){
        return static_cast<T>(curand_uniform(state));
    }
    static constexpr int count = 1;
};

// instance
template<>
struct UniformDistribution_<float>{
    __device__ float4 operator() (curandStatePhilox4_32_10_t *state){
        return curand_uniform4(state);
    }
    static constexpr int count = 4;
};


template<typename T>
struct DstMaskFunctor{
    const float prob_;
    const bool is_upscale_in_train_;
    float inv_prop_;
    __device__ DstMaskFunctor(const float dropout_prob, bool is_upscale_in_train) : prob_(dropout_prob), is_upscale_in_train_(is_upscale_in_train){
        inv_prop_ = 1.0f / (1.0f - dropout_prob);
    }

    __device__ void operator() (const T* src, T* dst, const T* rand){
        static constexpr int count = UniformDistribution_<T>::count;
        for(int i = 0; i < count; i++){
            if(rand[i] < prob_){
                dst[i] = static_cast<T>(0);
                dst[i + count] = dst[i];
            }
            else{
                dst[i] = is_upscale_in_train_ ? static_cast<T>(src[i] * inv_prop_) : src[i];
                dst[i + count] = static_cast<T>(1);
            }
        }
    }
};
// vectorized dsk mask: d_dst = d_src * d_mask / (1 - dropout)
template<typename T>
__global__ void VectorizedDstMask(const T* d_src,
                                    T* d_dst,
                                    uint8_t* d_mask,
                                    const int num_elem,
                                    const float dropout_prob,
                                    bool is_upscale_in_train,
                                    const int main_offset,
                                    const int seed,
                                    const int increment
                                    ){
    // get vec size
    constexpr int vec_size = UniformDistribution_<float>::count;

    // get thread/block id
    int tid = threadIdx.x;

    // compute block offset
    int block_offset = blockIdx.x * blockDim.x;

    // compute start_offset for current thread
    int start = tid * vec_size + vec_size * blockIdx.x * blockDim.x;

    // compute number of data what all block handle in a iternative phase
    int stride = blockDim.x * gridDim.x * vec_size;

    // define vector type
    using vec_type = Vec<float, vec_size>;
    using mask_vec_type = Vec<uint8_t, vec_size>;

    // initlaize rand function
    curandStatePhilox4_32_10_t state;
    curand_init(seed, block_offset + tid, increment, &state);

    // define rand fucntion
    using Rand = UniformDistribution_<float>;

    // define dst_mask, 0~vec_size : dst, vec_size~2*vec_size : mask
    T dst_mask[vec_size * 2];

    // define rand register
    float rand[vec_size];

    // define mask_res register
    uint8_t mask_result[vec_size];

    // declare dst functor
    auto mask_functor = DstMaskFunctor<T>(dropout_prob, is_upscale_in_train);

    for(; start < num_elem; start += stride){
        // vetorized load src
        const vec_type *vec_input = reinterpret_cast<const vec_type*>(d_src + start);

        // generarte random number
        auto rand_tuple = Rand()(&state);

        // compute d_dst with d_mask
        for(int i = 0; i < vec_size; i++){
            dst_mask[i] = *(reinterpret_cast<const T*>(vec_input) + i);
            rand[i] = static_cast<float>((&rand_tuple.x)[i]);
        }

        // compute d_dst with d_mask, dropout
        mask_functor(&dst_mask[0], &dst_mask[0], &rand[0]);

        // store res on d_dst
        T* res = d_dst + start;
        vec_type *vec_output = reinterpret_cast<vec_type*>(res);
        vec_output[0] = *(reinterpret_cast<vec_type*>(&dst_mask[0]));

        for(int i = 0; i < vec_size; i++){
            mask_result[i] = static_cast<uint8_t>(dst_mask[i + vec_size]);
        }

        uint8_t *mask_res = d_mask + start;
        mask_vec_type *mask_output = reinterpret_cast<mask_vec_type*>(mask_res);
        mask_output[0] = *(reinterpret_cast<mask_vec_type*>(mask_result));
    }

    // remain part
    const int remain = num_elem - start;
    if(remain > 0){
        // load src
        const T *src_remain = d_src + start;

        // generarte random number
        auto rand_tuple = Rand()(&state);

        // compute d_dst with d_mask
        for(int i = 0; i < vec_size; i++){
            if(i < remain){
                dst_mask[i] = src_remain[i];
                rand[i] = static_cast<float>((&rand_tuple.x)[i]);
            }
        }
        mask_functor(&dst_mask[0], &dst_mask[0], &rand[0]);

        // store res on d_dst
        T* res = d_dst + start;
        uint8_t *mask_res = d_mask + start;
        for(int i = 0; i < vec_size; i++){
            if(i < remain){
                res[i] = dst_mask[i];
                mask_res[i] = static_cast<uint8_t>(dst_mask[i + vec_size]);
            }
        }
    }
}

// dropout kernel: d_dst = d_src * d_mask / (1 - dropout)
template<typename T>
void DropoutKernel(const T* d_src,
                    T* d_dst,
                    uint8_t* d_mask,
                    const int num_elem,
                    const float dropout_prob,
                    bool is_upscale_in_train,
                    bool is_test,
                    int seed){
    // determinate whether train or test mode
    if(!is_test){
        # // determinate dropout_prob
        if(dropout_prob == 1.0f){
            cudaMemset(d_dst, 0, num_elem);
            return;
        }

        // allocate grid/block size
        constexpr int num_block = 2;
        constexpr int num_thread = 256;
        dim3 grid_size(num_block);
        dim3 block_size(num_thread);

        // compute main offset for not align data part
        const int vec_size = UniformDistribution_<T>::count;
        const int main_offset = num_elem / (num_block * num_thread * vec_size) * (num_block * num_thread * vec_size);
        constexpr int increment = 0;

        // create cuda event
        float kernel_time;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // start record cuda event
        cudaEventRecord(start);

        // launch cuda kernel
        VectorizedDstMask<T><<<grid_size, block_size>>>(d_src,
                                                        d_dst,
                                                        d_mask,
                                                        num_elem,
                                                        dropout_prob,
                                                        is_upscale_in_train,
                                                        main_offset,
                                                        seed,
                                                        increment);

        // stop record cuda event
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&kernel_time, start, stop);
        printf("dropout kernel latency = %f ms\n", kernel_time);
    }
    else{
        cudaMemcpy(d_dst, d_src, num_elem * sizeof(T), cudaMemcpyDeviceToDevice);
    }
}

int main(){
    // 512 * 4 + 2
    constexpr int num_elem = 2050;
    // allocate CPU memory for src, dst, mask
    float *h_src = (float*)malloc(sizeof(float) * num_elem);
    float *h_dst = (float*)malloc(sizeof(float) * num_elem);
    uint8_t *h_mask = (uint8_t*)malloc(sizeof(uint8_t) * num_elem);

    // allocate GPU memory for src, dst, mask
    float *d_src, *d_dst;
    uint8_t *d_mask;
    cudaMalloc((void**)&d_src, sizeof(float) * num_elem);
    cudaMalloc((void**)&d_dst, sizeof(float) * num_elem);
    cudaMalloc((void**)&d_mask, sizeof(uint8_t) * num_elem);
    
    // initialize CPU src
    for(int i = 0; i < num_elem; i++){
        h_src[i] = 1.0;
    }

    constexpr float dropout_prob = 0.5;

    constexpr bool is_upscale_in_train = true;

    constexpr bool is_test = false;

    constexpr int seed = 10000;

    // copy CPU src/dst/mask to GPU src/dst/mask
    cudaMemcpy(d_src, h_src, num_elem * sizeof(float), cudaMemcpyHostToDevice);

    // call cuda kernel function
    DropoutKernel<float>(d_src,
                        d_dst,
                        d_mask,
                        num_elem,
                        dropout_prob,
                        is_upscale_in_train,
                        is_test,
                        seed);

    // copy GPU dst to CPU dst
    cudaMemcpy(h_dst, d_dst, sizeof(float) * num_elem, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_mask, d_mask, sizeof(uint8_t) * num_elem, cudaMemcpyDeviceToHost);

    // compute ground truth of the last 3 element
    for(int i = num_elem - 3; i < num_elem; i++){
        printf("%d y = %f \n", i, h_dst[i]);
        printf("%d mask = %d \n", i, h_mask[i]);
    }

    // free memory
    cudaFree(d_src);
    cudaFree(d_dst);
    cudaFree(d_mask);
    free(h_src);
    free(h_dst);
    free(h_mask);

    return 0;
}
