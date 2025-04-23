#include "rms_norm.cuh"


template<typename T>
void gemv_cpu(T* mat, T* vec, float* ground_true, const int size){
    // cpu do not support half type, so we need to change the data type from half to float
    for(int i = 0; i < size; i++){
        ground_true[i] = 0.0f;
        ground_true[i] += (float)mat[i] * (float)vec[i];
        if(i < 5){
            printf("cpu res = %f\n", ground_true[i]);
        }
    }
}


template<typename T>
bool check_result(T* out, float* ground_true, const int size){
    for(int i = 0; i < size; i++){
        if((float)out[i] != ground_true[i]){
            printf("res is wrong at %d: GPU=%f and CPU=%f \n", i, (float)out[i], ground_true[i]);
            return false;
        }
    }
    return true;
}

template<typename T>
void rms_norm_launch(){
    
    constexpr int N = 2050;
    constexpr int M = 256;

    // allocate memory to vec, mat
    T *d_vec;
    T *vec = (T*)malloc(sizeof(T) * M * N);
    cudaMalloc((void**)&d_vec, sizeof(T) * M * N);

    T *d_mat;
    T *mat = (T*)malloc(sizeof(T) * M * N);
    cudaMalloc((void**)&d_mat, sizeof(T) * M * N);

    T* d_res;
    T* res = (T*)malloc(sizeof(T) * M * N);
    cudaMalloc((void**)&d_res, sizeof(T) * M * N);

    // initialize vec, mat
    for(int i = 0; i < M * N; i++){
        mat[i] = (T)1;
    }

    for(int i = 0; i < N * M; i++){
        vec[i] = (T)1;
    }

    // copy data from CPU to GPU
    cudaMemcpy(d_vec, vec, M * N * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mat, mat, M * N * sizeof(T), cudaMemcpyHostToDevice);

    // allocate grid_size, block_size
    dim3 grid(1);
    dim3 block(1024);

    // create cuda event
    float cuda_time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    row_rmsnorm_f32<<<grid, block>>>(d_vec, d_mat, d_res, M * N, 0);
    // TODO: cudaError_t error;
    printf("called\n");
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cuda_time, start, stop);
    printf("rms norm latency = %f\n", cuda_time);
    cudaMemcpy(res, d_res, sizeof(T) * M * N, cudaMemcpyDeviceToHost);
    printf("GPU-res %f \n", res[4096]);

    // check result, cpu don't has FP16 type, so it need to change FP16 to FP32
    float* ground_true = (float*)malloc(sizeof(float) * M * N);
    // implicit template instantiation
    gemv_cpu(mat, vec, ground_true, M * N);
    bool is_right = check_result(res, ground_true, M * N);
    if(is_right){
        printf("the result is right \n");
    }
    else{
        printf("the result is wrong \n");
    }

    // free memory
    cudaFree(d_vec);
    cudaFree(d_mat);
    cudaFree(d_res);
    free(vec);
    free(mat);
    free(res);
    free(ground_true);
}

// explicit template instantiation
template void rms_norm_launch<float>();
// template void gemv_launch<__half>();

int main(int argc, char** argv){
    rms_norm_launch<float>();
    return 0;
}
