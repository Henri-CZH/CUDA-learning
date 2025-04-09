#include "L50_gemv.cuh"

template<typename T>
void gemv_cpu(T* mat, T* vec, float* ground_true, const int rows, const int cols){
    // cpu do not support half type, so we need to change the data type from half to float
    for(int i = 0; i < rows; i++){
        ground_true[i] = 0.0f;
        for(int j = 0; j < cols; j++){
            ground_true[i] += (float)mat[i * cols + j] * (float)vec[j];
        }
        if(i < 5){
            printf("cpu res = %f\n", ground_true[i]);
        }
    }
}


template<typename T>
bool check_result(T* out, float* ground_true, const int rows){
    for(int i = 0; i < rows; i++){
        if((float)out[i] != ground_true[i]){
            printf("res is wrong at %d: GPU=%f and CPU=%f \n", i, (float)out[i], ground_true[i]);
            return false;
        }
    }
    return true;
}


// (M,N)x(N,1)->(M,1), 1 block compute one col data, intra warp reduce->inter warp reduce
template<typename T>
void gemv_launch(){
    
    // vec: Nx1; mat: MxN; mat * vec = (M,N)x(N,1)->(M,1)
    constexpr int N = 2048;
    constexpr int M = 256;

    // allocate memory to vec, mat
    T *d_vec, *d_mat, *d_res;
    T *vec = (T*)malloc(sizeof(T) * N);
    cudaMalloc((void**)&d_vec, sizeof(T) * N);

    T *mat = (T*)malloc(sizeof(T) * M * N);
    cudaMalloc((void**)&d_mat, sizeof(T) * M * N);

    T* res = (T*)malloc(sizeof(T) * M);
    cudaMalloc((void**)&d_res, sizeof(T) * M);

    // initialize vec, mat
    for(int i = 0; i < M * N; i++){
        mat[i] = (T)1;
    }

    for(int i = 0; i < N; i++){
        vec[i] = (T)1;
    }

    // copy data from CPU to GPU
    cudaMemcpy(d_vec, vec, N * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mat, mat, M * N * sizeof(T), cudaMemcpyHostToDevice);

    // allocate grid_size, block_size
    dim3 grid(M);
    dim3 block(256);

    // allocate vec size, vec per thread
    constexpr int vec_size = Vec<T>::vec_size;
    constexpr int vec_per_thread = (N / 256) / vec_size;

    // create cuda event
    float cuda_time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    gemv_kernel<vec_size, vec_per_thread><<<grid, block>>>(d_vec, d_mat, d_res, M, N);
    // TODO: cudaError_t error;
    printf("called\n");
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cuda_time, start, stop);
    printf("gemv latency = %f\n", cuda_time);
    cudaMemcpy(res, d_res, sizeof(T) * M, cudaMemcpyDeviceToHost);
    printf("GPU-res %f \n", __half2float(res[0]));

    // check result, cpu has FP16 type, so it need to change FP16 to FP32
    float* ground_true = (float*)malloc(sizeof(float) * M);
    // implicit template instantiation
    gemv_cpu(mat, vec, ground_true, M, N);
    bool is_right = check_result(res, ground_true, M);
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
template void gevm_launch<float>();
template void gevm_launch<__half>();

int main(int argc, char** argv){
    if(argv[1]){
        gemv_launch<float>();
    }
    else{
        gemv_launch<__half>();
    }
    return 0;
}
