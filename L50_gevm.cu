#include "L50_gemv.cuh"

template<typename T>
void gemv_cpu(T* mat, T* vec, float* ground_true, int rows, int cols){
    // cpu do not support half type, so we need to change the data type from half to float
    for(int i = 0; i < cols; i++){
        ground_true[i] = 0.0f;
        for(int j = 0; j < rows; j++){
            ground_true[j] += (float)mat[j * cols + i] * (float)vec[i];
        }
    }
}

template<typename T>
bool check_result(T* res, float* ground_true, int cols){
    for(int i = 0; i < cols; i++){
        if((float)res[i] != ground_true[i]){
            printf("res is wrong: GPU=%f and CPU=%f \n", (float)res[i], ground_true[i]);
            return false;
        }
    }
    return true;
}


// (1,M)x(M,N)->(1,N), THREAD_PER_ROW compute cols data, one block compute row_per_iter data, iternatively compute all data
template<typename T>
void gevm_launch(){
    
    // vec: 1xM; mat: MxN; vec * mat = (1,M)x(M,N)->(1,N)
    constexpr int N = 256;
    constexpr int M = 256;

    // allocate memory to vec, mat
    T *d_vec, *d_mat, *d_res;
    T *vec = (T*)malloc(sizeof(T) * M);
    cudaMalloc((void**)&d_vec, sizeof(T) * M);

    T *mat = (T*)malloc(sizeof(T) * M * N);
    cudaMalloc((void**)&d_mat, sizeof(T) * M * N);

    T* res = (T*)malloc(sizeof(T) * N);
    cudaMalloc((void**)&d_res, sizeof(T) * N);

    // initialize vec, mat
    for(int i = 0; i < M * N; i++){
        mat[i] = (T)1;
    }

    for(int i = 0; i < M; i++){
        vec[i] = (T)1;
    }

    // copy data from CPU to GPU
    cudaMemcpy(d_vec, vec, M * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mat, mat, M * N * sizeof(T), cudaMemcpyHostToDevice);

    // allocate grid_size, block_size
    dim3 grid(1);
    dim3 block(256);

    // allocate vec size, thread_per_block, thread_per_row; 4:FP32, 8:FP16;
    constexpr int vec_size = Vec<T>::vec_size; 
    constexpr int thread_per_row = get_threads_per_mat_row<M, T>::value;
    constexpr int thread_per_block = 256;

    // create cuda event
    float cuda_time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    gevm_kernel<thread_per_row, thread_per_block, vec_size><<<grid, block>>>(d_vec, d_mat, d_res, M, N);
    // TODO: cudaError_t error;
    printf("called\n");
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cuda_time, start, stop);
    printf("gemv latency = %f\n", cuda_time);
    // store result from GPU to CPU
    cudaMemcpy(res, d_res, sizeof(T) * N, cudaMemcpyDeviceToHost);
    printf("GPU-res %f \n", __half2float(res[0]));

    // check result, cpu has FP16 type, so it need to change FP16 to FP32
    float* ground_true = (float*)malloc(sizeof(float) * N);
    // implicit template instantiation
    gemv_cpu(mat, vec, ground_true, M, N);
    bool is_right = check_result(res, ground_true, N);
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
        gevm_launch<float>();
    }
    else{
        gevm_launch<__half>();
    }
    return 0;
}


