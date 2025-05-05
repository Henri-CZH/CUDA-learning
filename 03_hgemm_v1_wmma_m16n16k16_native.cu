#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include "common/tester.h"

// only one warp per block process matrix mat (16,16)*(16,16)
template<int WMMA_M =16, int WMMA_N = 16, int WMMA_K = 16>
__global__ void hgemm_wmma_m16n16k16_naive_kernel(__half *A, __half *B, __half *C,
                                                  int M, int N, int K) {
    // define iternation times in K axis
    const int NUM_K_TIMES = div_ceil(K, WMMA_K);

    // define offset for fragment A in M axis
    const int offset_A = blockIdx.y * WMMA_M;

    // define offset for fragment B in N axis
    const int offset_B = blockIdx.x * WMMA_N;

    if (offset_A >= M && offset_B >= N) {
        return;
    }

    // define fragment C and initialize
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> C_frag;
    nvcuda::wmma::fill_fragment(C_frag, 0.0);
    
    // process
#pragma unroll
    for (int k_id = 0; k_id < NUM_K_TIMES; ++k_id) {
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, nvcuda::wmma::row_major> A_frag;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, nvcuda::wmma::row_major> B_frag;

        // load data from matrix A into fragment A
        nvcuda::wmma::load_matrix_sync(A_frag, A + offset_A * K + k_id * WMMA_K, K);

        // load data from matrix B into fragment B
        nvcuda::wmma::load_matrix_sync(B_frag, B + k_id * WMMA_K * N + offset_B, N);

        nvcuda::wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);

        __syncthreads();
    }

    // store result in matrix C
    nvcuda::wmma::store_matrix_sync(C + offset_A * N + offset_B, C_frag, N, nvcuda::wmma::mem_row_major);

}

void hgemm_wmma_m16n16k16_naive(__half *A, __half *B, __half *C, int M, int N, int K) {
    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;
    dim3 block(32);
    dim3 grid(div_ceil(N, WMMA_N), div_ceil(M, WMMA_M));

    hgemm_wmma_m16n16k16_naive_kernel<WMMA_M, WMMA_N, WMMA_K><<<grid, block>>>(A, B, C, M, N, K);
}


int main(int argc, char *argv[]) {
    Tester tester(512, 2048, 1024, 1, 10, 100, true);
    tester.evaluate(hgemm_wmma_m16n16k16_naive, "hgemm_wmma_m16n16k16_naive");
}