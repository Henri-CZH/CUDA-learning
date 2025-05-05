#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include "common/tester.h"

#define LDST64BITS(value) (reinterpret_cast<float2*>(&(value))[0])
#define LDST32BITS(value) (reinterpret_cast<__half2*>(&(value))[0])

// only 8 warp per block process matrix mat (64,16)*(16,32)
template<int WMMA_M = 16, int WMMA_N = 16, int WMMA_K = 16, int WMMA_TILE_M = 4, int WMMA_TILE_N = 2>
__global__ void hgemm_wmma_m16n16k16_mma4x2_kernel(__half *A, __half *B, __half *C,
                                                  int M, int N, int K) {
    // define iternation times in K axis
    const int NUM_K_TIMES = div_ceil(K, WMMA_K);

    // define bid in M, N axis
    const int bid_M = blockIdx.y;
    const int bid_N = blockIdx.x;

    // define tid in M, N axis
    const int tid = threadIdx.x + blockDim.x * threadIdx.y;

    // define define warp id in M, N axis->4 warp in M axis, 2 warp in N warp within a block
    const int warp_id = tid / 32; // 0~7
    const int lane_id = tid % 32; // 0~31

    const int warp_m = warp_id / 2; // 0~3
    const int warp_n = warp_id % 2; // 0~1

    // define size of fragment C->(BM, BN)
    constexpr int BM = WMMA_M * WMMA_TILE_M; // 16x4=64
    constexpr int BN = WMMA_N * WMMA_TILE_N; // 16x2=32
    constexpr int BK = WMMA_K; // 16

    // define shared memory w/. [BM][BN]
    __shared__ __half s_a[BM][BK], s_b[BK][BN]; // 64x16x2=2k, 16x32x2=1k

    // define shared memory offset in M, K, N axis
    // 256 thread load s_a = (64,16), s_b = (16,32)
    // s_a, 4 half/thread->4 threads/row, 64 rows->256 threads
    // s_b, 2 half/thread->16 threads/row, 16 rows->256 threads
    const int load_smem_a_m = tid / 4; // 0~63
    const int load_smem_a_k = (tid % 4) * 4; // 0, 4, 8, 12

    const int load_smem_b_k = tid / 16; // 0~15
    const int load_smem_b_n = (tid % 16) * 2; // 0, 2, 4, 6, ..., 30

    // define global offset in M, N axis
    const int load_gmem_a_m = BM * bid_M + load_smem_a_m; // global m
    const int load_gmem_b_n = BN * bid_N + load_smem_b_n; // global n

    if (load_gmem_a_m >= M && load_gmem_b_n >= N) {
        return;
    }

    // define fragment C and initialize
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> C_frag;
    nvcuda::wmma::fill_fragment(C_frag, 0.0);
    
    // process
#pragma unroll
    for (int k_id = 0; k_id < NUM_K_TIMES; ++k_id) {
        // define global col offset of matrix A
        int load_gmem_a_k = k_id * WMMA_K + load_smem_a_k;

        // define global row offset of matrix B
        int load_gmem_b_k = k_id * WMMA_K + load_smem_b_k;

        // load data from global memory in shared memory
        LDST64BITS(s_a[load_smem_a_m][load_smem_a_k]) = LDST64BITS(A[load_gmem_a_m * K + load_gmem_a_k]);

        LDST32BITS(s_b[load_smem_b_k][load_smem_b_n]) = LDST32BITS(B[load_gmem_b_k * N + load_gmem_b_n]);

        __syncthreads();

        // define fragment A, B
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, nvcuda::wmma::row_major> A_frag;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, nvcuda::wmma::row_major> B_frag;

        // load data from s_a into fragment A
        nvcuda::wmma::load_matrix_sync(A_frag, &s_a[warp_m * WMMA_M][0], BK); // (WMMA_M,BK)

        // load data from matrix B into fragment B
        nvcuda::wmma::load_matrix_sync(B_frag, &s_b[0][warp_n * WMMA_N], BN); // (BK,WMMA_N)

        nvcuda::wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);

        __syncthreads();
    }

    // store result in matrix C
    const int store_gmem_c_m = bid_M * BM + warp_m * WMMA_M;
    const int store_gmem_c_n = bid_N * BN + warp_n * WMMA_N;
    nvcuda::wmma::store_matrix_sync(C + store_gmem_c_m * N + store_gmem_c_n, C_frag, N, nvcuda::wmma::mem_row_major); // (WMMA_M,WMMA_N)

}

void hgemm_wmma_m16n16k16_mma4x2(__half *A, __half *B, __half *C, int M, int N, int K) {
    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;
    constexpr int WMMA_TILE_M = 4;
    constexpr int WMMA_TILE_N = 2;
    dim3 block(256);
    dim3 grid(div_ceil(N, WMMA_N * WMMA_TILE_N), div_ceil(M, WMMA_M * WMMA_TILE_M));

    hgemm_wmma_m16n16k16_mma4x2_kernel<WMMA_M, WMMA_N, WMMA_K, WMMA_TILE_M, WMMA_TILE_N><<<grid, block>>>(A, B, C, M, N, K);
}


int main(int argc, char *argv[]) {
    Tester tester(512, 2048, 1024, 1, 10, 100, true);
    tester.evaluate(hgemm_wmma_m16n16k16_mma4x2, "hgemm_wmma_m16n16k16_mma4x2");
}