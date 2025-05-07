#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include "../../include/common/tester.h"
#include "../../include/common/ptx.h"

#define LDST64BITS(value) (reinterpret_cast<float2*>(&(value))[0])
#define LDST32BITS(value) (reinterpret_cast<__half2*>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4*>(&(value))[0])

// only 8 warps[WMMA_TILE_M][WMMA_TILE_N] per block process matrix mat (128,16)*(16,128), one warp process 8 fragments[WARP_TILE_M][WARP_TILE_N]
// double buffer
template<int WMMA_M = 16, int WMMA_N = 16, int WMMA_K = 16, 
        int WMMA_TILE_M = 4, int WMMA_TILE_N = 2, 
        int WARP_TILE_M = 2, int WARP_TILE_N = 4,
        int OFFSET = 0>
__global__ void hgemm_wmma_m16n16k16_mma4x2_warp2x4_dbuf_async_kernel(__half *A, __half *B, __half *C,
                                                                      int M, int N, int K) {
    // define iternation times in K axis
    const int NUM_K_TIMES = div_ceil(K, WMMA_K);

    // define bid in M, N axis
    const int bid_M = blockIdx.y;
    const int bid_N = blockIdx.x;

    // define tid in M, N axis
    const int tid = threadIdx.x + blockDim.x * threadIdx.y; // 0~255

    // define define warp id in M, N axis->4 warp in M axis, 2 warp in N warp within a block
    const int warp_id = tid / 32; // 0~7
    const int lane_id = tid % 32; // 0~31

    const int warp_m = warp_id / 2; // 0~3
    const int warp_n = warp_id % 2; // 0~1

    // define size of fragment C->(BM, BN)
    constexpr int BM = WMMA_M * WMMA_TILE_M * WARP_TILE_M; // 16x4x2=128
    constexpr int BN = WMMA_N * WMMA_TILE_N * WARP_TILE_N; // 16x2x4=128
    constexpr int BK = WMMA_K; // 16

    // define shared memory w/. [2][BM][BK], [2][BK][BN]
    __shared__ __half s_a[2][BM][BK + OFFSET], s_b[2][BK][BN + OFFSET]; // 128x16x2=4k 4+4=8, 16x128x2=4k, 4+4=8; padding to reduce bank conflicts

    // define shared memory offset in M, K, N axis
    // 256 thread load s_a = (128,16), s_b = (16,128)
    // s_a, 8 half/thread->2 threads/row, 128 rows->256 threads
    // s_b, 8 half/thread->16 threads/row, 16 rows->256 threads
    const int load_smem_a_m = tid / 2; // 0~127
    const int load_smem_a_k = (tid % 2) * 8; // 0, 8

    const int load_smem_b_k = tid / 16; // 0~15
    const int load_smem_b_n = (tid % 16) * 8; // 0, 16, 32, 48, ..., 120

    // define global memory offset in M, N axis
    const int load_gmem_a_m = BM * bid_M + load_smem_a_m; // global m of matrix A, C
    const int load_gmem_b_n = BN * bid_N + load_smem_b_n; // global n of matrix B, C

    if (load_gmem_a_m >= M || load_gmem_b_n >= N) {
        return;
    }

    // define fragment C and initialize
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> C_frag[WARP_TILE_M][WARP_TILE_N];
#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
            nvcuda::wmma::fill_fragment(C_frag[i][j], 0.0);
        }
    }

     // k_id = 0, load data of k_id = 0 in buffer 0
    {
        // define global col offset of matrix A
        int load_gmem_a_k = load_smem_a_k;
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;

        // define global row offset of matrixB
        int load_gmem_b_k = load_smem_b_k;
        int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;

        // load data from global memory in shared memory w/. buffer 0
        uint32_t load_smem_a_ptr = 
            __cvta_generic_to_shared(&s_a[0][load_smem_a_m][load_smem_a_k]);
        CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);

        uint32_t load_smem_b_ptr = 
            __cvta_generic_to_shared(&s_b[0][load_smem_b_k][load_smem_b_n]);
        CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);

        // send command
        CP_ASYNC_COMMIT_GROUP();
        // synchronize all asynchronized command
        CP_ASYNC_WAIT_GROUP(0);
    }

    __syncthreads();

    // process 0 ~ (k_id - 1) wmma
    // iternative load k_id_th tile's data, process (k_id - 1)_th wmma
#pragma unroll
    for (int k_id = 1; k_id < NUM_K_TIMES; ++k_id) {
        // define selected shared memory buffer
        int smem_sel = (k_id - 1) & 1; // k_id=1->0, K_id=2->1, k_id=3->0
        int smem_sel_next = k_id & 1; // k_id=1->1, K_id=2->0, k_id=3->1

        // define global col offset of matrix A
        int load_gmem_a_k = k_id * WMMA_K + load_smem_a_k;

        // define global row offset of matrix B
        int load_gmem_b_k = k_id * WMMA_K + load_smem_b_k;

        // load data from global memory in shared memory
        uint32_t load_smem_a_ptr = 
            __cvta_generic_to_shared(&s_a[smem_sel_next][load_smem_a_m][load_smem_a_k]);
        CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_m * K + load_gmem_a_k], 16);

        uint32_t load_smem_b_ptr = 
            __cvta_generic_to_shared(&s_b[smem_sel_next][load_smem_b_k][load_smem_b_n]);
        CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_k * N + load_gmem_b_n], 16);

        // send command
        CP_ASYNC_COMMIT_GROUP();

        // define fragment A, B
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, nvcuda::wmma::row_major> A_frag[WARP_TILE_M];
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, nvcuda::wmma::row_major> B_frag[WARP_TILE_N];

        // load 2 tile->reg, smem a->frags a, warp_m 0~3
#pragma unroll
        for (int i = 0; i < WARP_TILE_M; ++i) {
            const int warp_smem_a_m = warp_m * WMMA_M * WARP_TILE_M + i * WMMA_M;
            nvcuda::wmma::load_matrix_sync(A_frag[i], &s_a[smem_sel][warp_smem_a_m][0], BK + OFFSET); // (WMMA_M,BK)
        }

        // load 4 tile->reg, smem b->frags b, warp_n 0~1
#pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
            const int warp_smem_b_n = warp_n * WMMA_N * WARP_TILE_N + j * WMMA_N;
            nvcuda::wmma::load_matrix_sync(B_frag[j], &s_b[smem_sel][0][warp_smem_b_n], BN + OFFSET); // (BK,WMMA_N)
        }

#pragma unroll
        for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
            for (int j = 0; j < WARP_TILE_N; ++j) {
                // mma: 2 mma in m axis, 4 mma in n axis
                nvcuda::wmma::mma_sync(C_frag[i][j], A_frag[i], B_frag[j], C_frag[i][j]);
            }
        }

        // synchronize all asynchronized command
        CP_ASYNC_WAIT_GROUP(0);

        __syncthreads();
    }


    // process last kth wmma
    {
        // define fragment A, B
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, nvcuda::wmma::row_major> A_frag[WARP_TILE_M];
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, nvcuda::wmma::row_major> B_frag[WARP_TILE_N];

        // load 2 tile->reg, smem a->frags a, warp_m 0~3
#pragma unroll
        for (int i = 0; i < WARP_TILE_M; ++i) {
            const int warp_smem_a_m = warp_m * WMMA_M * WARP_TILE_M + i * WMMA_M;
            nvcuda::wmma::load_matrix_sync(A_frag[i], &s_a[1][warp_smem_a_m][0], BK + OFFSET); // (WMMA_M,BK)
        }

        // load 4 tile->reg, smem b->frags b, warp_n 0~1
#pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
            const int warp_smem_b_n = warp_n * WMMA_N * WARP_TILE_N + j * WMMA_N;
            nvcuda::wmma::load_matrix_sync(B_frag[j], &s_b[1][0][warp_smem_b_n], BN + OFFSET); // (BK,WMMA_N)
        }

#pragma unroll
        for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
            for (int j = 0; j < WARP_TILE_N; ++j) {
                // mma: 2 mma in m axis, 4 mma in n axis
                nvcuda::wmma::mma_sync(C_frag[i][j], A_frag[i], B_frag[j], C_frag[i][j]);
            }
        }
    }

    // store result in matrix C
#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
            // store: 2 matrix (16,16) in m axis, 4 matrix (16,16) in n axis
            const int store_gmem_c_m = bid_M * BM + warp_m * WMMA_M * WARP_TILE_M + i * WMMA_M;
            const int store_gmem_c_n = bid_N * BN + warp_n * WMMA_N * WARP_TILE_N + j * WMMA_N;
            nvcuda::wmma::store_matrix_sync(C + store_gmem_c_m * N + store_gmem_c_n, C_frag[i][j], N, nvcuda::wmma::mem_row_major); // (WMMA_M,WMMA_N)
        }
    }

}

void hgemm_wmma_m16n16k16_mma4x2_warp2x4_dbuf_async(__half *A, __half *B, __half *C, int M, int N, int K) {
    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;
    constexpr int WMMA_TILE_M = 4;
    constexpr int WMMA_TILE_N = 2;
    constexpr int WARP_TILE_M = 2;
    constexpr int WARP_TILE_N = 4;
    constexpr int OFFSET = 0;
    dim3 block(256);
    dim3 grid(div_ceil(N, WMMA_N * WMMA_TILE_N * WARP_TILE_N), div_ceil(M, WMMA_M * WMMA_TILE_M * WARP_TILE_M));

    hgemm_wmma_m16n16k16_mma4x2_warp2x4_dbuf_async_kernel<WMMA_M, WMMA_N, WMMA_K,
                                                          WMMA_TILE_M, WMMA_TILE_N,
                                                          WARP_TILE_M, WARP_TILE_N,
                                                          OFFSET><<<grid, block>>>(A, B, C, M, N, K);
}

int main(int argc, char *argv[]) {
    Tester tester(512, 2048, 1024, 1, 10, 100, true);
    tester.evaluate(hgemm_wmma_m16n16k16_mma4x2_warp2x4_dbuf_async, "my_hgemm_v4_wmma_m16n16k16_mma4x2_warp2x4_async");
}