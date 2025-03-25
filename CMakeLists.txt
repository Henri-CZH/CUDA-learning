cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(CUDA-Learning LANGUAGES CXX CUDA)

find_package(CUDA 10.0 REQUIRED)
# 需下载安装cuda toolkit，直接搜即可
set(CUDA_PATH ${CUDA_TOOLKIT_ROOT_DIR})

# set compiler flags
set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  -Xcompiler -Wall")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  \
                        -gencode=arch=compute_70,code=\\\"sm_70,compute_70\\\" \
                        -gencode=arch=compute_75,code=\\\"sm_75,compute_75\\\" \
                        -gencode=arch=compute_80,code=\\\"sm_80,compute_80\\\" \
                        -gencode=arch=compute_86,code=\\\"sm_86,compute_86\\\" \
                        ")
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)
#输出的可执行文件保存地址
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
    
set(COMMON_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}
    # ${PROJECT_SOURCE_DIR}/15_gemv
    ${CUDA_PATH}/include
)

set(COMMON_LIB_DIRS
    ${CUDA_PATH}/lib64
)
#指定头文件路径
include_directories(
    ${COMMON_HEADER_DIRS}
)
#指定待链接的lib路径
link_directories(
    ${COMMON_LIB_DIRS}
)
#1
add_executable(L1_hello_world
    L1_hello_world.cu
)
target_link_libraries(
    L1_hello_world PUBLIC
    -lcudart
    -lcudadevrt)
#2
add_executable(L5_sum
    L5_sum.cu
)
target_link_libraries(
    L5_sum PUBLIC
    -lcudart
    -lcudadevrt)
#3
add_executable(L7_vec_add
    L7_vec_add.cu.cu
)
target_link_libraries(
    L7_vec_add PUBLIC
    -lcudart
    -lcudadevrt)
add_executable(L11_vectorized_vec_add
    L11_vectorized_vec_add.cu
)
target_link_libraries(
    L11_vectorized_vec_add PUBLIC
    -lcudart
    -lcudadevrt)
#4
add_executable(L12_device_query
    L12_device_query.cu
)
target_link_libraries(
    L12_device_query PUBLIC
    -lcudart
    -lcudadevrt)
#5.reduce
add_executable(L18_reduce_baseline
    L18_reduce_baseline.cu
)
target_link_libraries(
    L18_reduce_baseline PUBLIC
    -lcudart
    -lcudadevrt)
add_executable(L19_reduce_v0
    L19_reduce_v0.cu
)
target_link_libraries(
    L19_reduce_v0 PUBLIC
    -lcudart
    -lcudadevrt)
add_executable(L20_reduce_v1
    L20_reduce_v1.cu
)
target_link_libraries(
    L20_reduce_v1 PUBLIC
    -lcudart
    -lcudadevrt)
add_executable(L21_reduce_v2
    L21_reduce_v2.cu
)
target_link_libraries(
    L21_reduce_v2.cu PUBLIC
    -lcudart
    -lcudadevrt)
add_executable(L22_reduce_v3
    L22_reduce_v3.cu
)
target_link_libraries(
    L22_reduce_v3 PUBLIC
    -lcudart
    -lcudadevrt)
add_executable(L23_reduce_v4.cu
    L23_reduce_v4.cu
)
target_link_libraries(
    L23_reduce_v4 PUBLIC
    -lcudart
    -lcudadevrt)
add_executable(L24_reduce_v5
    L24_reduce_v5.cu
)
target_link_libraries(
    L24_reduce_v5 PUBLIC
    -lcudart
    -lcudadevrt)
add_executable(L25_reduce_v6
    L25_reduce_v6.cu
)
target_link_libraries(
    L25_reduce_v6 PUBLIC
    -lcudart
    -lcudadevrt)
#6
add_executable(L26_warp_level_reduce
    L26_warp_level_reduce.cu
)
target_link_libraries(
    L26_warp_level_reduce PUBLIC
    -lcudart
    -lcudadevrt)
#7
add_executable(L28_histogram
    L28_histogram.cu
)
target_link_libraries(
    L28_histogram PUBLIC
    -lcudart
    -lcudadevrt)

add_executable(L29_histogram_
    L29_histogram_.cu
)
target_link_libraries(
    L29_histogram_ PUBLIC
    -lcudart
    -lcudadevrt)
#8
add_executable(L30_copy_if
    L30_copy_if.cu
)
target_link_libraries(
    L30_copy_if PUBLIC
    -lcudart
    -lcudadevrt)
#9
add_executable(L32_FP16_gelu
    L32_FP16_gelu.cu
)
target_link_libraries(
    L32_FP16_gelu
    -lcudart
    -lcudadevrt)
#10 fp32
add_executable(L34_fused_bias_mask_scale_add_FP16
    L34_fused_bias_mask_scale_add_FP16.cu
)
target_link_libraries(
    L34_fused_bias_mask_scale_add_FP16 PUBLIC
    -lcudart
    -lcudadevrt)
#10 fp16
add_executable(L34_fused_bias_mask_scale_add
    L34_fused_bias_mask_scale_add.cu
)
target_link_libraries(
    L34_fused_bias_mask_scale_add PUBLIC
    -lcudart
    -lcudadevrt)
#11
add_executable(L36_softmax_reduce
    L36_softmax_reduce.cu
)
target_link_libraries(
    L36_softmax_reduce PUBLIC
    -lcudart
    -lcudadevrt)
#12
add_executable(L41_gpu_peak_performance_measure
    L41_gpu_peak_performance_measure.cu
)
target_link_libraries(
    L41_gpu_peak_performance_measure PUBLIC
    -lcudart
    -lcudadevrt)
#13 CUDA stream
add_executable(L45_cuda_stream_overlap
    L45_cuda_stream_overlap.cu
)
target_link_libraries(
    L45_cuda_stream_overlap PUBLIC
    -lcudart
    -lcudadevrt)
add_executable(L45_cuda_stream_vector_add
    L45_cuda_stream_vector_add.cu
)
target_link_libraries(
    L45_cuda_stream_vector_add PUBLIC
    -lcudart
    -lcudadevrt)
# #14
# add_executable(14_quantize
#     14_quantize.cu
# )
# target_link_libraries(
#     14_quantize PUBLIC
#     -lcudart
#     -lcudadevrt)
# #15 gemv
# add_executable(15_1_gemv
#     15_gemv/15_1_fp32_fp16_gemv.cu
# )
# target_link_libraries(
#     15_1_gemv PUBLIC
#     -lcudart
#     -lcudadevrt)
# add_executable(15_2_gemv
#     15_gemv/15_2_fp32_fp16_gemv.cu
# )
# target_link_libraries(
#     15_2_gemv PUBLIC
#     -lcudart
#     -lcudadevrt)
# #16 dropout
# add_executable(16_dropout
#     16_fused_dropout/16_dropout.cu
# )
# target_link_libraries(
#     16_dropout PUBLIC
#     -lcudart
#     -lcudadevrt)