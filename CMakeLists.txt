cmake_minimum_required(VERSION 3.25)
project(GradCraft CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# default to rtx 30 series
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

find_package(CUDAToolkit REQUIRED)
find_package(OpenBLAS REQUIRED)
find_package(OpenMP REQUIRED)

# core lib
add_library(GradCore STATIC
    gradc/gradc.hpp
    gradc/backend/cuda/kernels_math.cu
    gradc/backend/cuda/kernels_utils.cu
)

target_link_libraries(GradCore PUBLIC 
    CUDA::cudart
    CUDA::cublas
    CUDA::curand
    CUDA::nvrtc
    OpenBLAS::OpenBLAS
    OpenMP::OpenMP_CXX
)

target_include_directories(GradCore PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
)

if(MSVC)
    target_compile_options(GradCore PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/Zi /W4 /bigobj /O2 /fp:fast>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/bigobj,-O2,-fp:fast -O3 -use_fast_math>
    )
endif()

# execs
add_executable(train train.cpp)
target_link_libraries(train PRIVATE GradCore)

add_executable(tokenizer tokenize.cpp)
target_link_libraries(tokenizer PRIVATE GradCore)

add_executable(inference inference.cpp)
target_link_libraries(inference PRIVATE GradCore)

# benchmarks
add_executable(bench_cpu_gpu benchmarks/bench_cpu_vs_gpu.cpp)
target_link_libraries(bench_cpu_gpu PRIVATE GradCore)

add_executable(bench_training benchmarks/bench_training.cpp)
target_link_libraries(bench_training PRIVATE GradCore)

add_executable(bench_kernels benchmarks/bench_kernels.cpp)
target_link_libraries(bench_kernels PRIVATE GradCore)

if(MSVC)
    target_link_options(train PRIVATE /INCREMENTAL:NO)
    target_link_options(tokenizer PRIVATE /INCREMENTAL:NO)
    target_link_options(inference PRIVATE /INCREMENTAL:NO)
    target_link_options(bench_cpu_gpu PRIVATE /INCREMENTAL:NO)
    target_link_options(bench_training PRIVATE /INCREMENTAL:NO)
    target_link_options(bench_kernels PRIVATE /INCREMENTAL:NO)
    
endif()