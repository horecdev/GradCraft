cmake_minimum_required(VERSION 3.25)
project(GradCraft CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)

find_package(CUDAToolkit REQUIRED)
find_package(OpenBLAS REQUIRED)
find_package(OpenMP COMPONENTS CXX REQUIRED)

# Core Library
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

# exects
add_executable(train train.cpp)
target_link_libraries(train PRIVATE GradCore)

add_executable(tokenize tokenize.cpp)
target_link_libraries(tokenize PRIVATE GradCore)

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
    set(EXEC_TARGETS train tokenize inference bench_cpu_gpu bench_training bench_kernels)
    foreach(target ${EXEC_TARGETS})
        target_link_options(${target} PRIVATE /INCREMENTAL:NO)
    endforeach()
endif()