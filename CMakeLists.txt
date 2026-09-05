cmake_minimum_required(VERSION 3.25)
set(CMAKE_TOOLCHAIN_FILE "C:/Development/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "Vcpkg toolchain file")
project(CalculationGraphEngine CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 86)

find_package(CUDAToolkit REQUIRED)
find_package(OpenBLAS REQUIRED)

# core
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
)

target_include_directories(GradCore PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
)

if(MSVC)
    target_compile_options(GradCore PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/Zi /W4 /bigobj>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/bigobj>
    )
endif()

# train
add_executable(train train.cpp)
target_link_libraries(train PRIVATE GradCore)

if(MSVC)
    target_link_options(train PRIVATE /INCREMENTAL:NO)
endif()

# tokenizer
add_executable(tokenizer tokenize.cpp)
target_link_libraries(tokenizer PRIVATE GradCore)

if(MSVC)
    target_link_options(tokenizer PRIVATE /INCREMENTAL:NO)
endif()