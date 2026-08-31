cmake_minimum_required(VERSION 3.25)
set(CMAKE_TOOLCHAIN_FILE "C:/Development/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "Vcpkg toolchain file")
project(CalculationGraphEngine CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 86)

add_executable(tensor 
    main.cpp
    gradc/gradc.hpp
    gradc/backend/cuda/kernels_math.cu
    gradc/backend/cuda/kernels_utils.cu
    gradc/backend/cuda/cudnn_ops.cu
)

find_package(CUDAToolkit REQUIRED)
find_package(OpenBLAS REQUIRED)

set(CUDNN_ROOT "C:/Program Files/NVIDIA/CUDNN/v9.25")

target_include_directories(tensor PRIVATE "${CUDNN_ROOT}/include/13.4")

target_link_directories(tensor PRIVATE "${CUDNN_ROOT}/lib/13.4/x64")

target_link_libraries(tensor PRIVATE 
    CUDA::cudart
    CUDA::cublas
    CUDA::curand
    CUDA::nvrtc
    OpenBLAS::OpenBLAS
    cudnn
)

target_include_directories(tensor PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/cudnn_frontend/include
)

if(MSVC)
    target_compile_options(tensor PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/Zi /W4>)
    target_link_options(tensor PRIVATE /INCREMENTAL:NO)
else()
    target_compile_options(tensor PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-fsanitize=address -g -fno-omit-frame-pointer>)
    target_link_options(tensor PRIVATE -fsanitize=address)
endif()