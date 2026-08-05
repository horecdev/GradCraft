cmake_minimum_required(VERSION 3.25)
project(CalculationGraphEngine CXX CUDA)

# Keep your C++ frontend on C++23
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Force CUDA compilation units (.cu files) to use C++20 standard internally 
# This prevents CMake from looking for a broken "CUDA23" dialect
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 86)

add_executable(tensor 
    main.cpp
    gradc/gradc.hpp
    gradc/backend/cuda/kernels_math.cu
    gradc/backend/cuda/kernels_utils.cu
)

# Find the CUDA toolkit installation
find_package(CUDAToolkit REQUIRED)
find_package(OpenBLAS REQUIRED)

# Link the CUDA runtime so .cpp and .hpp files can see <cuda_runtime.h>
target_link_libraries(tensor PRIVATE 
    CUDA::cudart
    OpenBLAS::OpenBLAS
)

# Tell the compiler to look in the root directory, so "gradc/..." paths work
target_include_directories(tensor PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

if(MSVC)
    # Generator expressions prevent passing /fsanitize=address to nvcc
    target_compile_options(tensor PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/Zi /W4>)
    target_link_options(tensor PRIVATE /INCREMENTAL:NO)
else()
    target_compile_options(tensor PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-fsanitize=address -g -fno-omit-frame-pointer>)
    target_link_options(tensor PRIVATE -fsanitize=address)
endif()