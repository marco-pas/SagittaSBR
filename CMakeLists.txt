cmake_minimum_required(VERSION 3.25)

project(RT-RCS
    VERSION 0.1.0
    LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89;90")
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

find_package(CUDAToolkit REQUIRED)

option(enableWarningsAsErrors "Treat compiler warnings as errors" OFF)

add_library(projectOptions INTERFACE)

target_include_directories(projectOptions
    INTERFACE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/model"
)

target_compile_features(projectOptions
    INTERFACE
        cxx_std_23
)

target_compile_options(projectOptions
    INTERFACE
        -Wall -Wextra 
)

if(enableWarningsAsErrors)
    target_compile_options(projectOptions
        INTERFACE
            -Werror
    )
endif()

# target_compile_options(projectOptions
#     INTERFACE
#         $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
# )

target_link_libraries(projectOptions
    INTERFACE
        CUDA::cudart
        CUDA::cuda_driver
        CUDA::cublas
        CUDA::curand
)

add_executable(RT-RCS
    src/RT-RCS.cpp
    src/modelLoader.cpp
    src/cuda/rayPool.cu
    src/cuda/tracing.cu
)

target_link_libraries(RT-RCS
    PRIVATE
        projectOptions
)