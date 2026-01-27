cmake_minimum_required(VERSION 3.25)

# Option to enable HIP (AMD GPU) instead of CUDA (NVIDIA GPU)
option(USE_HIP "Build with HIP for AMD GPUs instead of CUDA" OFF)

if(USE_HIP)
    project(RT-RCS
        VERSION 0.1.0
        LANGUAGES C CXX HIP)
else()
    project(RT-RCS
        VERSION 0.1.0
        LANGUAGES C CXX CUDA)
endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(USE_HIP)
    set(CMAKE_HIP_STANDARD 17)
    set(CMAKE_HIP_STANDARD_REQUIRED ON)
    
    if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
        set(CMAKE_HIP_ARCHITECTURES "gfx906;gfx908;gfx90a;gfx1030;gfx1100")
    endif()
    
    find_package(hip REQUIRED)
    find_package(rocrand REQUIRED)
    find_package(hipblas REQUIRED)
else()
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
    
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89;90")
    endif()
    
    find_package(CUDAToolkit REQUIRED)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

find_package(OpenMP)

option(enableWarningsAsErrors "Treat compiler warnings as errors" OFF)

add_library(projectOptions INTERFACE)

target_include_directories(projectOptions
    INTERFACE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/model"
)

# Add HIP compatibility header when building with HIP
if(USE_HIP)
    target_compile_definitions(projectOptions INTERFACE USE_HIP)
    target_include_directories(projectOptions
        INTERFACE
            "${CMAKE_CURRENT_SOURCE_DIR}/include/hip"
    )
endif()

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

if(USE_HIP)
    target_link_libraries(projectOptions
        INTERFACE
            hip::host
            roc::rocrand
            roc::hipblas
    )
else()
    target_link_libraries(projectOptions
        INTERFACE
            CUDA::cudart
            CUDA::cuda_driver
            CUDA::cublas
            CUDA::curand
    )
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(projectOptions
        INTERFACE
            OpenMP::OpenMP_CXX
    )
endif()

set(CUDA_SOURCES
    src/app/deviceBuffers.cu
    src/app/rtRcsEntry.cu
    src/app/sweep.cu
    src/cuda/cudaUtils.cu
    src/cuda/rtKernels.cu
    src/cuda/poKernels.cu
)

add_executable(RT-RCS
    src/RT-RCS.cpp
    src/app/config.cpp
    src/app/print.cpp
    src/modelLoader.cpp
    src/scene/bvhBuilder.cpp
    ${CUDA_SOURCES}
)

# Set language for .cu files when using HIP
if(USE_HIP)
    set_source_files_properties(${CUDA_SOURCES} PROPERTIES LANGUAGE HIP)
endif()

target_link_libraries(RT-RCS
    PRIVATE
        projectOptions
)
