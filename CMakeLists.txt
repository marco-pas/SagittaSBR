cmake_minimum_required(VERSION 3.25)

# ============================================================================
# Build Configuration - Default to Release with high optimizations
# ============================================================================
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
endif()

# Option to enable HIP (AMD GPU) instead of CUDA (NVIDIA GPU)
option(USE_HIP "Build with HIP for AMD GPUs instead of CUDA" OFF)

# Option to enable double precision floating point
option(USE_DOUBLE "Use double precision (64-bit) instead of single precision (32-bit)" OFF)


project(RT-RCS
    VERSION 0.1.0
    LANGUAGES C CXX
)

if(USE_HIP)
    enable_language(HIP)
else()
    enable_language(CUDA)
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

find_package(MPI REQUIRED)

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

# Add double precision compile definition
if(USE_DOUBLE)
    target_compile_definitions(projectOptions INTERFACE USE_DOUBLE)
    message(STATUS "Building with DOUBLE precision (64-bit)")
else()
    message(STATUS "Building with SINGLE precision (32-bit)")
endif()

target_compile_features(projectOptions
    INTERFACE
        cxx_std_23
)

# ============================================================================
# Optimization Flags - High performance by default
# ============================================================================
# Release optimization flags for C/C++
target_compile_options(projectOptions
    INTERFACE
        $<$<CONFIG:Release>:-O3 -march=native -funroll-loops>
        $<$<CONFIG:RelWithDebInfo>:-O2 -g>
        $<$<CONFIG:Debug>:-O0 -g>
)

# ============================================================================
# GPU-specific optimization flags
# ============================================================================
#
# The AABB slab test computes 1.0/direction → ±Inf for axis-aligned rays.
# IEEE 754 guarantees fmin(Inf,x)=x and fmax(-Inf,x)=x, which the BVH
# traversal relies on for correctness.
#
# SAFE flags (no physics impact):
#   -ffp-contract=fast    FMA fusion — more accurate, never less
#   -fno-signed-zeros     Treats +0 == -0 (irrelevant for ray tracing)
#   -fno-trapping-math    No FP traps (we never trap)
#   -munsafe-fp-atomics   HW FP atomics on gfx90a instead of CAS loop

if(USE_HIP)
    # HIP (AMD) — same safe flags for both FP32 and FP64
    # MI250X has 1:1 FP64:FP32 throughput, so no need to differentiate
    # compute-related flags. Only memory bandwidth matters.
    target_compile_options(projectOptions
        INTERFACE
            $<$<COMPILE_LANGUAGE:HIP>:
                -O3
                -ffp-contract=fast
                -fno-signed-zeros
                -fno-trapping-math
                -munsafe-fp-atomics
            >
    )
else()
    # CUDA (NVIDIA)
    if(USE_DOUBLE)
        target_compile_options(projectOptions
            INTERFACE
                $<$<COMPILE_LANGUAGE:CUDA>:-O3 --fmad=true>
        )
    else()
        # --use_fast_math on NVIDIA enables HW intrinsics (__fminf/__fmaxf)
        # that handle Inf correctly at the hardware level regardless of
        # the compiler flag, so it is safe for FP32 on NVIDIA.
        target_compile_options(projectOptions
            INTERFACE
                $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
        )
    endif()
endif()

# Warning flags (separate from optimization)
target_compile_options(projectOptions
    INTERFACE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>
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

target_link_libraries(RT-RCS PRIVATE MPI::MPI_CXX)
