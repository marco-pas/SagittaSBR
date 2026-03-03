#ifndef GPU_CONFIG_CUH
#define GPU_CONFIG_CUH

// @@
// GPU CONFIGURATION - Centralized settings for AMD and NVIDIA GPUs
// All GPU-specific tuning parameters should be defined here for maintainability.
// Different architectures have different optimal configurations.
// @@

#if defined(USE_HIP) || defined(__HIP_PLATFORM_AMD__)
    // @@
    // AMD GPU Configuration (MI250X / CDNA2 / gfx90a)
    // - Wavefront size: 64 threads (vs 32 for NVIDIA)
    // - 110 compute units, each can run up to 32 wavefronts
    // - Higher register count per thread (256 VGPRs)
    // - Prefer block sizes that are multiples of 64
    // @@
    
    #define GPU_WARP_SIZE 64
    
    // --- Ray Tracing Kernel (2D grid, BVH traversal)
    // 16x16 = 256 threads = 4 wavefronts per block
    // MI250X needs multiple wavefronts per CU to hide HBM latency
    // during BVH traversal (random memory access pattern).
    // 1 wavefront/block (64 threads) starves the memory pipeline.
    #define GPU_RT_BLOCK_X 8
    #define GPU_RT_BLOCK_Y 16
    #define GPU_RT_BLOCK_SIZE (GPU_RT_BLOCK_X * GPU_RT_BLOCK_Y)  // 256
    
    // --- Physical Optics Kernel (1D grid, reduction)
    // 256 threads = 4 wavefronts per block
    // Benefits: Good occupancy, efficient warp-level reductions
    #define GPU_PO_BLOCK_SIZE 256
    
    // --- Generic 1D kernels
    #define GPU_BLOCK_SIZE_1D 256
    
    // --- Maximum warps per block for shared memory sizing
    // 256 threads / 64 wavefront = 4 warps max
    #define GPU_MAX_WARPS_PER_BLOCK 4
    
    // --- Warp shuffle intrinsics
    #define GPU_SHFL_DOWN(val, offset) __shfl_down(val, offset)
    #define GPU_SHFL_XOR(val, mask) __shfl_xor(val, mask)
    
#else
    // @@
    // NVIDIA GPU Configuration (Ampere/Ada/Hopper)
    // - Warp size: 32 threads
    // - Requires explicit sync mask for warp shuffles
    // - Standard 16x16 blocks work well for most kernels
    // @@
    
    #define GPU_WARP_SIZE 32
    
    // --- Ray Tracing Kernel (2D grid, BVH traversal)
    // 16x16 = 256 threads = 8 warps per block
    // NVIDIA handles divergence well with independent thread scheduling
    #define GPU_RT_BLOCK_X 16
    #define GPU_RT_BLOCK_Y 16
    #define GPU_RT_BLOCK_SIZE (GPU_RT_BLOCK_X * GPU_RT_BLOCK_Y)  // 256
    
    // --- Physical Optics Kernel (1D grid, reduction)
    // 256 threads = 8 warps per block
    #define GPU_PO_BLOCK_SIZE 256
    
    // --- Generic 1D kernels
    #define GPU_BLOCK_SIZE_1D 256
    
    // --- Maximum warps per block for shared memory sizing
    // 256 threads / 32 warp = 8 warps max
    #define GPU_MAX_WARPS_PER_BLOCK 8
    
    // --- Warp shuffle intrinsics (require sync mask on NVIDIA)
    #define GPU_SHFL_DOWN(val, offset) __shfl_down_sync(0xFFFFFFFF, val, offset)
    #define GPU_SHFL_XOR(val, mask) __shfl_xor_sync(0xFFFFFFFF, val, mask)
    
#endif

// Helper Macros (Platform-Independent)

// Grid size calculation: ceiling division
#define GPU_GRID_SIZE(n, blockSize) (((n) + (blockSize) - 1) / (blockSize))

// 2D grid size calculation for RT kernel
#define GPU_RT_GRID_X(nx) (((nx) + GPU_RT_BLOCK_X - 1) / GPU_RT_BLOCK_X)
#define GPU_RT_GRID_Y(ny) (((ny) + GPU_RT_BLOCK_Y - 1) / GPU_RT_BLOCK_Y)

// Legacy compatibility defines (deprecated, use specific kernel defines)
#define GPU_BLOCK_SIZE_2D_X GPU_RT_BLOCK_X
#define GPU_BLOCK_SIZE_2D_Y GPU_RT_BLOCK_Y

#endif // GPU_CONFIG_CUH
