#ifndef GPU_CONFIG_CUH
#define GPU_CONFIG_CUH

// Platform-specific warp/wavefront size
// NVIDIA uses 32-thread warps, AMD uses 64-thread wavefronts
#if defined(USE_HIP) || defined(__HIP_PLATFORM_AMD__)
    #define GPU_WARP_SIZE 64
    // AMD MI250X (gfx90a) optimal configuration:
    // - 64-thread wavefronts
    // - 110 compute units, each can run up to 32 wavefronts
    // - Prefer block sizes that are multiples of 64
    #define GPU_BLOCK_SIZE_1D 256
    // For 2D kernels, use 8x8=64 (one wavefront) or 16x16=256 (4 wavefronts)
    // 8x8 minimizes divergence within a wavefront for spatial locality
    #define GPU_BLOCK_SIZE_2D_X 16
    #define GPU_BLOCK_SIZE_2D_Y 16
#else
    #define GPU_WARP_SIZE 32
    // NVIDIA optimal configuration:
    // - 32-thread warps
    // - Standard 16x16 = 256 threads works well
    #define GPU_BLOCK_SIZE_1D 256
    #define GPU_BLOCK_SIZE_2D_X 16
    #define GPU_BLOCK_SIZE_2D_Y 16
#endif

// Helper macro for grid calculation
#define GPU_GRID_SIZE(n, blockSize) (((n) + (blockSize) - 1) / (blockSize))

// Compile-time warp size for device code
#ifdef __CUDA_ARCH__
    // CUDA provides warpSize as a built-in, but it's a runtime value
    // For compile-time constants, use GPU_WARP_SIZE
#endif

#ifdef __HIP_DEVICE_COMPILE__
    // HIP wavefront size - typically 64 for AMD GCN/CDNA
    // MI250X uses CDNA2 architecture with 64-wide wavefronts
#endif

#endif // GPU_CONFIG_CUH
