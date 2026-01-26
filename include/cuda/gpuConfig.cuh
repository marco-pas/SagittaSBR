#ifndef GPU_CONFIG_CUH
#define GPU_CONFIG_CUH

// Platform-specific warp/wavefront size
// NVIDIA uses 32-thread warps, AMD uses 64-thread wavefronts
#if defined(USE_HIP) || defined(__HIP_PLATFORM_AMD__)
    #define GPU_WARP_SIZE 64
#else
    #define GPU_WARP_SIZE 32
#endif

// Common block sizes that work well on both platforms
// These are multiples of both 32 and 64
#define GPU_BLOCK_SIZE_1D 256
#define GPU_BLOCK_SIZE_2D_X 16
#define GPU_BLOCK_SIZE_2D_Y 16

// Helper macro for grid calculation
#define GPU_GRID_SIZE(n, blockSize) (((n) + (blockSize) - 1) / (blockSize))

// Compile-time warp size for device code
#ifdef __CUDA_ARCH__
    // CUDA provides warpSize as a built-in, but it's a runtime value
    // For compile-time constants, use GPU_WARP_SIZE
#endif

#ifdef __HIP_DEVICE_COMPILE__
    // HIP wavefront size - typically 64 for AMD GCN/RDNA
#endif

#endif // GPU_CONFIG_CUH
