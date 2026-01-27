#ifndef _HIPIFLY_HPP_
#define _HIPIFLY_HPP_


// Error Handling
#define cudaError_t hipError_t
#define cudaSuccess hipSuccess
#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError

// Initialization and Device Management
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaDeviceReset hipDeviceReset
#define cudaDeviceGetAttribute hipDeviceGetAttribute
#define cudaSetDevice hipSetDevice
#define cudaGetDevice hipGetDevice
#define cudaGetDeviceCount hipGetDeviceCount

// Memory Management
#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaMemcpy hipMemcpy
#define cudaMemcpyToSymbol hipMemcpyToSymbol
#define cudaMemcpyFromSymbol hipMemcpyFromSymbol
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemset hipMemset
#define cudaHostAlloc hipHostMalloc
#define cudaFreeHost hipHostFree
#define cudaHostAllocDefault hipHostMallocDefault
#define cudaMallocAsync hipMallocAsync
#define cudaHostRegisterDefault hipHostRegisterDefault
#define cudaHostRegister hipHostRegister
#define cudaHostUnregister hipHostUnregister
#define cudaMemsetAsync hipMemsetAsync
#define cudaFreeAsync hipFreeAsync
#define cudaMallocHost hipHostMalloc // hipMallocHost is deprecated, and there is no cudaHostMalloc but cudaHostAlloc
#define cudaMallocManaged hipMallocManaged

// Memory Types
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaMemcpyDefault hipMemcpyDefault

// Stream Management
#define cudaStream_t hipStream_t
#define cudaStreamCreate hipStreamCreate
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStreamWaitEvent hipStreamWaitEvent

// Event Management
#define cudaEvent_t hipEvent_t
#define cudaEventCreate hipEventCreate
#define cudaEventDestroy hipEventDestroy
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaEventElapsedTime hipEventElapsedTime
#define cudaEventDisableTiming hipEventDisableTiming
#define cudaEventCreateWithFlags hipEventCreateWithFlags
#define cudaEventBlockingSync hipEventBlockingSync

// Texture and Surface References
#define cudaTextureObject_t hipTextureObject_t
#define cudaCreateTextureObject hipCreateTextureObject
#define cudaDestroyTextureObject hipDestroyTextureObject
#define cudaResourceDesc hipResourceDesc
#define cudaTextureDesc hipTextureDesc

// Unified Memory Management
// NOTE: hipMallocManaged has VERY poor performance on AMD GPUs!
// Prefer explicit device memory + memcpy for performance-critical code.
// #define cudaMallocManaged hipMallocManaged  // Already defined above
#define cudaMemPrefetchAsync hipMemPrefetchAsync

// Cooperative Groups
#define cudaLaunchCooperativeKernel hipLaunchCooperativeKernel

// Kernel Launch Configuration
#define cudaLaunchKernel hipLaunchKernelGGL


// Warp/Wavefront primitives
// AMD GPUs use 64-thread wavefronts vs NVIDIA's 32-thread warps
// Note: HIP intrinsics don't require a mask parameter (it's implicit)
// __shfl_down_sync -> __shfl_down (mask is ignored on AMD)
#define __shfl_down_sync(mask, val, delta) __shfl_down(val, delta)
#define __shfl_sync(mask, val, srcLane) __shfl(val, srcLane)
#define __shfl_up_sync(mask, val, delta) __shfl_up(val, delta)
#define __shfl_xor_sync(mask, val, laneMask) __shfl_xor(val, laneMask)
#define __ballot_sync(mask, predicate) __ballot(predicate)
#define __any_sync(mask, predicate) __any(predicate)
#define __all_sync(mask, predicate) __all(predicate)
#define __activemask() __ballot(1)
// AMD wavefronts execute in lockstep, so __syncwarp is effectively a no-op
// However, for memory visibility we should use __threadfence_block()
#define __syncwarp(mask) __threadfence_block()

// Complex number types
#define cuFloatComplex hipFloatComplex
#define cuDoubleComplex hipDoubleComplex
#define make_cuFloatComplex make_hipFloatComplex
#define make_cuDoubleComplex make_hipDoubleComplex
#define cuCrealf hipCrealf
#define cuCimagf hipCimagf
#define cuCreal hipCreal
#define cuCimag hipCimag
#define cuCaddf hipCaddf
#define cuCsubf hipCsubf
#define cuCmulf hipCmulf
#define cuCdivf hipCdivf
#define cuConjf hipConjf
#define cuCabsf hipCabsf
#define cuCadd hipCadd
#define cuCsub hipCsub
#define cuCmul hipCmul
#define cuCdiv hipCdiv
#define cuConj hipConj
#define cuCabs hipCabs

// Device properties
#define cudaDeviceProp hipDeviceProp_t
#define cudaGetDeviceProperties hipGetDeviceProperties


#endif // _HIPIFLY_HPP_