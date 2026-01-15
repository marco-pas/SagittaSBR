# Change: Move world preprocessing and BVH build to CPU

## Why
Large meshes stall during GPU-side world creation due to single-threaded kernel construction and device-side dynamic allocation. Moving preprocessing and BVH construction to the CPU enables faster, more scalable setup and allows the GPU to focus on ray traversal.

## What Changes
- Build triangle geometry and BVH on the CPU after model loading.
- Copy compact BVH and triangle buffers to GPU memory for tracing.
- Remove GPU-side dynamic allocation and `hitable` construction kernels.
- Require a model path for simulation; remove the default sphere fallback.
- Use iterative BVH traversal on GPU with an explicit stack.

## Impact
- Affected specs:
  - `cpu-bvh-preprocess` (new)
  - `gpu-bvh-tracing` (new)
- Affected code:
  - `src/app/rtRcsEntry.cu`
  - `src/scene/world.cu`
  - `include/scene/world.cuh`
  - `include/RT/hitable.h`
  - `include/RT/hitable_list.h`
  - `include/RT/triangle.h`
  - `include/cuda/rtKernels.cuh`
  - `src/cuda/rtKernels.cu`
  - `include/app/config.hpp`
  - `src/app/config.cpp`
  - `README.md`
