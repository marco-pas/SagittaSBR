## Context
The current mesh path creates one `triangle` per face on the GPU using a single kernel and device-side `new`, then relies on `hitableList` with virtual dispatch for intersection tests. This stalls or times out on large meshes and does not scale.

## Goals / Non-Goals
- Goals:
  - Build triangle geometry and BVH on the CPU.
  - Upload compact, contiguous buffers to the GPU for tracing.
  - Use iterative BVH traversal with an explicit stack in GPU kernels.
  - Remove GPU-side dynamic allocation and `hitable` polymorphism.
  - Require a model path for all simulations (no default sphere fallback).
- Non-Goals:
  - SAH or other expensive BVH build optimizations (use a simpler splitter first).
  - Multi-model scenes or instancing.
  - CPU-only tracing.

## Decisions
- Use a simple CPU BVH builder (median split or equal-count partition) to create a binary tree.
- Store triangles and BVH nodes in contiguous GPU buffers with a layout aligned to 16 bytes for coalesced access.
- Represent BVH nodes with bounding boxes plus child indices or leaf ranges (industry-standard flat array layout).
- Traverse BVH on GPU with an explicit stack, avoiding recursion.
- Remove `createWorld`/`createWorldFromMesh` kernels and related `hitable` allocations from the runtime path.

## Risks / Trade-offs
- CPU preprocessing time increases, but overall wall-clock time improves for large meshes.
- Iterative traversal requires a fixed stack size; deep trees may need a conservative limit.

## Migration Plan
- Introduce new CPU-side BVH builder and GPU buffers while keeping the public CLI stable.
- Remove the default sphere fallback and require `--model` for simulation runs.

## Open Questions
- None.
