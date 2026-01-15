## 1. Implementation
- [x] 1.1 Add CPU-side BVH data structures and builder (median split) for triangle meshes.
- [x] 1.2 Replace GPU `hitable` world creation with flat triangle and BVH buffers uploaded to device memory.
- [x] 1.3 Update GPU ray traversal kernels to intersect triangles via iterative BVH traversal with an explicit stack.
- [x] 1.4 Remove default sphere world; require `--model` and report an error if missing.
- [x] 1.5 Update documentation to describe the new preprocessing flow and BVH-based tracing.

## 2. Validation
- [x] 2.1 Run a medium mesh (e.g., box or semisphere) to confirm BVH build completes on CPU and tracing runs.
- [x] 2.2 Run a larger mesh to verify the world setup no longer stalls on GPU.
