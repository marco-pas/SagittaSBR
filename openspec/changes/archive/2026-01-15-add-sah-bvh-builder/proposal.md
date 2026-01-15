# Change: Upgrade CPU BVH build with SAH and parallelization

## Why
The current CPU BVH uses a simple median split, which is fast but yields suboptimal trees for large meshes. A binned SAH builder with parallel construction will improve traversal performance and reduce total runtime on complex models.

## What Changes
- Replace the median-split BVH builder with a binned SAH splitter.
- Parallelize BVH construction on the CPU using OpenMP or standard threads.
- Introduce a builder options interface to tune leaf size, bin count, and parallel thresholds (code-level configuration).
- Provide a code-level switch to select between the simple splitter and the SAH splitter.
- Keep GPU traversal unchanged (still iterative with explicit stack).

## Impact
- Affected specs:
  - `cpu-bvh-preprocess` (modified)
- Affected code:
  - `include/scene/bvhBuilder.hpp`
  - `src/scene/bvhBuilder.cpp`
  - `src/app/rtRcsEntry.cu`
  - `CMakeLists.txt`
  - `README.md`
