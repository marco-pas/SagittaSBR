## 1. Implementation
- [x] 1.1 Add a `bvhBuildOptions` struct (leaf size, bin count, parallel depth/threshold) and update `buildBvh` signature.
- [x] 1.2 Implement a binned SAH splitter for BVH construction.
- [x] 1.3 Add a code-level switch to select between the simple splitter and the SAH splitter.
- [x] 1.4 Add CPU parallelization for BVH build with OpenMP tasks or std::thread fallback.
- [x] 1.5 Update call sites to pass options and document the new defaults.
- [x] 1.6 Update build configuration to enable OpenMP when available.

## 2. Validation
- [x] 2.1 Build the project with and without OpenMP enabled.
- [x] 2.2 Run a medium and large mesh to confirm BVH build completes and performance improves or remains stable.
