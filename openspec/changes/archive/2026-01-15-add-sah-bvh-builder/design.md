## Context
The current BVH builder uses a simple median split, which is easy to implement but produces higher traversal cost on complex meshes. The project prioritizes performance and will likely scale to larger models or multi-node workflows.

## Goals / Non-Goals
- Goals:
  - Build higher-quality BVHs using a binned SAH splitter.
  - Parallelize BVH construction on CPU.
  - Provide a stable options interface for later tuning.
- Non-Goals:
  - GPU-side BVH rebuilds.
  - SAH over full primitive sets without binning.
  - Changing the GPU traversal algorithm.

## Decisions
- Use a binned SAH builder with a fixed bin count (default 16) and leaf size (default 4) stored in a `bvhBuildOptions` struct.
- Keep a simple splitter available and add a code-level switch to select the BVH algorithm.
- Implement parallel construction with OpenMP tasks (fallback to single-thread when OpenMP is unavailable).
- Keep a deterministic build order for reproducibility when parallelism is disabled.
- Expose options through a CPU-side interface (no CLI yet), leaving room for future tuning.

## Risks / Trade-offs
- SAH build time is higher than median split, but expected to reduce traversal cost on large meshes.
- Parallel recursion needs cutoffs to avoid overhead; defaults will be conservative.

## Migration Plan
- Update `buildBvh` to accept options and use SAH; adjust call sites accordingly.
- Add OpenMP detection in build configuration.

## Open Questions
- None.
