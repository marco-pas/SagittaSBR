## Context
The current scene setup builds a single sphere on the GPU in `createWorld`. A CPU-side model loader exists (`modelLoader`) but is not connected to the simulation. The ray tracer uses `hitable` objects and a `hitableList` to perform linear hit tests.

## Goals / Non-Goals
- Goals:
  - Load OBJ/GLTF/GLB meshes via `modelLoader` and use them as simulation geometry.
  - Add a triangle primitive with correct ray-triangle intersection.
  - Keep the implementation minimal and compatible with the current `hitable` workflow.
- Non-Goals:
  - BVH or any acceleration structure.
  - Materials, textures, or polarization modeling.
  - Mesh instancing or complex scene graphs.

## Decisions
- Load mesh data on the CPU in `runRcsApp` using `modelLoader`, then transfer positions and indices to GPU-accessible memory.
- Introduce a `triangle` class that implements `hitable::hit` using a standard Moller-Trumbore intersection test.
- Build the world by instantiating one `triangle` per face and storing them in `hitableList` (linear traversal).
- Add `modelPath` and `modelScale` fields to `simulationConfig`, populated from CLI flags (no config file changes required).
- Validate the file extension in CLI parsing to select OBJ vs GLTF/GLB, and reject unsupported extensions before loading.
- If per-vertex normals are missing, compute face normals from triangle vertices for hit records.

## Risks / Trade-offs
- Linear triangle traversal will be slow for large meshes, but this aligns with the minimal feature goal.
- GPU memory usage will grow with triangle count; large meshes may exceed available memory.

## Migration Plan
- No migration is required. Existing workflows remain valid if no `modelPath` is provided.

## Open Questions
- None.
