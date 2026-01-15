## 1. Implementation
- [x] 1.1 Extend `simulationConfig` with `modelPath` (string) and `modelScale` (float), and update CLI parsing to populate them while preserving the current positional frequency argument.
- [x] 1.2 Add a GPU-friendly `triangle` primitive that implements `hitable::hit` with a ray-triangle intersection test.
- [x] 1.3 Add device memory ownership for mesh vertex and index buffers, plus a `createWorldFromMesh` kernel that builds a `hitableList` from triangles.
- [x] 1.4 Integrate model loading in `runRcsApp`: validate extension, load mesh with `modelLoader`, apply `modelScale`, transfer buffers, and choose between mesh or default sphere based on `modelPath`.
- [x] 1.5 Update documentation to show CLI usage for mesh-based simulation (model path and optional scale).

## 2. Validation
- [x] 2.1 Run a minimal mesh (single triangle OBJ) and confirm `rcs_results.csv` is produced without CUDA errors.
- [x] 2.2 Run the default sphere path to confirm unchanged behavior when `modelPath` is empty.
