# Change: Integrate model loading and triangle geometry

## Why
The simulator currently only supports a hard-coded sphere. To run RCS simulations on real models, the application needs to load mesh files and trace rays against triangle geometry. This change enables basic mesh-based simulations first, without acceleration structures.

## What Changes
- Integrate `modelLoader` so the app can load OBJ/GLTF/GLB meshes from a CLI-specified path.
- Validate the model file extension to select loader type and reject unsupported files early.
- Add triangle geometry support to the ray tracer and build the world from mesh triangles.
- Keep the first implementation simple: no BVH or mesh acceleration, just linear triangle lists.
- Add an optional uniform model scale parameter (CLI) to fit meshes inside the illumination grid.

## Impact
- Affected specs:
  - `model-loading` (new)
  - `triangle-geometry` (new)
- Affected code:
  - `src/app/rtRcsEntry.cu`
  - `src/app/config.cpp`
  - `include/app/config.hpp`
  - `src/scene/world.cu`
  - `include/scene/world.cuh`
  - `include/RT/hitable.h`
  - `include/RT/hitable_list.h`
  - `include/RT/sphere.h`
  - `include/RT/triangle.h`
  - `include/model/modelLoader.hpp`
  - `src/modelLoader.cpp`
