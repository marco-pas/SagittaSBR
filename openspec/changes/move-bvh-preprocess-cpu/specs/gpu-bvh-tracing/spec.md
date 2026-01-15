## ADDED Requirements
### Requirement: Iterative BVH traversal
The system SHALL traverse the BVH on the GPU using an explicit stack and SHALL avoid recursion.

#### Scenario: BVH traversal on GPU
- **WHEN** a ray is traced through the scene
- **THEN** the kernel iteratively traverses the BVH using a fixed-size stack

### Requirement: Triangle intersection via BVH
The system SHALL compute ray-triangle intersections by traversing BVH nodes and testing candidate triangles in leaf nodes.

#### Scenario: Leaf triangle testing
- **WHEN** a ray reaches a BVH leaf
- **THEN** all triangles in the leaf are tested and the closest hit is selected

### Requirement: No device-side dynamic allocation
The system SHALL avoid device-side `new` or virtual dispatch for world construction and traversal.

#### Scenario: GPU kernel launch
- **WHEN** ray tracing kernels launch
- **THEN** they operate on prebuilt buffers without allocating geometry or BVH nodes on the GPU
