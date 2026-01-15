# cpu-bvh-preprocess Specification

## Purpose
TBD - created by archiving change move-bvh-preprocess-cpu. Update Purpose after archive.
## Requirements
### Requirement: CPU BVH construction
The system SHALL build a BVH for the loaded triangle mesh on the CPU before launching any GPU kernels.

#### Scenario: Mesh preprocessing
- **WHEN** a model is loaded successfully
- **THEN** a CPU BVH is built and the resulting node and triangle buffers are prepared for GPU upload

### Requirement: Contiguous GPU buffers
The system SHALL upload BVH nodes and triangle data as contiguous buffers optimized for GPU access.

#### Scenario: Buffer upload
- **WHEN** preprocessing completes
- **THEN** BVH node and triangle buffers are copied to GPU memory without device-side dynamic allocation

### Requirement: Model path required
The system SHALL require a model path for simulation and SHALL not fall back to a default sphere.

#### Scenario: Missing model path
- **WHEN** the user omits `--model`
- **THEN** the application reports an error and exits without running the sweep

