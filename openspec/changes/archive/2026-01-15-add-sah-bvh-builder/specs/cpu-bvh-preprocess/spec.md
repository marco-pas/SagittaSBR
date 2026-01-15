## MODIFIED Requirements
### Requirement: CPU BVH construction
The system SHALL build a BVH for the loaded triangle mesh on the CPU using a binned SAH splitter before launching any GPU kernels.

#### Scenario: Mesh preprocessing with SAH
- **WHEN** a model is loaded successfully
- **THEN** a binned SAH BVH is built and the resulting node and triangle buffers are prepared for GPU upload

### Requirement: Contiguous GPU buffers
The system SHALL upload BVH nodes and triangle data as contiguous buffers optimized for GPU access.

#### Scenario: Buffer upload
- **WHEN** preprocessing completes
- **THEN** BVH node and triangle buffers are copied to GPU memory without device-side dynamic allocation

## ADDED Requirements
### Requirement: Parallel BVH construction
The system SHALL support parallel BVH construction on the CPU when available.

#### Scenario: Parallel build enabled
- **WHEN** BVH construction runs with CPU parallelism enabled
- **THEN** BVH nodes are built using multiple CPU threads

### Requirement: Tunable BVH build options
The system SHALL expose code-level options to tune BVH build parameters.

#### Scenario: Default build options
- **WHEN** no custom options are provided
- **THEN** the builder uses default bin count, leaf size, and parallel cutoffs suitable for general meshes

### Requirement: Selectable BVH algorithm
The system SHALL allow selecting between a simple splitter and the SAH splitter via a code-level option.

#### Scenario: Simple splitter selected
- **WHEN** the build option selects the simple splitter
- **THEN** BVH construction uses the non-SAH algorithm

#### Scenario: SAH splitter selected
- **WHEN** the build option selects the SAH splitter
- **THEN** BVH construction uses the binned SAH algorithm
