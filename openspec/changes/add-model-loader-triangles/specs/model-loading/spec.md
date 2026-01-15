## ADDED Requirements
### Requirement: CLI model selection
The system SHALL accept a command line argument to specify a model file path for simulation.

#### Scenario: User passes a model path
- **WHEN** the user provides `--model <path>` on the command line
- **THEN** the application attempts to load the mesh at that path for the simulation

### Requirement: Model file type validation
The system SHALL determine the model type from the file extension and reject unsupported extensions.

#### Scenario: Supported OBJ file
- **WHEN** `--model` ends with `.obj`
- **THEN** the OBJ loader is used to parse the mesh

#### Scenario: Supported GLTF or GLB file
- **WHEN** `--model` ends with `.gltf` or `.glb`
- **THEN** the GLTF loader is used to parse the mesh

#### Scenario: Unsupported extension
- **WHEN** `--model` has an unsupported extension
- **THEN** the application reports an error and exits before starting the sweep

### Requirement: Uniform model scaling
The system SHALL allow an optional uniform scale factor for model vertices.

#### Scenario: Scale specified
- **WHEN** the user provides `--model-scale <float>`
- **THEN** mesh positions are multiplied by that factor before building triangles

#### Scenario: Scale omitted
- **WHEN** no model scale is provided
- **THEN** a scale factor of 1.0 is applied

### Requirement: Model load failure handling
The system SHALL report a clear error and abort the simulation when the mesh cannot be loaded or contains no triangles.

#### Scenario: Invalid or empty mesh
- **WHEN** `modelPath` is unreadable, unsupported, or produces zero triangle indices
- **THEN** the application prints an error and exits without running the sweep

### Requirement: Default geometry fallback
The system SHALL use the existing default geometry when `modelPath` is empty.

#### Scenario: No model path configured
- **WHEN** `modelPath` is empty or not provided
- **THEN** the application runs the simulation with the default sphere world
