# Project Context

## Purpose
Develop a high-accuracy and high-performance electromagnetic simulation tool for radar RCS using the SBR ray-tracing method, with GPU acceleration as a core focus.

## Tech Stack
- C++23 (primary language, modern features encouraged)
- CUDA (GPU acceleration, currently CUDA cores; RT cores planned)
- CMake (build system)
- Planned model loaders: tinyOBJLoader, tinyGLTF (not integrated yet)

## Project Conventions

### Code Style
- Use lower camel naming convention for identifiers unless language or library conventions require otherwise.
- Use English for all code and documentation annotations.
- Prefer clear, self-explanatory code; add brief comments only for non-obvious logic.

### Architecture Patterns
- Core focus on accurate SBR ray-tracing physics and GPU performance.
- Keep simulation kernels and math utilities modular to enable future RT core integration.
- Separate model loading/parsing from simulation execution; loaders are optional modules.

### Testing Strategy
- Prioritize correctness validation for known analytic cases (e.g., sphere RCS).
- Add performance regression checks for GPU kernels where practical.
- TBD: automated unit and integration test framework selection.

### Git Workflow
- TBD: define branching strategy and commit message conventions.

## Domain Context
- The simulator targets electromagnetic RCS computation using SBR (Shooting and Bouncing Rays).
- Current geometry support is limited to simple spheres; complex mesh support is planned.

## Important Constraints
- Accuracy and performance are top priorities; avoid changes that trade correctness for convenience.
- GPU acceleration is mandatory; CPU-only fallbacks are out of scope.

## External Dependencies
- CUDA toolkit and GPU drivers.
- Planned: tinyOBJLoader and tinyGLTF for model loading (not yet coupled).
