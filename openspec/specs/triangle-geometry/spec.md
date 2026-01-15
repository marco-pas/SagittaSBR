# triangle-geometry Specification

## Purpose
TBD - created by archiving change add-model-loader-triangles. Update Purpose after archive.
## Requirements
### Requirement: Triangle hit testing
The system SHALL support triangle primitives for ray intersection and populate hit records with the closest triangle hit.

#### Scenario: Ray hits a triangle
- **WHEN** a ray intersects a triangle inside the `tMin` and `tMax` range
- **THEN** `hit` returns true and populates `hitRecord` with intersection position, distance, and normal

### Requirement: Mesh world construction
The system SHALL build a simulation world from the triangle list by creating one `hitable` per triangle and aggregating them in a `hitableList`.

#### Scenario: World built from mesh triangles
- **WHEN** a mesh with N triangles is loaded
- **THEN** the world contains N triangle hitables and hit testing checks all N triangles

### Requirement: Triangle normal derivation
The system SHALL compute triangle normals from vertex positions when evaluating hits.

#### Scenario: Triangle without vertex normals
- **WHEN** a mesh provides positions but no normals
- **THEN** the triangle normal is computed from the cross product of its edges

