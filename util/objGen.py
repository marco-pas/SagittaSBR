import trimesh
# Create a sphere with a specific radius and subdivision level
sphere = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
sphere.export('./assets/test_models/sphere2.obj')

