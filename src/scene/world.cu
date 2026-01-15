#include "scene/world.cuh"

#include "RT/hitable_list.h"
#include "RT/sphere.h"
#include "RT/triangle.h"

__global__ void createWorld(hitable** deviceList, hitable** deviceWorld) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        deviceList[0] = new sphere(vec3(0.0f, 0.0f, 0.0f), 1.0f);
        *deviceWorld = new hitableList(deviceList, 1);
    }
}

__global__ void freeWorld(hitable** deviceList, hitable** deviceWorld) {
    delete deviceList[0];
    delete *deviceWorld;
}

__global__ void createWorldFromMesh(hitable** deviceList, hitable** deviceWorld,
                                    const vec3* vertices, const int* indices,
                                    int triangleCount) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int i = 0; i < triangleCount; ++i) {
            int indexOffset = i * 3;
            int i0 = indices[indexOffset + 0];
            int i1 = indices[indexOffset + 1];
            int i2 = indices[indexOffset + 2];
            deviceList[i] = new triangle(vertices[i0], vertices[i1], vertices[i2]);
        }
        *deviceWorld = new hitableList(deviceList, triangleCount);
    }
}

__global__ void freeWorldFromMesh(hitable** deviceList, hitable** deviceWorld,
                                  int triangleCount) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int i = 0; i < triangleCount; ++i) {
            delete deviceList[i];
        }
        delete *deviceWorld;
    }
}
