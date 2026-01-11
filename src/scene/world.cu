#include "scene/world.cuh"

#include "RT/hitable_list.h"
#include "RT/sphere.h"

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
