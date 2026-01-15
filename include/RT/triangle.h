#ifndef TRIANGLEH
#define TRIANGLEH

#include <math.h>

#include "hitable.h"

class triangle : public hitable {
    public:
        __device__ triangle() {}
        __device__ triangle(vec3 a, vec3 b, vec3 c) : v0(a), v1(b), v2(c) {}
        __device__ virtual bool hit(const ray& r, float tMin, float tMax, hitRecord& rec) const;

        vec3 v0;
        vec3 v1;
        vec3 v2;
};

__device__ bool triangle::hit(const ray& r, float tMin, float tMax, hitRecord& rec) const {
    const float eps = 1.0e-6f;
    vec3 edge1 = v1 - v0;
    vec3 edge2 = v2 - v0;
    vec3 pvec = cross(r.direction(), edge2);
    float det = dot(edge1, pvec);

    if (fabsf(det) < eps) {
        return false;
    }

    float invDet = 1.0f / det;
    vec3 tvec = r.origin() - v0;
    float u = dot(tvec, pvec) * invDet;
    if (u < 0.0f || u > 1.0f) {
        return false;
    }

    vec3 qvec = cross(tvec, edge1);
    float v = dot(r.direction(), qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) {
        return false;
    }

    float t = dot(edge2, qvec) * invDet;
    if (t < tMin || t > tMax) {
        return false;
    }

    rec.t = t;
    rec.p = r.pointAtParameter(t);
    rec.normal = unitVector(cross(edge1, edge2));
    return true;
}

#endif
