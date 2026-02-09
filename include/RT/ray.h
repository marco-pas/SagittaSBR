#ifndef RAYH
#define RAYH
#include "vec3.h"

// only device functions here

template<typename T>
class ray_t
{
    public:
        __device__ ray_t() {}
        __device__ ray_t(const vec3_t<T>& origin, const vec3_t<T>& direction) {
            originValue = origin;
            directionValue = direction;
        }
        __device__ vec3_t<T> origin() const       { return originValue; }
        __device__ vec3_t<T> direction() const    { return directionValue; }
        __device__ vec3_t<T> pointAtParameter(T t) const { return originValue + t * directionValue; }

        vec3_t<T> originValue;
        vec3_t<T> directionValue;
};

// Type alias - follows precision from vec3.h / precision.h
using ray = ray_t<Real>;

#endif
