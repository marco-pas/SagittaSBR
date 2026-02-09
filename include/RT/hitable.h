#ifndef HITABLEH
#define HITABLEH

#include "ray.h"

template<typename T>
struct hitRecord_t
{
    T t;                // distance along ray
    vec3_t<T> p;        // hit point in 3d space (x,y,z)
    vec3_t<T> normal;   // surface normal at hit point
};

// Type alias
using hitRecord = hitRecord_t<Real>;

template<typename T>
class hitable_t  {
    public:
        __device__ virtual bool hit(const ray_t<T>& r, T tMin, T tMax, hitRecord_t<T>& rec) const = 0;
};

using hitable = hitable_t<Real>;

#endif
