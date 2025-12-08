#ifndef HITABLEH
#define HITABLEH

#include "ray.h"

struct hit_record
{
    float t;            // distance along ray
    vec3 p;             // hit point in 3d space (x,y,z)
    vec3 normal;        // surface normal at hit point
};

class hitable  {
    public:
        __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const = 0;
};

#endif
