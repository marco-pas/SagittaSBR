#ifndef RAYH
#define RAYH
#include "vec3.h"

// only device functions here

class ray
{
    public:
        __device__ ray() {}
        __device__ ray(const vec3& a, const vec3& b) { A = a; B = b; }
        __device__ vec3 origin() const       { return A; } // @@ this has to be updated after the hit
        __device__ vec3 direction() const    { return B; } // @@ this has to be updated after every hit
        __device__ vec3 point_at_parameter(float t) const { return A + t*B; }

        vec3 A;
        vec3 B;
};

#endif
