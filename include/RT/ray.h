#ifndef RAYH
#define RAYH
#include "vec3.h"

// only device functions here

class ray
{
    public:
        __device__ ray() {}
        __device__ ray(const vec3& origin, const vec3& direction) {
            originValue = origin;
            directionValue = direction;
        }
        __device__ vec3 origin() const       { return originValue; } // @@ this has to be updated after the hit
        __device__ vec3 direction() const    { return directionValue; } // @@ this has to be updated after every hit
        __device__ vec3 pointAtParameter(float t) const { return originValue + t * directionValue; }
        // __device__ float distance() const;

        /*
        __device__ float distTravelled() const    { return t; }
        __device__ float rayArea() const  { return area; }
        __device__ int hitCount() const { return hits; }
        __device__ float rayPolariz() const { return polariz; }

        */

        vec3 originValue;
        vec3 directionValue;
};

#endif
