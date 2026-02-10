#ifndef HITABLELISTH
#define HITABLELISTH

#include "hitable.h"


// this inherits from the base "hitable" class --> polymorphism
class hitableList: public hitable  {  
    public:
        __device__ hitableList() {}                                               // default constructor     
        __device__ hitableList(hitable **l, int n) {list = l; listSize = n; }    // initialization constructor

        // declaration of a virtual function "hit" for the GPU
        // this creates a virtual fucntion table (vtable)
        __device__ virtual bool hit(const ray& r, float tMin, float tMax, hitRecord& rec) const;

        hitable **list;     
        int listSize;     
};


// actual implementation of "hit" for the "hitable_list" class to get the CLOSEST HIT OBJECT

__device__ bool hitableList::hit(const ray& r, float tMin, float tMax, hitRecord& rec) const {
        hitRecord tempRec;
        bool hitAnything = false;                  // at the start no hit
        float closestSoFar = tMax;

        // loop on all the objects
        for (int i = 0; i < listSize; i++) {
            

            if (list[i]->hit(r, tMin, closestSoFar, tempRec)) { 
                hitAnything = true;                // if we hit something we set this to true
                closestSoFar = tempRec.t;        // update closest distance
                rec = tempRec;                     // save the info of the hit; @@ for SBR to get currents
            }
        }
        return hitAnything;
}

#endif
