#ifndef HITABLELISTH
#define HITABLELISTH

#include "hitable.h"

/*

This is used to get the intersections with multiple objects

*/


// this inherits from the base "hitable" class --> polymorphism
class hitableList: public hitable  {  
    public:
        __device__ hitableList() {}                                               // default constructor     
        __device__ hitableList(hitable **l, int n) {list = l; listSize = n; }    // initialization constructor

        // declaration of a virtual function "hit" for the GPU
        // this creates a virtual fucntion table (vtable)
        __device__ virtual bool hit(const ray& r, float tMin, float tMax, hitRecord& rec) const;

        hitable **list;     // this is an array of pointers to hitable objects
        int listSize;      // @@ how many elements in our scene? in SBR this is going to be 1 or 2, but can keep the generality
};


// actual implementation of "hit" for the "hitable_list" class to get the CLOSEST HIT OBJECT
// important if you have objects in front of each other
// @@ in SBR we treat each triangle as an object (maybe)
// @@ we only care about the closes hit of course
// @@ after knowing which is the closest hit then you can reflect
__device__ bool hitableList::hit(const ray& r, float tMin, float tMax, hitRecord& rec) const {
        hitRecord tempRec;
        bool hitAnything = false;                  // at the start no hit
        float closestSoFar = tMax;

        // loop on all the objects
        // @@ super important for SBR
        for (int i = 0; i < listSize; i++) {
            
            /*

            In the IF statement below:
            1. get object pointer: "list[i]""
            2. follow vtable pointer: "->""
            3. lookup function in vtable: "hit(...)" ; @@ this could be a sphere hit, a triangle hit, etc.

            We basically find the implementation of "hit(...)" for that particular object!

            */

            if (list[i]->hit(r, tMin, closestSoFar, tempRec)) { 
                hitAnything = true;                // if we hit something we set this to true
                closestSoFar = tempRec.t;        // update closest distance
                rec = tempRec;                     // save the info of the hit; @@ for SBR to get currents
            }
        }
        return hitAnything;
}

#endif
