/*
* This is used to get the intersections with multiple objects
*/

#ifndef HITABLE_LIST_H
#define HITABLE_LIST_H

#include "hitable.h"

// this inherits from the base "hitable" class --> polymorphism
class hitable_list: public hitable  {  
    public:
        __device__ hitable_list() {}                                               // default constructor     
        __device__ hitable_list(hitable **l, int n) {list = l; list_size = n; }    // initialization constructor

        // declaration of a virtual function "hit" for the GPU
        // this creates a virtual function table (vtable)
        __device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;

        hitable **list;     // this is an array of pointers to hitable objects
        int list_size;      // @@ how many elements in our scene? in SBR this is going to be 1 or 2, but can keep the generality
};

#endif // HITABLE_LIST_H