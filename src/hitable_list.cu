/*
* Implementation of hitable_list class methods for ray-object intersections.
*/

#include "hitable_list.h"

// actual implementation of "hit" for the "hitable_list" class to get the CLOSEST HIT OBJECT
// important if you have objects in front of each other
// @@ in SBR we treat each triangle as an object (maybe)
// @@ we only care about the closes hit of course
// @@ after knowing which is the closest hit then you can reflect
__device__ bool hitable_list::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    hit_record temp_rec;
    bool hit_anything = false;                  // at the start no hit
    float closest_so_far = t_max;

    // loop on all the objects
    // @@ super important for SBR
    for (int i = 0; i < list_size; i++) {
        
        /*

        In the IF statement below:
        1. get object pointer: "list[i]"
        2. follow vtable pointer: "->"
        3. lookup function in vtable: "hit(...)" ; @@ this could be a sphere hit, a triangle hit, etc.

        We basically find the implementation of "hit(...)" for that particular object!

        */

        if (list[i]->hit(r, t_min, closest_so_far, temp_rec)) { 
            hit_anything = true;                // if we hit something we set this to true
            closest_so_far = temp_rec.t;        // update closest distance
            rec = temp_rec;                     // save the info of the hit; @@ for SBR to get currents
        }
    }
    return hit_anything;
}
