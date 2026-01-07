#ifndef BOXH
#define BOXH

#include "hitable.h"

class box : public hitable {
public:
    __device__ box() {}
    
    // Simple axis-aligned box constructor
    // min_point: minimum corner (e.g., vec3(-1, -1, -1))
    // max_point: maximum corner (e.g., vec3(1, 1, 1))
    __device__ box(vec3 min_point, vec3 max_point) : box_min(min_point), box_max(max_point) {}
    
    __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const;
    
    vec3 box_min;
    vec3 box_max;
};

__device__ bool box::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    // Ray-box intersection using slab method
    float tmin = t_min;
    float tmax = t_max;
    
    vec3 inv_dir = vec3(1.0f / r.direction().x(), 
                        1.0f / r.direction().y(), 
                        1.0f / r.direction().z());
    
    int sign[3];
    sign[0] = (inv_dir.x() < 0);
    sign[1] = (inv_dir.y() < 0);
    sign[2] = (inv_dir.z() < 0);
    
    vec3 bounds[2];
    bounds[0] = box_min;
    bounds[1] = box_max;
    
    // Track which face we hit for normal calculation
    int hit_face = -1;
    
    // X slab
    float tx_min = (bounds[sign[0]].x() - r.origin().x()) * inv_dir.x();
    float tx_max = (bounds[1-sign[0]].x() - r.origin().x()) * inv_dir.x();
    
    if (tx_min > tmin) {
        tmin = tx_min;
        hit_face = sign[0] ? 1 : 0; // 0 = +X face, 1 = -X face
    }
    if (tx_max < tmax) tmax = tx_max;
    
    // Y slab
    float ty_min = (bounds[sign[1]].y() - r.origin().y()) * inv_dir.y();
    float ty_max = (bounds[1-sign[1]].y() - r.origin().y()) * inv_dir.y();
    
    if (ty_min > tmin) {
        tmin = ty_min;
        hit_face = sign[1] ? 3 : 2; // 2 = +Y face, 3 = -Y face
    }
    if (ty_max < tmax) tmax = ty_max;
    
    // Z slab
    float tz_min = (bounds[sign[2]].z() - r.origin().z()) * inv_dir.z();
    float tz_max = (bounds[1-sign[2]].z() - r.origin().z()) * inv_dir.z();
    
    if (tz_min > tmin) {
        tmin = tz_min;
        hit_face = sign[2] ? 5 : 4; // 4 = +Z face, 5 = -Z face
    }
    if (tz_max < tmax) tmax = tz_max;
    
    // Check if we have a valid intersection
    if (tmin > tmax || tmax < t_min || tmin > t_max) {
        return false;
    }
    
    // Record the hit
    rec.t = tmin;
    rec.p = r.point_at_parameter(rec.t);
    
    // Set the normal based on which face we hit
    switch(hit_face) {
        case 0: rec.normal = vec3(1, 0, 0); break;   // +X
        case 1: rec.normal = vec3(-1, 0, 0); break;  // -X
        case 2: rec.normal = vec3(0, 1, 0); break;   // +Y
        case 3: rec.normal = vec3(0, -1, 0); break;  // -Y
        case 4: rec.normal = vec3(0, 0, 1); break;   // +Z
        case 5: rec.normal = vec3(0, 0, -1); break;  // -Z
    }
    
    return true;
}

#endif