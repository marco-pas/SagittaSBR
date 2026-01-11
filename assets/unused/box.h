// #ifndef BOXH
// #define BOXH

// #include "hitable.h"

// #ifndef M_PI
// #define M_PI 3.14159265358979323846
// #endif

// class box : public hitable {
// public:
//     __device__ box() {}

//     // cen:   Center position
//     // L:     Width
//     // H:     Height
//     // theta: Polar angle (degrees)
//     // phi:   Azimuthal angle (degrees)
//     __device__ box(vec3 cen, float L, float H, float theta, float phi) {
        
//         float t_rad = theta * (M_PI / 180.0f);
//         float p_rad = phi * (M_PI / 180.0f);

//         // 1. Calculate Normal from spherical coordinates
//         float nx = sinf(t_rad) * cosf(p_rad);
//         float ny = sinf(t_rad) * sinf(p_rad);
//         float nz = cosf(t_rad);
        
//         normal = unit_vector(vec3(nx, ny, nz));

//         // 2. Construct local basis (u and v)
//         vec3 world_up(0, 1, 0);
//         if (fabsf(dot(normal, world_up)) > 0.99f) {
//             world_up = vec3(1, 0, 0); 
//         }

//         vec3 u_dir = unit_vector(cross(world_up, normal));
//         vec3 v_dir = unit_vector(cross(normal, u_dir));

//         u = u_dir * L;
//         v = v_dir * H;
        
//         // Pre-calculate squared lengths for the hit function optimization
//         u_len_sq = L * L;
//         v_len_sq = H * H;

//         // 3. Calculate Bottom-Left Corner Q
//         Q = cen - (u / 2.0f) - (v / 2.0f);

//         // 4. Plane constant D (distance from origin along normal)
//         D = dot(normal, Q);
//     }

//     __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const;

//     vec3 Q;         // Corner
//     vec3 u, v;      // Side vectors
//     vec3 normal;    // Surface normal
//     float D;        // Plane constant
//     float u_len_sq; // Optimization: Squared length of u
//     float v_len_sq; // Optimization: Squared length of v
// };

// __device__ bool box::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    
//     // 1. Ray-Plane Intersection
//     float denom = dot(normal, r.direction());

//     // Parallel check (ray is parallel to plane)
//     if (fabsf(denom) < 1e-8f) {
//         return false;
//     }

//     // Solve for t: ray equation is P = O + t*D
//     // Plane equation is: N·P = N·Q (or N·P = D)
//     // Substituting: N·(O + t*D) = N·Q
//     // t = (N·Q - N·O) / (N·D)
//     float t = (D - dot(normal, r.origin())) / denom;

//     // Check if t is within valid range
//     if (t < t_min || t > t_max) {
//         return false;
//     }

//     // 2. Interior Check (Projection Method)
//     vec3 intersection = r.point_at_parameter(t);
//     vec3 planar_hit_vector = intersection - Q;
    
//     // Project the hit vector onto the side vectors u and v
//     // alpha = (P-Q) · u / |u|^2
//     // beta  = (P-Q) · v / |v|^2
//     float alpha = dot(planar_hit_vector, u) / u_len_sq;
//     float beta  = dot(planar_hit_vector, v) / v_len_sq;

//     // Check if within rectangle bounds [0,1] x [0,1]
//     if (alpha < 0.0f || alpha > 1.0f || beta < 0.0f || beta > 1.0f) {
//         return false;
//     }

//     // 3. Record Hit
//     rec.t = t;
//     rec.p = intersection;
    
//     // Make sure normal points against the ray direction (front-face hit)
//     rec.normal = (denom < 0.0f) ? normal : -normal;
    
//     return true;
// }

// #endif


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