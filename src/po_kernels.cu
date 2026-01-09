/*
 * Implementation of Physical Optics integration kernels.
 */

#include "po_kernels.cuh"
#include <cmath>

__global__ void integral_PO_singlebounce(vec3* hit_pos, vec3* hit_normal, float* hit_dist, int* hit_flag, 
                            int n, float k, vec3 k_inc, float ray_area, cuFloatComplex* accum) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n || !hit_flag[idx]) return;

    // 1. Obliquity factor: In monostatic backscatter, this is dot(n, -k_inc)
    float cos_theta = dot(hit_normal[idx], -k_inc);
    if (cos_theta <= 0.0f) return;

    // 2. Monostatic Phase
    float phase = 2.0f * k * hit_dist[idx];
    cuFloatComplex e = make_cuFloatComplex(cosf(phase), sinf(phase));

    // 3. THE FIX: Multiply by 2.0f because J = 2 * (n x H)
    // The standard PO factor is (j * k / (4 * PI)) * 2.0
    float mag = (k * ray_area / (4.0f * M_PI)) * 2.0f;
    
    cuFloatComplex contrib = make_cuFloatComplex(-mag * sinf(phase), mag * cosf(phase)); 
    // representing j * factor * exp(j*phase)

    atomicAdd(&accum->x, contrib.x);    // real part 
    atomicAdd(&accum->y, contrib.y);    // imaginary part
}

// OLD version

// __global__ void integral_PO_multibounce(vec3* hit_pos, vec3* hit_normal, float* hit_dist, 
//                                        int* hit_count, int n, float k, vec3 k_inc, 
//                                        float ray_area, cuFloatComplex* accum, 
//                                        float reflection_const) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;

//     if (idx >= n || hit_count[idx] == 0) return;

//     float total_refl_coeff = powf(reflection_const, (float)hit_count[idx]);

//     // 1. Obliquity factor: In monostatic backscatter, this is dot(n, -k_inc)
//     float cos_theta = dot(hit_normal[idx], -k_inc);
//     if (cos_theta <= 0.0f) return;

//     // 2. Monostatic Phase
//     float phase = 2.0f * k * hit_dist[idx];
//     cuFloatComplex e = make_cuFloatComplex(cosf(phase), sinf(phase));

//     // 3. THE FIX: Multiply by 2.0f because J = 2 * (n x H)
//     // The standard PO factor is (j * k / (4 * PI)) * 2.0
//     float mag = (k * ray_area / (4.0f * M_PI)) * 2.0f * total_refl_coeff;
    
//     cuFloatComplex contrib = make_cuFloatComplex(-mag * sinf(phase), mag * cosf(phase)); 
//     // representing j * factor * exp(j*phase)

//     atomicAdd(&accum->x, contrib.x);    // real part 
//     atomicAdd(&accum->y, contrib.y);    // imaginary part
// }


__global__ void integral_PO(vec3* hit_pos, vec3* hit_normal, float* hit_dist, 
                                       int* hit_count, int n, float k, vec3 k_inc, 
                                       float ray_area, cuFloatComplex* accum, 
                                       float reflection_const) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n || hit_count[idx] == 0) return;

    // Total reflection coefficient for multi-bounce
    float total_refl_coeff = powf(reflection_const, (float)hit_count[idx]);

    // Obliquity factor (monostatic PO)
    float cos_theta = dot(hit_normal[idx], -k_inc);
    if (cos_theta <= 0.0f) return;

    // Monostatic round-trip phase (note the minus sign)
    float phase = -2.0f * k * hit_dist[idx];
    phase = fmodf(phase, 2.0f * M_PI); // make sure to bound the phase from 0 to 2*pi

    float c = cosf(phase);
    float s = sinf(phase);

    // PO magnitude factor:
    // (j k ΔA / 4π) * 2 * (n · -k_inc)
    float mag = (k * ray_area / (4.0f * M_PI))
                * 2.0f
                * cos_theta
                * total_refl_coeff;

    // j * mag * exp(j phase)
    cuFloatComplex contrib = make_cuFloatComplex(
        -mag * s,   // real part
         mag * c    // imaginary part
    );

    atomicAdd(&accum->x, contrib.x);
    atomicAdd(&accum->y, contrib.y);
}


__global__ void integral_PO_vector(
    vec3* hit_pos, 
    vec3* hit_normal, 
    vec3* hit_dir,      // The direction the ray was traveling at the hit
    vec3* hit_pol,      // The E-field polarization vector (complex or real)
    float* hit_dist, 
    int* hit_count, 
    int n, 
    float k, 
    vec3 obs_dir,       // Direction to the observer
    vec3 obs_theta,     // Unit vector Theta (for polarization basis)
    vec3 obs_phi,       // Unit vector Phi (for polarization basis)
    float ray_area, 
    cuFloatComplex* accum_theta, // Accumulator for Theta component
    cuFloatComplex* accum_phi,    // Accumulator for Phi component
    float reflection_const
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n || hit_count[idx] == 0) return;

    // 1. Calculate Phase and Reflection
    float kr = k * hit_dist[idx];
    float refl = powf(reflection_const, (float)hit_count[idx]); // Matching your C++ logic

    // 2. Define Aperture E and H Fields (apE and apH)
    // apE = exp(i*kr) * pol * refl
    float c_kr = cosf(kr);
    float s_kr = sinf(kr);
    
    // Complex E-field vector (Real and Imaginary parts)
    vec3 apE_re = hit_pol[idx] * (c_kr * refl);
    vec3 apE_im = hit_pol[idx] * (s_kr * refl);

    // apH = -cross(apE, hit_dir)
    vec3 apH_re = -cross(apE_re, hit_dir[idx]);
    vec3 apH_im = -cross(apE_im, hit_dir[idx]);

    // 3. Project onto Observation Basis (BU and BR)
    // BU = Dot(-(Cross(apE, -obs_phi) + Cross(apH, obs_theta)), hit_dir)
    // This looks complex, but it's essentially calculating the far-field 
    // contribution of the surface currents.
    
    // Intermediate cross products
    vec3 termU_re = -(cross(apE_re, -obs_phi) + cross(apH_re, obs_theta));
    vec3 termU_im = -(cross(apE_im, -obs_phi) + cross(apH_im, obs_theta));
    
    vec3 termR_re = -(cross(apE_re, obs_theta) + cross(apH_re, obs_phi));
    vec3 termR_im = -(cross(apE_im, obs_theta) + cross(apH_im, obs_phi));

    float BU_re = dot(termU_re, hit_dir[idx]);
    float BU_im = dot(termU_im, hit_dir[idx]);
    float BR_re = dot(termR_re, hit_dir[idx]);
    float BR_im = dot(termR_im, hit_dir[idx]);

    // 4. Calculate Phase Factor: exp(-i * k * dot(obs_dir, hit_pos))
    float phase_factor = -k * dot(obs_dir, hit_pos[idx]);
    float cp = cosf(phase_factor);
    float sp = sinf(phase_factor);
    
    // Constant magnitude: (k * area) / (4 * PI)
    // The "i" in the C++ code (complex<T>(0, 1)) shifts the phase by 90 degrees
    float mag = (k * ray_area) / (4.0f * M_PI);

    // 5. Accumulate Results (Complex multiplication: (BU_re + iBU_im) * (0 + i*mag) * (cp + i*sp))
    // Simplifies to: BU * mag * (-sp + i*cp)
    cuFloatComplex factor = make_cuFloatComplex(-mag * sp, mag * cp);
    
    // Theta component accumulation
    atomicAdd(&accum_theta->x, (BU_re * factor.x - BU_im * factor.y));
    atomicAdd(&accum_theta->y, (BU_re * factor.y + BU_im * factor.x));

    // Phi component accumulation
    atomicAdd(&accum_phi->x, (BR_re * factor.x - BR_im * factor.y));
    atomicAdd(&accum_phi->y, (BR_re * factor.y + BR_im * factor.x));
}
