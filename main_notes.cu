#include <iostream>
#include <time.h>

#include <complex.h>        // for the complex fields

#include <float.h>          // we also add this now

#include "vec3.h"           // for vectors (color RGBs)
#include "ray.h"            // ray class
#include "sphere.h"         // implements the hit for the sphere 
#include "box.h"         // implements the hit for the sphere 
#include "hitable_list.h"   // implements the closest hit for general object

/*

In Part 5 the we create a world of spheres on the device
@@ this is not really useful in our implementation as we would just like to have 1 single object!

In "hitable.h" we will later implement the normal of our surface.
@@ this will be crucial in our SBR implementation.
@@ on thing to note here is that if we have a sphere we can have the analytical formula so we can get the normal quite easily
@@ when you have a complex geometry defined my meshes we have to see how to deal with this

*/


// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

// called from the GPU, runs on the GPU
__device__ vec3 color(const ray& r, hitable **world) {
    hit_record rec;

    // if hit
    if ((*world)->hit(r, 0.0, FLT_MAX, rec)) {
        return 0.5f*vec3(rec.normal.x()+1.0f, rec.normal.y()+1.0f, rec.normal.z()+1.0f);
    }
    // if not hit here then we set the background gradient
    // @@ here if no hit then we wouldn't do anything in out SB
    else {
        vec3 unit_direction = unit_vector(r.direction());
        float t = 0.5f*(unit_direction.y() + 1.0f);
        return (1.0f-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
    }
}

// called from the GPU, runs on the GPU
__device__ vec3 collide(const ray& r, hitable **world) {
    hit_record rec;

    // if hit
    if ((*world)->hit(r, 0.0, FLT_MAX, rec)) { 
        return t rec.p.x rec.p.y rec.p.z rec.normal.x rec.normal.y rec.normal.z;
    }
    // if not hit here then we set the background gradient
    // @@ here if no hit then we wouldn't do anything in out SB
    else {
        return;
    }
}

// // called from the CPU, runs on the GPU
// __global__ void render(vec3 *fb, int max_x, int max_y,
//                        vec3 lower_left_corner, vec3 horizontal, vec3 vertical, vec3 origin,
//                        hitable **world // we use a pointer to a pointer here to maintain generality, good for extensibility!
//                     ) {
//     int i = threadIdx.x + blockIdx.x * blockDim.x;                          // get position x
//     int j = threadIdx.y + blockIdx.y * blockDim.y;                          // get position y
//     if((i >= max_x) || (j >= max_y)) return;                                // make sure we are inside the window
//     int pixel_index = j*max_x + i;                                          // row major
    
//     // now the color is set via the device function 
//     float u = float(i) / float(max_x);
//     float v = float(j) / float(max_y);
//     ray r(origin, lower_left_corner + u * horizontal + v * vertical);

//     fb[pixel_index] = color(r, world);  // this is what updates the color (@@ change)

//     // @@ register collisions here instead of the color 
//     // no need for teh pixel info, we need the space info

//     // @@ also need to do have a kernel for the phase calculation
//     /*
//     if ((*world)->hit(r, 0.0, FLT_MAX, rec)) { 
        
//     }
//     */

// }


__global__ void launcher(float *distance_buffer, vec3 *position_buffer, vec3 *normal_buffer
                       int max_x, int max_y,
                       vec3 lower_left_corner, vec3 horizontal, vec3 vertical, vec3 origin,
                       hitable **world // we use a pointer to a pointer here to maintain generality, good for extensibility!
                    ) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;                          // get position x
    int j = threadIdx.y + blockIdx.y * blockDim.y;                          // get position y
    if((i >= max_x) || (j >= max_y)) return;                                // make sure we are inside the window
    int pixel_index = j*max_x + i;                                          // row major

    ray r(origin, lower_left_corner + u * horizontal + v * vertical);       // initial ray

    ray cur_ray = r;
    
    for (int i = 0; i <= max_hit; i++) { // @@ set max_hit to 10
        hit_record rec;
        if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) { 

            /*
            return t rec.p.x rec.p.y rec.p.z rec.normal.x rec.normal.y rec.normal.z;

            distance_buffer[pixel_index] = rec.t        // t : this is the distance in between hit points
            position_buffer[pixel_index] = rec.p        // ( rec.p.x, rec.p.y, rec.p.z ) : this is the hit point
            normal_buffer[pixel_index] = rec.normal     // ( rec.p.x, rec.p.y, rec.p.z ) : this is the normal at the hit point
            */

            
            r.origin = rec.p;       // r.origin = vec3(rec.p.x(), rec.p.y(), rec.p.z());
            r.hitCount += 1;        // r.hitCount gets increased at every hit
            r.distance += rec.t     // count distance
            // r.rayPolarization = stays the same

            cur_ray = ray(rec.p, rec.normal);

        }
    }

}


// for(int i = 0; i < 50; i++) {
//         hit_record rec;
//         if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
//             // what creates the diffusion is the + random_in_unit_sphere(local_rand_state);
//             vec3 target = rec.p + rec.normal + random_in_unit_sphere(local_rand_state); // this is the new part
            
//             // this is the attenuation set
//             cur_attenuation *= 0.5f;

//             // creates new ray which is the new input
//             cur_ray = ray(rec.p, target-rec.p);
//         }
//         else {
//             vec3 unit_direction = unit_vector(cur_ray.direction());
//             float t = 0.5f*(unit_direction.y() + 1.0f);
//             vec3 c = (1.0f-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
//             return cur_attenuation * c;
//         }
//     }
//     return vec3(0.0,0.0,0.0); // exceeded recursion




__global__ void integral_PO(float *distance_buffer, vec3 *position_buffer, vec3 *normal_buffer
                       int max_x, int max_y,
                       vec3 lower_left_corner, vec3 horizontal, vec3 vertical, vec3 origin,
                       float freq,
                       float phi, float theta
                    ) {

    int i = threadIdx.x + blockIdx.x * blockDim.x;                          // get position x
    int j = threadIdx.y + blockIdx.y * blockDim.y;                          // get position y
    if((i >= max_x) || (j >= max_y)) return;                                // make sure we are inside the window
    int pixel_index = j*max_x + i;                                          // row major

    // @@ these should be done outside of the kernel
    lambda = 1 / f;
    w = 2 * PI * f;
    k = 2 * PI / lambda;

    // you should scan on all phi and theta, so these should be inputs
    phi = 0.5;
    theta = 0.1; 

    cos_phi = cos(phi);
    sin_phi = sin(phi);
    cos_theta = cos(theta);
    sin_theta = sin(theta);

    // unit vectors
    x_uv = vec3(1.0, 0.0, 0.0);
    y_uv = vec3(0.0, 1.0, 0.0);
    z_uv = vec3(0.0, 0.0, 1.0);
    phi_uv = vec3(-sin_phi, cos_phi, 0.0);
    theta_uv = vec3(cos_phi * cos_theta, sin_phi * cos_theta, - sin_theta);

    // wave vector in spherical coordinates
    k_vec_sphere = k * ( (x_uv * cos_phi + y_uv * sin_phi) * sin_theta + z_uv * cos_theta );

    complex AU = 0.0; // what is AU ?
    complex AR = 0.0; // what is Ar ?

    complex i(0.0, 1.0); // is this correct ?

    // @@ implment distanc in ray class / ray record so that it sums the distance travelled < max reflection
    // @@ implemnet the polarization in to the ray
    // @@ implment the numebr of hits summing up in the ray / record so that it summs up
    // @@ get the ray direction for the last hit in the ray class
    // @@ get the area also inside the ray as if it was a tube

    for (rays) {
        // give pointer to the ray
        if (ray has at least one hit) {
            k_r = k * ray.distance; // phase
            refl_coeff = (1.0) ** ray.hits; // we assume here PEC = 1.0 @@ do the right power caluclation.
            E_field = exp( i * k_r) * ray.polarization * refl_coeff; // this is a vector!
            H_field = - cross(ampl_E, ray.direction); // this is a vector!

            // contribute of one polarization ?
            BU = dot(
                - cross(E_field, -phi_uv) + cross(H_field, theta_uv),
                ray.direction
            )

            // contribute of the other polarization ?
            BR = dot(
                - cross(E_field, theta_uv) + cross(H_field, phi_uv),
                ray.direction
            )

            // factor
            complex factor = complex(
                0.0, 
                ( ( k * ray.area ) / ( 4.0 * pi ) ) ) * exp( -i * dot( k_vec_sphere, ray.position )
            ); // @@ what is the ray.position ? is that the position of the last hit ?'

            // accumulate
            AU += BU * factor;
            AR += BR * factor;
        }
    }
        
    rcs = 4.0 * pi * ( abs( AU ) * abs( AU ) + abs( AR ) * abs( AR ) ) );

}

__global__ void create_world(hitable **d_list, hitable **d_world) { // (@@ keep)
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *(d_list)   = new sphere(vec3(0.0, 6, -15.0), 2);
        *(d_list+1) = new sphere(vec3(0.5, -0.3, -2.0), 0.6);

        // horizontal, vertical, distance from view (negative)
        *(d_list+2) = new box( vec3(-0.95, -0.2, -2.0) ,     // min point (<)
                               vec3(-0.5 ,  0.5, -1.0) );    // max point (>)
        // 
        
        *d_world    = new hitable_list(d_list,3); // 3 is the number of elements
    }
}

__global__ void free_world(hitable **d_list, hitable **d_world) {
    delete *(d_list);
    delete *(d_list+1);
    delete *(d_list+2);
    delete *d_world;
}

// (@@ change: to calculate the various theta position we need to rotate the thing around!)
// rotate the objects or rotate the view, maybe easier
// for the start just do 1 position
// this should be done in a spheric view
int main() { 

    // starting parameters
    int nx = 300;       // pixel horiz
    int ny = 200;       // pixel vert
    int tx = 8;         // block horiz
    int ty = 8;         // block vert
    int max_hit = 10;   // max number of hits
    float f = 10E9      // frequency

    float phiValue = 0.5f;      // phi for sphere integration
    float thetaValue = 0.3f;    // theta for sphere integration


    // launching surface

    float V = 2.0f;   // meters
    float H = 4.0f;   // meters

    vec3 surface_llc(-2.0f, -1.0f, 0.0f);     // lower-left corner
    vec3 surface_origin(0.0f, 0.0f, 0.0f);    // origin
    vec3 surface_vertical(V, 0.0f, 0.0f);     // vertical vector
    vec3 surface_horizontal(0.0f, H, 0.0f);   // horizontal vector


    // (@@ change)
    // std::cerr << "Rendering a " << nx << "x" << ny << " image ";
    // std::cerr << "in " << tx << "x" << ty << " blocks.\n";
    std::cerr << "Calculating RCS for a " << nx << "x" << ny << " image ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";
    std::cerr << "Frequency set to " << f << " Hz.\n";
    std::cerr << "There are " << ny * nx << " rays. Density is " << (nx * ny) / (V * H)  << "(rays / m^2).\n";

    int num_pixels = nx*ny;

    // allocate Distance Buffer
    float *distance_buffer;
    size_t distance_buffer_size = max_hit * num_pixels * sizeof(float);
    checkCudaErrors(cudaMallocManaged((void**)&distance_buffer, distance_buffer_size));

    // allorate Hit-Position Buffer
    vec3 *position_buffer;
    size_t position_buffer_size = sizeof(vec3) * max_hit * num_pixels;
    checkCudaErrors(cudaMallocManaged((void**)&position_buffer, position_buffer_size));

    // allocate Normal-to-Surface Buffer
    vec3 *normal_buffer;
    size_t normal_buffer_size = sizeof(vec3) * max_hit * num_pixels;
    checkCudaErrors(cudaMallocManaged((void**)&normal_buffer, normal_buffer_size));

    // // allocate Frame Buffer
    // vec3 *fb;
    // size_t fb_size = num_pixels * sizeof(vec3);
    // checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

    ///////////////
    std::cerr << "Creating World!";
    ///////////////

    // make our world of hitables: this is the new part!!!
    hitable **d_list;
    checkCudaErrors(cudaMalloc((void **)&d_list, 2*sizeof(hitable *)));
    hitable **d_world;
    checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hitable *)));
    create_world<<<1,1>>>(d_list,d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    clock_t start, stop;
    start = clock();
    // Render our buffer
    dim3 blocks(nx/tx+1,ny/ty+1);
    dim3 threads(tx,ty);

    ///////////////
    std::cerr << "Launching Rays!";
    ///////////////

    // - - - - 
    // @@ this is where you would rotate the view
    launcher<<<blocks, threads>>>(
                                distance_buffer,            // store the distances
                                position_buffer,            // store the hit positions
                                normal_buffer,              // store the normals
                                nx,                         // horiz points
                                ny,                         // vert points
                                surface_llc,                // lower left corner
                                surface_horizontal,         // horizontal
                                surface_vertical,           // vertical
                                surface_origin,             // origin
                                d_world                     // world of hitable objects
                                ); 

    // lower left corner just defines the camera in space... it is an arbitrary point!
    // - - - - 



    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "Took " << timer_seconds << " seconds for the collisions.\n";


    // Integral calculation
    dim3 blocks(nx/tx+1,ny/ty+1);
    dim3 threads(tx,ty);
    clock_t start, stop;
    start = clock();
    ///////////////
    std::cerr << "Calculating Integral!";
    ///////////////

    integral_PO<<<blocks, threads>>>(
                                distance_buffer,            // store the distances
                                position_buffer,            // store the hit positions
                                normal_buffer,              // store the normals
                                nx,                         // horiz points
                                ny,                         // vert points
                                f,                          // frequency
                                phiValue,
                                thetaValue,
                                ); 


    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "Took " << timer_seconds << " seconds for the integral calculation.\n";


    ///////////////
    std::cerr << "Cleaning up!";
    ///////////////

    // clean up
    checkCudaErrors(cudaDeviceSynchronize());
    free_world<<<1,1>>>(d_list,d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_world));


    checkCudaErrors(cudaFree(distance_buffer))
    VcheckCudaErrors(cudaFree(position_buffer))
    checkCudaErrors(cudaFree(normal_buffer))
    //checkCudaErrors(cudaFree(fb)); // @@ clean the correct things at the end, this wont be needed

    // useful for cuda-memcheck --leak-check full
    cudaDeviceReset();

    // @@ last part to consider:
    // we launch rays from a position but we dont have to calculate the RCS at the same position.
    // we can make sure that the ray-launching surface is far enough not to interact with the
    // hitables in the world. cus it will just not matter.

    // @@ additional: 
    // - we can keep the .ppm generation just to visualize fast the scene
    // - we can put the opacity thing

    // @@ last part is moving to crazy objects with the mesh and BVH approach

    // @@ for the results it could be cool to see a difference between the analytical sphere and the meshed sphere
}


/*

implement:

1) max number of reflections
2) maybe can implement the opache thingy but not really necessary
3) when you have hit instead of going to the color thing you can 
just give back to the host the points, the distance between hits
caluclate the phase and do the integral

*/


