#include <iostream>
#include <time.h>

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
        return;
    }
    // if not hit here then we set the background gradient
    // @@ here if no hit then we wouldn't do anything in out SB
    else {
        return;
    }
}

// called from the CPU, runs on the GPU
__global__ void render(vec3 *fb, int max_x, int max_y,
                       vec3 lower_left_corner, vec3 horizontal, vec3 vertical, vec3 origin,
                       hitable **world // we use a pointer to a pointer here to maintain generality, good for extensibility!
                    ) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;                          // get position x
    int j = threadIdx.y + blockIdx.y * blockDim.y;                          // get position y
    if((i >= max_x) || (j >= max_y)) return;                                // make sure we are inside the window
    int pixel_index = j*max_x + i;                                          // row major
    
    // now the color is set via the device function 
    float u = float(i) / float(max_x);
    float v = float(j) / float(max_y);
    ray r(origin, lower_left_corner + u * horizontal + v * vertical);

    fb[pixel_index] = color(r, world);  // this is what updates the color (@@ change)

    // @@ register collisions here instead of the color 
    // no need for teh pixel info, we need the space info

    // @@ also need to do have a kernel for the phase calculation
    /*
    if ((*world)->hit(r, 0.0, FLT_MAX, rec)) { 
        
    }
    */

}


__global__ void render(float *distance_buffer, vec3 *position_buffer, vec3 *normal_buffer
                       int max_x, int max_y,
                       vec3 lower_left_corner, vec3 horizontal, vec3 vertical, vec3 origin,
                       hitable **world // we use a pointer to a pointer here to maintain generality, good for extensibility!
                    ) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;                          // get position x
    int j = threadIdx.y + blockIdx.y * blockDim.y;                          // get position y
    if((i >= max_x) || (j >= max_y)) return;                                // make sure we are inside the window
    int pixel_index = j*max_x + i;                                          // row major
    
    // now the color is set via the device function 
    float u = float(i) / float(max_x);
    float v = float(j) / float(max_y);
    ray r(origin, lower_left_corner + u * horizontal + v * vertical);

    fb[pixel_index] = color(r, world);  // this is what updates the color (@@ change)

    // hits

    // (...)



    distance_buffer[pixel_index] = ...
    position_buffer[pixel_index] = ...
    normal_buffer[pixel_index] = ...


    // @@ register collisions here instead of the color 
    // no need for teh pixel info, we need the space info

    // @@ also need to do have a kernel for the phase calculation
    /*
    if ((*world)->hit(r, 0.0, FLT_MAX, rec)) { 
        
    }
    */

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

// (@@ change: to calculate the various tehta position we need to rotate the thing around!)
// rotate the objects or rotate the view, maybe easier
// for the start just do 1 position
// this should be done in a spheric view
int main() { 
    int nx = 300;     // pixel horiz
    int ny = 200;     // pixel vert
    int tx = 8;       // block horiz
    int ty = 8;       // block vert
    int max_hit = 10; // max number of hits

    // (@@ change)
    std::cerr << "Rendering a " << nx << "x" << ny << " image ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";
    // std::cerr << "Calculating RCS for a " << nx << "x" << ny << " image ";
    // std::cerr << "in " << tx << "x" << ty << " blocks.\n";

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
    // you should make sure that the object are fixed in space though if you want to rotate
    render<<<blocks, threads>>>(fb, nx, ny,             // KERNEL!
                                vec3(-2.0, -1.0, -1.0), // lower left corner
                                vec3( 4.0,  0.0,  0.0), // horizontal
                                vec3( 0.0,  2.0,  0.0), // vertical
                                vec3( 0.0,  0.0,  0.0), // origin
                                d_world);               //

    launcher<<<blocks, threads>>>(distance_buffer, position_buffer, normal_buffer,  // NEW KERNEL!
                                nx, ny,             
                                vec3(-2.0, -1.0, -1.0), // lower left corner
                                vec3( 4.0,  0.0,  0.0), // horizontal
                                vec3( 0.0,  2.0,  0.0), // vertical
                                vec3( 0.0,  0.0,  0.0), // origin
                                d_world);               //

    // lower left corner just defines the camera in space... it is an arbitrary point!

    // - - - - 



    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "Took " << timer_seconds << " seconds.\n";

    // @@ here instead of outputing the image we would calucluate the PO integral with a kernel
    // Output FB as Image 
    std::cout << "P3\n" << nx << " " << ny << "\n255\n";
    for (int j = ny-1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            size_t pixel_index = j*nx + i;
            int ir = int(255.99*fb[pixel_index].r());
            int ig = int(255.99*fb[pixel_index].g());
            int ib = int(255.99*fb[pixel_index].b());
            std::cout << ir << " " << ig << " " << ib << "\n";
        }
    }

    // clean up
    checkCudaErrors(cudaDeviceSynchronize());
    free_world<<<1,1>>>(d_list,d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(fb)); // @@ clean the correct things at the end, this wont be needed

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


