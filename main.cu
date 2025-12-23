#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <cmath>
#include <ctime>
#include <cfloat>
#include <string>
#include <iomanip>

#include <cuda_runtime.h>
#include <cuComplex.h>

#include "vec3.h"
#include "ray.h"
#include "sphere.h"
#include "hitable_list.h"

// CUDA error checking
#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)

void check_cuda(cudaError_t result, char const* func, const char* file, int line) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result)
                  << " at " << file << ":" << line
                  << " '" << func << "'\n";
        cudaDeviceReset();
        exit(1);
    }
}

// functions for the fancy printing

void printGPUInfo() {
    cudaDeviceProp prop;
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    if (deviceCount == 0) {
        std::cerr << "No CUDA-capable devices found\n";
        return;
    }
    
    cudaGetDeviceProperties(&prop, 0);
    
    std::cerr << "╔═══════════════════════════════════════════════════╗\n";
    std::cerr << "║                   GPU INFORMATION                 ║\n";
    std::cerr << "╠═══════════════════════════════════════════════════╣\n";
    std::cerr << "║ Device: " << prop.name << "\n";
    std::cerr << "║ Compute Capability: " << prop.major << "." << prop.minor << "\n";
    std::cerr << "║ Total Global Memory: " 
              << prop.totalGlobalMem / (1024 * 1024 * 1024.0) << " GB\n";
    std::cerr << "║ Multiprocessors: " << prop.multiProcessorCount << "\n";
    std::cerr << "║ Max Threads per Block: " << prop.maxThreadsPerBlock << "\n";
    std::cerr << "╚═══════════════════════════════════════════════════╝\n";
    std::cerr << "\n";
}

void printSeparator(const std::string& title = "") {
    std::cerr << "\n┌── " << std::left << std::setw(46) << title << " ┐\n";
}

void printEndSeparator() {
    std::cerr << "└──────────────────────────────────────────────────┘\n";
}

template<typename T>
void printKV(const std::string& key, const T& value, const std::string& unit = "") {
    std::cerr << "│ " << std::left << std::setw(20) << key << ": " << value << " " << unit << "\n";
}

std::string formatTime(double seconds) {
    if (seconds < 0.001) return std::to_string(seconds * 1e6) + " µs";
    if (seconds < 1.0) return std::to_string(seconds * 1000) + " ms";
    return std::to_string(seconds) + " s";
}

// ------------



// Parser to read config.txt file
std::map<std::string, float> loadConfig(const std::string& filename) {
    std::map<std::string, float> config;
    std::ifstream file(filename);
    std::string line;

    if (!file.is_open()) {
        std::cerr << "Warning: Could not open " << filename << ". Using defaults." << std::endl;
        return config;
    }

    while (std::getline(file, line)) {
        // 1. Strip out comments (anything after #)
        size_t commentPos = line.find('#');
        if (commentPos != std::string::npos) {
            line = line.substr(0, commentPos);
        }

        // 2. Extract key and value from the remaining string
        std::stringstream ss(line);
        std::string key;
        float value;
        if (ss >> key >> value) {
            config[key] = value;
        }
    }
    return file.close(), config;
}

// CUDA kernels

__global__ void launcher(vec3* hit_pos, vec3* hit_normal, float* hit_dist, int* hit_flag, int nx, int ny, vec3 llc, vec3 horiz, vec3 vert, vec3 ray_dir, hitable** world) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= nx || j >= ny) return; // check we in the range
    int idx = j * nx + i;

    float u = (i + 0.5f) / float(nx);
    float v = (j + 0.5f) / float(ny);

    vec3 ray_start = llc + u * horiz + v * vert; // start the ray launch

    ray r(ray_start, ray_dir);

    // now here we still only allow for 1 hit
    hit_record rec;
    if ((*world)->hit(r, 0.001f, FLT_MAX, rec)) {

        hit_flag[idx] = 1; 
        hit_pos[idx] = rec.p; 
        hit_normal[idx] = rec.normal; 
        hit_dist[idx] = rec.t;

    } else { hit_flag[idx] = 0; }
}

__global__ void launcher_multibounce(vec3* hit_pos, vec3* hit_normal, float* hit_dist, int* hit_count, int nx, int ny, vec3 llc, vec3 horiz, vec3 vert, vec3 ray_dir, hitable** world, int max_bounces) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= nx || j >= ny) return;
    int idx = j * nx + i;

    float u = (i + 0.5f) / float(nx);
    float v = (j + 0.5f) / float(ny);

    vec3 current_origin = llc + u * horiz + v * vert;
    vec3 current_dir = ray_dir; 
    
    // hit_flag[idx] = 0;
    hit_dist[idx] = 0.0f;
    hit_count[idx] = 0;

    for (int k = 0; k < max_bounces; k++) {
        ray r(current_origin, current_dir);
        hit_record rec;

        if ((*world)->hit(r, 0.001f, FLT_MAX, rec)) {
            // hit_flag[idx] = 1;
            hit_count[idx]++;
            
            // Store the very first hit position/normal for the buffer
            if (k == 0) {
                hit_pos[idx] = rec.p;
                hit_normal[idx] = rec.normal;
            }

            hit_dist[idx] += rec.t;

            // Update for next bounce
            // Example: Simple reflection logic
            current_origin = rec.p; 

            /*
            __device__ vec3 reflect(const vec3& i, const vec3& n) {
                // i = incident vector
                // n = normal vector (must be normalized)
                return i - 2.0f * dot(i, n) * n;
            }
            */
            current_dir = current_dir - 2.0f * dot(current_dir, rec.normal) * rec.normal;

        } else {
            break; // Ray missed everything, stop bouncing
        }
    }
}

__global__ void integral_PO(vec3* hit_pos, vec3* hit_normal, float* hit_dist, int* hit_flag, int n, float k, vec3 k_inc, float ray_area, cuFloatComplex* accum) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // physical optics integration !

    if (idx >= n || !hit_flag[idx]) return;

    // 1. Obliquity factor: In monostatic backscatter, this is dot(n, -k_inc)
    float cos_theta = dot(hit_normal[idx], -k_inc);
    if (cos_theta <= 0.0f) return;

    // 2. Monostatic Phase
    float phase = 2.0f * k * hit_dist[idx];
    cuFloatComplex e = make_cuFloatComplex(cosf(phase), sinf(phase));

    // 3. THE FIX: Multiply by 2.0f because J = 2 * (n x H)
    // The standard PO factor is (j * k / (4 * PI)) * 2.0
    float mag = (k * ray_area / (4.0f * M_PI)) * 2.0f; // it was missing the 2.0 here
    
    // Optional: If you want to be extremely precise with curvature, 
    // some include the cos_theta here, but with a flat ray-grid, 
    // ray_area is already the 'projected' area.
    // @@ here we dont consider the ray area diverging
    
    cuFloatComplex contrib = make_cuFloatComplex(-mag * sinf(phase), mag * cosf(phase)); // representing j * factor * exp(j*phase)

    atomicAdd(&accum->x, contrib.x);    // real part 
    atomicAdd(&accum->y, contrib.y);    // imaginary part
}

__global__ void integral_PO_multibounce(vec3* hit_pos, vec3* hit_normal, float* hit_dist, int* hit_count, int n, float k, vec3 k_inc, float ray_area, cuFloatComplex* accum, float reflection_const) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // physical optics integration !

    if (idx >= n || hit_count[idx] == 0) return;

    float total_refl_coeff = powf(reflection_const, (float)hit_count[idx]);

    // 1. Obliquity factor: In monostatic backscatter, this is dot(n, -k_inc)
    float cos_theta = dot(hit_normal[idx], -k_inc);
    if (cos_theta <= 0.0f) return;

    // 2. Monostatic Phase
    float phase = 2.0f * k * hit_dist[idx];
    cuFloatComplex e = make_cuFloatComplex(cosf(phase), sinf(phase));

    // 3. THE FIX: Multiply by 2.0f because J = 2 * (n x H)
    // The standard PO factor is (j * k / (4 * PI)) * 2.0
    float mag = (k * ray_area / (4.0f * M_PI)) * 2.0f * total_refl_coeff; // it was missing the 2.0 here
    
    // Optional: If you want to be extremely precise with curvature, 
    // some include the cos_theta here, but with a flat ray-grid, 
    // ray_area is already the 'projected' area.
    // @@ here we dont consider the ray area diverging
    
    cuFloatComplex contrib = make_cuFloatComplex(-mag * sinf(phase), mag * cosf(phase)); // representing j * factor * exp(j*phase)

    atomicAdd(&accum->x, contrib.x);    // real part 
    atomicAdd(&accum->y, contrib.y);    // imaginary part
}


// @@ here we need something more well done
// like a configuration file or something like that
__global__ void create_world(hitable** d_list, hitable** d_world) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_list[0] = new sphere(vec3(0.0f, 0.0f, 0.0f), 1.0f); // we center our object at (0.0f, 0.0f, 0.0f) !!
        *d_world  = new hitable_list(d_list, 1);

        // d_list[0] = new sphere(vec3(1.2f, 0.0f, 0.0f), 0.5f);
        // d_list[1] = new sphere(vec3(-1.2f, 0.0f, 0.0f), 0.5f);
        // *d_world  = new hitable_list(d_list, 2);
    }
}

__global__ void free_world(hitable** d_list, hitable** d_world) {
    delete d_list[0]; delete *d_world;
}

// --- MAIN ---

int main() {
    std::cerr << "╔═══════════════════════════════════════════════════╗\n";
    std::cerr << "║  MONOSTATIC RCS SWEEP - SHOOTING & BOUNCING RAYS  ║\n";
    std::cerr << "╚═══════════════════════════════════════════════════╝\n";
    
    // GPU info
    printGPUInfo(); 

    // load configuration
    auto cfg = loadConfig("config.txt");

    // if no value, then put the standard values
    float phi_start = cfg.count("phi_start") ? cfg["phi_start"] : 0.0f;
    float phi_end   = cfg.count("phi_end")   ? cfg["phi_end"]   : 180.0f;
    int phi_samples = cfg.count("phi_samples") ? (int)cfg["phi_samples"] : 181;
    float theta_start = cfg.count("theta_start") ? cfg["theta_start"] : 90.0f;
    float theta_end   = cfg.count("theta_end")   ? cfg["theta_end"]   : 90.0f;
    int theta_samples = cfg.count("theta_samples") ? (int)cfg["theta_samples"] : 1;
    float freq      = cfg.count("freq")      ? cfg["freq"]      : 10.0e9f;
    float grid_size = cfg.count("grid_size") ? cfg["grid_size"] : 3.0f;
    int nx          = cfg.count("nx")        ? (int)cfg["nx"]   : 400;
    int ny          = cfg.count("ny")        ? (int)cfg["ny"]   : 400;
    int max_bounces = cfg.count("max_bounces")    ? (int)cfg["max_bounces"]   : 20;
    float reflection_const = cfg.count("reflection_const")    ? (int)cfg["reflection_const"] : 1.0;

    // constants
    const float c0 = 299792458.0f;
    const float lambda = c0 / freq;
    const float k = 2.0f * M_PI / lambda;
    const float ray_area = (grid_size * grid_size) / (nx * ny);
    const int N = nx * ny;
    const float distance_offset = 10.0f;

    // start prints
    printSeparator("SIMULATION PARAMETERS");
    printKV("Frequency", freq / 1e9, "GHz");
    printKV("Wavelength", lambda * 1000, "mm");
    printKV("Wave Number k", k, "rad/m");
    printKV("Illumination Area", grid_size * grid_size, "m² (make sure this is larger than your world objects)");
    printKV("Grid Size", std::to_string(nx) + "x" + std::to_string(ny));
    printKV("Total Rays", N);
    printKV("Ray Density", N / (grid_size * grid_size), "rays/m² (make sure this is dense enough)");
    printKV("Ray Area", ray_area, "m²");
    printKV("Phi Range", std::to_string(phi_start) + " to " + std::to_string(phi_end));
    printEndSeparator();

    // memory allocation on the Device

    // our buffers to save all the data
    printSeparator("MEMORY ALLOCATION");
    vec3 *hit_pos, *hit_normal; 
    float *hit_dist; 
    int *hit_flag, *hit_count; 
    cuFloatComplex *accum;
    checkCudaErrors(cudaMallocManaged(&hit_pos, N * sizeof(vec3)));
    checkCudaErrors(cudaMallocManaged(&hit_normal, N * sizeof(vec3)));
    checkCudaErrors(cudaMallocManaged(&hit_dist, N * sizeof(float)));
    checkCudaErrors(cudaMallocManaged(&hit_flag, N * sizeof(int)));
    checkCudaErrors(cudaMallocManaged(&hit_count, N * sizeof(int)));
    checkCudaErrors(cudaMallocManaged(&accum, sizeof(cuFloatComplex)));

    // our world of hitable objects
    hitable **d_list, **d_world;
    checkCudaErrors(cudaMalloc(&d_list, sizeof(hitable*)));
    checkCudaErrors(cudaMalloc(&d_world, sizeof(hitable*)));
    create_world<<<1,1>>>(d_list, d_world);
    checkCudaErrors(cudaDeviceSynchronize());
    std::cerr << "│  Managed memory and world setup complete.\n";
    printEndSeparator();

    // the output file that is read in python
    std::ofstream outFile("rcs_results.csv");
    outFile << "# Frequency: " << freq << "\n# Grid: " << nx << "x" << ny << "\n";
    outFile << "# GridSize: " << grid_size << "\n# ThetaSamples: " << theta_samples << "\n";
    outFile << "# PhiSamples: " << phi_samples << "\n" << "theta,phi,rcs_m2,rcs_dbsm\n";

    // define the theads and the blocks
    // @@ we may need some changes here to make everything more efficient
    dim3 threads(16, 16);
    dim3 blocks((nx + 15) / 16, (ny + 15) / 16);

    // here we start the calculation for the MONOSTATIC calculation
    // we move the launcher and the receiver (which coincide) on theta and phi

    printSeparator("EXECUTING SWEEP");
    clock_t total_sweep_start = clock();
    int total_iterations = 0;

    // theta loop
    for (int t = 0; t < theta_samples; ++t) {
        float theta_deg = theta_start + (theta_samples > 1 ? t * (theta_end - theta_start) / (theta_samples - 1) : 0);
        float theta_rad = theta_deg * M_PI / 180.0f;

        // phi loop
        for (int p = 0; p < phi_samples; ++p) {
            clock_t iter_start = clock();
            
            float phi_deg = phi_start + (phi_samples > 1 ? p * (phi_end - phi_start) / (phi_samples - 1) : 0);
            float phi_rad = phi_deg * M_PI / 180.0f;

            vec3 dir(sinf(theta_rad)*cosf(phi_rad), sinf(theta_rad)*sinf(phi_rad), cosf(theta_rad));
            vec3 ray_dir = -dir;

            vec3 up(0, 1, 0);
            if (fabs(dir.y()) > 0.9f) up = vec3(1, 0, 0);
            vec3 u_vec = unit_vector(cross(up, dir)) * grid_size;
            vec3 v_vec = unit_vector(cross(dir, u_vec)) * grid_size;
            vec3 llc = (dir * distance_offset) - 0.5f * u_vec - 0.5f * v_vec; // here we move the lower left corner around

            accum->x = 0.0f; 
            accum->y = 0.0f;

            // we launch the rays

            // 1) single reflection
            // launcher<<<blocks, threads>>>(hit_pos, hit_normal, hit_dist, hit_flag, nx, ny, llc, u_vec, v_vec, ray_dir, d_world);
            // cudaDeviceSynchronize();


            // 2) multiple reflection
            launcher_multibounce<<<blocks, threads>>>(hit_pos, hit_normal, hit_dist, hit_count, nx, ny, llc, u_vec, v_vec, ray_dir, d_world, max_bounces);
            cudaDeviceSynchronize();

            // we compute the PO integral for the scattered field

            // 1) single reflection
            // integral_PO<<<(N+255)/256, 256>>>(hit_pos, hit_normal, hit_dist, hit_flag, N, k, ray_dir, ray_area, accum);
            // cudaDeviceSynchronize();


            // 2) multiple reflection
            integral_PO_multibounce<<<(N+255)/256, 256>>>(hit_pos, hit_normal, hit_dist, hit_count, N, k, ray_dir, ray_area, accum, reflection_const);
            cudaDeviceSynchronize();

            // we accumulate the fields
            float sigma = 4.0f * M_PI * (accum->x * accum->x + accum->y * accum->y);        // RCS in m^2
            float rcs_dBsm = 10.0f * log10f(fmaxf(sigma, 1e-10f));                          // RCS in dBsm

            // write in .csv for plotting
            outFile << theta_deg << "," << phi_deg << "," << sigma << "," << rcs_dBsm << "\n";
            
            clock_t iter_end = clock();
            double iter_time = double(iter_end - iter_start) / CLOCKS_PER_SEC;


            // @@ this can be done more efficiently
            // Stats regarding the hit count
            int max_h = 0;
            int zero_count = 0;
            long long sum_hits = 0; // 64 bit integer
            int non_zero_count = 0;

            for (int n = 0; n < N; n++) {
                int val = hit_count[n];
                if (val == 0) {
                    zero_count++;
                } else {
                    if (val > max_h) max_h = val;
                    sum_hits += val;
                    non_zero_count++;
                }
            }

            float avg_hits = (non_zero_count > 0) ? (float)sum_hits / non_zero_count : 0.0f;

            // prints
            std::cerr << "[" << std::setw(3) << total_iterations + 1 << "/" << phi_samples * theta_samples << "]"
                << " | Θ: " << std::setw(3) << (int)theta_deg << "°"
                << " | Φ: " << std::setw(3) << (int)phi_deg << "°"
                << " | " << formatTime(iter_time) 
                << " | Hit %: " << std::setprecision(1) << 100.0f - (100.0f * zero_count / N)  << "%" 
                << " | Avg of bounces: " << std::fixed << std::setprecision(2) << avg_hits // not counting zero bounces
                << " | Max bounces: " << max_h << " (" << max_bounces << ")"
                << "\n";

                        
            total_iterations++;
        }
    }
    
    clock_t total_sweep_end = clock();
    double total_time = double(total_sweep_end - total_sweep_start) / CLOCKS_PER_SEC;
    printEndSeparator();

    // summary
    printSeparator("PERFORMANCE SUMMARY");
    printKV(" Total Sweep Time", formatTime(total_time));
    printKV(" Total Points", total_iterations);
    printKV(" Average Time/Point", formatTime(total_time / total_iterations));
    printEndSeparator();

    // free memory
    printSeparator("CLEANUP");
    free_world<<<1,1>>>(d_list, d_world);
    cudaDeviceSynchronize();
    cudaFree(hit_pos); cudaFree(hit_normal); cudaFree(hit_dist);
    cudaFree(hit_flag); cudaFree(accum); cudaFree(d_list); cudaFree(d_world);
    cudaFree(hit_count);
    std::cerr << "│  Device memory released. Success.\n";
    printEndSeparator();

    return 0;
}




/*

Some TODOs:

- Make geometry reading more general
- Due to low precision there is a dependency of the results with the frequency which is problematic...
- Verify results for double sphere with thesis
- check result for the sphere with results at: https://www.flexcompute.com/tidy3d/examples/notebooks/PECSphereRCS/ (Mie series)
    i.  Make sure to do in 3D
    ii. Can we also get analytical results for the 2-sphere system? Not really, but can compare with other simulations
- issue: for the sphere there should be multiple hits (max value should always be 1...), maybe set a threshold ?

- also get an optimized Makefile for:
    a. debugging / profiling
    b. with optimization flags

- for the CUDA part i believe there could be a lot of omtimization that can be done!




*/