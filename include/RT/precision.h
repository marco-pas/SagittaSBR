// #ifndef PRECISION_H
// #define PRECISION_H

// // ----------------------------------------------------------------------------
// // PRECISION CONFIGURATION
// // Controls single (float) vs double precision throughout the simulation.
// // Define USE_DOUBLE at compile time to switch all computations to double.
// //
// // Usage (CMake):
// //   cmake -DUSE_DOUBLE=ON ..
// //
// // Usage (manual):
// //   nvcc -DUSE_DOUBLE ...
// // ----------------------------------------------------------------------------

// #include <cmath>
// #include <cfloat>

// // Complex types are only available in CUDA/HIP compilation units
// #if defined(__CUDACC__) || defined(__HIPCC__)
//     #if defined(USE_HIP)
//     #include <hip/hip_complex.h>
//     #include "hip/hipifly.hpp"
//     #else
//     #include <cuComplex.h>
//     #endif
// #endif

// #ifdef USE_DOUBLE
//     using Real = double;

//     #if defined(__CUDACC__) || defined(__HIPCC__)
//     using cuRealComplex = cuDoubleComplex;
//     #define make_cuRealComplex make_cuDoubleComplex
//     #endif

//     #define REAL_CONST(x) x

//     // ---- Host math wrappers
//     inline double realSqrt(double x) { return std::sqrt(x); }
//     inline double realAbs(double x) { return std::fabs(x); }
//     inline double realFmax(double a, double b) { return std::fmax(a, b); }
//     inline double realFmin(double a, double b) { return std::fmin(a, b); }
//     inline double realLog10(double x) { return std::log10(x); }
//     inline double realPow(double x, double y) { return std::pow(x, y); }
//     inline double realFmod(double x, double y) { return std::fmod(x, y); }
//     inline double realSin(double x) { return std::sin(x); }
//     inline double realCos(double x) { return std::cos(x); }

//     // ---- Device math wrappers
//     #if defined(__CUDACC__) || defined(__HIPCC__)
//     __device__ __forceinline__ double devAbs(double x) { return fabs(x); }
//     __device__ __forceinline__ double devSqrt(double x) { return sqrt(x); }
//     __device__ __forceinline__ double devFmax(double a, double b) { return fmax(a, b); }
//     __device__ __forceinline__ double devFmin(double a, double b) { return fmin(a, b); }
//     __device__ __forceinline__ double devPow(double x, double y) { return pow(x, y); }
//     __device__ __forceinline__ double devFmod(double x, double y) { return fmod(x, y); }
//     __device__ __forceinline__ void devSinCos(double x, double* s, double* c) { sincos(x, s, c); }
//     __device__ __forceinline__ double devLog10(double x) { return log10(x); }
//     #endif

//     #define REAL_FLT_MAX 1.7976931348623158e+308
//     #define REAL_MPI_TYPE MPI_DOUBLE

// #else
//     using Real = float;

//     #if defined(__CUDACC__) || defined(__HIPCC__)
//     using cuRealComplex = cuFloatComplex;
//     #define make_cuRealComplex make_cuFloatComplex
//     #endif

//     #define REAL_CONST(x) x##f

//     // ---- Host math wrappers 
//     inline float realSqrt(float x) { return std::sqrtf(x); }
//     inline float realAbs(float x) { return std::fabsf(x); }
//     inline float realFmax(float a, float b) { return std::fmaxf(a, b); }
//     inline float realFmin(float a, float b) { return std::fminf(a, b); }
//     inline float realLog10(float x) { return std::log10f(x); }
//     inline float realPow(float x, float y) { return std::powf(x, y); }
//     inline float realFmod(float x, float y) { return std::fmodf(x, y); }
//     inline float realSin(float x) { return std::sinf(x); }
//     inline float realCos(float x) { return std::cosf(x); }

//     // ---- Device math wrappers 
//     #if defined(__CUDACC__) || defined(__HIPCC__)
//     __device__ __forceinline__ float devAbs(float x) { return fabsf(x); }
//     __device__ __forceinline__ float devSqrt(float x) { return sqrtf(x); }
//     __device__ __forceinline__ float devFmax(float a, float b) { return fmaxf(a, b); }
//     __device__ __forceinline__ float devFmin(float a, float b) { return fminf(a, b); }
//     __device__ __forceinline__ float devPow(float x, float y) { return powf(x, y); }
//     __device__ __forceinline__ float devFmod(float x, float y) { return fmodf(x, y); }
//     __device__ __forceinline__ void devSinCos(float x, float* s, float* c) { sincosf(x, s, c); }
//     __device__ __forceinline__ float devLog10(float x) { return log10f(x); }
//     #endif

//     #define REAL_FLT_MAX FLT_MAX
//     #define REAL_MPI_TYPE MPI_FLOAT

// #endif

// #endif // PRECISION_H




#ifndef PRECISION_H
#define PRECISION_H

// ----------------------------------------------------------------------------
// PRECISION CONFIGURATION
// Controls single (float) vs double precision throughout the simulation.
// Define USE_DOUBLE at compile time to switch all computations to double.
//
// Usage (CMake):
//   cmake -DUSE_DOUBLE=ON ..
//
// Usage (manual):
//   nvcc -DUSE_DOUBLE ...
// ----------------------------------------------------------------------------

#include <cmath>
#include <cfloat>

// Complex types are only available in CUDA/HIP compilation units
#if defined(__CUDACC__) || defined(__HIPCC__)
    #if defined(USE_HIP)
    #include <hip/hip_complex.h>
    #include "hip/hipifly.hpp"
    #else
    #include <cuComplex.h>
    #endif
#endif

#ifdef USE_DOUBLE
    using Real = double;

    #if defined(__CUDACC__) || defined(__HIPCC__)
    using cuRealComplex = cuDoubleComplex;
    #define make_cuRealComplex make_cuDoubleComplex
    #endif

    #define REAL_CONST(x) x

    // ---- Host math wrappers
    inline double realSqrt(double x) { return std::sqrt(x); }
    inline double realAbs(double x) { return std::fabs(x); }
    inline double realFmax(double a, double b) { return std::fmax(a, b); }
    inline double realFmin(double a, double b) { return std::fmin(a, b); }
    inline double realLog10(double x) { return std::log10(x); }
    inline double realPow(double x, double y) { return std::pow(x, y); }
    inline double realFmod(double x, double y) { return std::fmod(x, y); }
    inline double realSin(double x) { return std::sin(x); }
    inline double realCos(double x) { return std::cos(x); }

    // ---- Device math wrappers
    #if defined(__CUDACC__) || defined(__HIPCC__)
    __device__ __forceinline__ double devAbs(double x) { return fabs(x); }
    __device__ __forceinline__ double devSqrt(double x) { return sqrt(x); }
    __device__ __forceinline__ double devFmax(double a, double b) { return fmax(a, b); }
    __device__ __forceinline__ double devFmin(double a, double b) { return fmin(a, b); }
    __device__ __forceinline__ double devPow(double x, double y) { return pow(x, y); }
    __device__ __forceinline__ double devFmod(double x, double y) { return fmod(x, y); }
    __device__ __forceinline__ void devSinCos(double x, double* s, double* c) { sincos(x, s, c); }
    __device__ __forceinline__ double devLog10(double x) { return log10(x); }
    #endif

    #define REAL_FLT_MAX 1.7976931348623158e+308
    #define REAL_MPI_TYPE MPI_DOUBLE

#else
    using Real = float;

    #if defined(__CUDACC__) || defined(__HIPCC__)
    using cuRealComplex = cuFloatComplex;
    #define make_cuRealComplex make_cuFloatComplex
    #endif

    #define REAL_CONST(x) x##f

    // ---- Host math wrappers 
    inline float realSqrt(float x) { return std::sqrt(x); }
    inline float realAbs(float x) { return std::fabs(x); }
    inline float realFmax(float a, float b) { return std::fmax(a, b); }
    inline float realFmin(float a, float b) { return std::fmin(a, b); }
    inline float realLog10(float x) { return std::log10(x); }
    inline float realPow(float x, float y) { return std::pow(x, y); }
    inline float realFmod(float x, float y) { return std::fmod(x, y); }
    inline float realSin(float x) { return std::sin(x); }
    inline float realCos(float x) { return std::cos(x); }

    // ---- Device math wrappers 
    #if defined(__CUDACC__) || defined(__HIPCC__)
    __device__ __forceinline__ float devAbs(float x) { return fabsf(x); }
    __device__ __forceinline__ float devSqrt(float x) { return sqrtf(x); }
    __device__ __forceinline__ float devFmax(float a, float b) { return fmaxf(a, b); }
    __device__ __forceinline__ float devFmin(float a, float b) { return fminf(a, b); }
    __device__ __forceinline__ float devPow(float x, float y) { return powf(x, y); }
    __device__ __forceinline__ float devFmod(float x, float y) { return fmodf(x, y); }
    __device__ __forceinline__ void devSinCos(float x, float* s, float* c) { sincosf(x, s, c); }
    __device__ __forceinline__ float devLog10(float x) { return log10f(x); }
    #endif

    #define REAL_FLT_MAX FLT_MAX
    #define REAL_MPI_TYPE MPI_FLOAT

#endif

#endif // PRECISION_H