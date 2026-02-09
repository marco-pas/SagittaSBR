#ifndef VEC3H
#define VEC3H

#include <math.h>
#include <stdlib.h>
#include <iostream>

#include "precision.h"

#if !defined(__CUDACC__) && !defined(__HIPCC__)
#define __host__
#define __device__
#endif

template<typename T>
class vec3_t  {

/* 

__host__ __device__ tells the compiler to produce a CPU version and GPU version

Templated on T for single/double precision support.

*/

public:
    __host__ __device__ vec3_t() {}
    __host__ __device__ vec3_t(T e0, T e1, T e2) { e[0] = e0; e[1] = e1; e[2] = e2; }
    __host__ __device__ inline T x() const { return e[0]; }
    __host__ __device__ inline T y() const { return e[1]; }
    __host__ __device__ inline T z() const { return e[2]; }
    __host__ __device__ inline T r() const { return e[0]; }
    __host__ __device__ inline T g() const { return e[1]; }
    __host__ __device__ inline T b() const { return e[2]; }

    __host__ __device__ inline const vec3_t& operator+() const { return *this; }
    __host__ __device__ inline vec3_t operator-() const { return vec3_t(-e[0], -e[1], -e[2]); }
    __host__ __device__ inline T operator[](int i) const { return e[i]; }
    __host__ __device__ inline T& operator[](int i) { return e[i]; };

    __host__ __device__ inline vec3_t& operator+=(const vec3_t &v2);
    __host__ __device__ inline vec3_t& operator-=(const vec3_t &v2);
    __host__ __device__ inline vec3_t& operator*=(const vec3_t &v2);
    __host__ __device__ inline vec3_t& operator/=(const vec3_t &v2);
    __host__ __device__ inline vec3_t& operator*=(const T t);
    __host__ __device__ inline vec3_t& operator/=(const T t);

    __host__ __device__ inline T length() const { return sqrt(e[0]*e[0] + e[1]*e[1] + e[2]*e[2]); }
    __host__ __device__ inline T squaredLength() const { return e[0]*e[0] + e[1]*e[1] + e[2]*e[2]; }
    __host__ __device__ inline void makeUnitVector();

    T e[3];
};


template<typename T>
inline std::istream& operator>>(std::istream &is, vec3_t<T> &t) {
    is >> t.e[0] >> t.e[1] >> t.e[2];
    return is;
}

template<typename T>
inline std::ostream& operator<<(std::ostream &os, const vec3_t<T> &t) {
    os << t.e[0] << " " << t.e[1] << " " << t.e[2];
    return os;
}

template<typename T>
__host__ __device__ inline void vec3_t<T>::makeUnitVector() {
    T k = T(1.0) / sqrt(e[0]*e[0] + e[1]*e[1] + e[2]*e[2]);
    e[0] *= k; e[1] *= k; e[2] *= k;
}

template<typename T>
__host__ __device__ inline vec3_t<T> operator+(const vec3_t<T> &v1, const vec3_t<T> &v2) {
    return vec3_t<T>(v1.e[0] + v2.e[0], v1.e[1] + v2.e[1], v1.e[2] + v2.e[2]);
}

template<typename T>
__host__ __device__ inline vec3_t<T> operator-(const vec3_t<T> &v1, const vec3_t<T> &v2) {
    return vec3_t<T>(v1.e[0] - v2.e[0], v1.e[1] - v2.e[1], v1.e[2] - v2.e[2]);
}

template<typename T>
__host__ __device__ inline vec3_t<T> operator*(const vec3_t<T> &v1, const vec3_t<T> &v2) {
    return vec3_t<T>(v1.e[0] * v2.e[0], v1.e[1] * v2.e[1], v1.e[2] * v2.e[2]);
}

template<typename T>
__host__ __device__ inline vec3_t<T> operator/(const vec3_t<T> &v1, const vec3_t<T> &v2) {
    return vec3_t<T>(v1.e[0] / v2.e[0], v1.e[1] / v2.e[1], v1.e[2] / v2.e[2]);
}

template<typename T>
__host__ __device__ inline vec3_t<T> operator*(T t, const vec3_t<T> &v) {
    return vec3_t<T>(t*v.e[0], t*v.e[1], t*v.e[2]);
}

template<typename T>
__host__ __device__ inline vec3_t<T> operator/(vec3_t<T> v, T t) {
    return vec3_t<T>(v.e[0]/t, v.e[1]/t, v.e[2]/t);
}

template<typename T>
__host__ __device__ inline vec3_t<T> operator*(const vec3_t<T> &v, T t) {
    return vec3_t<T>(t*v.e[0], t*v.e[1], t*v.e[2]);
}

template<typename T>
__host__ __device__ inline T dot(const vec3_t<T> &v1, const vec3_t<T> &v2) {
    return v1.e[0] *v2.e[0] + v1.e[1] *v2.e[1]  + v1.e[2] *v2.e[2];
}

template<typename T>
__host__ __device__ inline vec3_t<T> cross(const vec3_t<T> &v1, const vec3_t<T> &v2) {
    return vec3_t<T>( (v1.e[1]*v2.e[2] - v1.e[2]*v2.e[1]),
                (-(v1.e[0]*v2.e[2] - v1.e[2]*v2.e[0])),
                (v1.e[0]*v2.e[1] - v1.e[1]*v2.e[0]));
}

template<typename T>
__host__ __device__ inline vec3_t<T>& vec3_t<T>::operator+=(const vec3_t &v){
    e[0]  += v.e[0];
    e[1]  += v.e[1];
    e[2]  += v.e[2];
    return *this;
}

template<typename T>
__host__ __device__ inline vec3_t<T>& vec3_t<T>::operator*=(const vec3_t &v){
    e[0]  *= v.e[0];
    e[1]  *= v.e[1];
    e[2]  *= v.e[2];
    return *this;
}

template<typename T>
__host__ __device__ inline vec3_t<T>& vec3_t<T>::operator/=(const vec3_t &v){
    e[0]  /= v.e[0];
    e[1]  /= v.e[1];
    e[2]  /= v.e[2];
    return *this;
}

template<typename T>
__host__ __device__ inline vec3_t<T>& vec3_t<T>::operator-=(const vec3_t& v) {
    e[0]  -= v.e[0];
    e[1]  -= v.e[1];
    e[2]  -= v.e[2];
    return *this;
}

template<typename T>
__host__ __device__ inline vec3_t<T>& vec3_t<T>::operator*=(const T t) {
    e[0]  *= t;
    e[1]  *= t;
    e[2]  *= t;
    return *this;
}

template<typename T>
__host__ __device__ inline vec3_t<T>& vec3_t<T>::operator/=(const T t) {
    T k = T(1.0)/t;

    e[0]  *= k;
    e[1]  *= k;
    e[2]  *= k;
    return *this;
}

template<typename T>
__host__ __device__ inline vec3_t<T> unitVector(vec3_t<T> v) {
    return v / v.length();
}

// Type aliases - use Real from precision.h for seamless precision switching
using vec3 = vec3_t<Real>;
using vec3f = vec3_t<float>;
using vec3d = vec3_t<double>;

#endif
