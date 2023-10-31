#include "level1/level1.h"
#include "stdio.h"
#include "math.h"

__global__ void myblas_SROTG(float *a, float *b, float *c, float *s){
    int r = abs(*a) > abs(*b) ? 
        ((*a > 0) - (*a < 0)) * sqrt((*a) * (*a) + (*b) * (*b)) :
        ((*b > 0) - (*b < 0)) * sqrt((*a) * (*a) + (*b) * (*b));

    
    *c = r != 0 ? *a / r : 1;
    *s = r != 0 ? (*b)/r : 0;

    int z;
    if(abs(*a) > abs(*b)){
        z = (*s);
    } else if(abs(*a) <= abs(*b) && (*c) != 0 && r != 0){
        z = 1/(*c);
    } else if(abs(*a) <= abs(*b) && c == 0 && r != 0){
        z = 1;
    } else {
        z = 0;
    }

    *a = r;
    *b = z;
}

// Not implemented, cannot find good documentation, rarely used, not parallelizable
__global__ void myblas_SROTMG(float *d1, float *d2, float *x1, const float *y1, float *param){}

// Not implemented, rarely used, not parallelizable
__global__ void myblas_SROT(int n, float *x, int incx, float *y, int incy, const float *c, const float *s){}

// Not implemented, rarely used, not parallelizable
__global__ void myblas_ROTM(int n, float  *x, int incx, float  *y, int incy, const float*  param){}

__global__ void myblas_SSWAP(int n, float *x, int incx, float *y, int incy){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    float t = x[i * incx];
    x[i * incx] = y[i * incy];
    y[i * incy] = t;
}

__global__ void myblas_SSCAL(int n, const float *alpha, float* x, int incx){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    x[i * incx] =  (*alpha) * x[i * incx];
}

__global__ void myblas_SAXPY(int n, const float *alpha, const float* x, int incx, float *y, int incy){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    y[i * incy] = (*alpha) * x[i * incx] + y[i * incy];
}