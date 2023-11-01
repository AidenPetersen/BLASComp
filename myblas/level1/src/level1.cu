#include <math.h>
#include <stdio.h>

#include <cub/cub.cuh>

#include "level1/level1.h"

__global__ void myblas_SROTG(float *a, float *b, float *c, float *s) {
    int r = abs(*a) > abs(*b)
                ? ((*a > 0) - (*a < 0)) * sqrt((*a) * (*a) + (*b) * (*b))
                : ((*b > 0) - (*b < 0)) * sqrt((*a) * (*a) + (*b) * (*b));

    *c = r != 0 ? *a / r : 1;
    *s = r != 0 ? (*b) / r : 0;

    int z;
    if (abs(*a) > abs(*b)) {
        z = (*s);
    } else if (abs(*a) <= abs(*b) && (*c) != 0 && r != 0) {
        z = 1 / (*c);
    } else if (abs(*a) <= abs(*b) && c == 0 && r != 0) {
        z = 1;
    } else {
        z = 0;
    }

    *a = r;
    *b = z;
}

// Not implemented, cannot find good documentation, rarely used, not
// parallelizable
__global__ void myblas_SROTMG(float *d1, float *d2, float *x1, const float *y1,
                              float *param) {}

// Not implemented, rarely used, not parallelizable
__global__ void myblas_SROT(int n, float *x, int incx, float *y, int incy,
                            const float *c, const float *s) {}

// Not implemented, rarely used, not parallelizable
__global__ void myblas_ROTM(int n, float *x, int incx, float *y, int incy,
                            const float *param) {}

__global__ void myblas_SSWAP(int n, float *x, int incx, float *y, int incy) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    float t = x[i * incx];
    x[i * incx] = y[i * incy];
    y[i * incy] = t;
}

__global__ void myblas_SSCAL(int n, const float *alpha, float *x, int incx) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    x[i * incx] = (*alpha) * x[i * incx];
}

__global__ void myblas_SAXPY(int n, const float *alpha, const float *x,
                             int incx, float *y, int incy) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    y[i * incy] = (*alpha) * x[i * incx] + y[i * incy];
}

__global__ void myblas_SDOT(int n, const float *x, int incx, const float *y,
                            int incy, float *result) {
    __shared__ float *s;

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i == 0) {
        s = new float(n);
    }
    __syncthreads();

    s[i] = x[i * incx] * y[i * incy];

    void *d_temp_storage = NULL;

    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, s, result, n);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, s, result, n);
    cudaFree(&d_temp_storage);
    if (i == 0) {
        delete s;
    }
    __syncthreads();
}

__global__ void myblas_SNRM2(int n, const float *x, int incx, float *result) {
    __shared__ float *s;
    __shared__ float *r;

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i == 0) {
        s = new float(n);
        r = new float(1);
    }
    __syncthreads();
    s[i] = x[i * incx] * x[i * incx];

    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, s, r, n);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, s, r, n);
    cudaFree(&d_temp_storage);
    if (i == 0) {
        *result = sqrt(*r);
        delete r;
        delete s;
    }
    __syncthreads();
}

__global__ void myblas_SASUM(int n, const float *x, int incx, float *result) {
    __shared__ float *s;

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i == 0) {
        s = new float(n);
    }
    __syncthreads();
    s[i] = abs(x[i * incx]);

    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, s, result, n);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, s, result, n);
    cudaFree(&d_temp_storage);
    if (i == 0) {
        delete s;
    }
    __syncthreads();
}

__global__ void myblas_SAMAX(int n, const float *x, int incx, int *result) {
    __shared__ float *s;
    __shared__ cub::KeyValuePair<int, int> *r;

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i == 0) {
        s = new float(n);
        r = new cub::KeyValuePair<int, int>();
    }
    __syncthreads();
    s[i] = abs(x[i * incx]);

    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::ArgMax(d_temp_storage, temp_storage_bytes, s, r, n);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceReduce::ArgMax(d_temp_storage, temp_storage_bytes, s, r, n);
    cudaFree(&d_temp_storage);
    if (i == 0) {
        *result = r->key;
        delete s;
        delete r;
    }
    __syncthreads();
}

__global__ void myblas_SAMIN(int n, const float *x, int incx, int *result) {
    __shared__ float *s;
    __shared__ cub::KeyValuePair<int, int> *r;

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i == 0) {
        s = new float(n);
        r = new cub::KeyValuePair<int, int>();
    }
    __syncthreads();
    s[i] = abs(x[i * incx]);

    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::ArgMin(d_temp_storage, temp_storage_bytes, s, r, n);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceReduce::ArgMin(d_temp_storage, temp_storage_bytes, s, r, n);
    cudaFree(&d_temp_storage);
    if (i == 0) {
        *result = r->key;
        delete s;
        delete r;
    }
    __syncthreads();
}