#pragma once

__global__ void myblas_SROTG(float *a, float *b, float *c, float *s);

__global__ void myblas_SROTMG(float *d1, float *d2, float *x1, const float *y1,
                              float *param);

__global__ void myblas_SROT(int n, float *x, int incx, float *y, int incy,
                            const float *c, const float *s);

__global__ void myblas_ROTM(int n, float *x, int incx, float *y, int incy,
                            const float *param);

__global__ void myblas_SSWAP(int n, float *x, int incx, float *y, int incy);

__global__ void myblas_SSCAL(int n, const float *alpha, float *x, int incx);

__global__ void myblas_SAXPY(int n, const float *alpha, const float *x,
                             int incx, float *y, int incy);

__global__ void myblas_SDOT(int n, const float *x, int incx, const float *y,
                            int incy, float *result);

__global__ void myblas_SNRM2(int n, const float *x, int incx, float *result);

__global__ void myblas_SASUM(int n, const float *x, int incx, float *result);

__global__ void myblas_SAMAX(int n, const float *x, int incx, int *result);

__global__ void myblas_SAMIN(int n, const float *x, int incx, int *result);