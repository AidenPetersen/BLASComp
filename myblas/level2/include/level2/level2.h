#pragma once

__global__ void myblas_SGEMV(char trans, int m, int n, const float *alpha,
                             const float *A, int lda, const float *x, int incx,
                             const float *beta, float *y, int incy);

__global__ void myblas_SGBMV(char trans, int m, int n, int kl, int ku,
                             const float *alpha, const float *A, int lda,
                             const float *x, int incx, const float *beta,
                             float *y, int incy);

__global__ void myblas_SSYMV(char uplo, int n, const float *alpha,
                             const float *A, int lda, const float *x, int incx,
                             const float *beta, float *y, int incy);

__global__ void myblas_SSBMV(char uplo, int n, int k, const float *alpha,
                             const float *A, int lda, const float *x, int incx,
                             const float *beta, float *y, int incy);

// Not implemented. These packed functions are destroying my mind.
__global__ void myblas_SSPMV(char uplo, int n, const float *alpha,
                             const float *AP, const float *x, int incx,
                             const float *beta, float *y, int incy);

__global__ void myblas_STBMV(char uplo, char trans, int n, int k,
                             const float *A, int lda, float *x, int incx);

__global__ void myblas_STPMV(char uplo, char trans, char diag, int n,
                             const float *AP, float *x, int incx);

__global__ void myblas_STRSV(char uplo, char trans, char diag, int n,
                             const float *A, int lda, float *x, int incx);

__global__ void myblas_STBSV(char uplo, char trans, char diag, int n, int k,
                             const float *A, int lda, float *x, int incx);

__global__ void myblas_STPSV(char uplo, char trans, char diag, int n,
                             const float *AP, float *x, int incx);

__global__ void myblas_SGER(int m, int n, const float *alpha, const float *x,
                            int incx, const float *y, int incy, float *A,
                            int lda);

__global__ void myblas_SSYR(char uplo, int n, const float *alpha,
                            const float *x, int incx, float *A, int lda);

__global__ void myblas_SSYR2(char uplo, int n, const float *alpha,
                             const float *x, int incx, const float *y, int incy,
                             float *A, int lda);