#include <math.h>
#include <stdio.h>

#include <cub/cub.cuh>

#include "level2/level2.h"

__global__ void myblas_SGEMV(char trans, int m, int n, const float *alpha,
                             const float *A, int lda, const float *x, int incx,
                             const float *beta, float *y, int incy) {
    // Not transpose
    if (trans == 'N') {
        int row = blockDim.x * blockIdx.x + threadIdx.x;
        y[row] = (*beta) * y[row * incy];
        for (int i = 0; i < m; i++) {
            y[row] += (*alpha) * A[row * n + i] * x[i * incx];
        }
    }
    // transpose
    else {
        int col = blockDim.x * blockIdx.x + threadIdx.x;
        y[col] = (*beta) * y[col];
        for (int i = 0; i < m; i++) {
            y[col] += (*alpha) * A[i * n + col] * x[i * incx];
        }
    }
}

__global__ void myblas_SGBMV(char trans, int m, int n, int kl, int ku,
                             const float *alpha, const float *A, int lda,
                             const float *x, int incx, const float *beta,
                             float *y, int incy) {
    // Not transpose
    if (trans == 'N') {
        int row = blockDim.x * blockIdx.x + threadIdx.x;
        for (int i = -kl; i <= ku; i++) {
            y[row] = (*beta) * y[row * incy];
            if (i + row >= 0 && i + row < n) {
                y[row] += (*alpha) * A[row * n + i + row] * x[(i + row) * incx];
            }
        }
    }
    // transpose
    else {
        int col = blockDim.x * blockIdx.x + threadIdx.x;
        for (int i = -ku; i <= kl; i++) {
            y[col] = (*beta) * y[col * incy];
            if (i + col >= 0 && i + col < n) {
                y[col] += (*alpha) * A[col * n + i + col] * x[(i + col) * incx];
            }
        }
    }
}

__global__ void myblas_SSYMV(char uplo, int n, const float *alpha,
                             const float *A, int lda, const float *x, int incx,
                             const float *beta, float *y, int incy) {
    // Upper portion
    if (uplo == 'U') {
        int row = blockDim.x * blockIdx.x + threadIdx.x;
        y[row] = (*beta) * y[row * incy];
        for (int i = row; i < row + n; i++) {
            // normal case
            if (i < n) {
                y[row] += (*alpha) * A[row * n + i] * x[i * incx];
            }
            // do transpose indexing when out of bounds
            else {
                y[row] += (*alpha) * A[(i % n) * n + row] * x[(i % n) * incx];
            }
        }
    }
    // Lower triangle
    else {
        int row = blockDim.x * blockIdx.x + threadIdx.x;
        y[row] = (*beta) * y[row * incy];
        for (int i = 0; i < n; i++) {
            // normal case
            if (i <= row) {
                y[row] += (*alpha) * A[row * n + i] * x[i * incx];
            }
            // do transpose indexing when out of bounds
            else {
                y[row] += (*alpha) * A[i * n + row] * x[i * incx];
            }
        }
    }
}

__global__ void myblas_SSBMV(char uplo, int n, int k, const float *alpha,
                             const float *A, int lda, const float *x, int incx,
                             const float *beta, float *y, int incy) {
    // Upper portion
    if (uplo == 'U') {
        int row = blockDim.x * blockIdx.x + threadIdx.x;
        y[row] = (*beta) * y[row * incy];
        for (int i = row - k; i <= row + k; i++) {
            if (i >= 0 && i < n) {
                // normal case
                if (i >= row) {
                    y[row] += (*alpha) * A[row * n + i] * x[i * incx];
                    printf("thread: %d row: %d col: %d\n", row, row, i);
                }
                // do transpose indexing when out of bounds
                else {
                    y[row] += (*alpha) * A[i * n + row] * x[i * incx];
                    printf("thread: %d row: %d col: %d\n", row, i, row);
                }
            }
        }
    }
    // Lower triangle
    else {
        int row = blockDim.x * blockIdx.x + threadIdx.x;
        y[row] = (*beta) * y[row * incy];
        for (int i = row - k; i <= row + k; i++) {
            if (i >= 0 && i < n) {
                // normal case
                if (i <= row) {
                    y[row] += (*alpha) * A[row * n + i] * x[i * incx];
                    printf("thread: %d row: %d col: %d\n", row, row, i);
                }
                // do transpose indexing when out of bounds
                else {
                    y[row] += (*alpha) * A[i * n + row] * x[i * incx];
                    printf("thread: %d row: %d col: %d\n", row, i, row);
                }
            }
        }
    }
}

__global__ void myblas_SSPMV(char uplo, int n, const float *alpha,
                             const float *AP, const float *x, int incx,
                             const float *beta, float *y, int incy) {
    // Not implmented (I am too dumb)
}

__global__ void myblas_STBMV(char uplo, char trans, int n, int k,
                             const float *A, int lda, float *x, int incx) {
    if (uplo == 'U') {
        // Not transpose
        if (trans == 'N') {
            int row = blockDim.x * blockIdx.x + threadIdx.x;
            float sum = 0;
            for (int i = 0; i <= k; i++) {
                if (i + row >= 0 && i + row < n) {
                    sum += A[row * n + i + row] * x[(i + row) * incx];
                }
            }
            x[row] = sum;
        }
        // transpose
        else {
            int col = blockDim.x * blockIdx.x + threadIdx.x;
            float sum = 0;
            for (int i = -k; i <= 0; i++) {
                if (i + col >= 0 && i + col < n) {
                    sum += A[col * n + i + col] * x[(i + col) * incx];
                }
            }
            x[col] = sum;
        }
    } else {
        // Not transpose
        if (trans == 'N') {
            int row = blockDim.x * blockIdx.x + threadIdx.x;
            float sum = 0;
            for (int i = -k; i <= 0; i++) {
                if (i + row >= 0 && i + row < n) {
                    sum += A[row * n + i + row] * x[(i + row) * incx];
                }
            }
            x[row] = sum;
        }
        // transpose
        else {
            int col = blockDim.x * blockIdx.x + threadIdx.x;
            float sum = 0;
            for (int i = 0; i <= k; i++) {
                if (i + col >= 0 && i + col < n) {
                    sum += A[col * n + i + col] * x[(i + col) * incx];
                }
            }
            x[col] = sum;
        }
    }
}

__global__ void myblas_STPMV(char uplo, char trans, char diag, int n,
                             const float *AP, float *x, int incx) {
    // packed is still too complex
}

__global__ void myblas_STRSV(char uplo, char trans, char diag, int n,
                             const float *A, int lda, float *x, int incx) {}