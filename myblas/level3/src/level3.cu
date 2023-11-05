#include <math.h>
#include <stdio.h>

#include <cub/cub.cuh>

#include "level3/level3.h"

__global__ void myblas_SGEMM(char transa, char transb, int m, int n, int k,
                             const float *alpha, const float *A, int lda,
                             const float *B, int ldb, const float *beta,
                             float *C, int LDC) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    float sum = (*beta) * C[row * k + col];
    for (int i = 0; i < k; i++) {
        sum += A[k * row + i] * B[k * i + col];
    }
    C[k * row + col] = sum;
}

// I'm at the point where my sanity is disappearing and I don't want to
// implement different versions of the same functions. I'm skipping the next 4.
__global__ void myblas_SSYMM(char side, char uplo, int m, int n,
                             const float *alpha, const float *A, int lda,
                             const float *B, int ldb, const float *beta,
                             float *c, int ldc) {}

__global__ void myblas_SSYRK(char uplo, char trans, int n, int k,
                             const float *alpha, const float *A, int lda,
                             const float *beta, float *C, int ldc) {}

__global__ void myblas_SSYR2K(char uplo, char trans, int n, int k,
                              const float *alpha, const float *A, int lda,
                              const float *B, int ldb, const float *beta,
                              float *C, int ldc) {}

__global__ void myblas_STRMM(char side, char uplo, char transa, char diag,
                             int m, int n, const float *alpha, const float *A,
                             int lda, float *B, int ldb) {}

// This function is actually interesting.
__global__ void myblas_STRSM(char side, char uplo, char transa, char diag,
                             int m, int n, const float *alpha, const float *A,
                             int lda, float *B, int ldb) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = blockDim.x * blockIdx.y + threadIdx.y;
    float sum = B[row * n + col];
    if (uplo == 'L') {
        for (int i = 0; i < n; i++) {
            if (i == row) {
                if (diag != 'U') {
                    B[i * m + col] = sum / A[row * n + i];
                } else {
                    B[i * m + col] = sum;
                }
            }
            __syncthreads();
            if (i < row) {
                sum -= B[i * m + col] * A[row * n + i];
            }
            __syncthreads();
        }
    } else {
        for (int i = n - 1; i >= 0; i--) {
            if (i == row) {
                if (diag != 'U') {
                    B[i * m + col] = sum / A[row * n + i];
                } else {
                    B[i * m + col] = sum;
                }
            }
            __syncthreads();
            if (i < row) {
                sum -= B[i * m + col] * A[row * n + i];
            }
            __syncthreads();
        }
    }
}