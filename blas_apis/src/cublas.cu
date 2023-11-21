#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include "cublas.h"
// ================ LEVEL 1 APIS ================
void cublas_api_SAXPY(int n, const float *alpha, const float *x, float *y) {
    cublasHandle_t cublasH = NULL;

    cublasCreate(&cublasH);

    // Run function
    cublasSaxpy(cublasH, n, alpha, x, 1, y, 1);

    cublasDestroy(cublasH);
}

void cublas_api_SDOT(int n, const float *x, const float *y, float *result) {
    cublasHandle_t cublasH = NULL;
    cublasCreate(&cublasH);

    // Run function
    cublasSdot(cublasH, n, x, 1, y, 1, result);

    cublasDestroy(cublasH);
}

// ================ LEVEL 2 APIS ================
void cublas_api_SGEMV(int m, int n, const float *alpha, const float *A,
                      const float *x, const float *beta, float *y) {
    cublasHandle_t cublasH = NULL;
    cublasCreate(&cublasH);

    // Run function
    cublasSgemv(cublasH, CUBLAS_OP_N, m, n, alpha, A, m, x, 1, beta, y, 1);

    cublasDestroy(cublasH);
}

// Assumed to be upper triangular and not unit triangular
void cublas_api_STRSV(int n, const float *A, float *x) {
    cublasHandle_t cublasH = NULL;
    cublasCreate(&cublasH);

    // Run function
    cublasStrsv(cublasH, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                CUBLAS_DIAG_NON_UNIT, n, A, n, x, 1);

    cublasDestroy(cublasH);
}

// ================ LEVEL 3 APIS ================
void cublas_api_SGEMM(int m, int n, int k, const float *alpha, const float *A,
                      const float *B, const float *beta, float *C) {
    cublasHandle_t cublasH = NULL;
    cublasCreate(&cublasH);

    // Run kernel
    cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, alpha, A, m, B, k,
                beta, C, m);

    cublasDestroy(cublasH);
}

// Assumed to be upper triangular and not unit triangular
void cublas_api_STRSM(int m, int n, const float *alpha, const float *A,
                      float *B) {
    cublasHandle_t cublasH = NULL;
    cublasCreate(&cublasH);

    // Run function
    cublasStrsm(cublasH, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                CUBLAS_DIAG_NON_UNIT, m, n, alpha, A, m, B, m);

    cublasDestroy(cublasH);
}