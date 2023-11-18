#include "openblas.h"

// ================ LEVEL 1 APIS ================
void openblas_api_SAXPY(int n, const float *alpha, const float *x, float *y) {
    cblas_saxpy(n, *alpha, x, 1, y, 1);
}

void openblas_api_SDOT(int n, const float *x, const float *y, float *result) {
    cblas_sdot(n, x, 1, y, 1);
}

// ================ LEVEL 2 APIS ================
void openblas_api_SGEMV(int m, int n, const float *alpha, const float *A,
                        const float *x, const float *beta, float *y) {
    cblas_sgemv(CblasRowMajor, CblasNoTrans, m, n, *alpha, A, m, x, 1, *beta, y,
                1);
}

void openblas_api_STRSV(int n, const float *A, float *x) {
    cblas_strsv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit, n, A, n,
                x, 1);
}

// ================ LEVEL 3 APIS ================
void openblas_api_SGEMM(int m, int n, int k, const float *alpha, const float *A,
                        const float *B, const float *beta, float *C) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, *alpha, A,
                m, B, k, *beta, C, m);
}

void openblas_api_STRSM(int m, int n, const float *alpha, const float *A,
                        float *B) {
    cblas_strsm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans,
                CblasNonUnit, m, n, *alpha, A, m, B, m);
}