#pragma once

// ================ LEVEL 1 APIS ================
void cublas_api_SAXPY(int n, const float *alpha, const float *x, float *y);

void cublas_api_SDOT(int n, const float *x, const float *y, float *result);

// ================ LEVEL 2 APIS ================
void cublas_api_SGEMV(int m, int n, const float *alpha, const float *A,
                      const float *x, const float *beta, float *y);

// Assumed to be upper triangular and not unit triangular
void cublas_api_STRSV(int n, const float *A, float *x);

// ================ LEVEL 3 APIS ================
void cublas_api_SGEMM(int m, int n, int k, const float *alpha, const float *A,
                      const float *B, const float *beta, float *C);

// Assumed to be upper triangular and not unit triangular
void cublas_api_STRSM(int m, int n, const float *alpha, const float *A,
                      float *B);