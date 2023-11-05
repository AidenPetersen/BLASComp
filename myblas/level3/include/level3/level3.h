#pragma once

__global__ void myblas_SGEMM(char transa, char transb, int m, int n, int k,
                             const float *alpha, const float *A, int lda,
                             const float *B, int ldb, const float *beta,
                             float *C, int LDC);

__global__ void myblas_SSYMM(char side, char uplo, int m, int n,
                             const float *alpha, const float *A, int lda,
                             const float *B, int ldb, const float *beta,
                             float *c, int ldc);

__global__ void myblas_SSYRK(char uplo, char trans, int n, int k,
                             const float *alpha, const float *A, int lda,
                             const float *beta, float *C, int ldc);

__global__ void myblas_SSYR2K(char uplo, char trans, int n, int k,
                              const float *alpha, const float *A, int lda,
                              const float *B, int ldb, const float *beta,
                              float *C, int ldc);

__global__ void myblas_STRMM(char side, char uplo, char transa, char diag,
                             int m, int n, const float *alpha, const float *A,
                             int lda, float *B, int ldb);

__global__ void myblas_STRSM(char side, char uplo, char transa, char diag,
                             int m, int n, const float *alpha, const float *A,
                             int lda, float *B, int ldb);