#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "cublas.h"
// ================ LEVEL 1 APIS ================
void cublas_api_SAXPY(int n, const float *alpha, const float *x, float *y) {
    cublasHandle_t cublasH = NULL;
    cublasCreate(&cublasH);

    int size = n * sizeof(float);
    // Allocate memory to GPU
    float *d_x;
    cudaMalloc(&d_x, size);
    float *d_y;
    cudaMalloc(&d_y, size);
    float *d_alpha;
    cudaMalloc(&d_alpha, sizeof(float));

    // Copy memory to GPU
    cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_alpha, alpha, sizeof(float), cudaMemcpyHostToDevice);

    // run
    cublasSaxpy(cublasH, n, d_alpha, d_x, 1, d_y, 1);

    // Copy memory back
    cudaMemcpy(y, d_y, size, cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_alpha);

    cublasDestroy(cublasH);
}

void cublas_api_SDOT(int n, const float *x, const float *y, float *result) {
    cublasHandle_t cublasH = NULL;
    cublasCreate(&cublasH);

    int size = n * sizeof(float);
    // Allocate memory to GPU
    float *d_x;
    cudaMalloc(&d_x, size);
    float *d_y;
    cudaMalloc(&d_y, size);
    float *d_result;
    cudaMalloc(&d_result, sizeof(float));

    // Copy memory to GPU
    cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, size, cudaMemcpyHostToDevice);

    // Initialize kernel and run function
    int threadsPerBlock = n;
    int numBlocks = 1;
    cublasSdot(cublasH, n, d_x, 1, d_y, 1, d_result);

    // Copy memory back
    cudaMemcpy(result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_result);
    cublasDestroy(cublasH);
}

// ================ LEVEL 2 APIS ================
void cublas_api_SGEMV(int m, int n, const float *alpha, const float *A,
                      const float *x, const float *beta, float *y) {
    cublasHandle_t cublasH = NULL;
    cublasCreate(&cublasH);
    int size_v = m * sizeof(float);
    int size_m = n * m * sizeof(float);

    // Allocate memory to gpu
    float *d_A;
    cudaMalloc(&d_A, size_m);
    float *d_x;
    cudaMalloc(&d_x, size_v);
    float *d_y;
    cudaMalloc(&d_y, size_v);
    float *d_alpha;
    cudaMalloc(&d_alpha, sizeof(float));
    float *d_beta;
    cudaMalloc(&d_beta, sizeof(float));

    // Copy memory to GPU
    cudaMemcpy(d_A, A, size_m, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, size_v, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, size_v, cudaMemcpyHostToDevice);
    cudaMemcpy(d_alpha, alpha, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, beta, sizeof(float), cudaMemcpyHostToDevice);

    // Run Kernel
    cublasSgemv(cublasH, CUBLAS_OP_N, m, n, d_alpha, d_A, 1, d_x, 1, d_beta,
                d_y, 1);
    // Copy memory back from GPU
    cudaMemcpy(y, d_y, size_v, cudaMemcpyDeviceToHost);

    // Free memory in GPU
    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_alpha);
    cudaFree(d_beta);
    cublasDestroy(cublasH);
}

// Assumed to be upper triangular and not unit triangular
void cublas_api_STRSV(int n, const float *A, float *x) {
    cublasHandle_t cublasH = NULL;
    cublasCreate(&cublasH);
    int size_v = n * sizeof(float);
    int size_m = n * n * sizeof(float);

    // Allocate memory in GPU
    float *d_A;
    cudaMalloc(&d_A, size_m);
    float *d_x;
    cudaMalloc(&d_x, size_v);

    // Copy memory to GPU
    cudaMemcpy(d_A, A, size_m, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, size_v, cudaMemcpyHostToDevice);

    // Execute kernel
    cublasStrsv(CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n,
                d_A, 1, d_x, 1);

    // Free memory in GPU
    cudaFree(d_A);
    cudaFree(d_x);
    cublasDestroy(cublasH);
}

// ================ LEVEL 3 APIS ================
void cublas_api_SGEMM(int m, int n, int k, const float *alpha, const float *A,
                      const float *B, const float *beta, float *C) {
    cublasHandle_t cublasH = NULL;
    cublasCreate(&cublasH);
    int size_A = m * k * sizeof(float);
    int size_B = n * k * sizeof(float);
    int size_C = m * n * sizeof(float);
    // Allocate memory in GPU
    float *d_A;
    cudaMalloc(&d_A, size_A);
    float *d_B;
    cudaMalloc(&d_B, size_B);
    float *d_C;
    cudaMalloc(&d_C, size_C);
    float *d_alpha;
    cudaMalloc(&d_alpha, sizeof(float));
    float *d_beta;
    cudaMalloc(&d_beta, sizeof(float));
    // Copy memory to GPU
    cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, size_C, cudaMemcpyHostToDevice);
    cudaMemcpy(d_alpha, alpha, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, beta, sizeof(float), cudaMemcpyHostToDevice);

    // Run kernel
    cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, d_alpha, d_A, m,
                d_B, k, d_beta, d_C, m);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_alpha);
    cudaFree(d_beta);
    cublasDestroy(cublasH);
}

// Assumed to be upper triangular and not unit triangular
void cublas_api_STRSM(int m, int n, const float *alpha, const float *A,
                      float *B) {
    cublasHandle_t cublasH = NULL;
    cublasCreate(&cublasH);
    int size = n * m * sizeof(float);
    // Allocate memory in GPU
    float *d_A;
    cudaMalloc(&d_A, size);
    float *d_B;
    cudaMalloc(&d_B, size);
    float *d_alpha;
    cudaMalloc(&d_alpha, sizeof(float));

    // Copy memory to GPU
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_alpha, alpha, sizeof(float), cudaMemcpyHostToDevice);

    // Run kernel
    cublasStrsm(cublasH, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                CUBLAS_DIAG_NON_UNIT, m, n, d_alpha, d_A, m, d_B, m);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_alpha);
    cublasDestroy(cublasH);
}