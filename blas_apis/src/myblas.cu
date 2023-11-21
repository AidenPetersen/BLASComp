
#include <stdio.h>

#include "myblas.h"

#define gpuErrchk(ans) \
    { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
                line);
        // if (abort) exit(code);
    }
}

// ================ LEVEL 1 APIS ================
void myblas_api_SAXPY(int n, const float *alpha, const float *x, float *y) {
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

    // Initialize kernel and run function
    myblas_SAXPY<<<n / 128, 128>>>(n, d_alpha, d_x, 1, d_y, 1);

    // Copy memory back
    cudaMemcpy(y, d_y, size, cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_alpha);
}

void myblas_api_SDOT(int n, const float *x, const float *y, float *result) {
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
    myblas_SDOT<<<n / 128, 128>>>(n, d_x, 1, d_y, 1, d_result);

    // Copy memory back
    cudaMemcpy(result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_result);
}

// ================ LEVEL 2 APIS ================
void myblas_api_SGEMV(int m, int n, const float *alpha, const float *A,
                      const float *x, const float *beta, float *y) {
    int size_v = m * sizeof(float);
    int size_m = n * m * sizeof(float);

    // create memory stream

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
    myblas_SGEMV<<<n / 128, 128>>>('N', m, n, d_alpha, d_A, 1, d_x, 1, d_beta,
                                   d_y, 1);
    // Copy memory back from GPU
    cudaMemcpy(y, d_y, size_v, cudaMemcpyDeviceToHost);

    // Free memory in GPU
    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_alpha);
    cudaFree(d_beta);
}

// Assumed to be upper triangular and not unit triangular
void myblas_api_STRSV(int n, const float *A, float *x) {
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
    myblas_STRSV<<<n / 128, 128>>>('U', 'N', 'N', n, d_A, 1, d_x, 1);

    // Free memory in GPU
    cudaFree(d_A);
    cudaFree(d_x);
}

// ================ LEVEL 3 APIS ================
void myblas_api_SGEMM(int m, int n, int k, const float *alpha, const float *A,
                      const float *B, const float *beta, float *C) {
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
    dim3 threadsPerBlock(32, 32);
    myblas_SGEMM<<<n / 32, threadsPerBlock>>>('N', 'N', m, n, k, d_alpha, d_A,
                                              1, d_B, 1, d_beta, d_C, 1);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_alpha);
    cudaFree(d_beta);
}

// Assumed to be upper triangular and not unit triangular
void myblas_api_STRSM(int m, int n, const float *alpha, const float *A,
                      float *B) {
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
    dim3 threadsPerBlock(n, n);
    myblas_STRSM<<<n / 128, threadsPerBlock>>>('L', 'L', 'N', 'N', m, n,
                                               d_alpha, d_A, 1, d_B, 1);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_alpha);
}