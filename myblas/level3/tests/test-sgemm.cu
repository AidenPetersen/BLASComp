#include <stdio.h>

#include "level3/level3.h"

int main() {
    int n = 4;
    int N = n * n;
    size_t size = N * sizeof(float);

    // Allocate h_A h_B in host memory
    float h_C_final[N] = {90,  100, 110, 120, 202, 228, 254, 280,
                          314, 356, 398, 440, 426, 484, 542, 600};
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);
    float* h_alpha = (float*)malloc(sizeof(float));
    float* h_beta = (float*)malloc(sizeof(float));

    // Initialize inputs
    *h_alpha = 1;
    *h_beta = 1;
    for (int i = 0; i < N; i++) {
        h_A[i] = i + 1;
        h_B[i] = i + 1;
        h_C[i] = 0;
    }
    // Allocate vectors in device memory
    float* d_A;
    cudaMalloc(&d_A, size);
    float* d_B;
    cudaMalloc(&d_B, size);
    float* d_C;
    cudaMalloc(&d_C, size);
    float* d_alpha;
    cudaMalloc(&d_alpha, sizeof(float));
    float* d_beta;
    cudaMalloc(&d_beta, sizeof(float));

    // Copy vectors from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_alpha, h_alpha, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, h_beta, sizeof(float), cudaMemcpyHostToDevice);

    int numBlocks = 1;
    dim3 threadsPerBlock(n, n);
    myblas_SGEMM<<<numBlocks, threadsPerBlock>>>(
        'N', 'N', 4, 4, 4, d_alpha, d_A, 1, d_B, 1, d_beta, d_C, 1);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    int result = 0;
    for (int i = 0; i < N; i++) {
        if (h_C[i] != h_C_final[i]) {
            result = 1;
        }
    }
    exit(result);
}