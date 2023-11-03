#include <stdio.h>

#include "level2/level2.h"

int main() {
    int N = 4;
    size_t size = N * sizeof(float);

    // Allocate h_A h_B in host memory
    static const float h_A[16] = {1, 2, 3, 0, 2, 1, 2, 3,
                                  3, 2, 1, 2, 0, 3, 2, 1};
    float* h_x = (float*)malloc(size);
    float* h_y = (float*)malloc(size);
    float* h_alpha = (float*)malloc(sizeof(float));
    float* h_beta = (float*)malloc(sizeof(float));

    // Initialize inputs
    *h_alpha = 1;
    *h_beta = 1;
    for (int i = 0; i < N; i++) {
        h_x[i] = i + 1;
        h_y[i] = 0;
    }
    // Allocate vectors in device memory
    float* d_A;
    cudaMalloc(&d_A, size * N);
    float* d_x;
    cudaMalloc(&d_x, size);
    float* d_y;
    cudaMalloc(&d_y, size);
    float* d_alpha;
    cudaMalloc(&d_alpha, sizeof(float));
    float* d_beta;
    cudaMalloc(&d_beta, sizeof(float));

    // Copy vectors from host to device
    cudaMemcpy(d_A, h_A, N * size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_alpha, h_alpha, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, h_beta, sizeof(float), cudaMemcpyHostToDevice);

    myblas_SSBMV<<<1, N>>>('U', N, 2, d_alpha, d_A, 1, d_x, 1, d_beta, d_y, 1);

    cudaMemcpy(h_y, d_y, size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; i++) {
        printf("%f\n", h_y[i]);
    }
    exit(!(h_y[0] == 14 && h_y[1] == 22 && h_y[2] == 18 && h_y[3] == 16));
}