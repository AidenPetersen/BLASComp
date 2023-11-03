#include <stdio.h>

#include "level2/level2.h"

int main() {
    int N = 6;
    size_t size = N * sizeof(float);

    // Allocate h_A h_B in host memory
    static const float h_A[36] = {1, 1, 0, 0, 0, 0, 1, 2, 1, 0, 0, 0,
                                  0, 1, 3, 1, 0, 0, 0, 0, 1, 4, 1, 0,
                                  0, 0, 0, 1, 5, 1, 0, 0, 0, 0, 1, 6};
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

    myblas_SGBMV<<<1, N>>>('N', N, N, 1, 1, d_alpha, d_A, 1, d_x, 1, d_beta,
                           d_y, 1);

    cudaMemcpy(h_y, d_y, size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; i++) {
        printf("%f\n", h_y[i]);
    }
    exit(!(h_y[0] == 3 && h_y[1] == 8 && h_y[2] == 15 && h_y[3] == 24 &&
           h_y[4] == 35 && h_y[5] == 41));
}