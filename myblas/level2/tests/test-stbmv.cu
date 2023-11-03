#include <stdio.h>

#include "level2/level2.h"

int main() {
    int N = 4;
    size_t size = N * sizeof(float);

    // Allocate h_A h_B in host memory
    static const float h_A[16] = {1, 2, 3, 0, 0, 1, 2, 3,
                                  0, 0, 1, 2, 0, 0, 0, 1};
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

    // Copy vectors from host to device
    cudaMemcpy(d_A, h_A, N * size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);

    myblas_STBMV<<<1, N>>>('U', 'N', N, 2, d_A, 1, d_x, 1);

    cudaMemcpy(h_x, d_x, size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; i++) {
        printf("%f\n", h_x[i]);
    }
    exit(!(h_x[0] == 14 && h_x[1] == 20 && h_x[2] == 11 && h_x[3] == 4));
}