#include <stdio.h>

#include "level2/level2.h"

int main() {
    int N = 4;
    size_t size = N * sizeof(float);

    // Allocate h_A h_B in host memory
    static float h_A[16] = {1, 2,  3,  4,  5,  6,  7,  8,
                            9, 10, 11, 12, 13, 14, 15, 16};
    float* h_x = (float*)malloc(size);
    float* h_y = (float*)malloc(size);
    float* h_alpha = (float*)malloc(sizeof(float));

    // Initialize inputs
    *h_alpha = 1;
    for (int i = 0; i < N; i++) {
        h_x[i] = 1;
        h_y[i] = 1;
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

    // Copy vectors from host to device
    cudaMemcpy(d_A, h_A, N * size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_alpha, h_alpha, sizeof(float), cudaMemcpyHostToDevice);
    int numBlocks = 1;
    dim3 threadsPerBlock(N, N);
    myblas_SGER<<<numBlocks, threadsPerBlock>>>(N, N, d_alpha, d_x, 1, d_y, 1,
                                                d_A, 1);

    cudaMemcpy(h_A, d_A, size * N, cudaMemcpyDeviceToHost);
    int result = 0;
    for (int i = 0; i < N * N; i++) {
        if (abs(((float)i + 2.0) - h_A[i]) > 0.01) {
            result = 1;
        }
    }
    exit(result);
}