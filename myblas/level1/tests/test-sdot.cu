#include <stdio.h>

#include "level1/level1.h"

int main() {
    int N = 5;
    size_t size = N * sizeof(float);

    // Allocate h_A h_B in host memory
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_result = (float*)malloc(sizeof(float));
    // Initialize inputs
    for (int i = 0; i < N; i++) {
        h_A[i] = i;
        h_B[i] = N - 1 - i;
    }

    // Allocate vectors in device memory
    float* d_A;
    cudaMalloc(&d_A, size);
    float* d_B;
    cudaMalloc(&d_B, size);
    float* d_result;
    cudaMalloc(&d_result, sizeof(float));

    // Copy vectors from host to device

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    myblas_SDOT<<<1, N>>>(N, d_A, 1, d_B, 1, d_result);

    cudaMemcpy(h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    exit(!(*h_result == 10.0));
}