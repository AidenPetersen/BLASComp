#include <math.h>
#include <stdio.h>

#include "level1/level1.h"

int main() {
    int N = 5;
    size_t size = N * sizeof(float);

    // Allocate h_A h_B in host memory
    float* h_A = (float*)malloc(size);
    int* h_result = (int*)malloc(sizeof(int));
    // Initialize inputs
    for (int i = 0; i < N; i++) {
        h_A[i] = i;
    }

    // Allocate vectors in device memory
    float* d_A;
    cudaMalloc(&d_A, size);
    int* d_result;
    cudaMalloc(&d_result, sizeof(int));

    // Copy vectors from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    myblas_SAMIN<<<1, N>>>(N, d_A, 1, d_result);

    cudaMemcpy(h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    float expected = 0;
    exit(!(*h_result == expected));
}