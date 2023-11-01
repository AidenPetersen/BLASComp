#include <math.h>
#include <stdio.h>

#include "level1/level1.h"

int main() {
    int N = 10;
    size_t size = N * sizeof(float);

    // Allocate h_A h_B in host memory
    float* h_A = (float*)malloc(size);
    float* h_result = (float*)malloc(sizeof(float));
    // Initialize inputs
    for (int i = 0; i < N; i++) {
        h_A[i] = i;
    }

    // Allocate vectors in device memory
    float* d_A;
    cudaMalloc(&d_A, size);
    float* d_result;
    cudaMalloc(&d_result, sizeof(float));

    // Copy vectors from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    myblas_SNRM2<<<1, N>>>(N, d_A, 1, d_result);

    cudaMemcpy(h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    float expected;
    for (int i = 0; i < N; i++) {
        expected += i * i;
    }
    expected = sqrt(expected);
    exit(!(*h_result == expected));
}