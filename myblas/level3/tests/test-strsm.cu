#include <stdio.h>

#include "level3/level3.h"

int main() {
    int n = 4;
    int N = n * n;
    size_t size = N * sizeof(float);

    // Allocate h_A h_B in host memory
    float h_B_final[N] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    static float h_A[16] = {1, 0, 0, 0, 2, 2, 0, 0, 5, 3, 3, 0, 7, 6, 4, 4};
    static float h_B[16] = {1, 0, 0, 0, 2, 2, 0, 0, 5, 3, 3, 0, 7, 6, 4, 4};
    float* h_alpha = (float*)malloc(sizeof(float));

    // Initialize inputs
    *h_alpha = 1;

    // Allocate vectors in device memory
    float* d_A;
    cudaMalloc(&d_A, size);
    float* d_B;
    cudaMalloc(&d_B, size);
    float* d_alpha;
    cudaMalloc(&d_alpha, sizeof(float));

    // Copy vectors from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_alpha, h_alpha, sizeof(float), cudaMemcpyHostToDevice);

    int numBlocks = 1;
    dim3 threadsPerBlock(n, n);
    myblas_STRSM<<<numBlocks, threadsPerBlock>>>('L', 'L', 'N', 'N', 4, 4,
                                                 d_alpha, d_A, 1, d_B, 1);

    cudaMemcpy(h_B, d_B, size, cudaMemcpyDeviceToHost);
    int result = 0;
    for (int i = 0; i < N; i++) {
        printf("%f\n", h_B[i]);
        if (h_B[i] != h_B_final[i]) {
            result = 1;
        }
    }
    exit(result);
}