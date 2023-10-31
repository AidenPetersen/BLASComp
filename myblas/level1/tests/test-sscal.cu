#include "level1/level1.h"
#include <stdio.h>

int main(){
    int N = 5;
    size_t size = N * sizeof(float);

    // Allocate h_x h_alpha in host memory
    float* h_x = (float*)malloc(size);
    float* h_alpha = (float*)malloc(sizeof(float));

    // Initialize inputs
    *h_alpha = 3;
    for(int i = 0; i < 5; i++){
        h_x[i] = i;
    }

    // Allocate vectors in device memory
    float* d_x;
    cudaMalloc(&d_x, size);
    float* d_alpha;
    cudaMalloc(&d_alpha, sizeof(float));

    // Copy vectors from host to device
    cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_alpha, h_alpha, sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    myblas_SSCAL<<<blocksPerGrid, threadsPerBlock>>>(N, d_alpha, d_x, 1);

    cudaMemcpy(h_x, d_x, size, cudaMemcpyDeviceToHost);

    for(int i = 0; i < 5; i++){
        if(h_x[i] != i * (*h_alpha)){
            exit(1);
        }
    }
    exit(0);
}