#include "level1/level1.h"
#include <stdio.h>

int main(){
    int N = 5;
    size_t size = N * sizeof(float);

    // Allocate h_A h_B in host memory
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);

    // Initialize inputs
    for(int i = 0; i < N; i++){
        h_A[i] = i;
        h_B[i] = N - 1 - i;
    }


    // Allocate vectors in device memory
    float* d_A;
    cudaMalloc(&d_A, size);
    float* d_B;
    cudaMalloc(&d_B, size);

    // Copy vectors from host to device

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    myblas_SSWAP<<<blocksPerGrid, threadsPerBlock>>>(N, d_A, 1, d_B, 1);

    cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_B, d_B, size, cudaMemcpyDeviceToHost);

    for(int i = 0; i < N; i++){
        if(h_A[i] != N - 1 - i || h_B[i] != i){
            exit(1);
        }
    }
    exit(0);
}