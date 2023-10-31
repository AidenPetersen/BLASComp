#include "level1/level1.h"
#include <stdio.h>

int main(){

    // Allocate h_A h_B in host memory
    float h_A = 0.0;
    float h_B = 2.0;
    float h_C;
    float h_S;


    // Allocate vectors in device memory
    float* d_A;
    cudaMalloc(&d_A, 1);
    float* d_B;
    cudaMalloc(&d_B, 1);
    float* d_C;
    cudaMalloc(&d_C, 1);    
    float* d_S;
    cudaMalloc(&d_S, 1);

    // Copy vectors from host to device
    cudaMemcpy(d_A, &h_A, 1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, &h_B, 1, cudaMemcpyHostToDevice);

    myblas_SROTG<<<1, 1>>>(d_A, d_B, d_C, d_S);
    cudaMemcpy(&h_A, d_A, 1, cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_B, d_B, 1, cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_C, d_C, 1, cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_S, d_S, 1, cudaMemcpyDeviceToHost);

    exit(h_A == 2.0 && h_B == 1.0 && h_C == 0.0 && h_S == 1.0);
}