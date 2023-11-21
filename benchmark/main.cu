#include <stdio.h>

#include <algorithm>
#include <iterator>
#include <vector>

#include "cublas.h"
#include "myblas.h"
#include "openblas.h"

#define NUM_TRIALS 100

float random_float() { return (float)rand() / (float)(RAND_MAX); }

void copy_arr(const float* src, float* dst, int n) {
    for (int i = 0; i < n; i++) {
        dst[i] = src[i];
    }
}

void benchmark_saxpy(std::vector<int>& sizes) {
    printf("========== SAXPY ==========\n");
    for (int n : sizes) {
        int myblas_cycles = 0;
        int openblas_cycles = 0;
        int cublas_cycles = 0;
        for (int t = 0; t < NUM_TRIALS; t++) {
            // generate dataset
            float alpha = random_float();
            float x[n];
            float y[n];
            float y_copy[n];
            for (int i = 0; i < n; i++) {
                x[i] = random_float();
                y[i] = random_float();
            }
            // warmup
            if (n == sizes[0]) {
                copy_arr(y, y_copy, n);
                myblas_api_SAXPY(n, &alpha, x, y_copy);
                copy_arr(y, y_copy, n);
                cublas_api_SAXPY(n, &alpha, x, y_copy);
            }
            clock_t start, diff;

            // myblas
            copy_arr(y, y_copy, n);
            start = clock();
            myblas_api_SAXPY(n, &alpha, x, y_copy);
            diff = clock() - start;
            myblas_cycles += diff;

            // openblas
            copy_arr(y, y_copy, n);
            start = clock();
            openblas_api_SAXPY(n, &alpha, x, y_copy);
            diff = clock() - start;
            openblas_cycles += diff;
            // cublas
            copy_arr(y, y_copy, n);
            start = clock();
            cublas_api_SAXPY(n, &alpha, x, y_copy);
            diff = clock() - start;
            cublas_cycles += diff;
        }
        printf("N=%d\n", n);
        printf("    myblas  : %d\n", myblas_cycles / NUM_TRIALS);
        printf("    openblas: %d\n", openblas_cycles / NUM_TRIALS);
        printf("    cublas  : %d\n", cublas_cycles / NUM_TRIALS);
    }
}
void benchmark_sdot(std::vector<int>& sizes) {
    printf("========== SDOT ==========\n");
    for (int n : sizes) {
        int myblas_cycles = 0;
        int openblas_cycles = 0;
        int cublas_cycles = 0;
        for (int t = 0; t < NUM_TRIALS; t++) {
            // generate dataset
            float x[n];
            float y[n];
            float result[n];
            for (int i = 0; i < n; i++) {
                x[i] = random_float();
                y[i] = random_float();
            }
            // warmup
            if (n == sizes[0]) {
                myblas_api_SDOT(n, x, y, result);
                cublas_api_SDOT(n, x, y, result);
            }
            clock_t start, diff;

            // myblas
            start = clock();
            myblas_api_SDOT(n, x, y, result);
            diff = clock() - start;
            myblas_cycles += diff;

            // openblas
            start = clock();
            openblas_api_SDOT(n, x, y, result);
            diff = clock() - start;
            openblas_cycles += diff;
            // cublas
            start = clock();
            cublas_api_SDOT(n, x, y, result);
            diff = clock() - start;
            cublas_cycles += diff;
        }
        printf("N=%d\n", n);
        printf("    myblas  : %d\n", myblas_cycles / NUM_TRIALS);
        printf("    openblas: %d\n", openblas_cycles / NUM_TRIALS);
        printf("    cublas  : %d\n", cublas_cycles / NUM_TRIALS);
    }
}

void benchmark_sgemv(std::vector<int>& sizes) {
    printf("========== SGEMV ==========\n");
    for (int n : sizes) {
        int myblas_cycles = 0;
        int openblas_cycles = 0;
        int cublas_cycles = 0;
        for (int t = 0; t < NUM_TRIALS; t++) {
            // generate dataset
            float A[n * n];
            float x[n];
            float y[n];
            float y_copy[n];
            float alpha = random_float();
            float beta = random_float();

            for (int i = 0; i < n; i++) {
                x[i] = random_float();
                y[i] = random_float();
            }
            for (int i = 0; i < n * n; i++) {
                A[i] = random_float();
            }
            // warmup
            if (n == sizes[0]) {
                copy_arr(y, y_copy, n);
                myblas_api_SGEMV(n, n, &alpha, A, x, &beta, y_copy);
                copy_arr(y, y_copy, n);
                cublas_api_SGEMV(n, n, &alpha, A, x, &beta, y_copy);
            }
            clock_t start, diff;

            // myblas
            copy_arr(y, y_copy, n);
            start = clock();
            myblas_api_SGEMV(n, n, &alpha, A, x, &beta, y_copy);
            diff = clock() - start;
            myblas_cycles += diff;

            // openblas
            copy_arr(y, y_copy, n);
            start = clock();
            openblas_api_SGEMV(n, n, &alpha, A, x, &beta, y_copy);
            diff = clock() - start;
            openblas_cycles += diff;

            // cublas
            copy_arr(y, y_copy, n);
            start = clock();
            cublas_api_SGEMV(n, n, &alpha, A, x, &beta, y_copy);
            diff = clock() - start;
            cublas_cycles += diff;
        }
        printf("N=%d\n", n);
        printf("    myblas  : %d\n", myblas_cycles / NUM_TRIALS);
        printf("    openblas: %d\n", openblas_cycles / NUM_TRIALS);
        printf("    cublas  : %d\n", cublas_cycles / NUM_TRIALS);
    }
}

void benchmark_strsv(std::vector<int>& sizes) {
    printf("========== STRSV ==========\n");
    for (int n : sizes) {
        int myblas_cycles = 0;
        int openblas_cycles = 0;
        int cublas_cycles = 0;
        for (int t = 0; t < NUM_TRIALS; t++) {
            // generate dataset
            float A[n * n];
            float x[n];
            float x_copy[n];

            for (int i = 0; i < n; i++) {
                x[i] = random_float();
            }
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n - i; j++) {
                    A[i * n + j] = random_float();
                }
            }
            // warmup
            if (n == sizes[0]) {
                copy_arr(x, x_copy, n);
                myblas_api_STRSV(n, A, x_copy);
                copy_arr(x, x_copy, n);
                cublas_api_STRSV(n, A, x_copy);
            }
            clock_t start, diff;

            // myblas
            copy_arr(x, x_copy, n);
            start = clock();
            myblas_api_STRSV(n, A, x_copy);
            diff = clock() - start;
            myblas_cycles += diff;

            // openblas
            copy_arr(x, x_copy, n);
            start = clock();
            openblas_api_STRSV(n, A, x_copy);
            diff = clock() - start;
            openblas_cycles += diff;

            // cublas
            copy_arr(x, x_copy, n);
            start = clock();
            cublas_api_STRSV(n, A, x_copy);
            diff = clock() - start;
            cublas_cycles += diff;
        }
        printf("N=%d\n", n);
        printf("    myblas  : %d\n", myblas_cycles / NUM_TRIALS);
        printf("    openblas: %d\n", openblas_cycles / NUM_TRIALS);
        printf("    cublas  : %d\n", cublas_cycles / NUM_TRIALS);
    }
}

void benchmark_sgemm(std::vector<int>& sizes) {
    printf("========== SGEMM ==========\n");
    for (int n : sizes) {
        int myblas_cycles = 0;
        int openblas_cycles = 0;
        int cublas_cycles = 0;
        for (int t = 0; t < NUM_TRIALS; t++) {
            // generate dataset
            float A[n * n];
            float B[n * n];
            float alpha = random_float();
            float beta = random_float();
            float C[n * n];
            float C_copy[n * n];

            for (int i = 0; i < n * n; i++) {
                A[i] = random_float();
                B[i] = random_float();
                C[i] = random_float();
            }
            // warmup
            if (n == sizes[0]) {
                copy_arr(C, C_copy, n * n);
                myblas_api_SGEMM(n, n, n, &alpha, A, B, &beta, C_copy);
                copy_arr(C, C_copy, n * n);
                cublas_api_SGEMM(n, n, n, &alpha, A, B, &beta, C_copy);
            }
            clock_t start, diff;

            // myblas
            copy_arr(C, C_copy, n * n);
            start = clock();
            myblas_api_SGEMM(n, n, n, &alpha, A, B, &beta, C_copy);
            diff = clock() - start;
            myblas_cycles += diff;

            // openblas
            copy_arr(C, C_copy, n * n);
            start = clock();
            openblas_api_SGEMM(n, n, n, &alpha, A, B, &beta, C_copy);
            diff = clock() - start;
            openblas_cycles += diff;

            // cublas
            copy_arr(C, C_copy, n * n);
            start = clock();
            cublas_api_SGEMM(n, n, n, &alpha, A, B, &beta, C_copy);
            diff = clock() - start;
            cublas_cycles += diff;
        }
        printf("N=%d\n", n);
        printf("    myblas  : %d\n", myblas_cycles / NUM_TRIALS);
        printf("    openblas: %d\n", openblas_cycles / NUM_TRIALS);
        printf("    cublas  : %d\n", cublas_cycles / NUM_TRIALS);
    }
}

void benchmark_strsm(std::vector<int>& sizes) {
    printf("========== STRSM ==========\n");
    for (int n : sizes) {
        int myblas_cycles = 0;
        int openblas_cycles = 0;
        int cublas_cycles = 0;
        for (int t = 0; t < NUM_TRIALS; t++) {
            // generate dataset
            float A[n * n];
            float B[n * n];
            float B_copy[n * n];

            float alpha = random_float();

            for (int i = 0; i < n * n; i++) {
                A[i] = random_float();
                B[i] = random_float();
            }
            // warmup
            if (n == sizes[0]) {
                copy_arr(B, B_copy, n * n);
                myblas_api_STRSM(n, n, &alpha, A, B_copy);
                copy_arr(B, B_copy, n * n);
                cublas_api_STRSM(n, n, &alpha, A, B_copy);
            }
            clock_t start, diff;

            // myblas
            copy_arr(B, B_copy, n * n);
            start = clock();
            myblas_api_STRSM(n, n, &alpha, A, B_copy);
            diff = clock() - start;
            myblas_cycles += diff;

            // openblas
            copy_arr(B, B_copy, n * n);
            start = clock();
            openblas_api_STRSM(n, n, &alpha, A, B_copy);
            diff = clock() - start;
            openblas_cycles += diff;

            // cublas
            copy_arr(B, B_copy, n * n);
            start = clock();
            cublas_api_STRSM(n, n, &alpha, A, B_copy);
            diff = clock() - start;
            cublas_cycles += diff;
        }
        printf("N=%d\n", n);
        printf("    myblas  : %d\n", myblas_cycles / NUM_TRIALS);
        printf("    openblas: %d\n", openblas_cycles / NUM_TRIALS);
        printf("    cublas  : %d\n", cublas_cycles / NUM_TRIALS);
    }
}

int main() {
    std::vector<int> sizes_level1 = {1024, 4096, 16384, 65536, 262144, 524288};
    benchmark_saxpy(sizes_level1);
    benchmark_sdot(sizes_level1);
    std::vector<int> sizes_level2 = {32, 64, 128, 256, 512, 1024};
    benchmark_sgemv(sizes_level2);
    benchmark_strsv(sizes_level2);
    std::vector<int> sizes_level3 = {32, 64, 128, 256, 512};
    benchmark_sgemm(sizes_level3);
    benchmark_strsm(sizes_level3);
}