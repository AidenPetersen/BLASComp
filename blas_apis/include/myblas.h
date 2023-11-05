#include "level1/level1.h"
#include "level2/level2.h"
#include "level3/level3.h"

// ================ LEVEL 1 APIS ================
template <int NB, int NT>
void myblas_api_SAXPY(int n, const float *alpha, const float *x, float *y);

template <int NB, int NT>
void myblas_api_SDOT(int n, const float *x, const float *y, float *result);

// ================ LEVEL 2 APIS ================
template <int NB, int NT>
void myblas_api_SGEMV(int m, int n, const float *alpha, const float *A,
                      const float *x, const float *beta, float *y);

// Assumed to be upper triangular and not unit triangular
template <int NB, int NT>
void myblas_api_STRSV(int n, const float *A, float *x);

// ================ LEVEL 3 APIS ================
template <int NB, int NT>
void myblas_SGEMM(int m, int n, int k, const float *alpha, const float *A,
                  const float *B, const float *beta, float *C);

// Assumed to be upper triangular and not unit triangular
template <int NB, int NT>
void myblas_api_STRSM(int m, int n, const float *alpha, const float *A,
                      float *B);