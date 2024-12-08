#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "timing.h"
#include "scan.h"

static inline void scan_c(float *output, float *input, int n) {
  float acc = 0;
  for (int i = 0; i < n; i++) {
    acc += input[i];
    output[i] = acc;
  }
}

int main() {
  int n = 100000;
  float *input       = calloc(n, sizeof(float));
  float *c_output    = calloc(n, sizeof(float));
  float *ispc_output = calloc(n, sizeof(float));

  for (int i = 0; i < n; i++) {
    input[i] = (float)rand()/RAND_MAX;
  }

  int runtime;
  int num_runs = 100;
  int total_runtime;
  total_runtime = 0;
  for (int i = 0; i < num_runs; i++) {
      TIMEIT(runtime) {
          scan_c(c_output, input, n);
      }
      total_runtime += runtime;
  }
  printf("C:                 %8d microseconds (average)\n", total_runtime / num_runs);
total_runtime = 0;
    for (int i = 0; i < num_runs; i++) {
        TIMEIT(runtime) {
            scan_ispc(ispc_output, input, n);
        }
        total_runtime += runtime;
    }
    printf("ISPC:              %8d microseconds (average)\n", total_runtime / num_runs);

    // Benchmark scan_ispc2
    total_runtime = 0;
    for (int i = 0; i < num_runs; i++) {
        TIMEIT(runtime) {
            scan_ispc2(ispc_output, input, n);
        }
        total_runtime += runtime;
    }
    printf("ISPC2:             %8d microseconds (average)\n", total_runtime / num_runs);
  for (int i = 0; i < n; i++) {
    // If necessary, fiddle with the tolerance here (recall that
    // floating-point addition is not actually associative).
    if (fabsf(1 - c_output[i] / ispc_output[i]) > 1.001) {
      fprintf(stderr, "Results differ at [%d]: %f != %f\n", i, c_output[i], ispc_output[i]);
      return 1;
    }
  }
}
