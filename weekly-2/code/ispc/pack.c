#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "timing.h"
#include "pack.h"

static inline int pack_c(int *output, int *input, int n) {
  int j = 1;
  int cur = input[0];
  output[0] = cur;
  for (int i = 1; i < n; i++) {
    int v = input[i];
    if (v != cur) {
      output[j++] = v;
    }
    cur = v;
  }
  return j;
}

int main() {
  int n = 100000;
  int *input = calloc(n, sizeof(int));
  int c_m;
  int *c_output = calloc(n, sizeof(int));
  int ispc_m;
  int ispc2_m;
  int *ispc_output = calloc(n, sizeof(int));

  for (int i = 0; i < n; i++) {
    input[i] = rand() % 2;
  }

  int runtime;
  int num_runs = 100;
  int total_runtime;
total_runtime = 0;
    for (int i = 0; i < num_runs; i++) {
        TIMEIT(runtime) {
            pack_c(c_output, input, n);
        }
        total_runtime += runtime;
    }
    printf("C:                 %8d microseconds (average)\n", total_runtime / num_runs);

    // Benchmark scan_ispc
    total_runtime = 0;
    for (int i = 0; i < num_runs; i++) {
        TIMEIT(runtime) {
            pack_ispc(ispc_output, input, n);
        }
        total_runtime += runtime;
    }
    printf("ISPC:              %8d microseconds (average)\n", total_runtime / num_runs);

    // Benchmark scan_ispc2
    total_runtime = 0;
    for (int i = 0; i < num_runs; i++) {
        TIMEIT(runtime) {
            pack_ispc2(ispc_output, input, n);
        }
        total_runtime += runtime;
    }
    printf("ISPC2:             %8d microseconds (average)\n", total_runtime / num_runs);
  for (int i = 0; i < c_m; i++) {
    if (c_output[i] != ispc_output[i]) {
      fprintf(stderr, "Results differ at [%d]: %d != %d\n", i, c_output[i], ispc_output[i]);
      return 1;
    }
  }
}
