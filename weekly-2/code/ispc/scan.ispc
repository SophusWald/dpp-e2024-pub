// First Implementation
export void scan_ispc(uniform float output[], uniform float input[], uniform int n) {
  float currentresult = 0;
  for (uniform int j = 0; j < n; j += programCount) {
    int idx = j + programIndex;
    output[idx] = currentresult + exclusive_scan_add(input[idx]) + input[idx];
    currentresult = broadcast(output[idx],programCount-1);
  }
}
//Second Implementation
export void scan_ispc2(uniform float output[], uniform float input[], uniform int n) {
  float currentresult = 0;
  foreach (j = 0 ... n) {
    int idx = j;
    output[idx] = currentresult + exclusive_scan_add(input[idx]) + input[idx];
    currentresult = broadcast(output[idx],programCount-1);
  }
}
