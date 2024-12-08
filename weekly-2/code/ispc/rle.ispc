export uniform int rle_ispc(uniform int output[], uniform int input[], uniform int n) {
  uniform int cur = input[0];
  uniform int count = 0;
  uniform int j = 0;
  foreach (i = 0 ... n) {
    int f = input[i];
    if (!all(f == cur)) {
      if (programIndex == 0) {
          for (int k = 0; k < programCount; k++) {
            int next = input[i+k];
            if (next != cur) {
              output[j++] = count;
              output[j++] = cur;
              cur = extract(next,0);
              count = 1;
            } else {
              count++;
        }}
    }} else {
    count += programCount;
    }
  }
  output[j++] = count;
  output[j++] = cur;
  return j;
}
