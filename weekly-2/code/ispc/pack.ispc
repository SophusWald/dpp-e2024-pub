//Stupid Implementation: just say you want the previous one from input
export uniform int pack_ispc(uniform int output[], uniform int input[], uniform int n) {
  uniform int m = 0;
  foreach (i = 0 ... n) {
    int j = input[i];
    int cur = input[i-1];
    int keep = j != cur;
    int offset = exclusive_scan_add(keep);
    if (!keep) {
      offset = programCount-1;
    }
    output[m + offset] = j;
    m += reduce_add(keep);
  }
  return m;
}
//(Maybe) Less Stupid Implementation: Use rotate, and manually fix the first index.
export uniform int pack_ispc2(uniform int output[], uniform int input[], uniform int n) {
  uniform int m = 0;
  uniform int last = input[0] + 1;
  foreach (i = 0 ... n) {
    int j = input[i];
    int cur = rotate(j,-1);
    cur = insert(cur,0,last);
    int keep = j != cur;
    int offset = exclusive_scan_add(keep);
    if (!keep) {
      offset = programCount-1;
    }
    output[m + offset] = j;
    last = extract(j,programCount-1);
    m += reduce_add(keep);
  }
  return m;
}
