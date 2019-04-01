#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <omp.h>

// Scan A array and write result into prefix_sum array;
// use long data type to avoid overflow
void scan_seq(long* prefix_sum, const long* A, long n) {
  if (n == 0) return;
  prefix_sum[0] = A[0];
  for (long i = 1; i < n; i++) {//Slightly changed code to include first element
    prefix_sum[i] = prefix_sum[i-1] + A[i];
  }
}

void scan_omp(long* prefix_sum, const long* A, long n) {
  long p = omp_get_max_threads();
  long work_sh = n/p; //Load per thread
  long rem = n - work_sh*p; //Remainder
  long i, base; int tid;
  //printf("p: %d, work_sh: %d, rem: %d\n", p,work_sh,rem);
  #pragma omp parallel private(base,tid,i)
  {
    tid = omp_get_thread_num();
    long base = tid*work_sh;
    prefix_sum[base] = A[base];
    for(i = base + 1; i < (tid+1)*work_sh; i++){
      prefix_sum[i] = prefix_sum[i-1] + A[i];
    }
  }

  if(rem > 0){//If p does not divide n
    long base = n-rem;
    prefix_sum[base] = A[base];
    for(long i = base+1; i < n; i++){
      prefix_sum[i] = prefix_sum[i-1] + A[i];
      //printf("Change prefix_sum[%d] to %d\n",i,prefix_sum[i]);
    }
  }

  //Add correction
  for(int i = work_sh; i < n-rem; i=i+work_sh){
    for(long j = i; j < i+work_sh; j++){//Iterate through subarray
      prefix_sum[j] += prefix_sum[i-1];
      //printf("Change prefix_sum[%d] to %d\n",j,prefix_sum[j]);
    }
  }

  for(int i = n-rem; i < n; i++){//Correction for remainder
    prefix_sum[i] += prefix_sum[n-rem-1];
    //printf("Change prefix_sum[%d] to %d\n",i,prefix_sum[i]);
  }




}

int main(int argc, char *argv[]) {
  if(argc > 1){omp_set_num_threads(atoi(argv[1]));} //If user passes argument, set omp threads
  //otherwise, use default number of threads.

  long N = 100000000;
  long* A = (long*) malloc(N * sizeof(long));
  long* B0 = (long*) malloc(N * sizeof(long));
  long* B1 = (long*) malloc(N * sizeof(long));
  for (long i = 0; i < N; i++) A[i] = rand();

  double tt = omp_get_wtime();
  scan_seq(B0, A, N);
  printf("sequential-scan = %fs\n", omp_get_wtime() - tt);

  tt = omp_get_wtime();
  scan_omp(B1, A, N);
  printf("parallel-scan   = %fs\n", omp_get_wtime() - tt);

  long err = 0;
  for (long i = 0; i < N; i++) err = std::max(err, std::abs(B0[i] - B1[i]));
  printf("error = %d\n", err);

  free(A);
  free(B0);
  free(B1);
  return 0;
}
