#include <algorithm>
#include <stdio.h>
#include <omp.h>
#include <string>
#include <math.h>

#define BLOCK_SIZE 1024


/*openmp functions to compute inner product*/
void vec_mult(double* c, const double* a, const double* b, long N){
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < N; i++) {
    c[i] = a[i]*b[i];
  }
}


void reduction(double* sum_ptr, const double* a, long N){
  double sum = 0;
  #pragma omp parallel for schedule(static) reduction(+:sum)
  for (long i = 0; i < N; i++) sum += a[i];
  *sum_ptr = sum;
}

void innerprod(double* sum_ptr, const double* a, const double *b, long N){
  double sum = 0;
  #pragma omp parallel for schedule(static) reduction(+:sum)
  for (long i = 0; i < N; i++) sum += a[i] * b[i];
  *sum_ptr = sum;
}

/*CUDA functions to compute dot product*/
__global__
void vec_mult_kernel(double* c, const double* a, const double* b, long N){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) c[idx] = a[idx] * b[idx];
}

void Check_CUDA_Error(const char *message){
  cudaError_t error = cudaGetLastError();
  if(error!=cudaSuccess) {
    fprintf(stderr,"ERROR: %s: %s\n", message, cudaGetErrorString(error) );
    exit(-1);
  }
}

__global__ void reduction_kernel2(double* sum, const double* a, long N){
  __shared__ double smem[BLOCK_SIZE];
  long idx = (blockIdx.x) * blockDim.x + threadIdx.x;

  if (idx < N) smem[threadIdx.x] = a[idx];
  else smem[threadIdx.x] = 0;

  __syncthreads();
  if (threadIdx.x < 512) smem[threadIdx.x] += smem[threadIdx.x + 512];
  __syncthreads();
  if (threadIdx.x < 256) smem[threadIdx.x] += smem[threadIdx.x + 256];
  __syncthreads();
  if (threadIdx.x < 128) smem[threadIdx.x] += smem[threadIdx.x + 128];
  __syncthreads();
  if (threadIdx.x <  64) smem[threadIdx.x] += smem[threadIdx.x +  64];
  __syncthreads();
  if (threadIdx.x <  32) {
    smem[threadIdx.x] += smem[threadIdx.x +  32];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +  16];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +   8];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +   4];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +   2];
    __syncwarp();
    if (threadIdx.x == 0) sum[blockIdx.x] = smem[0] + smem[1];
  }
}

__global__ void innerprod_kernel2(double* sum, const double* a, const double* b, long N){
  __shared__ double smem[BLOCK_SIZE];
  long idx = (blockIdx.x) * blockDim.x + threadIdx.x;

  if (idx < N) smem[threadIdx.x] = a[idx] * b[idx];
  else smem[threadIdx.x] = 0;

  __syncthreads();
  if (threadIdx.x < 512) smem[threadIdx.x] += smem[threadIdx.x + 512];
  __syncthreads();
  if (threadIdx.x < 256) smem[threadIdx.x] += smem[threadIdx.x + 256];
  __syncthreads();
  if (threadIdx.x < 128) smem[threadIdx.x] += smem[threadIdx.x + 128];
  __syncthreads();
  if (threadIdx.x <  64) smem[threadIdx.x] += smem[threadIdx.x +  64];
  __syncthreads();
  if (threadIdx.x <  32) {
    smem[threadIdx.x] += smem[threadIdx.x +  32];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +  16];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +   8];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +   4];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +   2];
    __syncwarp();
    if (threadIdx.x == 0) sum[blockIdx.x] = smem[0] + smem[1];
  }
}

void innerprod_CUDA(double* sumcuda, double* sum_d , const double* a, const double *b, long N){
  /*Vectors a and b should be located on the device*/
  long Nb = (N+BLOCK_SIZE-1)/(BLOCK_SIZE);
  innerprod_kernel2<<<Nb,BLOCK_SIZE>>>(sum_d, a, b, N);
  while (Nb > 1) {
    long N = Nb;
    Nb = (Nb+BLOCK_SIZE-1)/(BLOCK_SIZE);
    reduction_kernel2<<<Nb,BLOCK_SIZE>>>(sum_d + Nb, sum_d, N);
    sum_d += Nb;
  }

  cudaError_t error = cudaMemcpyAsync(sumcuda, sum_d, 1*sizeof(double), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  
}

int main(int argc, char* argv[]) {
  /*Iterative solving for linear systems
  argv[1]: represents dimension of matrix (long)*/
  
  long N = 1000; //Default value

  //If user passes no arguments, set defaults
  if(argc > 1){
    N = atoi(argv[1]);
  }

  double* x = (double*) malloc(N * sizeof(double));
  double* y = (double*) malloc(N * sizeof(double));
  double* z = (double*) malloc(N * sizeof(double));
  double* z_ref = (double*) malloc(N * sizeof(double));
  double* A = (double*) malloc(N * N * sizeof(double)); //Matrix with row major ordering
  double* v = (double*) malloc(N * sizeof(double)); //Result of matrix vector multiplication

  //AEM additions ~ variable declariations
  double sumomp = 0; double sumcuda = 0;

  #pragma omp parallel for schedule(static)
  for (long i = 0; i < N; i++) {
    x[i] = i+2;
    y[i] = 1.0/(i+1);
    z[i] = 0;
    z_ref[i] = 0;
    for(long j = 0; j < N; j++){
      A[N*i + j] = j;
    }
  }

 
  vec_mult(z_ref, x, y, N);
  double tt = omp_get_wtime();
  innerprod(&sumomp, x, y, N);
  printf("CPU Bandwidth Inner Product = %f GB/s\n", 2*N*sizeof(double) / (omp_get_wtime()-tt)/1e9);

  tt = omp_get_wtime();
  for(long i = 0; i < N; i++){ //Matrix mult between matrix A and vector y
    innerprod(v+i, A+(i*N), y, N);
    //for(long j = 0; j < N; j++){
      //printf("A[%d%d] = %f " , i, j, A[N*i+j]);
    //}
      //printf("v[%d] = %f\n",i,v[i]);
  }

  printf("CPU Bandwidth Matrix Vector Mult = %f GB/s\n", 2*N*N*sizeof(double) / (omp_get_wtime()-tt)/1e9);

  double *x_d, *y_d, *z_d, *A_d, *v_d, *temp_d;
  cudaMalloc(&x_d, N*sizeof(double));
  cudaMalloc(&y_d, N*sizeof(double));
  cudaMalloc(&z_d, N*sizeof(double));
  cudaMalloc(&A_d, N*sizeof(double)); //For CUDA matrix multiplication
  v_d = (double*) malloc(N * sizeof(double)); //For result of matrix mult (in CPU memory)
  cudaMalloc(&temp_d, N*sizeof(double)); //For temporary results



  cudaMemcpy(x_d, x, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(y_d, y, N*sizeof(double), cudaMemcpyHostToDevice);
  vec_mult_kernel<<<N/1024+1,1024>>>(z_d, x_d, y_d, N);
  cudaMemcpy(z, z_d, N*sizeof(double), cudaMemcpyDeviceToHost);


  //Perform CUDA reduction over vector z to obtain the inner product
  double *sum_d;
  cudaMalloc(&sum_d, ((N+BLOCK_SIZE-1)/BLOCK_SIZE)*sizeof(double));

  tt = omp_get_wtime();
  innerprod_CUDA(&sumcuda, sum_d, x_d, y_d, N);

  printf("GPU Bandwidth Inner Product = %f GB/s\n", 3*N*sizeof(double) / (omp_get_wtime()-tt)/1e9);


  /************************* CUDA Matrix Multiplication *****************************/
  tt = omp_get_wtime();
  cudaMemcpy(A_d, A, N*sizeof(double), cudaMemcpyHostToDevice); //Copy matrix to device for each row (assume rows are the same)
  for(long i = 0; i < N; i++){
    innerprod_CUDA(v_d+i, temp_d, A_d, y_d, N);
  }

  printf("GPU Bandwidth Matrix Vector Mult (Same rows) = %f GB/s\n", 3*N*N*sizeof(double) / (omp_get_wtime()-tt)/1e9);

  tt = omp_get_wtime();
  for(long i = 0; i < N; i++){
    cudaMemcpy(A_d, A+(i*N), N*sizeof(double), cudaMemcpyHostToDevice); //Copy matrix to device for each row
    innerprod_CUDA(v_d+i, temp_d, A_d, y_d, N);
  }

  printf("GPU Bandwidth Matrix Vector Mult (Diff rows) = %f GB/s\n", 5*N*N*sizeof(double) / (omp_get_wtime()-tt)/1e9);

  double err = 0, mat_err;
  for (long i = 0; i < N; i++){
    err += fabs(z[i]-z_ref[i]);
    mat_err += fabs(v[i]-v_d[i]);
  }
  printf("Error Vector Mult = %f\n", err);
  printf("Error Vector Matrix Mult = %f\n", mat_err);

  printf("Error Inner Product = %f\n", fabs(sumomp-sumcuda));

  double max_v = 0;
  for(int i = 0; i < N; i++)
    max_v = std::max(fabs(v_d[i]-v[i]),max_v);

  cudaFree(x_d);
  cudaFree(y_d);
  cudaFree(z_d);
  cudaFree(sum_d);
  cudaFree(temp_d);
  cudaFree(v_d);
  cudaFree(A_d);

  free(x);
  free(y);
  free(z);
  free(z_ref);
  free(A);
  free(v);

  return 0;
}

