#include <algorithm>
#include <stdio.h>
#include <omp.h>
#include <string>
#include <math.h>
#include "laplace2DHelper.h"

#define BLOCK_SIZE 32

__global__ void jacobi_kernel(double *u,double *result, double h, int N){
  long idx = (blockIdx.x) * blockDim.x + threadIdx.x;

    //Run one step of jacobi in 2D
  __shared__ double f_h;
  f_h = h*h; //Assumes f(x,y) = 1 everywhere
  
  if(idx <= N){return;} //If first row, then skip
  if(idx % N == 0){return;}//If first column, then skip
  if(idx % N == (N-1)){return;}//If last column, then skip
  if(idx >= N*N - N){return;}//If last row, then skip
  //if(result[idx] == 0){return;}

  result[idx] = (f_h + u[idx - N] + u[idx - 1] + u[idx + N] + u[idx + 1])/4;
}

__global__ void cudaRed(double *u,double *result, double h, int N){
  long idx = (blockIdx.x) * blockDim.x + threadIdx.x;

    //Run one step of jacobi in 2D
  __shared__ double f_h;
  f_h = h*h; //Assumes f(x,y) = 1 everywhere
  
  bool black = (idx/N % 2 == 0) && (idx % 2 == 1); //If true then black node
  black = black || (idx/N % 2 == 1) && (idx % 2 == 0);

  if(black){return;} //If on a black node
  if(idx <= N){return;} //If first row, then skip
  if(idx % N == 0){return;}//If first column, then skip
  if(idx % N == (N-1)){return;}//If last column, then skip
  if(idx >= N*N - N){return;}//If last row, then skip

  result[idx] = (f_h + u[idx - N] + u[idx - 1] + u[idx + N] + u[idx + 1])/4;
}

__global__ void cudaBlack(double *result, double h, int N){
  long idx = (blockIdx.x) * blockDim.x + threadIdx.x;

    //Run one step of jacobi in 2D
  __shared__ double f_h;
  f_h = h*h; //Assumes f(x,y) = 1 everywhere
  
  bool red = (idx/N % 2 == 0) && (idx % 2 == 0); //If true then red node
  red = red || (idx/N % 2 == 1) && (idx % 2 == 1);

  if(red){return;} //If on a black node
  if(idx <= N){return;} //If first row, then skip
  if(idx % N == 0){return;}//If first column, then skip
  if(idx % N == (N-1)){return;}//If last column, then skip
  if(idx >= N*N - N){return;}//If last row, then skip

  result[idx] = (f_h + result[idx - N] + result[idx - 1] + result[idx + N] + result[idx + 1])/4;
}

void Check_CUDA_Error(const char *message){
  cudaError_t error = cudaGetLastError();
  if(error!=cudaSuccess) {
    fprintf(stderr,"ERROR: %s: %s\n", message, cudaGetErrorString(error) );
    exit(-1);
  }
}

void gs2D_CUDA(double *u,double *result, double h, int N){
  //Perform red update then perform black update
  //If the dimension is not even, reject
  if(N % 2 == 1){
    fprintf(stderr,"ERROR: Dimension for Gauss-Seidel Must Be Even\n");
    exit(-1);
  }
  cudaRed<<<N*N/BLOCK_SIZE+1,BLOCK_SIZE>>>(u, result, h, N);
  cudaBlack<<<N*N/BLOCK_SIZE+1,BLOCK_SIZE>>>(result, h, N);
}

void jacobi_CUDA(double *u,double *result, double h, int N){
  //Call cuda kernal for jacobi
  jacobi_kernel<<<N*N/BLOCK_SIZE+1,BLOCK_SIZE>>>(u, result, h, N);
}

int main(int argc, char* argv[]) {
  /*Iterative solving for linear systems
  argv[1]: represents dimension of matrix (int)
  argv[2]: max number of iterations (int)
  argv[3]: set the number of omp threads
  argv[4]: Either Jacobi or GS
  default solver is Gauss-Seidel*/
  
  int N = 4;
  int num_iter = 100;
  int num_threads = 4;
  const char* sol_pick = "jacobi";

  //If user passes no arguments, set defaults
  if(argc > 1){
    N = atoi(argv[1]);
    num_iter = atoi(argv[2]); //Must be divisable by 2
    num_threads = atoi(argv[3]);
  }

  //Set omp threads
  #ifdef _OPENMP
  omp_set_num_threads(num_threads);
  #endif

  /*********************************************************************************************************
  *********************************** CPU Baseline   *******************************************************
  *********************************************************************************************************/

  //Allocate space for arrays
  double **f = (double **)malloc(N*sizeof(double));
  double **u_in = (double **)malloc(N*sizeof(double));
  double **u_out = (double **)malloc(N*sizeof(double));
  //Allocate second dimension
  for(int i = 0; i<N; i++){
    f[i] = (double *)malloc(N*sizeof(double));
    u_out[i] = (double *)malloc(N*sizeof(double));
    u_in[i] = (double *)malloc(N*sizeof(double));
  }
  
  //Declare solver to use for computation
  void (*solver)(double **u, double **result, double h, int N);
  solver = !strcmp("GS",sol_pick) ? gs2D : jacobi2D;
  //solver = jacobi2D;

  // printf("Starting %s solver with %d Dimensions and "
  //   "%d max iterations\n",argv[3],N,num_iter);

  double h = 1.0/(N+1);

  //Initalize problem statement
  iniLaplace2D(f,u_in,u_out,N);

  double tt = omp_get_wtime(); //Start timer

  //CPU implementation
  int i = 0;
  while(i<num_iter){
    if(i % 2 == 0){
      solver(u_in, u_out, h, N);
    }else{
      solver(u_out, u_in, h, N);
    }
    i++;
  }

  //Time results CPU
  printf("Run time CPU: %f Number of Iterations : %d Dimension : %d\n", omp_get_wtime()-tt,i,N);

  // //Print test results for small N:
  // for(int i = 0; i<N; i++){
  //   for(int j = 0; j<N; j++){
  //     printf("%f ",u_out[i][j]);
  //   }
  //   printf("\n");
  // }

  /*********************************************************************************************************
  *********************************** CUDA Solution  *******************************************************
  *********************************************************************************************************/

  double* U_in = (double*) malloc(N * N * sizeof(double)); //Matrix with row major ordering input
  double* U_out = (double*) malloc(N * N * sizeof(double)); //Matrix with row major ordering output
  iniRowMajor2D(U_in, N); //Convert initial matrix to vector with row major ordering
  iniRowMajor2D(U_out, N); //Convert initial matrix to vector with row major ordering

  double *U_in_d, *U_out_d;
  cudaMalloc(&U_in_d, N*N*sizeof(double));
  cudaMalloc(&U_out_d, N*N*sizeof(double));
  cudaMemcpy(U_in_d, U_in, N*N*sizeof(double), cudaMemcpyHostToDevice); //Copy memory to GPU
  cudaMemcpy(U_out_d, U_out, N*N*sizeof(double), cudaMemcpyHostToDevice); //Copy memory to GPU

  void (*solver_CUDA)(double *u, double *result, double h, int N);
  solver_CUDA = !strcmp("GS",sol_pick) ? gs2D_CUDA : jacobi_CUDA;

  tt = omp_get_wtime(); //Start timer

  i = 0; //GPU iterations
  while(i<num_iter){
    if(i % 2 == 0){
      solver_CUDA(U_in_d,U_out_d, h, N);
    }else{
      solver_CUDA(U_out_d,U_in_d, h, N);
    }
    i++;
  }

  cudaError_t error = cudaMemcpyAsync(U_in, U_in_d, N*N*sizeof(double), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  //Time results GPU
  printf("Run time GPU: %f Number of Iterations : %d Dimension : %d\n", omp_get_wtime()-tt,i,N);
  printf("GPU Speed = %f Gflops/s, GPU Bandwidth = %f GB/s\n", 9*num_iter*N*N / (omp_get_wtime()-tt)/1e9,
                   5*num_iter*N*N*sizeof(double) / (omp_get_wtime()-tt)/1e9);



  // //Print test results for small N:
  // for(int i = 0; i<N; i++){
  //   for(int j = 0; j<N; j++){
  //     printf("%f ",U_in[i*N + j]);
  //   }
  //   printf("\n");
  // }

  //Compute error between CPU and GPU versions (max difference in a single element)
  double max_e = 0;
  for (int i = 0; i < N; i++){
    for(int j = 0; j < N; j++)
      max_e = std::max(fabs(u_in[i][j]-U_in[N*i + j]),max_e);
  }

  printf("Max Error Among Elements: %f\n", max_e);

  //Free malloced memory
  for(int i = 0; i<N; i++){
    free(f[i]);
    free(u_in[i]);
    free(u_out[i]);
  }

  free(U_in);
  free(U_out);

  cudaFree(U_in_d);
  cudaFree(U_out_d);

  return 0;
}

