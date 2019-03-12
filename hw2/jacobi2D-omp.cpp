/*Anthony Maylath C code to solve 2D Laplace*/
#include <stdio.h>
#include <stdlib.h> 
#include <string.h>
#include "utils.h"
#include "laplace2DHelper.h"
#ifdef _OPENMP
#include <omp.h>
#endif


int main(int argc, char* argv[]){
	/*Iterative solving for linear systems
	argv[1]: represents dimension of matrix (int)
	argv[2]: max number of iterations (int)
	argv[3]: set the number of omp threads
	default solver is Gauss-Seidel*/
	
	int N = 5;
	int num_iter = 100;
	int num_threads = 4;

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
	//solver = !strcmp("GS",argv[3]) ? gs2D : jacobi2D;
	solver = jacobi2D;

	// printf("Starting %s solver with %d Dimensions and "
	// 	 "%d max iterations\n",argv[3],N,num_iter);

	double err0, err=10000, h = 1.0/(N+1), tol = 1e6;

	//Initalize problem statement
	iniLaplace2D(f,u_in,u_out,N);

	Timer t;
    t.tic(); //Start timer

	int i = 0;
	while(i<num_iter){
		if(i % 2 == 0){
			solver(u_in, u_out, h, N);
		}else{
			solver(u_out, u_in, h, N);
		}
		i++;
	}

	//Time results
	printf("Run time: %f Number of Iterations : %d Dimension : %d\n", t.toc(),i,N);

	// //Print test results for small N:
	// for(int i = 0; i<N; i++){
	// 	for(int j = 0; j<N; j++){
	// 		printf("%f ",u_in[i][j]);
	// 	}
	// 	printf("\n");
	// }

	//Free malloced memory
	for(int i = 0; i<N; i++){
		free(f[i]);
		free(u_in[i]);
		free(u_out[i]);
	}

}