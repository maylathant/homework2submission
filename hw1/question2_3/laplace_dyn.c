/*Anthony Maylath C code to solve 1D Laplace*/
#include <stdio.h>
#include <stdlib.h> 
#include <math.h>
#include <string.h>
#include "hw1q3_utils.h"
#include "utils.h"

//Function declarations
void iniLaplace(double *f, double **A, double *u, double h, int N);
void jacobi(double *f, double **A, double *u, int N);
void AuMult(double **A, double *u, double *b, int N);
void vecSub(double *u, double *f, double *b, int N);
double norm2(double *b, int N);
double error2(double **A, double *u, double *f, int N);

int main(int argc, char* argv[]){
	/*Iterative solving for linear systems
	argv[1]: represents dimension of matrix (int)
	argv[2]: mac number of iterations (int)
	argv[3]: represents type of solver. "jacobi" or "GS" (string) 
	default solver is Gauss-Seidel*/
	int N = atoi(argv[1]);
	int num_iter = atoi(argv[2]);

	//Allocate space for arrays
	double *f = (double *)malloc(N*sizeof(double));
	double *u = (double *)malloc(N*sizeof(double));
	double **A = (double **)malloc(N*sizeof(double));
	//Allocate second dimension
	for(int i = 0; i<N; i++)
		A[i] = (double *)malloc(N*sizeof(double));
	
	//Declare solver to use for computation
	void (*solver)(double *f, double **A, double *u, int N);
	solver = !strcmp("GS",argv[3]) ? solveGS : jacobi;

	printf("Starting %s solver with %d Dimensions and "
		 "%d max iterations\n",argv[3],N,num_iter);

	double err0, err=10000, h = 1.0/N, tol = 1e6;

	//Initalize problem statement
	iniLaplace(f,A,u,h,N);
	
	//Initial error
	err0 = error2(A,u,f,N);
	printf("Initial error is: %f\n",err0);

	Timer t;
    t.tic(); //Start timer
	int i = 0;
	while((i<num_iter) && (err0/err < tol)){
		solver(f,A,u,N);
		err = error2(A,u,f,N);
		//Print error for every 100 iterations
		if(i%100==0)
		 	printf("Error for iteration %d is %f\n", i, err);
		i++;
	}

	for(int i = 0; i<N; i++)
	 	printf("Entry %d is %f\n", i, u[i]);

	//Time results
	printf("Run time: %f\n Number of Iterations : %d\n", t.toc(),i);

	//Free malloced memory
	free(f); free(u);
	for(int i = 0; i<N; i++)
		free(A[i]);

}