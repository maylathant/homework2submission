//Helper functions for HW1 Question3
//Anthony Maylath 2/2/2019

//Function to initialize A matrix and f array
void iniLaplace(double *f, double **A, double *u, double h, int N){
	for(int i = 0; i<N; i++){
		//Initialize function and starting guess
		f[i] = 1.0; u[i] = 0.0;
		
		//Initialize second derivative matrix
		for(int j = 0; j<N; j++){
			if(i == j){A[i][j]=2.0/h/h;}
			else if(i == j-1){A[i][j]=-1.0/h/h;}
			else if(i == j+1){A[i][j]=-1.0/h/h;}
			else {A[i][j]=0;}
		}
	}
}

//Run one iteration of Jacobi method
void jacobi(double *f, double **A, double *u, int N){
	
	double sp = 0;
	double *tp = (double *)malloc(N*sizeof(double));
	for(int i = 0; i<N; i++) tp[i] = 0.0;

	for(int i = 0; i<N; i++){
		//Compute sumproduct for each i
		sp = 0;
		for(int j = 0; j<N; j++){
			if(i != j){sp += A[i][j]*u[j];}
		}

		//Compute new iteration
		tp[i] = (f[i] - sp)/A[i][i];
	}

	//Copy tp into u and free tp
	for(int i = 0; i<N; i++) u[i] = tp[i];
	free(tp);
}

//Run one iteration of Gauess-Seidel
void solveGS(double *f, double **A, double *u, int N){
	double sp = 0;

	for(int i = 0; i<N; i++){
		//Compute sumproduct for each i
		sp = 0;
		for(int j = 0; j<N; j++){
			if(i != j){sp += A[i][j]*u[j];}
		}

		//Compute new iteration
		u[i] = (f[i] - sp)/A[i][i];
	}
}

/*Vector matrix multiplication
Multiply matrix A and vector u;
store the result in b*/
void AuMult(double **A, double *u, double *b, int N){
	for(int i = 0; i<N; i++){
		b[i] = 0;
		for(int j = 0; j<N; j++)
			b[i] += u[j]*A[i][j];
	}
}

/*Vector subtracttion
Subtract vector f from u and store the result in b*/
void vecSub(double *u, double *f, double *b, int N){
	for(int i = 0; i<N; i++){
		b[i] = u[i] - f[i];
	}
}

/*Compute 2 norm of a vector b*/
double norm2(double *b, int N){
	double res = 0.0;
	for(int i = 0; i<N; i++)
		res += b[i]*b[i];
	return sqrt(res);
}

/*Compute error between Au and f*/
double error2(double **A, double *u, double *f, int N){
	double *temp = (double *)malloc(N*sizeof(double));
	double res;

	//Compute Au
	AuMult(A,u,temp,N);
	//Compute Au - f
	vecSub(temp,f,temp,N);
	//Return 2norm
	res = norm2(temp, N);
	//Free memory
	free(temp);

	return res;
}