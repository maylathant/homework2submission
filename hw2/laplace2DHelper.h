/**Helper functions for HW2q4*/
#ifdef _OPENMP
#include <omp.h>
#endif

//Function declarations
void jacobi2D(double **u,double **result, double h, int N){
	//Run one step of jacobi in 2D
	double f_h = h*h; //Assumes f(x,y) = 1 everywhere

	int i; int j;
	#ifdef _OPENMP
	#pragma omp parallel for private(i,j), shared(u,f_h)
	#endif
	for(int i = 1;i<N-1;i++){
		for(int j = 1;j<N-1;j++){
			result[i][j] = (f_h + u[i-1][j] + u[i][j-1] + u[i+1][j] + u[i][j+1])/4;
		}
	}

}

void assignRed(double **u,double **result, double h, int N){
	//Assign red nodes for 1 step of 2D Gauss-Seidel
	double f_h = h*h; //Assumes f(x,y) = 1 everywhere
	int j; //For skip iterations
	int i;
	
	#ifdef _OPENMP
	#pragma omp parallel for private(i,j), shared(u,f_h)
	#endif
	for(int i = 1;i<N-1;i++){
		j = 1;
		if(i % 2 == 0){j = 2;} //If even row, start with first element; otherwise, start with second
		for(;j<N-1;j=j+2){
			result[i][j] = (f_h + u[i-1][j] + u[i][j-1] + u[i+1][j] + u[i][j+1])/4;
		}
	}
}

void assignBlack(double **u,double **result, double h, int N){
	//Assign black nodes for 1 step of 2D Gauss-Seidel
	double f_h = h*h; //Assumes f(x,y) = 1 everywhere
	int j; //For skip iterations
	int i;
	
	#ifdef _OPENMP
	#pragma omp parallel for private(i,j), shared(f_h)
	#endif
	for(int i = 1;i<N-1;i++){
		j = 2;
		if(i % 2 == 0){j = 1;} //If even row, start with first element; otherwise, start with second
		for(;j<N-1;j=j+2){
			result[i][j] = (f_h + result[i-1][j] + result[i][j-1] + result[i+1][j] + result[i][j+1])/4;
		}
	}
}

void gs2D(double **u,double **result, double h, int N){
	//Perform red update then perform black update
	assignRed(u,result,h,N);
	assignBlack(u,result,h,N);
}

void iniLaplace2D(double **f, double **u, double **result, int N){
	//Initialize both f and u to 1
	for(int i = 0; i<N; i++){
		for(int j = 0; j<N; j++){
			//Assign boundary points zero and 1 everywhere else
			if (i == 0 || j == 0 || i == N-1 || j == N-1){
				f[i][j] = 1;
				u[i][j] = result[i][j] = 0;
			}else{
				f[i][j] = u[i][j] = result[i][j] = 1;
			}
		}
	}
}