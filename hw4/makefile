#By defaily, compile everything
all : q1 q2

#Question 1
q1 : vec-mat.cu
	nvcc -Xcompiler "-fopenmp" -O3 vec-mat.cu -o vec-mat

#Question 2
q2 : 2dJacobi.cu
	nvcc -Xcompiler "-fopenmp" -O3 2dJacobi.cu -o 2dJacobi

#Clean executables
clean :
	rm -rf vec-mat 2dJacobi