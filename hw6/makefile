#By default, compile everything
all : q1 q2 q3

#Question 1
q1 : jacobi_mpi.cpp
	mpicxx -O3 jacobi_mpi.cpp -o jacobi_mpi

#Question 2
q2 : sample_sort.cpp hw6helper.h
	mpic++ -O3 sample_sort.cpp -o sample_sort
	rm -rf output*

#Question 3
q3 : multigrid_mpi.cpp utils.h
	g++-8 -O3 -fopenmp multigrid_omp.cpp -o multigrid_omp

#Clean executables
clean :
	rm -rf jacobi_mpi sample_sort multigrid_omp output*