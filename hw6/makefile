#By default, compile everything
all : q1 q2 q3

#Question 1
q1 : jacobi_mpi.cpp
	mpicxx -O3 jacobi_mpi.cpp -o jacobi_mpi

#Question 2
q2 : sample_sort.cpp
	mpic++ -O3 sample_sort.cpp -o sample_sort
	rm -rf output*

#Question 3
q3 : multigrid_mpi.cpp utils.h
	mpic++ -O3 -std=c++11 multigrid_mpi.cpp -o multigrid_mpi

#Clean executables
clean :
	rm -rf jacobi_mpi sample_sort multigrid_mpi output*