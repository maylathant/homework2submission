#By default, compile everything
all : q1 q2

#Question 1
q1 : jacobi_mpi.cpp
	mpicxx -O3 jacobi_mpi.cpp -o jacobi_mpi

#Question 2
q2 : sample_sort.cpp
	mpic++ -O3 sample_sort.cpp -o sample_sort

#Clean executables
clean :
	rm -rf jacobi_mpi sample_sort