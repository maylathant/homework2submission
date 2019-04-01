#By defaily, compile everything
all : q1 q2

#Question 1
q1 : fast-sin_solved.cpp
	g++ --std=c++11 -O3 -o fast-sin_solved fast-sin_solved.cpp

#Question 2
q2 : omp-scan_solved.cpp
	g++-8 -O3 -fopenmp -o omp-scan_solved omp-scan_solved.cpp

#Clean executables
clean :
	rm -rf fast-sin_solved omp-scan_solved