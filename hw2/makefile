#By defaily, compile everything
all : q1a q1b q2 q32 q33 q34 q35 q36 q4jacobi q4gs

#Question 1
q1a : val_test01_solved.cpp
	g++ -O3 -g -o val_test01_solved val_test01_solved.cpp

q1b : val_test02_solved.cpp
	g++ -O3 -g -o val_test02_solved val_test02_solved.cpp

#Question 2
q2 : MMult1_solved.cpp
	g++-8 -O3 -fopenmp -o MMult1 MMult1_solved.cpp

#Question 3
q32 : omp_solved2.c
	gcc-8 -fopenmp -o omp_solved2 omp_solved2.c

q33 : omp_solved3.c
	gcc-8 -fopenmp -o omp_solved3 omp_solved3.c

q34 : omp_solved4.c
	gcc-8 -fopenmp -o omp_solved4 omp_solved4.c

q35 : omp_solved5.c
	gcc-8 -fopenmp -o omp_solved5 omp_solved5.c

q36 : omp_solved6.c
	gcc-8 -fopenmp -o omp_solved6 omp_solved6.c

q4jacobi : jacobi2D-omp.cpp
	g++-8 -fopenmp -o jacobi2D-omp jacobi2D-omp.cpp

q4gs : gs2D-omp.cpp
	g++-8 -fopenmp -o gs2D-omp gs2D-omp.cpp

#Clean executables
clean :
	rm val_test01_solved val_test02_solved MMult1 omp_solved2 omp_solved3 omp_solved4 omp_solved5 omp_solved6 jacobi2D-omp gs2D-omp