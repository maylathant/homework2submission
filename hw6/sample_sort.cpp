/* MPI-parallel Jacobi smoothing to solve -u''=f
 * Global vector has N unknowns, each processor works with its
 * part, which has lN = N/p unknowns.
 * Author: Georg Stadler
 */
#include <stdio.h>
#include <mpi.h>
#include <string.h>
#include <stdlib.h>
#include "hw6helper.h"
#include <unistd.h>
#include <algorithm>

void getRandList(double *v, long N, int mpirank){
  /*Populates the process' list with random numbers
  double v : array containing elements
  long N : size of the array*/
  
  srand(mpirank);
  for(int i = 0; i<N; i++){
    v[i] = (double)rand();
  }
}

int main(int argc, char * argv[]){
  int mpirank, p;
  long N;
  MPI_Status status, status1;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  /* get name of host running MPI process */
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);
  printf("Rank %d/%d running on %s.\n", mpirank, p, processor_name);

  sscanf(argv[1], "%ld", &N);

  /* compute number of unknowns handled by each process */
  long lN = N / p;
  if ((N % p != 0) && mpirank == 0 ) {
    printf("N: %ld, local N: %ld\n", N, lN);
    printf("Exiting. N must be a multiple of p\n");
    MPI_Abort(MPI_COMM_WORLD, 0);
  }

  /* timing */
  MPI_Barrier(MPI_COMM_WORLD);
  double tt = MPI_Wtime();
  long psq = (p-1)*p;


  double * v    = (double *)calloc(sizeof(double *), lN );
  double * s    = (double *)calloc(sizeof(double *), p-1 );

  double * sp    = (double *)calloc(sizeof(double *), psq );

  getRandList(v,lN,mpirank); //Populate Random Numbers
  
  //Select random P elements as splitters
  for(int i = 0; i < p-1; i++){
    s[i] = v[i];
  }

  quicksort(v,lN); //Local sort


  //Gather all s to mpirank == 0
  MPI_Gather(s,p-1,MPI_DOUBLE,sp,p-1,MPI_DOUBLE,0,MPI_COMM_WORLD);


  if(mpirank == 0){//Sort pre-splitters
    quicksort(sp,psq);
    sampleSplits(sp, s, psq, p-1);
    quicksort(s,p-1);
  }



  //Broadcast splitters
  MPI_Bcast(s,p-1,MPI_DOUBLE,0,MPI_COMM_WORLD);

  //Compute displacement and count for buckets
  long * sdispls  = (long *)calloc(sizeof(long *), p );
  long * bcounts  = (long *)calloc(sizeof(long *), p );
  long *allct = (long *)calloc(sizeof(long *), p*p);
  long temp = 0;
  sdispls[0] = 0;
  for(int i = 0; i < p-1; i++){//Compute splitters
      sdispls[i+1] = std::lower_bound(v, v+lN, s[i]) - v;
      bcounts[i] = (long)sdispls[i+1] - temp;
      printf("expression = %ld, bcounts = %ld\n", (long)sdispls[i+1] - temp, bcounts[i]);
      temp = (long)sdispls[i+1];
  }
  bcounts[p-1] = lN-temp;

  MPI_Alltoall(bcounts, p, MPI_LONG, allct, p, MPI_LONG, MPI_COMM_WORLD);


  if(mpirank==1){
    for(int i = 0; i < lN; i++){//Compute splitters
      printf("v[%d] = %f on process %d\n",i,v[i],mpirank);
    }
    for(int i = 0; i < p-1; i++){//Compute splitters
      printf("s[%d] = %f on process %d\n",i,s[i],mpirank);
    }
    for(int i = 0; i <= p-1; i++){//Compute splitters
      printf("displacement = %ld, count = %ld\n",sdispls[i],bcounts[i]);
    }

    for(long i = 0; i<p*p; i++){
      printf("allct[%ld] = %ld\n", i, allct[i]);
    }
  } 


  free(v);
  free(s);
  free(sp);
  free(sdispls);
  free(bcounts);
  free(allct);

  /* timing */
  MPI_Barrier(MPI_COMM_WORLD);
  double elapsed = MPI_Wtime() - tt;
  if (0 == mpirank) {
    printf("Time elapsed is %f seconds. rank = %d\n", elapsed, mpirank);
  }
  MPI_Finalize();
  return 0;
}