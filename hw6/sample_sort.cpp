#include <stdio.h>
#include <mpi.h>
#include <string.h>
#include <stdlib.h>
#include "hw6helper.h"
#include <unistd.h>
#include <algorithm>

void getRandList(int *v, long N, int mpirank){
  /*Populates the process' list with random numbers
  double v : array containing elements
  long N : size of the array*/
  
  srand(mpirank);
  for(int i = 0; i<N; i++){
    v[i] = rand();
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
  long psq = (p-1)*p;


  int * v    = (int *)calloc(sizeof(int *), lN );
  int * s    = (int *)calloc(sizeof(int *), p-1 );

  int * sp    = (int *)calloc(sizeof(int *), psq );

  getRandList(v,lN,mpirank); //Populate Random Numbers
  
  double tt = MPI_Wtime();

  //Select random P elements as splitters
  for(int i = 0; i < p-1; i++){
    s[i] = v[i];
  }

  quicksort(v,lN); //Local sort


  //Gather all s to mpirank == 0
  MPI_Gather(s,p-1,MPI_INT,sp,p-1,MPI_INT,0,MPI_COMM_WORLD);


  if(mpirank == 0){//Sort pre-splitters
    quicksort(sp,psq);
    sampleSplits(sp, s, psq, p-1);
    quicksort(s,p-1);
  }



  //Broadcast splitters
  MPI_Bcast(s,p-1,MPI_INT,0,MPI_COMM_WORLD);

  //Compute displacement and count for buckets
  int * sdispls  = (int *)calloc(sizeof(int *), p );
  int * bcounts  = (int *)calloc(sizeof(int *), p );
  int *allct = (int *)calloc(sizeof(int *), p);
  int *alldis = (int *)calloc(sizeof(int *), p);
  int *finBuck = (int *)calloc(sizeof(int *), 2*lN );
  int temp = 0;
  sdispls[0] = 0;
  for(int i = 0; i < p-1; i++){//Compute splitters
      sdispls[i+1] = std::lower_bound(v, v+lN, s[i]) - v;
      bcounts[i] = (int)sdispls[i+1] - temp;
      temp = (int)sdispls[i+1];
  }
  bcounts[p-1] = lN-temp;


  MPI_Alltoall(bcounts, 1, MPI_INT, allct, 1, MPI_INT, MPI_COMM_WORLD);
  for(int i = 1; i<p; i++) alldis[i] = allct[i-1] + alldis[i-1]; //Compute receiving displacements
  MPI_Alltoallv(v, bcounts, sdispls, MPI_INT, finBuck, allct, alldis, MPI_INT, MPI_COMM_WORLD);

  int total = 0;
  for(int i = 0; i<p; i++){//Get total meaningfull elements
    total += allct[i];
  }

  quicksort(finBuck,total); //Sort local bucket

  // if(mpirank==0){
  //   for(int i = 0; i < lN; i++){//Compute splitters
  //     printf("v[%d] = %d on process %d\n",i,v[i],mpirank);
  //   }
  //   for(int i = 0; i < p-1; i++){//Compute splitters
  //     printf("s[%d] = %d on process %d\n",i,s[i],mpirank);
  //   }
  //   for(int i = 0; i < p; i++){//Compute splitters
  //     printf("displacement = %d, count = %d\n",sdispls[i],bcounts[i]);
  //   }

  //   for(int i = 0; i<p; i++){
  //     printf("allct[%d] = %d, rank = %d\n", i, allct[i], mpirank);
  //   }
  //   for(int i = 0; i<p;i++){
  //     printf("alldis[%d] = %d, rank = %d\n", i,alldis[i], mpirank);
  //   }

  //     for(int i = 0; i<2*lN; i++){
  //       printf("finBuck[%d] = %d, rank = %d\n", i,finBuck[i],mpirank);
  //     }
  // }

  /* timing */
  MPI_Barrier(MPI_COMM_WORLD);
  double elapsed = MPI_Wtime() - tt;
  if (0 == mpirank) {
    printf("Time elapsed is %f seconds. rank = %d\n", elapsed, mpirank);
  }

  { // Write output to a file
    FILE* fd = NULL;
    char filename[256];
    snprintf(filename, 256, "output%02d.txt", mpirank);
    fd = fopen(filename,"w+");

    if(NULL == fd) {
      printf("Error opening file \n");
      return 1;
    }

  for(int i = 0; i<2*lN; i++){//Write results to file
    if((i > 0)&&(finBuck[i]<finBuck[i-1])){//If true then not part of original list
      break;
    }
    //printf("finBuck[%d] = %d, rank = %d\n", i,finBuck[i],mpirank);
    fprintf(fd, "  %d\n", finBuck[i]);
  }
      

    fclose(fd);
  }

  free(v);
  free(s);
  free(sp);
  free(sdispls);
  free(bcounts);
  free(allct);
  free(alldis);
  free(finBuck);


  MPI_Finalize();
  return 0;
}