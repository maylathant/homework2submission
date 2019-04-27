#include <stdio.h>
#include <unistd.h>
#include <mpi.h>
#include <stdlib.h>
#define msgSize 500000

void timeRing(int iter){
  int rank, csize, msg = 0;
  double lat = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &csize);
  MPI_Status status;

  MPI_Barrier(MPI_COMM_WORLD);
  double tt = MPI_Wtime();

  for(int k = 0; k<iter; k++){
    for(int i = 0; i<csize; i++){
      if(i == csize-1){//Send back to first process
          if(rank == csize-1){msg += rank; MPI_Send(&msg, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);}
          if(rank == 0){ MPI_Recv(&msg, 1, MPI_INT, csize-1, 0, MPI_COMM_WORLD, &status);}
      }else{
        if(rank == i){msg += rank; MPI_Send(&msg, 1, MPI_INT, rank+1, 0, MPI_COMM_WORLD);}
        if(rank-1 == i){MPI_Recv(&msg, 1, MPI_INT, rank-1, 0, MPI_COMM_WORLD, &status);}
      }
    }
  }

  if(rank == 0){
    double dur = MPI_Wtime()-tt;
    printf("Final message primative is: %d, Duration in seconds: %f, Average Latency = %f\n", msg,dur,dur/(iter*csize));
  }

}

void timeRingAr(int iter){
  int rank, csize;
  int  *msg = (int *)malloc(msgSize*sizeof(int));
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &csize);
  MPI_Status status;

  //Initalize array
  for(int i = 0; i<msgSize; i++)
    msg[i] = 0;

  MPI_Barrier(MPI_COMM_WORLD);
  double tt = MPI_Wtime();

  for(int k = 0; k<iter; k++){
    for(int i = 0; i<csize; i++){
      if(i == csize-1){//Send back to first process
          if(rank == csize-1){
            for(int j = 0; j<msgSize; j++) msg[j] += rank; //Increment all elements in array
            MPI_Send(msg, msgSize, MPI_INT, 0, 0, MPI_COMM_WORLD);
          }
          if(rank == 0){ MPI_Recv(msg, msgSize, MPI_INT, csize-1, 0, MPI_COMM_WORLD, &status);}
      }else{
        if(rank == i){
          for(int j = 0; j<msgSize; j++) msg[j] += rank; //Increment all elements in array by the rank
          MPI_Send(msg, msgSize, MPI_INT, rank+1, 0, MPI_COMM_WORLD);
      }
        if(rank-1 == i){MPI_Recv(msg, msgSize, MPI_INT, rank-1, 0, MPI_COMM_WORLD, &status);}
      }
    }
  }

  //Print a random element to verify
  if(rank == 0){
    int ranel = rand() % msgSize; double dur = MPI_Wtime()-tt;
    printf("Final message array is: %d, Duration in seconds: %f Bandwidth (GB/s) = %f\n", msg[ranel],dur,msgSize*3/dur/1e9);}  
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

    int c, args = 0; int iter;
    while((c = getopt(argc, argv, "i:")) != -1){
        args++;
        switch(c)
        {
            case 'i':
                iter = atoi(optarg);
                break;
            case '?':
                if (optopt == 'i')
                    fprintf(stderr, "option -%c requires an argument.\n", optopt);
                return 1;
        }
    }
    if(args<1){
        printf("-i [number of iterations] required arguments\n");
        return 1;
    }

  timeRing(iter);
  timeRingAr(iter);

  MPI_Finalize();
}
