#include "mpi.h"
#include <stdlib.h>
#include <stdio.h>
/*
This program tests MPI_Alltoallv by having processor i send different
amounts of data to each processor.
The first test sends i items to processor i from all processors.
*/
int main( int argc, char **argv )
{
    MPI_Comm comm;
    int *sbuf, *rbuf;
    int rank, size;
    int *sendcounts, *recvcounts, *rdispls, *sdispls;
    int i, j, *p, err;

    MPI_Init( &argc, &argv );
    err = 0;
    comm = MPI_COMM_WORLD;
    /* Create the buffer */
    MPI_Comm_size( comm, &size );
    MPI_Comm_rank( comm, &rank );
    sbuf = (int *)malloc( size * size * sizeof(int) );
    rbuf = (int *)malloc( size * size * sizeof(int) );
    if (!sbuf || !rbuf) {
        fprintf( stderr, "Could not allocated buffers!\n" );
        MPI_Abort( comm, 1 );
    }
    /* Load up the buffers */
    for (i=0; i<size*size; i++) {
        sbuf[i] = i + 100*rank;
        rbuf[i] = -i;
    }
    /* Create and load the arguments to alltoallv */
    sendcounts = (int *)malloc( size * sizeof(int) );
    recvcounts = (int *)malloc( size * sizeof(int) );
    rdispls = (int *)malloc( size * sizeof(int) );
    sdispls = (int *)malloc( size * sizeof(int) );
    if (!sendcounts || !recvcounts || !rdispls || !sdispls) {
        fprintf( stderr, "Could not allocate arg items!\n" );fflush(stderr);
        MPI_Abort( comm, 1 );
    }
    for (i=0; i<size; i++) {
        sendcounts[i] = i;
        recvcounts[i] = rank;
        rdispls[i] = i * rank;
        sdispls[i] = (i * (i+1))/2;
    }

    if(rank==2){
    for(int i = 0; i < size*size; i++){//Compute splitters
      printf("v[%d] = %d on process %d\n",i,sbuf[i],rank);
    }

    for(int i = 0; i < size; i++){//Compute splitters
      printf("displacement = %d, count = %d\n",sdispls[i],sendcounts[i]);
    }

    for(int i = 0; i<size; i++){
      printf("allct[%d] = %d, rank = %d\n", i, recvcounts[i], rank);
    }
    for(int i = 0; i<size;i++){
      printf("alldis[%d] = %d, rank = %d\n", i,rdispls[i], rank);
    }

      for(int i = 0; i<size*size; i++){
        printf("finBuck[%d] = %d, rank = %d\n", i,rbuf[i],rank);
      }
    }

    MPI_Alltoallv( sbuf, sendcounts, sdispls, MPI_INT,
                       rbuf, recvcounts, rdispls, MPI_INT, comm );
    

    if(rank==2){
    printf("AFTER:\n");
    for(int i = 0; i < size*size; i++){//Compute splitters
      printf("v[%d] = %d on process %d\n",i,sbuf[i],rank);
    }

    for(int i = 0; i < size; i++){//Compute splitters
      printf("displacement = %d, count = %d\n",sdispls[i],sendcounts[i]);
    }

    for(int i = 0; i<size; i++){
      printf("allct[%d] = %d, rank = %d\n", i, recvcounts[i], rank);
    }
    for(int i = 0; i<size;i++){
      printf("alldis[%d] = %d, rank = %d\n", i,rdispls[i], rank);
    }

      for(int i = 0; i<size*size; i++){
        printf("finBuck[%d] = %d, rank = %d\n", i,rbuf[i],rank);
      }
    }

    /* Check rbuf */
    for (i=0; i<size; i++) {
        p = rbuf + rdispls[i];
        for (j=0; j<rank; j++) {
            if (p[j] != i * 100 + (rank*(rank+1))/2 + j) {
                fprintf( stderr, "[%d] got %d expected %d for %dth\n",
                                    rank, p[j],(i*(i+1))/2 + j, j );
                fflush(stderr);
                err++;
            }
        }
    }
    free( sdispls );
    free( rdispls );
    free( recvcounts );
    free( sendcounts );
    free( rbuf );
    free( sbuf );
    MPI_Finalize();
    return 0;
}