/* MPI-parallel Jacobi smoothing to solve -u''=f
 * Global vector has N unknowns, each processor works with its
 * part, which has lN = N/p unknowns.
 * Author: Georg Stadler
 */
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <string.h>
#include <stdlib.h>

/* compuate global residual, assuming ghost values are updated */
double compute_residual(double *lu, int lN, double invhsq){
  double tmp, gres = 0.0, lres = 0.0;
  bool skip = false;

  for (long i = 0; i < (lN+2)*(lN+2); i++){
      skip = false;
      if(i <= (lN+2)){skip = true;} //If first row, then skip
      if(i % (lN+2) == 0){skip = true;}//If first column, then skip
      if(i % (lN+2) == ((lN+2)-1)){skip = true;}//If last column, then skip
      if(i >= (lN+2)*(lN+2) - (lN+2)){skip = true;}//If last row, then skip
      
      if(!skip){
        tmp = ((4.0*lu[i] - lu[i-1] - lu[i+1] - lu[i + lN+2] - lu[i - (lN+2)]) * invhsq - 1);
        lres += tmp * tmp;
      }
  }
  /* use allreduce for convenience; a reduce would also be sufficient */
  MPI_Allreduce(&lres, &gres, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return sqrt(gres);
}

void iniJac(double *lu, double *lunew, long bl_size, int sqrtp, int mpirank){
  /*Initialize solutions with zeros*/
  bool skip = false;

  for (long i = 0; i < bl_size*bl_size; i++){
      skip = false;
      if((i < bl_size) && (mpirank < sqrtp)){skip = true;} //If first row, then skip
      if((i % bl_size == 0) && (mpirank%sqrtp == 0)){skip = true;}//If first column, then skip
      if((i % bl_size == (bl_size-1)) && ((mpirank+1)%(sqrtp) == 0)){skip = true;}//If last column, then skip
      if((i >= bl_size*bl_size - bl_size) && (mpirank >= sqrtp*sqrtp - sqrtp)){skip = true;}//If last row, then skip
      
      if(!skip){
        lu[i] = 1;
      }else{
        lu[i] = 0;
      }
  }

}


int main(int argc, char * argv[]){
  int mpirank, p, N, lN, iter, max_iters;
  MPI_Status status, status1;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  /* get name of host running MPI process */
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);
  printf("Rank %d/%d running on %s.\n", mpirank, p, processor_name);

  sscanf(argv[1], "%d", &N);
  sscanf(argv[2], "%d", &max_iters);

  /* compute number of unknowns handled by each process */
  lN = N / p;
  if ((N % p != 0) && mpirank == 0 ) {
    printf("N: %d, local N: %d\n", N, lN);
    printf("Exiting. N must be a multiple of p\n");
    MPI_Abort(MPI_COMM_WORLD, 0);
  }
  /* timing */
  MPI_Barrier(MPI_COMM_WORLD);
  double tt = MPI_Wtime();


  /* Allocation of vectors, including surrounding ghost points */
  long block_sz = (lN + 2);
  // double ** lu    = (double **)calloc(sizeof(double *), block_sz );
  // double ** lunew = (double **)calloc(sizeof(double *), block_sz );
  double * lu    = (double *)calloc(sizeof(double *), block_sz*block_sz );
  double * lunew = (double *)calloc(sizeof(double *), block_sz*block_sz );
  double * lbuff = (double *)calloc(sizeof(double), block_sz); //Left buffer for column communication
  double * rbuff = (double *)calloc(sizeof(double), block_sz); //Right buffer for column communication
  double * lutemp;

  //Allocate second dimension
  // for(int i = 0; i<N; i++){
  //   lunew[i] = (double *)calloc(sizeof(double),block_sz);
  //   lu[i] = (double *)calloc(sizeof(double),block_sz);
  // }

  double h = 1.0 / (N + 1);
  double hsq = h * h;
  double invhsq = 1./hsq;
  int sqrtp = (int)sqrt(p);
  double gres, gres0, tol = 1e-5;


  iniJac(lu,lunew,block_sz, sqrtp, mpirank); //Initalize with zeros


  /* initial residual */
  gres0 = compute_residual(lu, lN, invhsq);
  gres = gres0;

  for (iter = 0; iter < max_iters && gres/gres0 > tol; iter++) {

    /* Jacobi step for local points assume f = 1*/
    for (long i = 0; i < (lN+2)*(lN+2); i++){
        bool skip = false;
        if(i <= (lN+2)){skip = true;} //If first row, then skip
        if(i % (lN+2) == 0){skip = true;}//If first column, then skip
        if(i % (lN+2) == ((lN+2)-1)){skip = true;}//If last column, then skip
        if(i >= (lN+2)*(lN+2) - (lN+2)){skip = true;}//If last row, then skip
        
        if(!skip){
          lunew[i]  = 0.25 * (hsq + lu[i - 1] + lu[i + 1] + lu[i + lN+2] + lu[i - (lN+2)]);
        }
    }

    // /* communicate ghost values */
    if (mpirank >= sqrtp) {//If not a top row, communicate top
      //printf("rank %d sending to rank %d\n",mpirank,mpirank-sqrtp );
      MPI_Send(&(lunew[lN+2]), lN+2, MPI_DOUBLE, mpirank-sqrtp, 124, MPI_COMM_WORLD);
      //printf("rank %d recieving from rank %d\n",mpirank,mpirank-sqrtp );
      MPI_Recv(&(lunew[0]), lN+2, MPI_DOUBLE, mpirank-sqrtp, 123, MPI_COMM_WORLD, &status);
    }
    if (mpirank < p - sqrtp) {//If not a bottom row, communicate bottom row
      //printf("rank %d recieving from rank %d\n",mpirank,mpirank+sqrtp );
      MPI_Recv(&(lunew[(lN+1)*(lN+2)]), lN+2, MPI_DOUBLE, mpirank+sqrtp, 124, MPI_COMM_WORLD, &status1);
      //printf("rank %d sending to rank %d\n",mpirank,mpirank+sqrtp );
      MPI_Send(&(lunew[lN*(lN+2)]), lN+2, MPI_DOUBLE, mpirank+sqrtp, 123, MPI_COMM_WORLD);
    }
    if (mpirank%sqrtp != 0) {//If not a left edge, communicate left column

      //Initialize buffer
      for(long i = 0; i < lN+2; i++){
        lbuff[i] = lunew[1 + i*(lN+2)];
      }

      MPI_Send(lbuff, lN+2, MPI_DOUBLE, mpirank-1, 125, MPI_COMM_WORLD);
      MPI_Recv(lbuff, lN+2, MPI_DOUBLE, mpirank-1, 126, MPI_COMM_WORLD, &status1);

      //Assign message
      for(long i = 0; i < lN+2; i++){
        lunew[i*(lN+2)] = lbuff[i];
      }
    }
    if((mpirank+1)%sqrtp != 0) {//If not a right edge, communicate right column

      MPI_Recv(rbuff, lN+2, MPI_DOUBLE, mpirank+1, 125, MPI_COMM_WORLD, &status1);
      //Assign message
      for(long i = 0; i < lN+2; i++){
        lunew[lN+1 + i*(lN+2)] = rbuff[i];
      }

      //Initialize buffer
      for(long i = 0; i < lN+2; i++){
        rbuff[i] = lunew[lN + i*(lN+2)];
      }

      MPI_Send(rbuff, lN+2, MPI_DOUBLE, mpirank+1, 126, MPI_COMM_WORLD);

    }

    /* copy newu to u using pointer flipping */
    lutemp = lu; lu = lunew; lunew = lutemp;
    if (0 == (iter % 100)) {
      gres = compute_residual(lu, lN, invhsq);
      if (0 == mpirank) {
  printf("Iter %d: Residual: %g\n", iter, gres);
      }
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);

  // /* Clean up */
  // for(long i = 0; i<block_sz; i++){
  //   free(lu[i]);
  //   free(lunew[i]);
  // }

  // if(mpirank == 1){
  //   int newline = 1;
  //   for(long i = 0; i < block_sz*block_sz; i++, newline++){
  //     printf("%f ", lu[i]);
  //     if(newline%(block_sz)==0){printf("\n");}
  //   }
  // }

  free(lu);
  free(lunew);
  free(rbuff);
  free(lbuff);

  /* timing */
  MPI_Barrier(MPI_COMM_WORLD);
  double elapsed = MPI_Wtime() - tt;
  if (0 == mpirank) {
    printf("Time elapsed is %f seconds.\n", elapsed);
  }
  MPI_Finalize();
  return 0;
}