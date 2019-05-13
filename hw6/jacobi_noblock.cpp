#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <string.h>
#include <unistd.h>
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

void localJacobi(double *lu, double *lunew, int lN, double hsq){
  /*Jacobi step for interior points
  lu : local solution
  lunew : next step in local solution
  lN : local dimension
  hsq : 1 divided by global dimension squared*/
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
}

void innerJacobi(double *lu, double *lunew, int lN, double hsq){
  /*Jacobi step for interior points that dont boarder the ghost
  lu : local solution
  lunew : next step in local solution
  lN : local dimension
  hsq : 1 divided by global dimension squared*/
  for (long i = 0; i < (lN+2)*(lN+2); i++){
      bool skip = false;
      if(i <= (lN+2)*2){skip = true;} //If first two rows, then skip
      if(i % (lN+2) < 2){skip = true;}//If first two column, then skip
      if(i % (lN+2) > ((lN+2)-3)){skip = true;}//If last two column, then skip
      if(i >= (lN+2)*(lN+2) - 2*(lN+2)){skip = true;}//If last two rows, then skip
      
      if(!skip){
        lunew[i]  = 0.25 * (hsq + lu[i - 1] + lu[i + 1] + lu[i + lN+2] + lu[i - (lN+2)]);
      }
  }
}

void outerJacobi(double *lu, double *lunew, int lN, double hsq){
  /*Jacobi step for interior points that boarder the ghost only
  lu : local solution
  lunew : next step in local solution
  lN : local dimension
  hsq : 1 divided by global dimension squared*/
  int offset1 = (lN+2), offset2 = (lN+2)*lN;
  for (long i = 1; i < lN+1; i++){//Update first and last non-ghost rows
      lunew[offset1 + i]  = 0.25 * (hsq + lu[offset1+ i - 1] + lu[offset1 +i + 1] + lu[offset1 + i + lN+2] + lu[offset1 +i - (lN+2)]);
      lunew[offset2 + i]  = 0.25 * (hsq + lu[offset2+ i - 1] + lu[offset2 +i + 1] + lu[offset2 + i + lN+2] + lu[offset2 +i - (lN+2)]);
  }

  for (long i = 1; i < lN+1; i++){//Update first and last non-ghost columns
      lunew[offset1 + (i-1)*offset1 + 1]  = 0.25 * (hsq + lu[offset1 + (i-1)*offset1] + lu[offset1 + (i-1)*offset1 + 2] + lu[offset1 + (i-1)*offset1+ lN+3] + lu[offset1 + (i-1)*offset1 - (lN+2) +1]);
      lunew[offset1*(1+i)-2]  = 0.25 * (hsq + lu[offset1*(1+i)-3] + lu[offset1*(1+i)-1] + lu[offset1*(1+i) + lN] + lu[offset1*(1+i)-2 - (lN+2)]);
  }

}


int main(int argc, char * argv[]){
  int mpirank, p, N, lN, iter, max_iters;
  MPI_Status status, status1;
  MPI_Request sendr, top, bottom, left, right;
  int topflag, bottomflag, leftflag, rightflag;

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
  double * lu    = (double *)calloc(sizeof(double *), block_sz*block_sz );
  double * lunew = (double *)calloc(sizeof(double *), block_sz*block_sz );
  double * lwait = (double *)calloc(sizeof(double *), block_sz*block_sz ); //To store data computed while waiting for message
  double * lbuff = (double *)calloc(sizeof(double), block_sz); //Left buffer for column communication
  double * rbuff = (double *)calloc(sizeof(double), block_sz); //Right buffer for column communication
  double * lutemp;


  double h = 1.0 / (N + 1);
  double hsq = h * h;
  double invhsq = 1./hsq;
  int sqrtp = (int)sqrt(p);
  double gres, gres0, tol = 1e-5;


  iniJac(lu,lunew,block_sz, sqrtp, mpirank); //Initalize with zeros

  /* initial residual */
  gres0 = compute_residual(lu, lN, invhsq);
  gres = gres0;


  bool skipiter = false; //flag to skip localJaboci call
  for (iter = 0; iter < max_iters && gres/gres0 > tol; iter++) {

    //Compute initial Jacobi step
    if(!skipiter){
      localJacobi(lu, lunew, lN, hsq);
    } 
    skipiter = false; //reset flag

    // /* communicate ghost values */
    if (mpirank >= sqrtp) {//If not a top row, communicate top
      //printf("rank %d sending to rank %d\n",mpirank,mpirank-sqrtp );
      MPI_Isend(&(lunew[lN+2]), lN+2, MPI_DOUBLE, mpirank-sqrtp, 124, MPI_COMM_WORLD,&sendr);
      //printf("rank %d recieving from rank %d\n",mpirank,mpirank-sqrtp );
      MPI_Irecv(&(lunew[0]), lN+2, MPI_DOUBLE, mpirank-sqrtp, 123, MPI_COMM_WORLD, &top);
    }
    if (mpirank < p - sqrtp) {//If not a bottom row, communicate bottom row
      //printf("rank %d recieving from rank %d\n",mpirank,mpirank+sqrtp );
      MPI_Irecv(&(lunew[(lN+1)*(lN+2)]), lN+2, MPI_DOUBLE, mpirank+sqrtp, 124, MPI_COMM_WORLD,&bottom);
      //printf("rank %d sending to rank %d\n",mpirank,mpirank+sqrtp );
      MPI_Isend(&(lunew[lN*(lN+2)]), lN+2, MPI_DOUBLE, mpirank+sqrtp, 123, MPI_COMM_WORLD,&sendr);
    }
    // if (mpirank%sqrtp != 0) {//If not a left edge, communicate left column
        //Nothing to do yet

    // }
    if((mpirank+1)%sqrtp != 0) {//If not a right edge, communicate right column

      MPI_Irecv(rbuff, lN+2, MPI_DOUBLE, mpirank+1, 125, MPI_COMM_WORLD, &right);

    }


    //Second wave of message passing
    if(iter<max_iters-1){innerJacobi(lunew, lwait, lN, hsq);} //Compute points not depending on ghost

    if(mpirank >= sqrtp){MPI_Wait(&top,&status);} //Wait for ghost points to communicate
    if(mpirank < p - sqrtp){MPI_Wait(&bottom,&status1);}
    if(mpirank%sqrtp != 0){//Left
      for(long i = 0; i < lN+2; i++){
        lbuff[i] = lunew[1 + i*(lN+2)];
        //if(mpirank==3){printf("lbuff[%ld] = %f\n",i,lunew[1 + i*(lN+2)] );}
      }

      MPI_Isend(lbuff, lN+2, MPI_DOUBLE, mpirank-1, 125, MPI_COMM_WORLD,&sendr);
      MPI_Irecv(lbuff, lN+2, MPI_DOUBLE, mpirank-1, 126, MPI_COMM_WORLD, &left);
      MPI_Wait(&left,&status);
      //Assign message
      for(long i = 0; i < lN+2; i++){
        lunew[i*(lN+2)] = lbuff[i];
      }
    }
    if((mpirank+1)%sqrtp != 0){//Right
      MPI_Wait(&right,&status1);
      //Assign message
      for(long i = 0; i < lN+2; i++){
        lunew[lN+1 + i*(lN+2)] = rbuff[i];
      }

      //Initialize buffer
      for(long i = 0; i < lN+2; i++){
        rbuff[i] = lunew[lN + i*(lN+2)];
      }

      MPI_Isend(rbuff, lN+2, MPI_DOUBLE, mpirank+1, 126, MPI_COMM_WORLD,&sendr);
    }
    if(iter<max_iters-1){
      outerJacobi(lunew, lwait, lN, hsq);
      skipiter = true; //Skip next jacobi iteration
    } //Compute points depending on ghost

    /* copy newu to u using pointer flipping */
    lutemp = lu; lu = lunew; lunew = lutemp;
    if (0 == (iter % 100)) {
      gres = compute_residual(lu, lN, invhsq);
      if (0 == mpirank) {
  printf("Iter %d: Residual: %g\n", iter, gres);
      }
    }

    if(skipiter){
      /* copy extra iteration to u using pointer flipping */
      lutemp = lunew; lunew = lwait; lwait = lutemp;
    }

  }

  MPI_Barrier(MPI_COMM_WORLD);


  // if(mpirank == 2){
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
  free(lwait);

  /* timing */
  MPI_Barrier(MPI_COMM_WORLD);
  double elapsed = MPI_Wtime() - tt;
  if (0 == mpirank) {
    printf("Time elapsed is %f seconds.\n", elapsed);
  }
  MPI_Finalize();
  return 0;
}