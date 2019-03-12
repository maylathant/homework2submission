/******************************************************************************
* FILE: omp_bug6.c
* DESCRIPTION:
*   This program compiles and runs fine, but produces the wrong result.
*   Compare to omp_orphan.c.
* AUTHOR: Blaise Barney  6/05
* LAST REVISED: 06/30/05
******************************************************************************/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define VECLEN 100

float a[VECLEN], b[VECLEN];

float dotprod (float *result)
{
int i,tid;
float sum = 0.0;

	#pragma omp parallel for reduction(+:sum) //Anthony Maylath Correction: start parellel region here
	  for (i=0; i < VECLEN; i++)
	    {
	    tid = omp_get_thread_num(); //Get tid for each iteration
	    sum = sum + (a[i]*b[i]);
	    printf("  tid= %d i=%d\n",tid,i);
	    }

	    *result = sum;
}


int main (int argc, char *argv[]) {
int i;
float sum;

for (i=0; i < VECLEN; i++)
  a[i] = b[i] = 1.0 * i;
sum = 0.0;

//#pragma omp parallel shared(sum) //Anthony Maylath Correction: No parellel in outer scope
  dotprod(&sum); //Anthony Maylath Edit: pass reference to sum instead of nothing

printf("Sum = %f\n",sum);

}

