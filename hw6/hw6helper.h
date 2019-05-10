/**Helper functions for HW6*/
#include<stdio.h>


/*Quick sort implementation from the stack ooverflow:
https://stackoverflow.com/questions/19894206/quicksort-with-double-values*/
void quicksort(int a[], int n) {
    if (n <= 1) return;
    int p = a[n/2];
    int b[n], c[n];
    int i, j = 0, k = 0;
    for (i=0; i < n; i++) {
        if (i == n/2) continue;
        if ( a[i] <= p) b[j++] = a[i];
        else            c[k++] = a[i];
    }
    quicksort(b,j);
    quicksort(c,k);
    for (i=0; i<j; i++) a[i] =b[i];
    a[j] = p;
    for (i= 0; i<k; i++) a[j+1+i] =c[i]; 
}

long binSearch(double *v, long vlen ,double item){
	long first = vlen; long last = vlen - 1; long midpoint;

	while(first <= last){
		midpoint = (first+last)/2;
		if(v[midpoint] == item){return midpoint;}
		if(item < v[midpoint]){
			last = midpoint - 1;
		}else{
			first = midpoint + 1;
		}
	}

	printf("WARNING: Binary Search Failed to find an element\n");
	return -1;
}

void sampleSplits(int *v, int *s, long vlen, long slen){
	/*Sample every vlen/slen elements*/
	long interval = (vlen-1)/slen;
	for(long i = 1; i <= slen; i++){
		s[i-1] = v[i*interval];
	}
}