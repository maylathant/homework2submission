#include <math.h>
#include <stdio.h>
#include "utils.h"
#include "intrin-wrapper.h"

// Headers for intrinsics
#ifdef __SSE__
#include <xmmintrin.h>
#endif
#ifdef __SSE2__
#include <emmintrin.h>
#endif
#ifdef __AVX__
#include <immintrin.h>
#endif




// coefficients in the Taylor series expansion of sin(x)
static constexpr double c3  = -1/(((double)2)*3);
static constexpr double c5  =  1/(((double)2)*3*4*5);
static constexpr double c7  = -1/(((double)2)*3*4*5*6*7);
static constexpr double c9  =  1/(((double)2)*3*4*5*6*7*8*9);
static constexpr double c11 = -1/(((double)2)*3*4*5*6*7*8*9*10*11);
// sin(x) = x + c3*x^3 + c5*x^5 + c7*x^7 + x9*x^9 + c11*x^11

//Anthony Maylath edit
// coefficients in the Taylor series expansion of cos(x)
static constexpr double c2  = -1/(((double)2));
static constexpr double c4  =  1/(((double)2)*3*4);
static constexpr double c6  = -1/(((double)2)*3*4*5*6);
static constexpr double c8  =  1/(((double)2)*3*4*5*6*7*8);
static constexpr double c10 = -1/(((double)2)*3*4*5*6*7*8*9*10);
// cos(x) = 1+ c2*x^2 + c4*x^4 + c6*x^6 + x8*x^8 + c10*x^10

static constexpr double twoPI = 2.0*M_PI; //Save calculation on 2pi
static constexpr double pi4 = M_PI/4;
static constexpr double pi34 = 3*M_PI/4;
static constexpr double pi54 = 5*M_PI/4;

void sin4_reference(double* sinx, const double* x) {
  for (long i = 0; i < 4; i++) sinx[i] = sin(x[i]);
}

void sin4_referenceEC(double* sinx, const double* x) {
  double theta = x[0];
  for (long i = 0; i < 4; i++) sinx[i] = sin(theta + i*M_PI/2);
}

void sin4_taylor(double* sinx, const double* x) {
  for (int i = 0; i < 4; i++) {
    double x1  = x[i];
    double x2  = x1 * x1;
    double x3  = x1 * x2;
    double x5  = x3 * x2;
    double x7  = x5 * x2;
    double x9  = x7 * x2;
    double x11 = x9 * x2;

    double s = x1;
    s += x3  * c3;
    s += x5  * c5;
    s += x7  * c7;
    s += x9  * c9;
    s += x11 * c11;
    sinx[i] = s;
  }
}

//Anthony Maylath addition for Extra credit
void sin4_taylorEC(double* sinx, const double* x) {
  double theta, sine, cosine;
  theta = x[0];
  //Evaluate sine of theta
  double x2  = theta * theta;
  double x3  = theta * x2;
  double x5  = x3 * x2;
  double x7  = x5 * x2;
  double x9  = x7 * x2;
  double x11 = x9 * x2;

  double s = theta;
  s += x3  * c3;
  s += x5  * c5;
  s += x7  * c7;
  s += x9  * c9;
  s += x11 * c11;
  sine = s;

  //Evaluate cosine of theta
  double x4  = x2 * x2;
  double x6  = x3 * x3;
  double x8  = x5 * x3;
  double x10  = x8 * x2;

  s = 1.0;
  s += x2  * c2;
  s += x4  * c4;
  s += x6  * c6;
  s += x8  * c8;
  s += x10 * c10;
  cosine = s;

  double coff;
  for (int i = 0; i < 4; i++) {//Evaluate sine(theta + i*pi/2) for i = 0,1,2,3
    if(i%2 == 0){//East/west in the unit circle
      coff = sine;
    }else{//North/south
      coff = cosine;
    }

    if(i > 1){coff = -1.0*coff;} //Mirror image

    sinx[i] = coff;

  }//End for
}


void sin4_intrin(double* sinx, const double* x) {
  // The definition of intrinsic functions can be found at:
  // https://software.intel.com/sites/landingpage/IntrinsicsGuide/#
#if defined(__AVX__)

  __m256d x1, x2, x3;
  x1  = _mm256_load_pd(x);
  x2  = _mm256_mul_pd(x1, x1);
  x3  = _mm256_mul_pd(x1, x2);

  //Anthony Maylath edit Add higher order terms
  __m256d x5, x7, x9, x11;
  x5 = _mm256_mul_pd(x2, x3); //Compute 5th power
  x7 = _mm256_mul_pd(x2, x5); //Compute 7th power
  x9 = _mm256_mul_pd(x2, x7); //Compute 9th power
  x11 = _mm256_mul_pd(x2, x9); //Compute 11th power

  __m256d s = x1;
  s = _mm256_add_pd(s, _mm256_mul_pd(x3 , _mm256_set1_pd(c3 )));
  s = _mm256_add_pd(s, _mm256_mul_pd(x5 , _mm256_set1_pd(c5 ))); //Add Higher powers
  s = _mm256_add_pd(s, _mm256_mul_pd(x7 , _mm256_set1_pd(c7 )));
  s = _mm256_add_pd(s, _mm256_mul_pd(x9 , _mm256_set1_pd(c9 )));
  s = _mm256_add_pd(s, _mm256_mul_pd(x11 , _mm256_set1_pd(c11 )));
  _mm256_store_pd(sinx, s);
#elif defined(__SSE2__)

  constexpr int sse_length = 2;
  for (int i = 0; i < 4; i+=sse_length) {
    __m128d x1, x2, x3;
    x1  = _mm_load_pd(x+i);
    x2  = _mm_mul_pd(x1, x1);
    x3  = _mm_mul_pd(x1, x2);

    __m128d x5, x7, x9, x11;
    x5 = _mm_mul_pd(x2,x3); //Anthony Maylath edit ~ compute higher order terms
    x7 = _mm_mul_pd(x2,x5);
    x9 = _mm_mul_pd(x2,x7);
    x11 = _mm_mul_pd(x2,x9);

    __m128 s = x1;
    s = _mm_add_pd(s, _mm_mul_pd(x3 , _mm_set1_pd(c3 )));
    s = _mm_add_pd(s, _mm_mul_pd(x5 , _mm_set1_pd(c5 ))); //Anthony Maylath edit ~ add higher order terms
    s = _mm_add_pd(s, _mm_mul_pd(x7 , _mm_set1_pd(c7 )));
    s = _mm_add_pd(s, _mm_mul_pd(x9 , _mm_set1_pd(c9 )));
    s = _mm_add_pd(s, _mm_mul_pd(x11 , _mm_set1_pd(c11 )));
    _mm_store_pd(sinx+i, s);
  }
#else
  sin4_reference(sinx, x);
#endif
}

void sin4_intrinEC(double* sinx, const double* x) {
  // The definition of intrinsic functions can be found at:
  // https://software.intel.com/sites/landingpage/IntrinsicsGuide/#
  #if defined(__SSE2__)

  constexpr int sse_length = 2;
  for (int i = 0; i < 2; i+=sse_length) {
    __m128d x1, x2, x3;
    x1  = _mm_load_pd(x+i);
    x2  = _mm_mul_pd(x1, x1);
    x3  = _mm_mul_pd(x1, x2);

    __m128d x5, x7, x9, x11;
    x5 = _mm_mul_pd(x2,x3); //Anthony Maylath edit ~ compute higher order terms
    x7 = _mm_mul_pd(x2,x5);
    x9 = _mm_mul_pd(x2,x7);
    x11 = _mm_mul_pd(x2,x9);

    __m128 s = x1;
    s = _mm_add_pd(s, _mm_mul_pd(x3 , _mm_set1_pd(c3 )));
    s = _mm_add_pd(s, _mm_mul_pd(x5 , _mm_set1_pd(c5 ))); //Anthony Maylath edit ~ add higher order terms
    s = _mm_add_pd(s, _mm_mul_pd(x7 , _mm_set1_pd(c7 )));
    s = _mm_add_pd(s, _mm_mul_pd(x9 , _mm_set1_pd(c9 )));
    s = _mm_add_pd(s, _mm_mul_pd(x11 , _mm_set1_pd(c11 )));
    _mm_store_pd(sinx+i, s);
  }

  for(int i = 2; i < 4; i++){//Use symmetry ~ take negation
    sinx[i] = -1*sinx[i-2];
  }

#else
  sin4_referenceEC(sinx, x);
#endif
}

void sin4_vector(double* sinx, const double* x) {
  // The Vec class is defined in the file intrin-wrapper.h
  typedef Vec<double,4> Vec4;
  Vec4 x1, x2, x3;
  x1  = Vec4::LoadAligned(x);
  x2  = x1 * x1;
  x3  = x1 * x2;

  Vec4 s = x1;
  s += x3  * c3 ;
  s.StoreAligned(sinx);
}

double err(double* x, double* y, long N) {
  double error = 0;
  for (long i = 0; i < N; i++) error = std::max(error, fabs(x[i]-y[i]));
  return error;
}

int main() {

  Timer tt;
  long N = 1000000;
  double* x = (double*) aligned_malloc(N*sizeof(double));
  double* x_sh = (double*) aligned_malloc(N*sizeof(double));
  double* sinx_ref = (double*) aligned_malloc(N*sizeof(double));
  double* sinx_taylor = (double*) aligned_malloc(N*sizeof(double));
  double* sinx_intrin = (double*) aligned_malloc(N*sizeof(double));
  double* sinx_vector = (double*) aligned_malloc(N*sizeof(double));
  double* sinx_refEC = (double*) aligned_malloc(N*sizeof(double));
  double* sinx_tayEC = (double*) aligned_malloc(N*sizeof(double)); //Taylor extra credit
  double* sinx_intrinEC = (double*) aligned_malloc(N*sizeof(double)); //Intrin extra credit

  for (long i = 0; i < N; i++) {
    x[i] = (drand48()-0.5) * M_PI/2; // [-pi/4,pi/4]
    sinx_ref[i] = 0;
    sinx_taylor[i] = 0;
    sinx_intrin[i] = 0;
    sinx_vector[i] = 0;
    sinx_refEC[i] = 0;
    sinx_tayEC[i] = 0;
  }

  for(long i = 0; i < N; i+=4){ //Shifted values for extra credit
    for(long j = 0; j < 4; j++){
      x_sh[i+j] = x[i] + j*M_PI/2;
    }
  }

  tt.tic();
  for (long rep = 0; rep < 1000; rep++) {
    for (long i = 0; i < N; i+=4) {
      sin4_reference(sinx_ref+i, x+i);
    }
  }
  printf("Reference time: %6.4f\n", tt.toc());

  tt.tic();
  for (long rep = 0; rep < 1000; rep++) {
    for (long i = 0; i < N; i+=4) {
      sin4_taylor(sinx_taylor+i, x+i);
    }
  }
  printf("Taylor time:    %6.4f      Error: %e\n", tt.toc(), err(sinx_ref, sinx_taylor, N));

  tt.tic();
  for (long rep = 0; rep < 1000; rep++) {
    for (long i = 0; i < N; i+=4) {
      sin4_intrin(sinx_intrin+i, x+i);
    }
  }
  printf("Intrin time:    %6.4f      Error: %e\n", tt.toc(), err(sinx_ref, sinx_intrin, N));

  tt.tic();
  for (long rep = 0; rep < 1000; rep++) {
    for (long i = 0; i < N; i+=4) {
      sin4_vector(sinx_vector+i, x+i);
    }
  }
  printf("Vector time:    %6.4f      Error: %e\n", tt.toc(), err(sinx_ref, sinx_vector, N));

  tt.tic();
  for (long rep = 0; rep < 1000; rep++) {
    for (long i = 0; i < N; i+=4) {
      sin4_referenceEC(sinx_refEC+i, x+i);
    }
  }

  tt.tic();
  for (long rep = 0; rep < 1000; rep++) {
    for (long i = 0; i < N; i+=4) {
      sin4_taylorEC(sinx_tayEC+i, x+i);
    }
  }
  printf("Taylor EC time:    %6.4f      Error: %e\n", tt.toc(), err(sinx_refEC, sinx_tayEC, N));

  // for(int i = 0; i < N; i++)
  //   printf("EC Ref %f EC Tay %f x = %f\n", sinx_refEC[i],sinx_tayEC[i],x[0]+i*M_PI/2);


  tt.tic();
  for (long rep = 0; rep < 1000; rep++) {
    for (long i = 0; i < N; i+=4) {
      sin4_intrinEC(sinx_intrinEC+i, x_sh+i);
    }
  }
  printf("Intrin EC time:    %6.4f      Error: %e\n", tt.toc(), err(sinx_refEC, sinx_intrinEC, N));

  // for(int i = 0; i < N; i++)
  //   printf("EC Ref %f EC Intrin %f x = %f Error = %e\n", sinx_refEC[i],sinx_intrinEC[i],x[0]+i*M_PI/2,sinx_intrinEC[i]-sinx_refEC[i]);

  aligned_free(x);
  aligned_free(sinx_ref);
  aligned_free(sinx_taylor);
  aligned_free(sinx_intrin);
  aligned_free(sinx_vector);
  aligned_free(sinx_tayEC);
  aligned_free(sinx_refEC);
  aligned_free(sinx_intrinEC);
}

