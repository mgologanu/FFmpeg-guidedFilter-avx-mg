#ifndef I_H

#define I_H

#define V 8

 // log2(V)
#define POW_V  3

//number of registers to use in a loop
#define NR 16   

// log2(V*NR)
#define POW_T  7  


/*
  block size for transpose.

  If value is changed,
  correct also POW_BL and BL_V below!!
*/

#define BL 64

// log2(BL)
#define POW_BL 6

//BL divided by V
#define BL_V 8

#define BL_V_HALF 4

typedef __m256 float_packed;

#define MUL(a,b)     _mm256_mul_ps(a,b)
#define DIV(a,b)     _mm256_div_ps(a,b)
#define ADD(a,b)     _mm256_add_ps(a,b)
#define SUB(a,b)     _mm256_sub_ps(a,b)
#define LOAD(a)      _mm256_load_ps(&a)
#define STORE(a,b)   _mm256_store_ps(&a,b)
#define BROADCAST(a) _mm256_broadcast_ss(&a)



extern const float zero;



void boxfilter1D(const float *x_in, float *x_out, size_t r, size_t n, size_t m, size_t ld);

void boxfilter1D_norm(const float *x_in, float *x_out, size_t r, size_t n, size_t m, size_t ld, const float * a_norm, const float * b_norm);


void transpose_8x8_2(float * a, float * b, size_t n, size_t m);
  
void transpose_8x8(float * a, float * b, size_t n, size_t m);

void transpose(float * in, float * out, size_t ld_n, size_t ld_m);


//int boxfilter(float *x_in, size_t r, size_t n, size_t m, size_t ld_n, size_t ld_m, const float *ai, const float *bi, float * work);


void matmul(const float *x1, const float *x2, float *y, size_t ld_n, size_t m);


void diffmatmul(float *x1, const float *x2, const float * x3, size_t ld_n, size_t m);

void addmatmul(float *x1, const float *x2, const float * x3, size_t ld_n, size_t m);

void matdivconst(float *x1, const float *x2, size_t ld_n, size_t m, float e);


#endif
