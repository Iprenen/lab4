/**
 * SSE matrix multiplication. Bonus assignment.
 *
 *
 * Course: Advanced Computer Architecture, Uppsala University
 * Course Part: Lab assignment 4
 *
 * Author: Andreas Sandberg <andreas.sandberg@it.uu.se>
 *
 * $Id: matmul.c 70 2011-11-22 10:07:10Z ansan501 $
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

#include <smmintrin.h>

#ifndef __SSE4_1__
#error This example requires SSE4.1
#endif

#include "util.h"

/* Size of the matrices to multiply */
#define SIZE 2048

#define SSE_BLOCK_SIZE 4

#ifndef L1_BLOCK_SIZE
#define L1_BLOCK_SIZE 64
#endif

#ifndef L2_BLOCK_SIZE
#define L2_BLOCK_SIZE 1024
#endif

/* A mode that controls how the matrix multiplication is optimized may
 * be specified at compile time. The following modes are defined:
 *
 * MODE_SSE - A simple, non-blocked, implementation of the matrix
 * multiplication.
 *
 * MODE_SSE_BLOCKED - A blocked matrix multiplication with implemented
 * using a 4x4 SSE block.
 *
 * MODE_BLOCKED - Blocked matrix mutliplication using ordinary
 * floating point math.
 */
#define MODE_SSE_BLOCKED 1
#define MODE_SSE 2
#define MODE_BLOCKED 3

#ifndef MODE
#define MODE MODE_SSE_BLOCKED
#endif


#define XMM_ALIGNMENT_BYTES 16

static float mat_a[SIZE][SIZE] __attribute__((aligned (XMM_ALIGNMENT_BYTES)));
static float mat_b[SIZE][SIZE] __attribute__((aligned (XMM_ALIGNMENT_BYTES)));
static float mat_c[SIZE][SIZE] __attribute__((aligned (XMM_ALIGNMENT_BYTES)));
static float mat_ref[SIZE][SIZE] __attribute__((aligned (XMM_ALIGNMENT_BYTES)));
static float mat_Tb[SIZE][SIZE] __attribute__((aligned (XMM_ALIGNMENT_BYTES)));

#if MODE == MODE_SSE_BLOCKED
/**
 * Blocked matrix multiplication, SSE block (4x4 matrix). Implement
 * your solution to the bonus assignment here.
 */


static inline void
matmul_block_sse(int i, int j, int k)
{
        /* BONUS TASK: Implement your SSE 4x4 matrix multiplication
         * block here. */
        /* HINT: You might find at least the following instructions
         * useful:
         *  - _mm_dp_ps
         *  - _MM_TRANSPOSE4_PS
         *
         * HINT: The result of _mm_dp_ps is scalar. The third
         * parameter can be used to restrict to which elements the
         * result is stored, all other elements are set to zero.
         */

          //Init variables needed (load points)
          __m128 a1, a2, a3, a4;
          __m128 b1, b2, b3, b4;
          __m128 c1, c2, c3, c4;

          __m128 mc1, mc2, mc3, mc4; //Temp for matmul part
          __m128 temp; //Used for storage of values, reused

          //Load values from matrix a
          a1 = _mm_load_ps(&mat_a[i][k]);
          a2 = _mm_load_ps(&mat_a[i+1][k]);
          a3 = _mm_load_ps(&mat_a[i+2][k]);
          a4 = _mm_load_ps(&mat_a[i+3][k]);

          //load valus from matrix B and transpose
          b1 = _mm_load_ps(&mat_b[k][j]);
          b2 = _mm_load_ps(&mat_b[k+1][j]);
          b3 = _mm_load_ps(&mat_b[k+2][j]);
          b4 = _mm_load_ps(&mat_b[k+3][j]);
          _MM_TRANSPOSE4_PS(b1,b2,b3,b4);


          //Load in values from previous iteration
          c1 = _mm_load_ps(&mat_c[i][j]);
          c2 = _mm_load_ps(&mat_c[i+1][j]);
          c3 = _mm_load_ps(&mat_c[i+2][j]);
          c4 = _mm_load_ps(&mat_c[i+3][j]);


          //Multiply using dot product instead of _mm_mul_ps
          mc1 = _mm_dp_ps(a1,b1,0xf1);
          mc2 = _mm_dp_ps(a1,b2,0xf2);
          mc3 = _mm_dp_ps(a1,b3,0xf4);
          mc4 = _mm_dp_ps(a1,b4,0xf8);

          //Combine results to temp
          temp =_mm_or_ps(mc1,mc2);
          temp=_mm_or_ps(temp,mc3);
          temp=_mm_or_ps(temp,mc4);
          c1=_mm_add_ps(c1,temp); //Add to c1

          //Repeat...
          mc1 = _mm_dp_ps(a2,b1,0xf1);
          mc2 = _mm_dp_ps(a2,b2,0xf2);
          mc3 = _mm_dp_ps(a2,b3,0xf4);
          mc4 = _mm_dp_ps(a2,b4,0xf8);

          temp =_mm_or_ps(mc1,mc2);
          temp=_mm_or_ps(temp,mc3);
          temp=_mm_or_ps(temp,mc4);
          c2=_mm_add_ps(c2,temp);


          mc1 = _mm_dp_ps(a3,b1,0xf1);
          mc2 = _mm_dp_ps(a3,b2,0xf2);
          mc3 = _mm_dp_ps(a3,b3,0xf4);
          mc4 = _mm_dp_ps(a3,b4,0xf8);

          temp =_mm_or_ps(mc1,mc2);
          temp =_mm_or_ps(temp,mc3);
          temp =_mm_or_ps(temp,mc4);
          c3=_mm_add_ps(c3,temp);

          mc1 = _mm_dp_ps(a4,b1,0xf1);
          mc2 = _mm_dp_ps(a4,b2,0xf2);
          mc3 = _mm_dp_ps(a4,b3,0xf4);
          mc4 = _mm_dp_ps(a4,b4,0xf8);

          temp =_mm_or_ps(mc1,mc2);
          temp =_mm_or_ps(temp,mc3);
          temp =_mm_or_ps(temp,mc4);
          c4 =_mm_add_ps(c4,temp);

          //Store values in their right location
          _mm_store_ps(&mat_c[i][j], c1);
          _mm_store_ps(&mat_c[i+1][j], c2);
          _mm_store_ps(&mat_c[i+2][j], c3);
          _mm_store_ps(&mat_c[i+3][j], c4);


}
#elif MODE == MODE_BLOCKED
/**
 * Blocked matrix multiplication, SSE block (4x4 matrix) implemented
 * using ordinary floating point math.
 */
static inline void
matmul_block_sse(int i, int j, int k)
{
        mat_c[i][j] +=
                mat_a[i][k] * mat_b[k][j]
                + mat_a[i][k + 1] * mat_b[k + 1][j]
                + mat_a[i][k + 2] * mat_b[k + 2][j]
                + mat_a[i][k + 3] * mat_b[k + 3][j];

        mat_c[i][j + 1] +=
                mat_a[i][k] * mat_b[k][j + 1]
                + mat_a[i][k + 1] * mat_b[k + 1][j + 1]
                + mat_a[i][k + 2] * mat_b[k + 2][j + 1]
                + mat_a[i][k + 3] * mat_b[k + 3][j + 1];

        mat_c[i][j + 2] +=
                mat_a[i][k] * mat_b[k][j + 2]
                + mat_a[i][k + 1] * mat_b[k + 1][j + 2]
                + mat_a[i][k + 2] * mat_b[k + 2][j + 2]
                + mat_a[i][k + 3] * mat_b[k + 3][j + 2];

        mat_c[i][j + 3] +=
                mat_a[i][k] * mat_b[k][j + 3]
                + mat_a[i][k + 1] * mat_b[k + 1][j + 3]
                + mat_a[i][k + 2] * mat_b[k + 2][j + 3]
                + mat_a[i][k + 3] * mat_b[k + 3][j + 3];



        mat_c[i + 1][j] +=
                mat_a[i + 1][k] * mat_b[k][j]
                + mat_a[i + 1][k + 1] * mat_b[k + 1][j]
                + mat_a[i + 1][k + 2] * mat_b[k + 2][j]
                + mat_a[i + 1][k + 3] * mat_b[k + 3][j];

        mat_c[i + 1][j + 1] +=
                mat_a[i + 1][k] * mat_b[k][j + 1]
                + mat_a[i + 1][k + 1] * mat_b[k + 1][j + 1]
                + mat_a[i + 1][k + 2] * mat_b[k + 2][j + 1]
                + mat_a[i + 1][k + 3] * mat_b[k + 3][j + 1];

        mat_c[i + 1][j + 2] +=
                mat_a[i + 1][k] * mat_b[k][j + 2]
                + mat_a[i + 1][k + 1] * mat_b[k + 1][j + 2]
                + mat_a[i + 1][k + 2] * mat_b[k + 2][j + 2]
                + mat_a[i + 1][k + 3] * mat_b[k + 3][j + 2];

        mat_c[i + 1][j + 3] +=
                mat_a[i + 1][k] * mat_b[k][j + 3]
                + mat_a[i + 1][k + 1] * mat_b[k + 1][j + 3]
                + mat_a[i + 1][k + 2] * mat_b[k + 2][j + 3]
                + mat_a[i + 1][k + 3] * mat_b[k + 3][j + 3];



        mat_c[i + 2][j] +=
                mat_a[i + 2][k] * mat_b[k][j]
                + mat_a[i + 2][k + 1] * mat_b[k + 1][j]
                + mat_a[i + 2][k + 2] * mat_b[k + 2][j]
                + mat_a[i + 2][k + 3] * mat_b[k + 3][j];

        mat_c[i + 2][j + 1] +=
                mat_a[i + 2][k] * mat_b[k][j + 1]
                + mat_a[i + 2][k + 1] * mat_b[k + 1][j + 1]
                + mat_a[i + 2][k + 2] * mat_b[k + 2][j + 1]
                + mat_a[i + 2][k + 3] * mat_b[k + 3][j + 1];

        mat_c[i + 2][j + 2] +=
                mat_a[i + 2][k] * mat_b[k][j + 2]
                + mat_a[i + 2][k + 1] * mat_b[k + 1][j + 2]
                + mat_a[i + 2][k + 2] * mat_b[k + 2][j + 2]
                + mat_a[i + 2][k + 3] * mat_b[k + 3][j + 2];

        mat_c[i + 2][j + 3] +=
                mat_a[i + 2][k] * mat_b[k][j + 3]
                + mat_a[i + 2][k + 1] * mat_b[k + 1][j + 3]
                + mat_a[i + 2][k + 2] * mat_b[k + 2][j + 3]
                + mat_a[i + 2][k + 3] * mat_b[k + 3][j + 3];



        mat_c[i + 3][j] +=
                mat_a[i + 3][k] * mat_b[k][j]
                + mat_a[i + 3][k + 1] * mat_b[k + 1][j]
                + mat_a[i + 3][k + 2] * mat_b[k + 2][j]
                + mat_a[i + 3][k + 3] * mat_b[k + 3][j];

        mat_c[i + 3][j + 1] +=
                mat_a[i + 3][k] * mat_b[k][j + 1]
                + mat_a[i + 3][k + 1] * mat_b[k + 1][j + 1]
                + mat_a[i + 3][k + 2] * mat_b[k + 2][j + 1]
                + mat_a[i + 3][k + 3] * mat_b[k + 3][j + 1];

        mat_c[i + 3][j + 2] +=
                mat_a[i + 3][k] * mat_b[k][j + 2]
                + mat_a[i + 3][k + 1] * mat_b[k + 1][j + 2]
                + mat_a[i + 3][k + 2] * mat_b[k + 2][j + 2]
                + mat_a[i + 3][k + 3] * mat_b[k + 3][j + 2];

        mat_c[i + 3][j + 3] +=
                mat_a[i + 3][k] * mat_b[k][j + 3]
                + mat_a[i + 3][k + 1] * mat_b[k + 1][j + 3]
                + mat_a[i + 3][k + 2] * mat_b[k + 2][j + 3]
                + mat_a[i + 3][k + 3] * mat_b[k + 3][j + 3];
}

#endif

#if MODE == MODE_SSE_BLOCKED || MODE == MODE_BLOCKED
/**
 * Blocked matrix multiplication, L1 block.
 */
static inline void
matmul_block_l1(int i, int j, int k)
{
        int ii, jj, kk;

        for (ii = i; ii < i + L1_BLOCK_SIZE; ii += SSE_BLOCK_SIZE)
                for (kk = k; kk < k + L1_BLOCK_SIZE; kk += SSE_BLOCK_SIZE)
                        for (jj = j; jj < j + L1_BLOCK_SIZE; jj += SSE_BLOCK_SIZE)
                                matmul_block_sse(ii, jj, kk);
}

/**
 * Blocked matrix multiplication, L2 block.
 */
static inline void
matmul_block_l2(int i, int j, int k)
{
        int ii, jj, kk;

        for (ii = i; ii < i + L2_BLOCK_SIZE; ii += L1_BLOCK_SIZE)
                for (kk = k; kk < k + L2_BLOCK_SIZE; kk += L1_BLOCK_SIZE)
                        for (jj = j; jj < j + L2_BLOCK_SIZE; jj += L1_BLOCK_SIZE)
                                matmul_block_l1(ii, jj, kk);
}

/**
 * Blocked matrix multiplication, entry function for multiplying two
 * matrices.
 */
static void
matmul_sse()
{
        int i, j, k;

        for (i = 0; i < SIZE; i += L2_BLOCK_SIZE)
                for (k = 0; k < SIZE; k += L2_BLOCK_SIZE)
                        for (j = 0; j < SIZE; j += L2_BLOCK_SIZE)
                                matmul_block_l2(i, j, k);
}

#elif MODE == MODE_SSE

/**
 * Matrix multiplication. This is the procedure you should try to
 * optimize.
 */
static void
matmul_sse()
{

    int i, j, k;

  __m128 vecReg;
  __m128 matrixVecReg;
  __m128 temp1, temp2, temp3, temp4;
  __m128 acc;
  __m128 out;

  /* Assume that the data size is an even multiple of the 128 bit
  * SSE vectors (i.e. 4 floats) */
  assert(!(SIZE & 0x3));
  //Initate transposed B
  for (i = 0; i < SIZE; i++) {
    for (j = 0; j < SIZE; j++) {
        mat_Tb[j][i] = mat_b[i][j];
    }
  }
  __m128 empty = _mm_setzero_ps();

  for (i = 0; i < SIZE; i++) { //row output
    for (j = 0; k < SIZE; j+=4) { //row input
          acc = _mm_setzero_ps();
        for (k = 0; k < SIZE; k += 4) { // Column
            vecReg = _mm_load_ps(&mat_a[i][j]); //Load row in A
            matrixVecReg1 = _mm_load_ps(&mat_Tb[k][j]); //Load row in transposed B
            matrixVecReg2 = _mm_load_ps(&mat_Tb[k+1][j]);
            matrixVecReg3 = _mm_load_ps(&mat_Tb[k+2][j]);
            matrixVecReg4 = _mm_load_ps(&mat_Tb[k+3][j]);
            temp1 = _mm_dp_ps(vecReg, matrixVecReg1, 0xf1); //Multiply them
            temp2 = _mm_dp_ps(vecReg, matrixVecReg2, 0xf2); //Multiply them
            temp3 = _mm_dp_ps(vecReg, matrixVecReg3, 0xf4); //Multiply them
            temp4 = _mm_dp_ps(vecReg, matrixVecReg4, 0xf8); //Multiply them
            temp1 = _mm_or_ps(temp1, temp2);
            temp1 = _mm_or_ps(temp1, temp3);
            out = _mm_or_ps(temp1, temp4);
            acc = _mm_add_ps(acc, out); //add result to acc

        }
        _mm_store_ps(mat_c[i][j], acc); //write to c
    }
  }


}

#else

#error Invalid mode

#endif

/**
 * Reference implementation of the matrix multiply algorithm. Used to
 * verify the answer from matmul_opt. Do NOT change this function.
 */
static void
matmul_ref()
{
        int i, j, k;

	for (i = 0; i < SIZE; i++) {
	    for (k = 0; k < SIZE; k++) {
		for (j = 0; j < SIZE; j++) {
                                mat_ref[i][j] += mat_a[i][k] * mat_b[k][j];
                        }
                }
        }
}

/**
 * Function used to verify the result. No need to change this one.
 */
static int
verify_result()
{
        float e_sum;
        int i, j;

        e_sum = 0;
        for (i = 0; i < SIZE; i++) {
                for (j = 0; j < SIZE; j++) {
                        e_sum += mat_c[i][j] < mat_ref[i][j] ?
                                mat_ref[i][j] - mat_c[i][j] :
                                mat_c[i][j] - mat_ref[i][j];
                }
        }

        printf("e_sum: %f\n", e_sum);

        return e_sum < 1E-6;
}

/**
 * Initialize mat_a and mat_b with "random" data. Write to every
 * element in mat_c to make sure that the kernel allocates physical
 * memory to every page in the matrix before we start doing
 * benchmarking.
 */
static void
init_matrices()
{
        int i, j;

        for (i = 0; i < SIZE; i++) {
                for (j = 0; j < SIZE; j++) {
                        mat_a[i][j] = ((i + j) & 0x0F) * 0x1P-4F;
                        mat_b[i][j] = (((i << 1) + (j >> 1)) & 0x0F) * 0x1P-4F;
                }
        }

        memset(mat_c, 0, sizeof(mat_c));
        memset(mat_ref, 0, sizeof(mat_ref));
}

static void
run_multiply()
{
        struct timespec ts_start, ts_stop;
        double runtime_ref, runtime_sse;

        printf("Starting SSE run...\n");
        util_monotonic_time(&ts_start);
        /* mat_c = mat_a * mat_b */
        matmul_sse();
        util_monotonic_time(&ts_stop);
        runtime_sse = util_time_diff(&ts_start, &ts_stop);
        printf("SSE run completed in %.2f s\n",
               runtime_sse);

        printf("Starting reference run...\n");
        util_monotonic_time(&ts_start);
	matmul_ref();
        util_monotonic_time(&ts_stop);
        runtime_ref = util_time_diff(&ts_start, &ts_stop);
        printf("Reference run completed in %.2f s\n",
               runtime_ref);

        printf("Speedup: %.2f\n",
               runtime_ref / runtime_sse);


	if (verify_result())
	    printf("OK\n");
	else
	    printf("MISMATCH\n");
}

int
main(int argc, char *argv[])
{
        /* Initialize the matrices with some "random" data. */
        init_matrices();

        run_multiply();

        return 0;
}


/*
 * Local Variables:
 * mode: c
 * c-basic-offset: 8
 * indent-tabs-mode: nil
 * c-file-style: "linux"
 * compile-command: "make -k"
 * End:
 */
