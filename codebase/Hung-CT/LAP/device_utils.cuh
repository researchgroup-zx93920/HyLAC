#pragma once
#include "cub/cub.cuh"

#define fundef template <typename data> \
__global__ void

__constant__ size_t SIZE;
__constant__ size_t SIZE2;
__constant__ uint NB4;
__constant__ uint NBR;
__constant__ uint DBS;
__constant__ uint L2DBS;

__managed__ __device__ int zeros_size; // The number fo zeros
__managed__ __device__ bool repeat_kernel, goto_5;
__managed__ __device__ long csr2_size, col_id_size, row_id_size;
__managed__ __device__ int nmatch_cur, nmatch_old;

// typedef uint ctype;
// typedef float data;
// typedef double data;

#define BLOCK_DIMX 256