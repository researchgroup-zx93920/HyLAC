#pragma once
#include "../include/logger.cuh"
#include "../include/timer.h"

#include "utils.cuh"
#include "device_utils.cuh"
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

template <typename data>
__device__ bool near_zero(data val)
{
  return ((val < epsilon) && (val > -epsilon));
}

__global__ void init(int *row_ass, int *col_ass, int *row_cover, int *col_cover)
{
  size_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < SIZE)
  {
    col_ass[i] = -1;
    row_ass[i] = -1;
    row_cover[i] = 0;
    col_cover[i] = 0;
  }
}

fundef compress_matrix(size_t *zeros, size_t *zeros_size_b, data *slack)
{
  size_t i = (size_t)blockDim.x * (size_t)blockIdx.x + (size_t)threadIdx.x;
  if (i < SIZE2)
  {
    if (near_zero(slack[i]))
    {

      size_t b = i >> L2DBS;
      size_t i0 = i & ~((size_t)DBS - 1);

      size_t j = (size_t)atomicAdd((uint64 *)&zeros_size_b[b], 1ULL);
      zeros[i0 + j] = i; // saves index of zeros in slack matrix per block
    }
  }
}

// Initial zero cover
__global__ void step2(const size_t *zeros, const size_t *zeros_size_b,
                      int *row_cover, int *col_cover, int *row_ass, int *col_ass)
{
  uint i = threadIdx.x;
  uint b = blockIdx.x;
  __shared__ bool repeat, s_repeat_kernel;
  if (i == 0)
    s_repeat_kernel = false;
  do
  {
    __syncthreads();
    if (i == 0)
      repeat = false;
    __syncthreads();
    for (uint j = i; j < zeros_size_b[b]; j += blockDim.x)
    {
      size_t z = zeros[(b << L2DBS) + j];
      uint l = z % SIZE;
      uint c = z / SIZE;
      if (row_cover[l] == 0 && col_cover[c] == 0)
      {
        if (!atomicExch((int *)&(row_cover[l]), 1))
        {
          // only one thread gets the line
          if (!atomicExch((int *)&(col_cover[c]), 1))
          {
            // only one thread gets the column
            row_ass[c] = l;
            col_ass[l] = c;
          }
          else
          {
            row_cover[l] = 0;
            repeat = true;
            s_repeat_kernel = true;
          }
        }
      }
    }
    __syncthreads();
  } while (repeat);
  if (s_repeat_kernel)
    repeat_kernel = true;
}
