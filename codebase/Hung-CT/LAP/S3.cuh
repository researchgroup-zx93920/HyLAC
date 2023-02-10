#pragma once
#include "../include/logger.cuh"
#include "../include/timer.h"

#include "utils.cuh"
#include "device_utils.cuh"

__global__ void step3(const int *row_ass, int *col_cover)
{
  size_t tid = threadIdx.x;
  size_t i = tid + (size_t)blockIdx.x * blockDim.x;
  __shared__ int matches;
  if (tid == 0)
    matches = 0;
  __syncthreads();
  if (i < SIZE)
  {
    if (row_ass[i] >= 0)
    {
      col_cover[i] = 1;
      atomicAdd((int *)&matches, 1);
    }
  }
  __syncthreads();
  if (tid == 0)
    atomicAdd((int *)&nmatch_cur, matches);
}