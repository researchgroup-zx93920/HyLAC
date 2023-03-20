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

template <typename data>
__global__ void get_obj(const int *row_ass, const data *cost, data *obj)
{
  __shared__ data b_obj;
  size_t tid = threadIdx.x;
  if (tid == 0)
    b_obj = 0;
  __syncthreads();
  size_t i = tid + (size_t)blockIdx.x * blockDim.x;
  if (i < SIZE)
  {
    atomicAdd(&b_obj, cost[i * SIZE + row_ass[i]]);
  }
  __syncthreads();
  if (tid == 0 && i < SIZE)
    atomicAdd(obj, b_obj);
}