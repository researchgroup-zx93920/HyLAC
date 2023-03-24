#pragma once
#include "../include/logger.cuh"
#include "../include/timer.h"

#include "utils.cuh"
#include "device_utils.cuh"

fundef row_reduce(double *row_min, data *slack)
{
  const size_t tid = threadIdx.x;
  const size_t rowID = (size_t)blockIdx.x * SIZE;
  data thread_min = (data)MAX_DATA;
  for (size_t i = tid; i < SIZE; i += blockDim.x)
  {
    thread_min = min(thread_min, slack[i + rowID]);
  }
  __syncthreads();
  typedef cub::BlockReduce<data, BLOCK_DIMX> BR;
  __shared__ typename BR::TempStorage temp_storage;
  thread_min = BR(temp_storage).Reduce(thread_min, cub::Min());
  if (threadIdx.x == 0)
    row_min[blockIdx.x] = (double)thread_min;
  __syncthreads();

  for (size_t i = tid; i < SIZE; i += blockDim.x)
  {
    slack[i + rowID] = slack[i + rowID] - (data)row_min[blockIdx.x];
  }
}

fundef col_min(const data *slack, double *col_min)
{
  size_t tid = (size_t)threadIdx.x;
  const size_t colID = blockIdx.x;
  data thread_min = (data)MAX_DATA;
  for (size_t i = tid; i < SIZE; i += blockDim.x)
  {
    thread_min = min(thread_min, slack[i * SIZE + colID]);
  }
  __syncthreads();
  typedef cub::BlockReduce<data, BLOCK_DIMX> BR;
  __shared__ typename BR::TempStorage temp_storage;
  thread_min = BR(temp_storage).Reduce(thread_min, cub::Min());

  if (threadIdx.x == 0)
    col_min[blockIdx.x] = (double)thread_min;
}

fundef col_sub(data *slack, double *col_min)
{
  size_t tid = threadIdx.x;
  const size_t rowID = (size_t)blockIdx.x * SIZE;
  for (size_t i = tid; i < SIZE; i += blockDim.x)
  {
    slack[i + rowID] = slack[i + rowID] - (data)col_min[i];
  }
}