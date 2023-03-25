#pragma once
#include "../include/logger.cuh"
#include "../include/timer.h"

#include "utils.cuh"
#include "device_utils.cuh"

namespace classical
{
  __global__ void S4_init(int *col_visited, int *row_visited)
  {
    size_t tid = threadIdx.x;
    size_t i = tid + (size_t)blockIdx.x * blockDim.x;
    if (i < SIZE)
    {
      col_visited[i] = -1;
      row_visited[i] = -1;
    }
  }
}
__global__ void S4(int *row_cover, int *col_cover, int *col_visited,
                   const size_t *zeros, const size_t *zeros_size_b, const int *col_ass)
{
  __shared__ bool s_found;
  __shared__ bool s_goto_5;
  __shared__ bool s_repeat_kernel;
  volatile int *v_row_cover = row_cover;
  volatile int *v_col_cover = col_cover;
  const size_t i = threadIdx.x;
  const size_t b = blockIdx.x;

  if (i == 0)
  {
    s_repeat_kernel = false;
    s_goto_5 = false;
  }
  do
  {
    __syncthreads();
    if (i == 0)
      s_found = false;
    __syncthreads();
    for (size_t j = threadIdx.x; j < zeros_size_b[b]; j += blockDim.x)
    {
      // each thread picks a zero!
      size_t z = zeros[(size_t)(b << (size_t)L2DBS) + j];
      int l = z % SIZE; // row
      int c = z / SIZE; // column
      int c1 = col_ass[l];

      if (!v_col_cover[c] && !v_row_cover[l])
      {
        s_found = true; // find uncovered zero
        s_repeat_kernel = true;
        col_visited[l] = c; // prime the uncovered zero

        if (c1 >= 0)
        {
          v_row_cover[l] = 1; // cover row
          __threadfence();
          v_col_cover[c1] = 0; // uncover column
        }
        else
        {
          s_goto_5 = true;
        }
      }
    }
    __syncthreads();
  } while (s_found && !s_goto_5);
  if (i == 0 && s_repeat_kernel)
    repeat_kernel = true;
  if (i == 0 && s_goto_5)
    goto_5 = true;
}

__global__ void S5a(int *col_visited, int *row_visited, const int *row_ass, const int *col_ass)
{
  size_t i = (size_t)blockDim.x * blockIdx.x + (size_t)threadIdx.x;
  if (i < SIZE)
  {
    int r_Z0, c_Z0;

    c_Z0 = col_visited[i];
    if (c_Z0 >= 0 && col_ass[i] < 0) // if primed and not covered
    {
      row_visited[c_Z0] = i; // mark the column as visited

      while ((r_Z0 = row_ass[c_Z0]) >= 0)
      {
        c_Z0 = col_visited[r_Z0];
        row_visited[c_Z0] = r_Z0;
      }
    }
  }
}

__global__ void S5b(int *row_visited, int *row_ass, int *col_ass)
{
  size_t j = (size_t)blockDim.x * blockIdx.x + (size_t)threadIdx.x;
  if (j < SIZE)
  {
    int r_Z0, c_Z0, c_Z2;
    r_Z0 = row_visited[j];
    if (r_Z0 >= 0 && row_ass[j] < 0)
    {

      c_Z2 = col_ass[r_Z0];

      col_ass[r_Z0] = j;
      row_ass[j] = r_Z0;

      while (c_Z2 >= 0)
      {
        r_Z0 = row_visited[c_Z2]; // row of Z2
        c_Z0 = c_Z2;              // col of Z2
        c_Z2 = col_ass[r_Z0];     // col of Z4

        // star Z2
        col_ass[r_Z0] = c_Z0;
        row_ass[c_Z0] = r_Z0;
      }
    }
  }
}

template <typename data = int, uint blockSize = BLOCK_DIMX>
__global__ void min_reduce_kernel1(volatile data *g_idata, volatile data *g_odata,
                                   const int *row_cover, const int *col_cover)
{
  __shared__ data sdata[blockSize];
  const uint tid = threadIdx.x;
  size_t i = (size_t)blockIdx.x * ((size_t)blockSize * 2) + (size_t)tid;
  size_t gridSize = (size_t)blockSize * 2 * (size_t)gridDim.x;
  sdata[tid] = MAX_DATA;
  while (i < SIZE2)
  {
    size_t i1 = i;
    size_t i2 = i + blockSize;
    size_t l1 = i1 % SIZE; // local index within the row
    size_t c1 = i1 / SIZE; // Row number
    data g1 = MAX_DATA, g2 = MAX_DATA;
    if (row_cover[l1] == 1 || col_cover[c1] == 1)
      g1 = MAX_DATA;
    else
      g1 = g_idata[i1];
    if (i2 < SIZE2)
    {
      size_t l2 = i2 % SIZE;
      size_t c2 = i2 / SIZE;
      if (row_cover[l2] == 1 || col_cover[c2] == 1)
        g2 = MAX_DATA;
      else
        g2 = g_idata[i2];
    }
    sdata[tid] = min(sdata[tid], min(g1, g2));
    i += gridSize;
  }
  __syncthreads();
  typedef cub::BlockReduce<data, blockSize> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  data val = sdata[tid];
  data minimum = BlockReduce(temp_storage).Reduce(val, cub::Min());
  if (tid == 0)
    g_odata[blockIdx.x] = minimum;
}

fundef S6_DualUpdate(const int *cover_row, const int *cover_column, const data *min_mat,
                     double *min_in_rows, double *min_in_cols)
{
  const size_t i = (size_t)blockDim.x * blockIdx.x + (size_t)threadIdx.x;
  if (i < SIZE)
  {
    if (cover_column[i] == 0)
      min_in_rows[i] += ((double)1.0 * min_mat[0]) / 2;
    else
      min_in_rows[i] -= ((double)1.0 * min_mat[0]) / 2;
    if (cover_row[i] == 0)
      min_in_cols[i] += ((double)1.0 * min_mat[0]) / 2;
    else
      min_in_cols[i] -= ((double)1.0 * min_mat[0]) / 2;
  }
}

fundef S6_update(data *slack, const int *row_cover, const int *col_cover,
                 const data *min_mat, size_t *zeros, size_t *zeros_size_b)
{
  const size_t i = (size_t)blockDim.x * blockIdx.x + (size_t)threadIdx.x;
  if (i < SIZE2)
  {
    const size_t l = i % SIZE;
    const size_t c = i / SIZE;
    data reg = slack[i];
    switch (row_cover[l] + col_cover[c])
    {
    case 2:
      reg += min_mat[0];
      slack[i] = reg;
      break;
    case 0:
      reg -= min_mat[0];
      slack[i] = reg;
      break;
    default:
      break;
    }

    // compress matrix
    if (near_zero(reg))
    {
      size_t b = i >> L2DBS;
      size_t i0 = i & ~((size_t)DBS - 1); // == b << log2_data_block_size
      size_t j = (size_t)atomicAdd((uint64 *)zeros_size_b + b, 1ULL);
      zeros[i0 + j] = i;
    }
  }
}
