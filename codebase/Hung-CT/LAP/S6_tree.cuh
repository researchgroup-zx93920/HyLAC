#pragma once
#include "../include/logger.cuh"
#include "../include/timer.h"

#include "utils.cuh"
#include "device_utils.cuh"
#include "structures.h"

template <typename data = uint>
__global__ void transfer_duals(double *row_duals, double *col_duals, double *row_duals_tree, double *col_duals_tree)
{
  size_t id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < SIZE)
  {
    row_duals_tree[id] = (double)row_duals[id];
    col_duals_tree[id] = (double)col_duals[id];
  }
}

namespace tree
{
  template <typename data = uint>
  __global__ void dualUpdate(double min_val, double *row_duals, double *col_duals, data *col_slacks,
                             int *row_covers, int *col_covers, int *col_parents, int *row_visited)
  {
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < SIZE)
    {
      int row_cover = row_covers[id];
      int col_cover = col_covers[id];

      if (row_cover == 0) // Row is labeled
        row_duals[id] += min_val;
      else
        row_duals[id] -= min_val;

      if (col_cover == 1)
        col_duals[id] -= min_val;
      else
      {
        col_duals[id] += min_val;
        col_slacks[id] -= (data)(2 * min_val);
        if (col_slacks[id] > -eps && col_slacks[id] < eps)
        {
          int par_rowid = col_parents[id];
          row_visited[par_rowid] = ACTIVE;
        }
      }
    }
  }
}