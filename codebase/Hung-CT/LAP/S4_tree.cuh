#pragma once
#include "../include/logger.cuh"
#include "../include/timer.h"

#include "utils.cuh"
#include "device_utils.cuh"
#include "structures.h"

#include <type_traits> //for comparing typenames

uint g_pre_DA_counts, g_post_DA_counts, g_DA_counts;
namespace tree
{
  // Kernel for initializing the row or column vertices, later used for recursive frontier update (in Step 3).
  template <typename data = uint>
  __global__ void Initialization(int *d_row_assignments,
                                 int *row_cover, int *col_cover,
                                 VertexData<data> row_data, VertexData<data> col_data)
  {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < SIZE)
    {
      int assignment = d_row_assignments[id];
      row_data.is_visited[id] = (assignment == -1) ? ACTIVE : DORMANT;
      row_cover[id] = (assignment == -1) ? 0 : 1;

      // Initializing rest of the memory
      col_cover[id] = 0;

      col_data.slack[id] = INFINITY;

      // row_data.is_visited[id] = DORMANT;
      col_data.is_visited[id] = DORMANT;

      row_data.parents[id] = -1;
      col_data.parents[id] = -1;

      row_data.children[id] = -1;
      col_data.children[id] = -1;
    }
  }

  __global__ void S4_init(int *vertices_csr1)
  {
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < SIZE)
      vertices_csr1[id] = id;
  }
}

__global__ void vertexPredicateConstructionCSR(Predicates vertex_predicates, int *vertices_csr1, int *visited)
{
  size_t id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < SIZE)
  {
    int vertexid = vertices_csr1[id];
    int visit = (vertexid != -1) ? visited[vertexid] : DORMANT;
    bool predicate = (visit == ACTIVE); // If vertex is not visited then it is added to frontier queue.
    long addr = predicate ? 1 : 0;

    vertex_predicates.predicates[id] = predicate;
    vertex_predicates.addresses[id] = addr;
  }
}

__global__ void vertexScatterCSR(int *d_vertex_ids_csr, int *d_vertex_ids,
                                 int *d_visited, const Predicates d_vertex_predicates)
{
  size_t id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < SIZE)
  {
    int vertexid = d_vertex_ids[id];
    bool predicate = d_vertex_predicates.predicates[id];
    // long compid = -1; // compaction id.
    if (predicate)
    {
      long compid = d_vertex_predicates.addresses[id]; // compaction id.
      d_vertex_ids_csr[compid] = vertexid;
      d_visited[id] = VISITED;
    }
  }
}

template <typename data = uint>
__device__ void __traverse(data *d_costs, double *row_duals, double *col_duals,
                           int *row_ass, int *col_ass, int *row_cover, int *col_cover,
                           int *d_row_parents, int *d_col_parents, int *d_row_visited,
                           int *d_col_visited, data *d_slacks,
                           int *d_start_ptr, int *d_end_ptr, const size_t colid)
{
  int *ptr1 = d_start_ptr;
  // bool L_goto_5 = false;
  while (ptr1 != d_end_ptr)
  {
    int rowid = *ptr1;
    data slack;
    if (std::is_same_v<data, uint> || std::is_same_v<data, int>)
      slack = (data)(d_costs[rowid * SIZE + colid] - (int)(row_duals[rowid] + col_duals[colid]));
    else
      slack = d_costs[rowid * SIZE + colid] - (data)(row_duals[rowid] + col_duals[colid]);
    int nxt_rowid = col_ass[colid];
    if (rowid != nxt_rowid && col_cover[colid] == 0)
    {
      if (slack < d_slacks[colid])
      {

        d_slacks[colid] = slack;
        d_col_parents[colid] = rowid;
      }
      if (near_zero(d_slacks[colid]))
      {

        if (nxt_rowid != -1)
        {
          d_row_parents[nxt_rowid] = colid; // update parent info

          row_cover[nxt_rowid] = 0;
          col_cover[colid] = 1;

          if (d_row_visited[nxt_rowid] != VISITED)
            d_row_visited[nxt_rowid] = ACTIVE;
        }

        else
        {
          d_col_visited[colid] = REVERSE;
          goto_5 = true;
          // goto_5 = true;
        }
      }
    }
    d_row_visited[rowid] = VISITED;
    ptr1++;
  }
}

template <typename data = uint>
__global__ void coverAndExpand(
    int *vertices_csr2, const size_t csr2_size,
    data *d_costs, double *row_duals, double *col_duals,
    int *row_ass, int *col_ass, int *row_cover, int *col_cover,
    VertexData<data> row_data, VertexData<data> col_data)
{
  size_t id = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t in_size = csr2_size;
  // Load values into local memory
  int *st_ptr = vertices_csr2;
  int *end_ptr = vertices_csr2 + in_size;

  // __shared__ S_goto_5;
  // bool L_goto_5 = false;
  // printf("start: %p, End: %p\n", st_ptr, end_ptr);
  if (id < SIZE)
  {
    __traverse(d_costs, row_duals, col_duals,
               row_ass, col_ass, row_cover, col_cover,
               row_data.parents, col_data.parents, row_data.is_visited, col_data.is_visited,
               col_data.slack, st_ptr, end_ptr, id);
  }
  // __syncthreads();
  // typedef cub::BlockReduce<bool, BLOCK_DIMX> BlockReduce;
  // __shared__ typename BlockReduce::TempStorage temp_storage;
  // __syncthreads();
  // L_goto_5 = BlockReduce(temp_storage).Sum(L_goto_5);
  // __syncthreads();
  // if (threadIdx.x == 0)
  // {
  //   atomicOr((int *)&goto_5, (int)L_goto_5);
  // }
}