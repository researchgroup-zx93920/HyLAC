#pragma once
#include "../include/logger.cuh"
#include "../include/timer.h"

#include "utils.cuh"
#include "device_utils.cuh"
#include "structures.h"

// Device function for traversing an alternating path from unassigned row to unassigned column.
__device__ void __reverse_traversal(int *d_row_visited, int *d_row_children, int *d_col_children, int *d_row_parents, int *d_col_parents, int init_colid)
{
  int cur_colid = init_colid;
  int cur_rowid = -1;

  while (cur_colid != -1)
  {
    d_col_children[cur_colid] = cur_rowid;

    cur_rowid = d_col_parents[cur_colid];

    d_row_children[cur_rowid] = cur_colid;
    cur_colid = d_row_parents[cur_rowid];
  }
  d_row_visited[cur_rowid] = AUGMENT;
}

__global__ void augmentPredicateConstruction(Predicates d_predicates, int *d_visited)
{
  size_t id = blockIdx.x * blockDim.x + threadIdx.x;
  // Copy the matrix into shared memory.
  int visited = (id < SIZE) ? d_visited[id] : DORMANT;
  bool predicate = (visited == REVERSE || visited == AUGMENT);
  long addr = predicate ? 1 : 0;
  if (id < SIZE)
  {
    d_predicates.predicates[id] = predicate;
    d_predicates.addresses[id] = addr;
  }
}

__global__ void augmentScatter(int *vertex_ids, Predicates predicates)
{
  size_t id = blockIdx.x * blockDim.x + threadIdx.x;

  bool predicate = (id < SIZE) ? predicates.predicates[id] : false;
  long compid = (predicate) ? predicates.addresses[id] : -1; // compaction id.

  if (id < SIZE)
  {
    if (predicate)
      vertex_ids[compid] = id;
  }
}

template <typename data = uint>
__global__ void reverseTraversal(int *col_vertices, VertexData<data> row_data, VertexData<data> col_data, size_t size)
{
  size_t id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < size)
  {
    int colid = col_vertices[id];
    __reverse_traversal(row_data.is_visited, row_data.children, col_data.children, row_data.parents, col_data.parents, colid);
  }
}

// Device function for augmenting the alternating path from unassigned column to unassigned row.
__device__ void __augment(int *d_row_assignments, int *d_col_assignments, int *d_row_children, int *d_col_children, int init_rowid)
{
  int cur_colid = -1;
  int cur_rowid = init_rowid;

  while (cur_rowid != -1)
  {
    cur_colid = d_row_children[cur_rowid];

    d_row_assignments[cur_rowid] = cur_colid;
    d_col_assignments[cur_colid] = cur_rowid;

    cur_rowid = d_col_children[cur_colid];
  }
}

template <typename data = uint>
__global__ void augment(int *row_ass, int *col_ass, int *row_vertices, VertexData<data> row_data, VertexData<data> col_data, size_t size)
{
  size_t id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < size)
  {
    int rowid = row_vertices[id];
    __augment(row_ass, col_ass, row_data.children, col_data.children, rowid);
  }
}