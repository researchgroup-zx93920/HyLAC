#pragma once
#include "../include/utils.cuh"
#include "device_utils.cuh"
#include "cub/cub.cuh"

#define fundef template <typename data = int> \
__global__ void

__constant__ uint SIZE;
__constant__ uint NB4;
__constant__ uint NBR;

__constant__ uint nrows;
__constant__ uint ncols;
__constant__ uint n_rows_per_block;
__constant__ uint n_cols_per_block;
__constant__ uint log2_n, log2_data_block_size, data_block_size;
__constant__ uint n_blocks_step_4;

const int max_threads_per_block = 1024;
const int columns_per_block_step_4 = 512;
const int n_threads_reduction = 256;
// const int n_blocks_step_4 = max(n / columns_per_block_step_4, 1);				 // Number of blocks in step 4 and 2

fundef init(GLOBAL_HANDLE<data> gh)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  // initializations
  // for step 2
  if (i < nrows)
  {
    gh.cover_row[i] = 0;
    gh.column_of_star_at_row[i] = -1;
  }
  if (i < ncols)
  {
    gh.cover_column[i] = 0;
    gh.row_of_star_at_column[i] = -1;
  }
}

template <typename data = int>
__device__ void min_in_rows_warp_reduce(volatile data *sdata, int tid)
{
  if (n_threads_reduction >= 64 && n_rows_per_block < 64 && tid + 32 < n_threads_reduction)
    sdata[tid] = min(sdata[tid], sdata[tid + 32]);
  if (n_threads_reduction >= 32 && n_rows_per_block < 32 && tid + 16 < n_threads_reduction)
    sdata[tid] = min(sdata[tid], sdata[tid + 16]);
  if (n_threads_reduction >= 16 && n_rows_per_block < 16 && tid + 8 < n_threads_reduction)
    sdata[tid] = min(sdata[tid], sdata[tid + 8]);
  if (n_threads_reduction >= 8 && n_rows_per_block < 8 && tid + 4 < n_threads_reduction)
    sdata[tid] = min(sdata[tid], sdata[tid + 4]);
  if (n_threads_reduction >= 4 && n_rows_per_block < 4 && tid + 2 < n_threads_reduction)
    sdata[tid] = min(sdata[tid], sdata[tid + 2]);
  if (n_threads_reduction >= 2 && n_rows_per_block < 2 && tid + 1 < n_threads_reduction)
    sdata[tid] = min(sdata[tid], sdata[tid + 1]);
}

fundef calc_min_in_rows(GLOBAL_HANDLE<data> gh)
{
  __shared__ data sdata[n_threads_reduction];

  const unsigned int gridSize = n_threads_reduction * NBR;
  uint tid = threadIdx.x;
  uint bid = blockIdx.x;
  // One gets the line and column from the blockID and threadID.
  unsigned int l = bid * n_rows_per_block + tid % n_rows_per_block;
  unsigned int c = tid / n_rows_per_block;
  unsigned int i = c * nrows + l;

  data thread_min = (data)MAX_DATA;
  while (i < (size_t)SIZE * SIZE)
  {
    thread_min = min(thread_min, gh.slack[i]);
    i += gridSize; // go to the next piece of the matrix...
                   // gridSize = 2^k * n, so that each thread always processes the same line or column
  }
  sdata[tid] = thread_min;
  __syncthreads();
  if (n_threads_reduction >= 1024 && n_rows_per_block < 1024)
  {
    if (tid < 512)
    {
      sdata[tid] = min(sdata[tid], sdata[tid + 512]);
    }
    __syncthreads();
  }
  if (n_threads_reduction >= 512 && n_rows_per_block < 512)
  {
    if (tid < 256)
    {
      sdata[tid] = min(sdata[tid], sdata[tid + 256]);
    }
    __syncthreads();
  }
  if (n_threads_reduction >= 256 && n_rows_per_block < 256)
  {
    if (tid < 128)
    {
      sdata[tid] = min(sdata[tid], sdata[tid + 128]);
    }
    __syncthreads();
  }
  if (n_threads_reduction >= 128 && n_rows_per_block < 128)
  {
    if (tid < 64)
    {
      sdata[tid] = min(sdata[tid], sdata[tid + 64]);
    }
    __syncthreads();
  }
  if (tid < 32)
    min_in_rows_warp_reduce(sdata, tid);

  if (tid < n_rows_per_block)
    gh.min_in_rows[bid * n_rows_per_block + tid] = sdata[tid];
}

fundef step_1_row_sub(GLOBAL_HANDLE<data> gh)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int l = i & gh.row_mask;
  gh.slack[i] = gh.slack[i] - gh.min_in_rows[l]; // subtract the minimum in row from that row
}

template <typename data = int>
__device__ void min_in_cols_warp_reduce(volatile data *sdata, int tid)
{
  if (n_threads_reduction >= 64 && n_cols_per_block < 64 && tid + 32 < n_threads_reduction)
    sdata[tid] = min(sdata[tid], sdata[tid + 32]);
  if (n_threads_reduction >= 32 && n_cols_per_block < 32 && tid + 16 < n_threads_reduction)
    sdata[tid] = min(sdata[tid], sdata[tid + 16]);
  if (n_threads_reduction >= 16 && n_cols_per_block < 16 && tid + 8 < n_threads_reduction)
    sdata[tid] = min(sdata[tid], sdata[tid + 8]);
  if (n_threads_reduction >= 8 && n_cols_per_block < 8 && tid + 4 < n_threads_reduction)
    sdata[tid] = min(sdata[tid], sdata[tid + 4]);
  if (n_threads_reduction >= 4 && n_cols_per_block < 4 && tid + 2 < n_threads_reduction)
    sdata[tid] = min(sdata[tid], sdata[tid + 2]);
  if (n_threads_reduction >= 2 && n_cols_per_block < 2 && tid + 1 < n_threads_reduction)
    sdata[tid] = min(sdata[tid], sdata[tid + 1]);
}

fundef calc_min_in_cols(GLOBAL_HANDLE<data> gh)
{
  __shared__ data sdata[n_threads_reduction];
  unsigned int tid = threadIdx.x;
  unsigned int bid = blockIdx.x;
  // One gets the line and column from the blockID and threadID.
  unsigned int c = bid * n_cols_per_block + tid % n_cols_per_block;
  unsigned int l = tid / n_cols_per_block;
  const unsigned int gridSize = n_threads_reduction * NBR;
  data thread_min = MAX_DATA;
  while (l < SIZE)
  {
    unsigned int i = c * nrows + l;
    thread_min = min(thread_min, gh.slack[i]);
    l += gridSize / SIZE; // go to the next piece of the matrix...
                          // gridSize = 2^k * n, so that each thread always processes the same line or column
  }
  sdata[tid] = thread_min;

  __syncthreads();
  if (n_threads_reduction >= 1024 && n_cols_per_block < 1024)
  {
    if (tid < 512)
    {
      sdata[tid] = min(sdata[tid], sdata[tid + 512]);
    }
    __syncthreads();
  }
  if (n_threads_reduction >= 512 && n_cols_per_block < 512)
  {
    if (tid < 256)
    {
      sdata[tid] = min(sdata[tid], sdata[tid + 256]);
    }
    __syncthreads();
  }
  if (n_threads_reduction >= 256 && n_cols_per_block < 256)
  {
    if (tid < 128)
    {
      sdata[tid] = min(sdata[tid], sdata[tid + 128]);
    }
    __syncthreads();
  }
  if (n_threads_reduction >= 128 && n_cols_per_block < 128)
  {
    if (tid < 64)
    {
      sdata[tid] = min(sdata[tid], sdata[tid + 64]);
    }
    __syncthreads();
  }
  if (tid < 32)
    min_in_cols_warp_reduce(sdata, tid);
  if (tid < n_cols_per_block)
    gh.min_in_cols[bid * n_cols_per_block + tid] = sdata[tid];
}

fundef step_1_col_sub(GLOBAL_HANDLE<data> gh)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int c = i >> log2_n;

  gh.slack[i] = gh.slack[i] - gh.min_in_cols[c]; // subtract the minimum in row from that row

  if (i == 0)
    zeros_size = 0;
  if (i < n_blocks_step_4)
    gh.zeros_size_b[i] = 0;
}

template <typename data = int>
__device__ bool near_zero(data val)
{
  return ((val < epsilon) && (val > -epsilon));
}

fundef compress_matrix(GLOBAL_HANDLE<data> gh)
{
  uint i = blockDim.x * blockIdx.x + threadIdx.x;
  if (near_zero(gh.slack[i]))
  {
    // atomicAdd(&zeros_size, 1);
    int b = i >> log2_data_block_size;
    int i0 = i & ~(data_block_size - 1); // == b << log2_data_block_size
    int j = atomicAdd(&gh.zeros_size_b[b], 1);
    gh.zeros[i0 + j] = i; // saves index of zeros in slack matrix per block
  }
}

fundef add_reduction(GLOBAL_HANDLE<data> gh)
{
  __shared__ int sdata[1024]; // hard coded need to change!
  const int i = threadIdx.x;
  sdata[i] = gh.zeros_size_b[i];
  __syncthreads();
  for (int j = blockDim.x >> 1; j > 0; j >>= 1)
  {
    if (i + j < blockDim.x)
      sdata[i] += sdata[i + j];
    __syncthreads();
  }
  if (i == 0)
  {
    zeros_size = sdata[0];
  }
}

fundef step_2(GLOBAL_HANDLE<data> gh)
{
  int i = threadIdx.x;
  int b = blockIdx.x;
  __shared__ bool repeat;
  __shared__ bool s_repeat_kernel;
  if (i == 0)
    s_repeat_kernel = false;

  do
  {
    __syncthreads();
    if (i == 0)
      repeat = false;
    __syncthreads();
    for (int j = i; j < gh.zeros_size_b[b]; j += blockDim.x)
    {
      int z = gh.zeros[(b << log2_data_block_size) + j];
      int l = z & gh.row_mask;
      int c = z >> log2_n;

      if (gh.cover_row[l] == 0 && gh.cover_column[c] == 0)
      {
        if (!atomicExch((int *)&(gh.cover_row[l]), 1))
        {
          // only one thread gets the line
          if (!atomicExch((int *)&(gh.cover_column[c]), 1))
          {
            // only one thread gets the column
            gh.row_of_star_at_column[c] = l;
            gh.column_of_star_at_row[l] = c;
          }
          else
          {
            gh.cover_row[l] = 0;
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

fundef step_3_init(GLOBAL_HANDLE<data> gh)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  gh.cover_row[i] = 0;
  gh.cover_column[i] = 0;
  if (i == 0)
    n_matches = 0;
}

fundef step_3(GLOBAL_HANDLE<data> gh)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  __shared__ int matches;
  if (threadIdx.x == 0)
    matches = 0;
  __syncthreads();
  if (gh.row_of_star_at_column[i] >= 0)
  {
    gh.cover_column[i] = 1;
    atomicAdd((int *)&matches, 1);
  }
  __syncthreads();
  if (threadIdx.x == 0)
    atomicAdd((int *)&n_matches, matches);
}

// STEP 4
// Find a noncovered zero and prime it. If there is no starred
// zero in the row containing this primed zero, go to Step 5.
// Otherwise, cover this row and uncover the column containing
// the starred zero. Continue in this manner until there are no
// uncovered zeros left. Save the smallest uncovered value and
// Go to Step 6.

fundef step_4_init(GLOBAL_HANDLE<data> gh)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  gh.column_of_prime_at_row[i] = -1;
  gh.row_of_green_at_column[i] = -1;
}

fundef step_4(GLOBAL_HANDLE<data> gh)
{
  __shared__ bool s_found;
  __shared__ bool s_goto_5;
  __shared__ bool s_repeat_kernel;
  volatile int *v_cover_row = gh.cover_row;
  volatile int *v_cover_column = gh.cover_column;

  const int i = threadIdx.x;
  const int b = blockIdx.x;
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
    for (int j = threadIdx.x; j < gh.zeros_size_b[b]; j += blockDim.x)
    {
      int z = gh.zeros[(b << log2_data_block_size) + j];
      int l = z & gh.row_mask; // row
      int c = z >> log2_n;     // column
      int c1 = gh.column_of_star_at_row[l];

      // for (int n = 0; n < 10; n++)	??
      // {

      if (!v_cover_column[c] && !v_cover_row[l])
      {
        s_found = true; // find uncovered zero
        s_repeat_kernel = true;
        gh.column_of_prime_at_row[l] = c; // prime the uncovered zero

        if (c1 >= 0)
        {
          v_cover_row[l] = 1; // cover row
          __threadfence();
          v_cover_column[c1] = 0; // uncover column
        }
        else
        {
          s_goto_5 = true;
        }
      }
      // } for(int n
    } // for(int j
    __syncthreads();
  } while (s_found && !s_goto_5);
  if (i == 0 && s_repeat_kernel)
    repeat_kernel = true;
  if (i == 0 && s_goto_5) // if any blocks needs to go to step 5, algorithm needs to go to step 5
    goto_5 = true;
}

template <typename data = int, uint blockSize = n_threads_reduction>
__global__ void min_reduce_kernel1(volatile data *g_idata, volatile data *g_odata,
                                   const uint n, GLOBAL_HANDLE<data> gh)
{
  __shared__ data sdata[blockSize];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * (blockSize * 2) + tid;
  unsigned int gridSize = blockSize * 2 * gridDim.x;
  sdata[tid] = MAX_DATA;
  while (i < n)
  {
    int i1 = i;
    int i2 = i + blockSize;
    int l1 = i1 & gh.row_mask;
    int c1 = i1 >> log2_n;
    data g1, g2;
    if (gh.cover_row[l1] == 1 || gh.cover_column[c1] == 1)
      g1 = MAX_DATA;
    else
      g1 = g_idata[i1];
    int l2 = i2 & gh.row_mask;
    int c2 = i2 >> log2_n;
    if (gh.cover_row[l2] == 1 || gh.cover_column[c2] == 1)
      g2 = MAX_DATA;
    else
      g2 = g_idata[i2];
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

fundef step_6_init(GLOBAL_HANDLE<data> gh)
{
  uint id = threadIdx.x + blockIdx.x * blockDim.x;
  if (threadIdx.x == 0)
    zeros_size = 0;
  if (id < n_blocks_step_4)
    gh.zeros_size_b[id] = 0;
}

/* STEP 5:
Construct a series of alternating primed and starred zeros as
follows:
Let Z0 represent the uncovered primed zero found in Step 4.
Let Z1 denote the starred zero in the column of Z0(if any).
Let Z2 denote the primed zero in the row of Z1(there will always
be one). Continue until the series terminates at a primed zero
that has no starred zero in its column. Unstar each starred
zero of the series, star each primed zero of the series, erase
all primes and uncover every line in the matrix. Return to Step 3.*/

// Eliminates joining paths
fundef step_5a(GLOBAL_HANDLE<data> gh)
{
  uint i = blockDim.x * blockIdx.x + threadIdx.x;
  int r_Z0, c_Z0;

  c_Z0 = gh.column_of_prime_at_row[i];
  if (c_Z0 >= 0 && gh.column_of_star_at_row[i] < 0) // if primed and not covered
  {
    gh.row_of_green_at_column[c_Z0] = i; // mark the column as green

    while ((r_Z0 = gh.row_of_star_at_column[c_Z0]) >= 0)
    {
      c_Z0 = gh.column_of_prime_at_row[r_Z0];
      gh.row_of_green_at_column[c_Z0] = r_Z0;
    }
  }
}

// Applies the alternating paths
fundef step_5b(GLOBAL_HANDLE<data> gh)
{
  uint j = blockDim.x * blockIdx.x + threadIdx.x;

  int r_Z0, c_Z0, c_Z2;

  r_Z0 = gh.row_of_green_at_column[j];

  if (r_Z0 >= 0 && gh.row_of_star_at_column[j] < 0)
  {

    c_Z2 = gh.column_of_star_at_row[r_Z0];

    gh.column_of_star_at_row[r_Z0] = j;
    gh.row_of_star_at_column[j] = r_Z0;

    while (c_Z2 >= 0)
    {
      r_Z0 = gh.row_of_green_at_column[c_Z2]; // row of Z2
      c_Z0 = c_Z2;                            // col of Z2
      c_Z2 = gh.column_of_star_at_row[r_Z0];  // col of Z4

      // star Z2
      gh.column_of_star_at_row[r_Z0] = c_Z0;
      gh.row_of_star_at_column[c_Z0] = r_Z0;
    }
  }
}

fundef step_6_add_sub_fused_compress_matrix(GLOBAL_HANDLE<data> gh)
{
  // STEP 6:
  /*STEP 6: Add the minimum uncovered value to every element of each covered
  row, and subtract it from every element of each uncovered column.
  Return to Step 4 without altering any stars, primes, or covered lines. */
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  const int l = i & gh.row_mask;
  const int c = i >> log2_n;
  auto reg = gh.slack[i];
  switch (gh.cover_row[l] + gh.cover_column[c])
  {
  case 2:
    reg += gh.d_min_in_mat[0];
    gh.slack[i] = reg;
    break;
  case 0:
    reg -= gh.d_min_in_mat[0];
    gh.slack[i] = reg;
    break;
  default:
    break;
  }

  // compress matrix
  if (near_zero(reg))
  {
    int b = i >> log2_data_block_size;
    int i0 = i & ~(data_block_size - 1); // == b << log2_data_block_size
    int j = atomicAdd(gh.zeros_size_b + b, 1);
    gh.zeros[i0 + j] = i;
  }
}