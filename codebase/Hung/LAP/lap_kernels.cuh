#pragma once
#include "../include/utils.cuh"
#include "device_utils.cuh"
#include "cub/cub.cuh"

#define fundef template <typename data = int> \
__global__ void

__constant__ size_t SIZE;
__constant__ size_t nrows;
__constant__ size_t ncols;

__constant__ uint NB4;
__constant__ uint NBR;
__constant__ uint n_rows_per_block;
__constant__ uint n_cols_per_block;
__constant__ uint log2_n, log2_data_block_size, data_block_size;
__constant__ uint n_blocks_step_4;

const int max_threads_per_block = 1024;
const int columns_per_block_step_4 = 512;
const int n_threads_reduction = 256;

fundef init(GLOBAL_HANDLE<data> gh)
{
  size_t i = (size_t)blockDim.x * (size_t)blockIdx.x + (size_t)threadIdx.x;
  // initializations
  // for step 2
  if (i < SIZE)
  {
    gh.cover_row[i] = 0;
    gh.column_of_star_at_row[i] = -1;
    gh.cover_column[i] = 0;
    gh.row_of_star_at_column[i] = -1;
  }
}

fundef calc_col_min(GLOBAL_HANDLE<data> gh)
{
  size_t i = (size_t)threadIdx.x * SIZE + (size_t)blockIdx.x;
  data thread_min = (data)MAX_DATA;

  while (i < SIZE * SIZE)
  {
    thread_min = min(thread_min, gh.slack[i]);
    i += (size_t)blockDim.x * SIZE;
  }
  __syncthreads();
  typedef cub::BlockReduce<data, n_threads_reduction> BR;
  __shared__ typename BR::TempStorage temp_storage;
  thread_min = BR(temp_storage).Reduce(thread_min, cub::Min());

  if (threadIdx.x == 0)
    gh.min_in_rows[blockIdx.x] = thread_min;
}

fundef col_sub(GLOBAL_HANDLE<data> gh)
{
  uint i = (size_t)blockDim.x * (size_t)blockIdx.x + (size_t)threadIdx.x;
  if (i < (size_t)SIZE * SIZE)
  {
    uint l = i % SIZE;
    gh.slack[i] = gh.slack[i] - gh.min_in_rows[l]; // subtract the minimum in row from that row
  }
}

fundef calc_row_min(GLOBAL_HANDLE<data> gh)
{
  data thread_min = MAX_DATA;
  size_t i = (size_t)blockIdx.x * SIZE + (size_t)threadIdx.x;
  while (i < SIZE * ((size_t)blockIdx.x + 1))
  {
    thread_min = min(thread_min, gh.slack[i]);
    i += blockDim.x;
  }
  typedef cub::BlockReduce<data, n_threads_reduction> BR;
  __shared__ typename BR::TempStorage temp_storage;
  thread_min = BR(temp_storage).Reduce(thread_min, cub::Min());

  if (threadIdx.x == 0)
  {
    // printf("%u, %d\n", blockIdx.x, thread_min);
    gh.min_in_cols[blockIdx.x] = thread_min;
  }
}

fundef row_sub(GLOBAL_HANDLE<data> gh)
{
  size_t i = (size_t)blockDim.x * (size_t)blockIdx.x + (size_t)threadIdx.x;
  size_t c = i / SIZE;
  if (i < SIZE * SIZE)
    gh.slack[i] = gh.slack[i] - gh.min_in_cols[c]; // subtract the minimum in row from that row

  if (i == 0)
    zeros_size = 0;
  if (i < n_blocks_step_4)
    gh.zeros_size_b[i] = 0;
}

template <typename data = int>
__device__ bool near_zero(data val)
{
  return ((val < eps) && (val > -eps));
}

fundef compress_matrix(GLOBAL_HANDLE<data> gh)
{
  size_t i = (size_t)blockDim.x * (size_t)blockIdx.x + (size_t)threadIdx.x;
  if (i < SIZE * SIZE)
  {
    if (near_zero(gh.slack[i]))
    {
      // atomicAdd(&zeros_size, 1);
      size_t b = i >> log2_data_block_size;
      size_t i0 = i & ~((size_t)data_block_size - 1); // == b << log2_data_block_size
      // if (i0 != 0 || b != 0)
      // {
      //   printf("This problem is big! %u\n", i);
      // }

      size_t j = (size_t)atomicAdd((uint64 *)&gh.zeros_size_b[b], 1ULL);
      gh.zeros[i0 + j] = i; // saves index of zeros in slack matrix per block
    }
  }
}

fundef add_reduction(GLOBAL_HANDLE<data> gh)
{
  __shared__ int sdata[1024]; // hard coded need to change!
  const int i = threadIdx.x;
  for (int j = 0; j < 1024; j += blockDim.x)
    sdata[j] = 0;
  __syncthreads();

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
  uint i = threadIdx.x;
  uint b = blockIdx.x;
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
      uint z = gh.zeros[(b << log2_data_block_size) + j];
      uint l = z % nrows;
      uint c = z / nrows;
      if (gh.cover_row[l] == 0 &&
          gh.cover_column[c] == 0)
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
  size_t i = (size_t)blockDim.x * (size_t)blockIdx.x + (size_t)threadIdx.x;
  if (i < nrows)
  {
    gh.cover_row[i] = 0;
    gh.cover_column[i] = 0;
  }
  if (i == 0)
    n_matches = 0;
}

fundef step_3(GLOBAL_HANDLE<data> gh)
{
  size_t i = (size_t)blockDim.x * (size_t)blockIdx.x + (size_t)threadIdx.x;
  __shared__ int matches;
  if (threadIdx.x == 0)
    matches = 0;
  __syncthreads();
  if (i < nrows)
  {
    if (gh.row_of_star_at_column[i] >= 0)
    {
      gh.cover_column[i] = 1;
      atomicAdd((int *)&matches, 1);
    }
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
  size_t i = (size_t)blockDim.x * (size_t)blockIdx.x + (size_t)threadIdx.x;
  if (i < SIZE)
  {
    gh.column_of_prime_at_row[i] = -1;
    gh.row_of_green_at_column[i] = -1;
  }
}

fundef step_4(GLOBAL_HANDLE<data> gh)
{
  __shared__ bool s_found;
  __shared__ bool s_goto_5;
  __shared__ bool s_repeat_kernel;
  volatile int *v_cover_row = gh.cover_row;
  volatile int *v_cover_column = gh.cover_column;

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
    for (size_t j = threadIdx.x; j < gh.zeros_size_b[b]; j += blockDim.x)
    {
      // each thread picks a zero!
      size_t z = gh.zeros[(size_t)(b << (size_t)log2_data_block_size) + j];
      int l = z % nrows; // row
      int c = z / nrows; // column
      int c1 = gh.column_of_star_at_row[l];

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
                                   const size_t n, GLOBAL_HANDLE<data> gh)
{
  __shared__ data sdata[blockSize];
  const uint tid = threadIdx.x;
  size_t i = (size_t)blockIdx.x * ((size_t)blockSize * 2) + (size_t)tid;
  size_t gridSize = (size_t)blockSize * 2 * (size_t)gridDim.x;
  sdata[tid] = MAX_DATA;
  while (i < n)
  {
    size_t i1 = i;
    size_t i2 = i + blockSize;
    size_t l1 = i1 % nrows; // local index within the row
    size_t c1 = i1 / nrows; // Row number
    data g1 = MAX_DATA, g2 = MAX_DATA;
    if (gh.cover_row[l1] == 1 || gh.cover_column[c1] == 1)
      g1 = MAX_DATA;
    else
      g1 = g_idata[i1];
    if (i2 < nrows * nrows)
    {
      size_t l2 = i2 % nrows;
      size_t c2 = i2 / nrows;
      if (gh.cover_row[l2] == 1 || gh.cover_column[c2] == 1)
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

fundef step_6_init(GLOBAL_HANDLE<data> gh)
{
  size_t id = (size_t)threadIdx.x + (size_t)blockIdx.x * (size_t)blockDim.x;
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
  size_t i = (size_t)blockDim.x * (size_t)blockIdx.x + (size_t)threadIdx.x;
  if (i < SIZE)
  {
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
}

// Applies the alternating paths
fundef step_5b(GLOBAL_HANDLE<data> gh)
{
  size_t j = (size_t)blockDim.x * (size_t)blockIdx.x + (size_t)threadIdx.x;
  if (j < SIZE)
  {
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
}

fundef step_6_add_sub_fused_compress_matrix(GLOBAL_HANDLE<data> gh)
{
  // STEP 6:
  /*STEP 6: Add the minimum uncovered value to every element of each covered
  row, and subtract it from every element of each uncovered column.
  Return to Step 4 without altering any stars, primes, or covered lines. */
  const size_t i = (size_t)blockDim.x * (size_t)blockIdx.x + (size_t)threadIdx.x;
  if (i < SIZE * SIZE)
  {
    const size_t l = i % nrows;
    const size_t c = i / nrows;
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
      size_t b = i >> log2_data_block_size;
      size_t i0 = i & ~((size_t)data_block_size - 1); // == b << log2_data_block_size
      size_t j = (size_t)atomicAdd((uint64 *)gh.zeros_size_b + b, 1ULL);
      gh.zeros[i0 + j] = i;
    }
  }
}