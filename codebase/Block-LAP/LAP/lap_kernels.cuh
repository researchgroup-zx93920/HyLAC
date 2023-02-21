#pragma once
#include "../include/utils.cuh"
#include "device_utils.cuh"
#include "cub/cub.cuh"

#define fundef template <typename data = int> \
__device__ __forceinline__

__constant__ size_t SIZE;
__constant__ uint NPROB;
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
const int n_threads_reduction = 512;

fundef void init(GLOBAL_HANDLE<data> &gh) // with single block
{
  // initializations
  // for step 2
  for (size_t i = threadIdx.x; i < SIZE; i += blockDim.x)
  {
    gh.cover_row[i] = 0;
    gh.column_of_star_at_row[i] = -1;
    gh.cover_column[i] = 0;
    gh.row_of_star_at_column[i] = -1;
  }
}

fundef void calc_col_min(GLOBAL_HANDLE<data> &gh) // with single block
{
  for (size_t col = 0; col < SIZE; col++)
  {
    size_t i = (size_t)threadIdx.x * SIZE + col;
    data thread_min = (data)MAX_DATA;

    while (i <= SIZE * (SIZE - 1) + col)
    {
      thread_min = min(thread_min, gh.slack[i]);
      i += (size_t)blockDim.x * SIZE;
    }
    __syncthreads();
    typedef cub::BlockReduce<data, n_threads_reduction> BR;
    __shared__ typename BR::TempStorage temp_storage;
    thread_min = BR(temp_storage).Reduce(thread_min, cub::Min());
    if (threadIdx.x == 0)
    {
      gh.min_in_cols[col] = thread_min;
    }
    __syncthreads();
  }
}

fundef void col_sub(GLOBAL_HANDLE<data> &gh) // with single block
{
  // uint i = (size_t)blockDim.x * (size_t)blockIdx.x + (size_t)threadIdx.x;
  for (size_t i = threadIdx.x; i < SIZE * SIZE; i += blockDim.x)
  {
    size_t l = i % SIZE;
    gh.slack[i] = gh.slack[i] - gh.min_in_cols[l]; // subtract the minimum in col from that col
  }
}

fundef void calc_row_min(GLOBAL_HANDLE<data> &gh) // with single block
{

  typedef cub::BlockReduce<data, n_threads_reduction> BR;
  __shared__ typename BR::TempStorage temp_storage;
  // size_t i = (size_t)blockIdx.x * SIZE + (size_t)threadIdx.x;
  for (size_t row = 0; row < SIZE; row++)
  {
    data thread_min = MAX_DATA;
    for (size_t i = threadIdx.x + row * SIZE; i < SIZE * (row + 1); i += blockDim.x)
    {
      thread_min = min(thread_min, gh.slack[i]);
    }
    __syncthreads();
    thread_min = BR(temp_storage).Reduce(thread_min, cub::Min());
    if (threadIdx.x == 0)
    {
      gh.min_in_rows[row] = thread_min;
    }
    __syncthreads();
  }
}

fundef void row_sub(GLOBAL_HANDLE<data> &gh, SHARED_HANDLE &sh) // with single block
{
  for (size_t i = threadIdx.x; i < SIZE * SIZE; i += blockDim.x)
  {
    size_t c = i / SIZE;
    gh.slack[i] = gh.slack[i] - gh.min_in_rows[c]; // subtract the minimum in row from that row
    if (i == 0)
      sh.zeros_size = 0;
  }
}

fundef bool near_zero(data val)
{
  return ((val < epsilon) && (val > -epsilon));
}

fundef void compress_matrix(GLOBAL_HANDLE<data> &gh, SHARED_HANDLE &sh) // with single block
{
  // size_t i = (size_t)blockDim.x * (size_t)blockIdx.x + (size_t)threadIdx.x;
  for (size_t i = threadIdx.x; i < SIZE * SIZE; i += blockDim.x)
  {
    if (near_zero(gh.slack[i]))
    {
      // atomicAdd(&zeros_size, 1);
      // size_t b = i >> log2_data_block_size;
      size_t i0 = i & ~((size_t)data_block_size - 1); // == b << log2_data_block_size
      size_t j = (size_t)atomicAdd(&sh.zeros_size, 1);
      gh.zeros[i0 + j] = i; // saves index of zeros in slack matrix per block
    }
  }
}

fundef void step_2(GLOBAL_HANDLE<data> &gh, uint temp_blockdim, SHARED_HANDLE &sh)
{
  uint i = threadIdx.x;
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
    for (int j = i; j < min(sh.zeros_size, temp_blockdim); j += blockDim.x)
    {
      uint z = gh.zeros[j];
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
    sh.repeat_kernel = true;
}

fundef void step_3_init(GLOBAL_HANDLE<data> &gh, SHARED_HANDLE &sh) // For single block
{
  for (size_t i = threadIdx.x; i < nrows; i += blockDim.x)
  {
    gh.cover_row[i] = 0;
    gh.cover_column[i] = 0;
  }
  if (threadIdx.x == 0)
    sh.n_matches = 0;
}

fundef void step_3(GLOBAL_HANDLE<data> &gh, SHARED_HANDLE &sh) // For single block
{
  // size_t i = (size_t)blockDim.x * (size_t)blockIdx.x + (size_t)threadIdx.x;
  __shared__ int matches;
  if (threadIdx.x == 0)
    matches = 0;
  __syncthreads();
  for (size_t i = threadIdx.x; i < nrows; i += blockDim.x)
  {
    if (gh.row_of_star_at_column[i] >= 0)
    {
      gh.cover_column[i] = 1;
      atomicAdd((int *)&matches, 1);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0)
    atomicAdd((int *)&sh.n_matches, matches);
}

// STEP 4
// Find a noncovered zero and prime it. If there is no starred
// zero in the row containing this primed zero, go to Step 5.
// Otherwise, cover this row and uncover the column containing
// the starred zero. Continue in this manner until there are no
// uncovered zeros left. Save the smallest uncovered value and
// Go to Step 6.

fundef void step_4_init(GLOBAL_HANDLE<data> &gh)
{
  for (size_t i = threadIdx.x; i < SIZE; i += blockDim.x)
  {
    gh.column_of_prime_at_row[i] = -1;
    gh.row_of_green_at_column[i] = -1;
  }
}

fundef void step_4(GLOBAL_HANDLE<data> &gh, uint temp_blockdim, SHARED_HANDLE &sh)
{
  __shared__ bool s_found;
  __shared__ bool s_goto_5;
  __shared__ bool s_repeat_kernel;
  volatile int *v_cover_row = gh.cover_row;
  volatile int *v_cover_column = gh.cover_column;

  const size_t i = threadIdx.x;
  // const size_t b = blockIdx.x;
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
    for (size_t j = threadIdx.x; j < sh.zeros_size; j += blockDim.x)
    {
      // each thread picks a zero!
      size_t z = gh.zeros[j];
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
    sh.repeat_kernel = true;
  if (i == 0 && s_goto_5) // if any blocks needs to go to step 5, algorithm needs to go to step 5
    sh.goto_5 = true;
}

template <typename data = int, uint blockSize = n_threads_reduction>
__device__ void min_reduce_kernel1(volatile data *g_idata, volatile data *g_odata,
                                   const size_t n, GLOBAL_HANDLE<data> &gh)
{
  __shared__ data sdata[blockSize];
  const uint tid = threadIdx.x;
  // size_t i = (size_t)blockIdx.x * ((size_t)blockSize * 2) + (size_t)tid;
  size_t i = tid;
  size_t gridSize = (size_t)blockSize * 2;
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
    *g_odata = minimum;
}

fundef void step_6_init(GLOBAL_HANDLE<data> &gh, SHARED_HANDLE &sh)
{
  // size_t id = (size_t)threadIdx.x + (size_t)blockIdx.x * (size_t)blockDim.x;
  if (threadIdx.x == 0)
    sh.zeros_size = 0;
  for (uint i = threadIdx.x; i < SIZE; i += blockDim.x)
  {
    if (gh.cover_column[i] == 0)
      gh.min_in_rows[i] += gh.d_min_in_mat[0] / 2;
    else
      gh.min_in_rows[i] -= gh.d_min_in_mat[0] / 2;
    if (gh.cover_row[i] == 0)
      gh.min_in_cols[i] += gh.d_min_in_mat[0] / 2;
    else
      gh.min_in_cols[i] -= gh.d_min_in_mat[0] / 2;
  }
  __syncthreads();
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
fundef void step_5a(GLOBAL_HANDLE<data> gh)
{
  // size_t i = (size_t)blockDim.x * (size_t)blockIdx.x + (size_t)threadIdx.x;
  for (size_t i = threadIdx.x; i < SIZE; i += blockDim.x)
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
fundef void step_5b(GLOBAL_HANDLE<data> &gh)
{
  // size_t j = (size_t)blockDim.x * (size_t)blockIdx.x + (size_t)threadIdx.x;
  for (size_t j = threadIdx.x; j < SIZE; j += blockDim.x)
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

fundef void step_6_add_sub_fused_compress_matrix(GLOBAL_HANDLE<data> &gh, SHARED_HANDLE &sh) // For single block
{
  // STEP 6:
  /*STEP 6: Add the minimum uncovered value to every element of each covered
  row, and subtract it from every element of each uncovered column.
  Return to Step 4 without altering any stars, primes, or covered lines. */
  // const size_t i = (size_t)blockDim.x * (size_t)blockIdx.x + (size_t)threadIdx.x;
  for (size_t i = threadIdx.x; i < SIZE * SIZE; i += blockDim.x)
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
      // size_t b = i >> log2_data_block_size;
      size_t i0 = i & ~((size_t)data_block_size - 1); // == b << log2_data_block_size
      int j = atomicAdd(&sh.zeros_size, 1);
      gh.zeros[i0 + j] = i;
    }
  }
}

fundef void printArray(data *idata, size_t len = SIZE, const char *message = NULL)
{
  __syncthreads();
#ifdef __DEBUG__D
  if (threadIdx.x == 0)
  {
    if (message != NULL)
      printf("%s: ", message);
    for (uint i = 0; i < len; i++)
    {
      printf("%f, ", (float)idata[i]);
    }
    printf("\n");
  }
  __syncthreads();
#endif
}

fundef void printMatrix(data *idata)
{
  __syncthreads();
#ifdef __DEBUG__D
  if (threadIdx.x == 0)
  {
    for (uint i = 0; i < SIZE; i++)
    {
      for (uint j = 0; j < SIZE; j++)
      {
        printf("%d, ", idata[SIZE * i + j]);
      }
      printf("\n");
    }
    printf("\n\n");
  }
  __syncthreads();
#endif
}

fundef void set_handles(TILED_HANDLE<data> &th, GLOBAL_HANDLE<data> &gh, uint &problemID)
{
  if (threadIdx.x == 0)
  {
    size_t b = blockIdx.x;
    problemID = atomicAdd(th.tail, 1);
    // problemID = b;
    if (problemID < NPROB)
    {
      gh.slack = &th.slack[(size_t)problemID * SIZE * SIZE];
      gh.column_of_star_at_row = &th.column_of_star_at_row[problemID * nrows];

      gh.min_in_rows = &th.min_in_rows[b * nrows];
      gh.min_in_cols = &th.min_in_cols[b * ncols];
      gh.zeros = &th.zeros[b * nrows * ncols];
      gh.zeros_size_b = &th.zeros_size_b[b * NB4];
      gh.row_of_star_at_column = &th.row_of_star_at_column[b * ncols];
      gh.cover_row = &th.cover_row[b * nrows];
      gh.cover_column = &th.cover_column[b * ncols];
      gh.column_of_prime_at_row = &th.column_of_prime_at_row[b * nrows];
      gh.row_of_green_at_column = &th.row_of_green_at_column[b * ncols];
      gh.max_in_mat_row = &th.max_in_mat_row[b * nrows];
      gh.max_in_mat_col = &th.max_in_mat_col[b * ncols];
      gh.d_min_in_mat_vect = &th.d_min_in_mat_vect[b * NBR];
      gh.d_min_in_mat = &th.d_min_in_mat[b * 1];
    }
  }
  __syncthreads();
}
#ifdef false
template <typename data = int, uint BLOCK_DIM_X>
__launch_bounds__(BLOCK_DIM_X)
    __global__ void BHA(GLOBAL_HANDLE<data> gh)
{
  __shared__ SHARED_HANDLE sh;
  init(gh);
  // printMatrix(gh.slack);000
  calc_col_min(gh);
  // printArray(gh.min_in_rows);
  __syncthreads();
  col_sub(gh);
  __syncthreads();
  // printMatrix(gh.slack);
  calc_row_min(gh);
  __syncthreads();
  row_sub(gh, sh);
  __syncthreads();
  compress_matrix(gh, sh);
  __syncthreads();
  // checkpoint();
  do
  {
    __syncthreads();
    if (threadIdx.x == 0)
      sh.repeat_kernel = false;
    __syncthreads();
    uint temp_blockdim = (gh.nb4 > 1 || sh.zeros_size > max_threads_per_block) ? max_threads_per_block : sh.zeros_size;
    step_2(gh, temp_blockdim, sh);
    __syncthreads();
  } while (sh.repeat_kernel);
  __syncthreads();
  // checkpoint();
  while (1)
  {
    __syncthreads();
    step_3_init(gh, sh);
    __syncthreads();
    step_3(gh, sh);
    __syncthreads();
    if (sh.n_matches >= SIZE)
      break;
    step_4_init(gh);
    __syncthreads();
    // checkpoint();
    while (1)
    {
      __syncthreads();
      do
      {
        if (threadIdx.x == 0)
        {
          sh.goto_5 = false;
          sh.repeat_kernel = false;
        }
        __syncthreads();
        uint temp_blockdim = (gh.nb4 > 1 || sh.zeros_size > max_threads_per_block) ? max_threads_per_block : sh.zeros_size;
        step_4(gh, temp_blockdim, sh);
        __syncthreads();
      } while (sh.repeat_kernel && !sh.goto_5);
      __syncthreads();
      if (sh.goto_5)
        break;
      // checkpoint();

      __syncthreads();

      min_reduce_kernel1<data, n_threads_reduction>(gh.slack, gh.d_min_in_mat, SIZE * SIZE, gh);
      __syncthreads();
      if (gh.d_min_in_mat[0] <= 0)
      {
        __syncthreads();
        if (threadIdx.x == 0)
          printf("minimum element in block %u is non positive\n%d", blockIdx.x, gh.d_min_in_mat[0]);
        return;
      }
      __syncthreads();
      step_6_init(gh, sh);
      __syncthreads();
      step_6_add_sub_fused_compress_matrix(gh, sh);
      __syncthreads();
    }
    __syncthreads();
    // checkpoint();
    step_5a(gh);
    __syncthreads();
    step_5b(gh);
    __syncthreads();
  }
}
#endif

fundef void BHA(GLOBAL_HANDLE<data> &gh, SHARED_HANDLE &sh, const uint problemID)
{

  init(gh);
  calc_row_min(gh);
  __syncthreads();
  row_sub(gh, sh);
  __syncthreads();
  calc_col_min(gh);
  __syncthreads();
  col_sub(gh);
  __syncthreads();
  // checkpoint();

  // printArray(gh.min_in_rows, SIZE, "rowmin");
  // printArray(gh.min_in_cols, SIZE, "colmin");
  compress_matrix(gh, sh);
  __syncthreads();

  // printArray(gh.cover_row, "cover row");
  // printArray(gh.cover_column, "cover column");

  do
  {
    __syncthreads();
    if (threadIdx.x == 0)
      sh.repeat_kernel = false;
    __syncthreads();
    uint temp_blockdim = (gh.nb4 > 1 || sh.zeros_size > max_threads_per_block) ? max_threads_per_block : sh.zeros_size;
    step_2(gh, temp_blockdim, sh);
    __syncthreads();
  } while (sh.repeat_kernel);
  __syncthreads();
  // printArray(gh.row_of_star_at_column, "row of star at column");
  // printArray(gh.column_of_star_at_row, "column of star at row");

  while (1)
  {
    __syncthreads();
    step_3_init(gh, sh);
    __syncthreads();
    step_3(gh, sh);
    __syncthreads();
    // printArray(gh.cover_column);
    // printArray(gh.cover_row);
    if (sh.n_matches >= SIZE)
      break;
    step_4_init(gh);
    __syncthreads();

    // checkpoint();

    while (1)
    {
      do
      {
        __syncthreads();
        if (threadIdx.x == 0)
        {
          sh.goto_5 = false;
          sh.repeat_kernel = false;
        }
        __syncthreads();
        uint temp_blockdim = (gh.nb4 > 1 || sh.zeros_size > max_threads_per_block) ? max_threads_per_block : sh.zeros_size;
        step_4(gh, temp_blockdim, sh);
        __syncthreads();
      } while (sh.repeat_kernel && !sh.goto_5);
      __syncthreads();
      if (sh.goto_5)
        break;
      // checkpoint();
      // printArray(&sh.zeros_size, 1, "zeros size:");
      // printArray(gh.cover_row, SIZE, "row cover");
      __syncthreads();
      min_reduce_kernel1<data, n_threads_reduction>(gh.slack, gh.d_min_in_mat, SIZE * SIZE, gh);
      __syncthreads();
      if (gh.d_min_in_mat[0] <= 0)
      {
        __syncthreads();
        if (threadIdx.x == 0)
        {

          printf("minimum element in problemID %u is non positive: %f\n", problemID, (float)gh.d_min_in_mat[0]);
          // printf("minimum element in problemID %u is non positive: %f\n", problemID, gh.d_min_in_mat[0]);
        }
        return;
      }
      __syncthreads();
      step_6_init(gh, sh); // Also does dual update
      __syncthreads();
      // printArray(gh.d_min_in_mat, 1, "min ");
      // printArray(gh.min_in_rows, SIZE, "row dual");
      step_6_add_sub_fused_compress_matrix(gh, sh);
      __syncthreads();
    }
    __syncthreads();
    // checkpoint();
    step_5a(gh);
    __syncthreads();
    step_5b(gh);
    __syncthreads();
  }
}

template <typename data, uint BLOCK_DIM_X>
__global__ void THA(TILED_HANDLE<data> th)
{
  __shared__ GLOBAL_HANDLE<data> gh;
  __shared__ SHARED_HANDLE sh;
  __shared__ uint problemID;
  // checkpoint();
  while (1)
  {
    set_handles(th, gh, problemID);
    __syncthreads();
    if (problemID >= NPROB)
      return;
    __syncthreads();

    BHA<data>(gh, sh, problemID);
    __syncthreads();
    // printArray(gh.min_in_rows, SIZE, "row duals");
    // printArray(gh.min_in_cols, SIZE, "col duals");
    // if (threadIdx.x == 0)
    //   printf("Problem ID: %u done\n", problemID);
  }
}