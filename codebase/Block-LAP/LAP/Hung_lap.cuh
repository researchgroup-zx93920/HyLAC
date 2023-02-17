#include "../include/defs.cuh"
#include "../include/logger.cuh"
#include "../include/Timer.h"
#include "lap_kernels.cuh"
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

template <typename data>
class BLAP
{
private:
  int dev_;
  size_t size_, h_nrows, h_ncols;
  data *cost_;

  uint num_blocks_4, num_blocks_reduction;

public:
  GLOBAL_HANDLE<data> gh;
  // constructor
  BLAP(data *cost, size_t size, int dev = 0) : cost_(cost), dev_(dev), size_(size)
  {
    h_nrows = size;
    h_ncols = size;

    // constant memory copies
    CUDA_RUNTIME(cudaSetDevice(dev_));
    CUDA_RUNTIME(cudaMemcpyToSymbol(SIZE, &size, sizeof(SIZE)));
    // memstatus("First");
    CUDA_RUNTIME(cudaMemcpyToSymbol(nrows, &h_nrows, sizeof(SIZE)));
    CUDA_RUNTIME(cudaMemcpyToSymbol(ncols, &h_ncols, sizeof(SIZE)));
    num_blocks_4 = max((uint)ceil((size * 1.0) / columns_per_block_step_4), 1);
    num_blocks_reduction = min(size, 512UL);
    CUDA_RUNTIME(cudaMemcpyToSymbol(NB4, &num_blocks_4, sizeof(NB4)));
    CUDA_RUNTIME(cudaMemcpyToSymbol(NBR, &num_blocks_reduction, sizeof(NBR)));
    const uint temp1 = ceil(size / num_blocks_reduction);
    CUDA_RUNTIME(cudaMemcpyToSymbol(n_rows_per_block, &temp1, sizeof(n_rows_per_block)));
    CUDA_RUNTIME(cudaMemcpyToSymbol(n_cols_per_block, &temp1, sizeof(n_rows_per_block)));
    const uint temp2 = (uint)ceil(log2(size_));
    CUDA_RUNTIME(cudaMemcpyToSymbol(log2_n, &temp2, sizeof(log2_n)));
    gh.row_mask = (1 << temp2) - 1;
    Log(debug, "log2_n %d", temp2);
    Log(debug, "row mask: %d", gh.row_mask);
    gh.nb4 = max((uint)ceil((size * 1.0) / columns_per_block_step_4), 1);
    CUDA_RUNTIME(cudaMemcpyToSymbol(n_blocks_step_4, &gh.nb4, sizeof(n_blocks_step_4)));
    const uint temp4 = columns_per_block_step_4 * pow(2, ceil(log2(size_)));
    Log(debug, "dbs: %u", temp4);
    CUDA_RUNTIME(cudaMemcpyToSymbol(data_block_size, &temp4, sizeof(data_block_size)));
    const uint temp5 = temp2 + (uint)ceil(log2(columns_per_block_step_4));
    Log(debug, "l2dbs: %u", temp5);
    CUDA_RUNTIME(cudaMemcpyToSymbol(log2_data_block_size, &temp5, sizeof(log2_data_block_size)));

    // memory allocations
    // CUDA_RUNTIME(cudaMalloc((void **)&gh.cost, size * size * sizeof(data)));
    // memstatus("Post constant");
    CUDA_RUNTIME(cudaMalloc((void **)&gh.slack, size * size * sizeof(data)));
    CUDA_RUNTIME(cudaMalloc((void **)&gh.min_in_rows, h_nrows * sizeof(data)));
    CUDA_RUNTIME(cudaMalloc((void **)&gh.min_in_cols, h_ncols * sizeof(data)));

    CUDA_RUNTIME(cudaMalloc((void **)&gh.zeros, h_nrows * h_ncols * sizeof(size_t)));
    CUDA_RUNTIME(cudaMalloc((void **)&gh.zeros_size_b, num_blocks_4 * sizeof(size_t)));
    CUDA_RUNTIME(cudaMalloc((void **)&gh.row_of_star_at_column, h_ncols * sizeof(int)));
    CUDA_RUNTIME(cudaMallocManaged((void **)&gh.column_of_star_at_row, h_nrows * sizeof(int)));
    CUDA_RUNTIME(cudaMalloc((void **)&gh.cover_row, h_nrows * sizeof(int)));
    CUDA_RUNTIME(cudaMalloc((void **)&gh.cover_column, h_ncols * sizeof(int)));
    CUDA_RUNTIME(cudaMalloc((void **)&gh.column_of_prime_at_row, h_nrows * sizeof(int)));
    CUDA_RUNTIME(cudaMalloc((void **)&gh.row_of_green_at_column, h_ncols * sizeof(int)));

    CUDA_RUNTIME(cudaMalloc((void **)&gh.max_in_mat_row, h_nrows * sizeof(data)));
    CUDA_RUNTIME(cudaMalloc((void **)&gh.max_in_mat_col, h_ncols * sizeof(data)));
    CUDA_RUNTIME(cudaMalloc((void **)&gh.d_min_in_mat_vect, num_blocks_reduction * sizeof(data)));
    CUDA_RUNTIME(cudaMalloc((void **)&gh.d_min_in_mat, 1 * sizeof(data)));

    CUDA_RUNTIME(cudaMemcpy(gh.slack, cost_, size * size * sizeof(data), cudaMemcpyDefault));
    // CUDA_RUNTIME(cudaMemcpy(gh.cost, cost_, size * size * sizeof(data), cudaMemcpyDefault));

    CUDA_RUNTIME(cudaDeviceSynchronize());
    // memstatus("Post all mallocs");
  };

  // destructor
  ~BLAP()
  {
    // Log(critical, "Destructor called");
    gh.clear();
  };
  void solve()
  {
    uint nprob = 1;
    const uint n_threads = (uint)min(size_, 512UL);
    const uint n_threads_full = (uint)min(size_ * size_, 512UL);
    const size_t n_blocks = (size_t)ceil((size_ * 1.0) / n_threads);

    execKernel(init, nprob, n_threads, dev_, false, gh);
    execKernel(calc_col_min, nprob, n_threads_reduction, dev_, false, gh);
    execKernel(col_sub, nprob, n_threads, dev_, false, gh);

    execKernel(calc_row_min, nprob, n_threads_reduction, dev_, false, gh);
    execKernel(row_sub, nprob, n_threads, dev_, false, gh);
    execKernel(compress_matrix, nprob, n_threads, dev_, false, gh);

    // use thrust instead of add reduction
    Log(debug, "b4: %d", gh.nb4);
    do
    {
      repeat_kernel = false;
      uint temp_blockdim = (gh.nb4 > 1 || zeros_size > max_threads_per_block) ? max_threads_per_block : zeros_size;
      execKernel(step_2, gh.nb4, temp_blockdim, dev_, false, gh);
    } while (repeat_kernel);
    Log(debug, "Zeros size: %d", zeros_size);

    while (1)
    {
      execKernel(step_3_init, nprob, n_threads, dev_, false, gh);
      execKernel(step_3, nprob, n_threads, dev_, false, gh);
      if (n_matches >= h_ncols)
        break;

      execKernel(step_4_init, nprob, n_threads, dev_, false, gh);

      while (1)
      {
        do
        {
          goto_5 = false;
          repeat_kernel = false;
          CUDA_RUNTIME(cudaDeviceSynchronize());

          uint temp_blockdim = (gh.nb4 > 1 || zeros_size > max_threads_per_block) ? max_threads_per_block : zeros_size;
          execKernel(step_4, gh.nb4, temp_blockdim, dev_, false, gh);
        } while (repeat_kernel && !goto_5);

        if (goto_5)
          break;

        // step 6
        // printDebugArray(gh.cover_column, size_, "Column cover");
        // printDebugArray(gh.cover_row, size_, "Row cover");
        execKernel((min_reduce_kernel1<data, n_threads_reduction>),
                   nprob, n_threads_reduction, dev_, false,
                   gh.slack, gh.d_min_in_mat, h_nrows * h_ncols, gh);

        // printDebugArray(gh.d_min_in_mat_vect, num_blocks_reduction, "min vector");
        // printDebugArray(gh.cover_column, size_, "Column cover");
        // printDebugArray(gh.cover_row, size_, "Row cover");

        if (!passes_sanity_test(gh.d_min_in_mat))
          exit(-1);

        execKernel(step_6_init, nprob, n_threads, dev_, false, gh);
        execKernel(step_6_add_sub_fused_compress_matrix, nprob, n_threads_full, dev_, false, gh);

        // printDebugArray(gh.zeros_size_b, num_blocks_4);
      } // repeat step 4 and 6

      execKernel(step_5a, nprob, n_threads, dev_, false, gh);
      execKernel(step_5b, nprob, n_threads, dev_, false, gh);
    } // repeat steps 3 to 6

    // CUDA_RUNTIME(cudaFree(d_temp_storage));

    // find objective
    double total_cost = 0;
    for (uint r = 0; r < h_nrows; r++)
    {
      int c = gh.column_of_star_at_row[r];
      if (c >= 0)
        total_cost += cost_[c * h_nrows + r];
      // printf("r = %d, c = %d\n", r, c);
    }
    printf("Total cost: \t %f \n", total_cost);
  };

  bool passes_sanity_test(data *d_min)
  {
    data temp;
    CUDA_RUNTIME(cudaMemcpy(&temp, d_min, 1 * sizeof(data), cudaMemcpyDeviceToHost));
    if (temp <= 0)
    {
      Log(critical, "minimum element in matrix is non positive => infinite loop condition !!!");
      Log(critical, "%d", temp);
      return false;
    }
    else
      return true;
  }
};