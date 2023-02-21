#include "../include/defs.cuh"
#include "../include/logger.cuh"
#include "../include/Timer.h"
#include "lap_kernels.cuh"
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
const uint nthr = 512;

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
    const uint n_threads = 512UL;
    const uint n_threads_full = (uint)min(size_ * size_, 512UL);
    const size_t n_blocks = (size_t)ceil((size_ * 1.0) / n_threads);

    execKernel((BHA<data, n_threads>), nprob, n_threads, dev_, false, gh);

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

template <typename data>
class TLAP
{
private:
  uint nprob_;
  int dev_, maxtile;
  size_t size_, h_nrows, h_ncols;
  data *Tcost_;
  uint num_blocks_4, num_blocks_reduction;

public:
  TILED_HANDLE<data> th;
  TLAP(uint nproblem, data *tcost, size_t size, int dev = 0)
      : nprob_(nproblem), Tcost_(tcost), dev_(dev), size_(size)
  {
    h_nrows = size;
    h_ncols = size;
    CUDA_RUNTIME(cudaSetDevice(dev_));
    CUDA_RUNTIME(cudaMemcpyToSymbol(NPROB, &nprob_, sizeof(NPROB)));
    CUDA_RUNTIME(cudaMemcpyToSymbol(SIZE, &size, sizeof(SIZE)));
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

    int max_active_blocks = 1;
    CUDAContext context;
    int num_SMs = context.num_SMs;

    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks,
                                                  THA<data, nthr>,
                                                  nthr, 0);
    max_active_blocks *= num_SMs;
    maxtile = min(nproblem, max_active_blocks);
    Log(debug, "Grid dimension %d\n", maxtile);

    th.row_mask = (1 << temp2) - 1;
    Log(debug, "log2_n %d", temp2);
    Log(debug, "row mask: %d", th.row_mask);
    th.nb4 = max((uint)ceil((size * 1.0) / columns_per_block_step_4), 1);
    CUDA_RUNTIME(cudaMemcpyToSymbol(n_blocks_step_4, &th.nb4, sizeof(n_blocks_step_4)));
    const uint temp4 = columns_per_block_step_4 * pow(2, ceil(log2(size_)));
    Log(debug, "dbs: %u", temp4);
    CUDA_RUNTIME(cudaMemcpyToSymbol(data_block_size, &temp4, sizeof(data_block_size)));
    const uint temp5 = temp2 + (uint)ceil(log2(columns_per_block_step_4));
    Log(debug, "l2dbs: %u", temp5);
    CUDA_RUNTIME(cudaMemcpyToSymbol(log2_data_block_size, &temp5, sizeof(log2_data_block_size)));

    // external memory
    CUDA_RUNTIME(cudaMalloc((void **)&th.slack, nproblem * size * size * sizeof(data)));
    CUDA_RUNTIME(cudaMallocManaged((void **)&th.column_of_star_at_row, nproblem * h_nrows * sizeof(int)));

    // internal memory
    CUDA_RUNTIME(cudaMalloc((void **)&th.min_in_rows, maxtile * h_nrows * sizeof(data)));
    CUDA_RUNTIME(cudaMalloc((void **)&th.min_in_cols, maxtile * h_ncols * sizeof(data)));
    CUDA_RUNTIME(cudaMalloc((void **)&th.zeros, maxtile * h_nrows * h_ncols * sizeof(size_t)));
    CUDA_RUNTIME(cudaMalloc((void **)&th.zeros_size_b, maxtile * num_blocks_4 * sizeof(size_t)));
    CUDA_RUNTIME(cudaMalloc((void **)&th.row_of_star_at_column, maxtile * h_ncols * sizeof(int)));
    CUDA_RUNTIME(cudaMalloc((void **)&th.cover_row, maxtile * h_nrows * sizeof(int)));
    CUDA_RUNTIME(cudaMalloc((void **)&th.cover_column, maxtile * h_ncols * sizeof(int)));
    CUDA_RUNTIME(cudaMalloc((void **)&th.column_of_prime_at_row, maxtile * h_nrows * sizeof(int)));
    CUDA_RUNTIME(cudaMalloc((void **)&th.row_of_green_at_column, maxtile * h_ncols * sizeof(int)));

    CUDA_RUNTIME(cudaMalloc((void **)&th.max_in_mat_row, maxtile * h_nrows * sizeof(data)));
    CUDA_RUNTIME(cudaMalloc((void **)&th.max_in_mat_col, maxtile * h_ncols * sizeof(data)));
    CUDA_RUNTIME(cudaMalloc((void **)&th.d_min_in_mat_vect, maxtile * num_blocks_reduction * sizeof(data)));
    CUDA_RUNTIME(cudaMalloc((void **)&th.d_min_in_mat, maxtile * 1 * sizeof(data)));

    CUDA_RUNTIME(cudaMalloc((void **)&th.tail, 1 * sizeof(uint)));
    CUDA_RUNTIME(cudaMemset(th.tail, 0, sizeof(uint)));
    // CUDA_RUNTIME(cudaDeviceSynchronize());
    // initialize slack
    CUDA_RUNTIME(cudaMemcpy(th.slack, tcost, nproblem * size * size * sizeof(data), cudaMemcpyDefault));
    CUDA_RUNTIME(cudaDeviceSynchronize());
  };
  // destructor
  ~TLAP()
  {
    th.clear();
  }

  void solve()
  {
    int nblocks = maxtile;
    Log(debug, "nblocks: %d\n", nblocks);
    Timer t;
    execKernel((THA<data, nthr>), nblocks, nthr, dev_, true, th);
    auto time = t.elapsed();
    Log(info, "kernel time %f s\n", time);
    for (int prob = 0; prob < nprob_; prob++)
    {
      data *cost = &Tcost_[prob * size_ * size_];
      // printDebugMatrix(cost, h_nrows, h_ncols, "cost");
      double total_cost = 0;
      for (int r = 0; r < h_nrows; r++)
      {
        int c = th.column_of_star_at_row[prob * size_ + r];
        // cout << c << ", ";
        if (c >= 0)
          total_cost += cost[c * h_nrows + r];
      }
      printf("problem %d \t obj: %f\n", prob, total_cost);
    }
  }
};
