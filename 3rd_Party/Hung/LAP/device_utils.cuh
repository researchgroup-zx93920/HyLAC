#pragma once
#include "../include/utils.cuh"

template <typename data = int>
struct GLOBAL_HANDLE
{
  data *cost;
  data *slack;
  data *min_in_rows;
  data *min_in_cols;

  int *zeros, *zeros_size_b;
  int *row_of_star_at_column, *column_of_star_at_row;
  int *cover_row, *cover_column;
  int *column_of_prime_at_row, *row_of_green_at_column;

  data *max_in_mat_row, *max_in_mat_col, *d_min_in_mat_vect, *d_min_in_mat;
  int row_mask;
  uint nb4;

  // ~GLOBAL_HANDLE()
  // {
  //   CUDA_RUNTIME(cudaFree(cost));
  //   CUDA_RUNTIME(cudaFree(slack));
  //   CUDA_RUNTIME(cudaFree(min_in_rows));
  //   CUDA_RUNTIME(cudaFree(min_in_cols));

  //   CUDA_RUNTIME(cudaFree(zeros));
  //   CUDA_RUNTIME(cudaFree(zeros_size_b));
  //   CUDA_RUNTIME(cudaFree(row_of_star_at_column));
  //   CUDA_RUNTIME(cudaFree(column_of_star_at_row));
  //   CUDA_RUNTIME(cudaFree(cover_row));
  //   CUDA_RUNTIME(cudaFree(cover_column));
  //   CUDA_RUNTIME(cudaFree(column_of_prime_at_row));
  //   CUDA_RUNTIME(cudaFree(row_of_green_at_column));

  //   CUDA_RUNTIME(cudaFree(max_in_mat_row));
  //   CUDA_RUNTIME(cudaFree(max_in_mat_col));
  //   CUDA_RUNTIME(cudaFree(d_min_in_mat_vect));
  // }
};

// struct UNIFIED_HANDLE
// {
//   int zeros_size;
//   int n_matches;
//   bool goto_5;
//   bool repeat_kernel;
// };

__managed__ __device__ int zeros_size;     // The number fo zeros
__managed__ __device__ int n_matches;      // Used in step 3 to count the number of matches found
__managed__ __device__ bool goto_5;        // After step 4, goto step 5?
__managed__ __device__ bool repeat_kernel; // Needs to repeat the step 2 and step 4 kernel?
