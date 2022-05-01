// STEP 1: Subtract the row minimum from each row. Subtract the column minimum from each column.
//
// STEP 2: Find a zero of the slack matrix. If there are no starred zeros in its column or row star the zero.
// Repeat for each zero.
//
// STEP 3: Cover each column with a starred zero. If all the columns are
// covered then the matching is maximum.
//
// STEP 4: Find a non-covered zero and prime it. If there is no starred zero in the row containing this primed zero,
// Go to Step 5. Otherwise, cover this row and uncover the column containing the starred zero.
// Continue in this manner until there are no uncovered zeros left.
// Save the smallest uncovered value and Go to Step 6.
//
// STEP 5: Construct a series of alternating primed and starred zeros as follows:
// Let Z0 represent the uncovered primed zero found in Step 4.
// Let Z1 denote the starred zero in the column of Z0(if any).
// Let Z2 denote the primed zero in the row of Z1(there will always be one).
// Continue until the series terminates at a primed zero that has no starred zero in its column.
// Un-star each starred zero of the series, star each primed zero of the series,
// erase all primes and uncover every row in the matrix. Return to Step 3.
//
// STEP 6: Add the minimum uncovered value to every element of each covered row,
// and subtract it from every element of each uncovered column.
// Return to Step 4 without altering any stars, primes, or covered rows.

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>
#include <stdlib.h>
#include <stdio.h>
#include <random>
#include <assert.h>
#include <chrono>
#include "defs.cuh"
#include "iostream"

#define klog2(n) ((n < 8) ? 2 : ((n < 16) ? 3 : ((n < 32) ? 4 : ((n < 64) ? 5 : ((n < 128) ? 6 : ((n < 256) ? 7 : ((n < 512) ? 8 : ((n < 1024) ? 9 : ((n < 2048) ? 10 : ((n < 4096) ? 11 : ((n < 8192) ? 12 : ((n < 16384) ? 13 : 0))))))))))))
#ifndef DYNAMIC
#define MANAGED __managed__
#define dh_checkCuda checkCuda
#define dh_get_globaltime get_globaltime
#define dh_get_timer_period get_timer_period
#else
#define dh_checkCuda d_checkCuda
#define dh_get_globaltime d_get_globaltime
#define dh_get_timer_period d_get_timer_period
#define MANAGED
#endif

#define kmin(x, y) ((x < y) ? x : y)
#define kmax(x, y) ((x > y) ? x : y)

// User inputs: These values should be changed by the user
const int user_n = 4096; // This is the size of the cost matrix as supplied by the user
const double frac = 10;
const double epsilon = 0.0001; // used for comparisons for floating point numbers
typedef int data;			   // data type of weight matrix

const int n = 1 << (klog2(user_n - 1) + 1); // The size of the cost/pay matrix used in the algorithm that is increased to a power of two
const double range = frac * user_n;			// defines the range of the random matrix.
const int n_tests = 1;						// defines the number of tests performed

// End of user inputs

bool __device__ near_zero(const double &val)
{
	return ((val < epsilon) && (val > -epsilon));
}

bool __device__ near_zero(int val)
{
	return val == 0;
}

const int log2_n = klog2(n);				  // log2(n)
const int n_threads = kmin(n, 64);			  // Number of threads used in small kernels grid size (typically grid size equal to n)
											  // Used in steps 3ini, 3, 4ini, 4a, 4b, 5a and 5b (64)
const int n_threads_reduction = kmin(n, 256); // Number of threads used in the redution kernels in step 1 and 6 (256)
const int n_blocks_reduction = kmin(n, 256);  // Number of blocks used in the redution kernels in step 1 and 6 (256)
const int n_threads_full = kmin(n, 512);	  // Number of threads used the largest grids sizes (typically grid size equal to n*n)
											  // Used in steps 2 and 6 (512)
const int seed = 45345;						  // Initialization for the random number generator

const int n_blocks = n / n_threads;										   // Number of blocks used in small kernels grid size (typically grid size equal to n)
const int n_blocks_full = n * n / n_threads_full;						   // Number of blocks used the largest gris sizes (typically grid size equal to n*n)
const int row_mask = (1 << log2_n) - 1;									   // Used to extract the row from tha matrix position index (matrices are column wise)
const int nrows = n, ncols = n;											   // The matrix is square so the number of rows and columns is equal to n
const int max_threads_per_block = 1024;									   // The maximum number of threads per block
const int columns_per_block_step_4 = 512;								   // Number of columns per block in step 4
const int n_blocks_step_4 = kmax(n / columns_per_block_step_4, 1);		   // Number of blocks in step 4 and 2
const int data_block_size = columns_per_block_step_4 * n;				   // The size of a data block. Note that this can be bigger than the matrix size.
const int log2_data_block_size = log2_n + klog2(columns_per_block_step_4); // log2 of the size of a data block. Note that klog2 cannot handle very large sizes

// No need to change this for data types
#define MAX_DATA INT_MAX
#define MIN_DATA INT_MIN

// Host Variables

// Some host variables start with h_ to distinguish them from the corresponding device variables
// Device variables have no prefix.

#ifndef USE_TEST_MATRIX
data h_cost[ncols][nrows];
#else
data h_cost[n][n] = {{1, 2, 3, 4}, {2, 4, 6, 8}, {3, 6, 9, 12}, {4, 8, 12, 16}};
#endif
int h_column_of_star_at_row[nrows];
int h_zeros_vector_size;
int h_n_matches;
bool h_found;
bool h_goto_5;

// Device Variables

__device__ data slack[nrows * ncols];		  // The slack matrix
__device__ data min_in_rows[nrows];			  // Minimum in rows
__device__ data min_in_cols[ncols];			  // Minimum in columns
__device__ int zeros[nrows * ncols];		  // A vector with the position of the zeros in the slack matrix
__device__ int zeros_size_b[n_blocks_step_4]; // The number of zeros in block i

__device__ int row_of_star_at_column[ncols];  // A vector that given the column j gives the row of the star at that column (or -1, no star)
__device__ int column_of_star_at_row[nrows];  // A vector that given the row i gives the column of the star at that row (or -1, no star)
__device__ int cover_row[nrows];			  // A vector that given the row i indicates if it is covered (1- covered, 0- uncovered)
__device__ int cover_column[ncols];			  // A vector that given the column j indicates if it is covered (1- covered, 0- uncovered)
__device__ int column_of_prime_at_row[nrows]; // A vector that given the row i gives the column of the prime at that row  (or -1, no prime)
__device__ int row_of_green_at_column[ncols]; // A vector that given the row j gives the column of the green at that row (or -1, no green)

__device__ data max_in_mat_row[nrows];				   // Used in step 1 to stores the maximum in rows
__device__ data min_in_mat_col[ncols];				   // Used in step 1 to stores the minimums in columns
__device__ data d_min_in_mat_vect[n_blocks_reduction]; // Used in step 6 to stores the intermediate results from the first reduction kernel
__device__ data d_min_in_mat;						   // Used in step 6 to store the minimum

MANAGED __device__ int zeros_size;	   // The number fo zeros
MANAGED __device__ int n_matches;	   // Used in step 3 to count the number of matches found
MANAGED __device__ bool goto_5;		   // After step 4, goto step 5?
MANAGED __device__ bool repeat_kernel; // Needs to repeat the step 2 and step 4 kernel?
#if defined(DEBUG) || defined(_DEBUG)
MANAGED __device__ int n_covered_rows;	  // Used in debug mode to check for the number of covered rows
MANAGED __device__ int n_covered_columns; // Used in debug mode to check for the number of covered columns
#endif

__shared__ extern data sdata[]; // For access to shared memory

// -------------------------------------------------------------------------------------
// Device code
// -------------------------------------------------------------------------------------

#if defined(DEBUG) || defined(_DEBUG)
__global__ void convergence_check()
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (cover_column[i])
		atomicAdd((int *)&n_covered_columns, 1);
	if (cover_row[i])
		atomicAdd((int *)&n_covered_rows, 1);
}

#endif

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline __device__ cudaError_t d_checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
	if (result != cudaSuccess)
	{
		printf("CUDA Runtime Error: %s\n",
			   cudaGetErrorString(result));
		assert(result == cudaSuccess);
	}
#endif
	return result;
};

__global__ void init()
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	// initializations
	// for step 2
	if (i < nrows)
	{
		cover_row[i] = 0;
		column_of_star_at_row[i] = -1;
	}
	if (i < ncols)
	{
		cover_column[i] = 0;
		row_of_star_at_column[i] = -1;
	}
}

// STEP 1.
// a) Subtracting the row by the minimum in each row
const int n_rows_per_block = n / n_blocks_reduction;

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

__global__ void calc_min_in_rows()
{
	__shared__ data sdata[n_threads_reduction]; // One temporary result for each thread.

	unsigned int tid = threadIdx.x;
	unsigned int bid = blockIdx.x;
	// One gets the line and column from the blockID and threadID.
	unsigned int l = bid * n_rows_per_block + tid % n_rows_per_block;
	unsigned int c = tid / n_rows_per_block;
	unsigned int i = c * nrows + l;
	const unsigned int gridSize = n_threads_reduction * n_blocks_reduction;
	data thread_min = MAX_DATA;

	while (i < n * n)
	{
		thread_min = min(thread_min, slack[i]);
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
		min_in_rows[bid * n_rows_per_block + tid] = sdata[tid];
}

// a) Subtracting the column by the minimum in each column
const int n_cols_per_block = n / n_blocks_reduction;

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

__global__ void calc_min_in_cols()
{
	__shared__ data sdata[n_threads_reduction]; // One temporary result for each thread

	unsigned int tid = threadIdx.x;
	unsigned int bid = blockIdx.x;
	// One gets the line and column from the blockID and threadID.
	unsigned int c = bid * n_cols_per_block + tid % n_cols_per_block;
	unsigned int l = tid / n_cols_per_block;
	const unsigned int gridSize = n_threads_reduction * n_blocks_reduction;
	data thread_min = MAX_DATA;

	while (l < n)
	{
		unsigned int i = c * nrows + l;
		thread_min = min(thread_min, slack[i]);
		l += gridSize / n; // go to the next piece of the matrix...
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
		min_in_cols[bid * n_cols_per_block + tid] = sdata[tid];
}

__global__ void step_1_row_sub()
{

	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int l = i & row_mask;

	slack[i] = slack[i] - min_in_rows[l]; // subtract the minimum in row from that row
}

__global__ void step_1_col_sub()
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int c = i >> log2_n;
	slack[i] = slack[i] - min_in_cols[c]; // subtract the minimum in row from that row

	if (i == 0)
		zeros_size = 0;
	if (i < n_blocks_step_4)
		zeros_size_b[i] = 0;
}

// Compress matrix
__global__ void compress_matrix()
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (near_zero(slack[i]))
	{
		// atomicAdd(&zeros_size, 1);
		int b = i >> log2_data_block_size;
		int i0 = i & ~(data_block_size - 1); // == b << log2_data_block_size
		int j = atomicAdd(&zeros_size_b[b], 1);
		zeros[i0 + j] = i; // saves index of zeros in slack matrix per block
	}
}

__global__ void add_reduction()
{
	__shared__ int sdata[n_blocks_step_4];
	const int i = threadIdx.x;
	sdata[i] = zeros_size_b[i];
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

// STEP 2
// Find a zero of slack. If there are no starred zeros in its
// column or row star the zero. Repeat for each zero.

// The zeros are split through blocks of data so we run step 2 with several thread blocks and rerun the kernel if repeat was set to true.
__global__ void step_2()
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

		for (int j = i; j < zeros_size_b[b]; j += blockDim.x)
		{
			int z = zeros[(b << log2_data_block_size) + j];
			int l = z & row_mask;
			int c = z >> log2_n;

			if (cover_row[l] == 0 && cover_column[c] == 0)
			{
				// thread tries to get the line
				if (!atomicExch((int *)&(cover_row[l]), 1))
				{
					// only one thread gets the line
					if (!atomicExch((int *)&(cover_column[c]), 1))
					{
						// only one thread gets the column
						row_of_star_at_column[c] = l;
						column_of_star_at_row[l] = c;
					}
					else
					{
						cover_row[l] = 0;
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

// STEP 3
// uncover all the rows and columns before going to step 3
__global__ void step_3ini()
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	cover_row[i] = 0;
	cover_column[i] = 0;
	if (i == 0)
		n_matches = 0;
}

// Cover each column with a starred zero. If all the columns are
// covered then the matching is maximum
__global__ void step_3()
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	__shared__ int matches;
	if (threadIdx.x == 0)
		matches = 0;
	__syncthreads();
	if (row_of_star_at_column[i] >= 0)
	{
		cover_column[i] = 1;
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

__global__ void step_4_init()
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	column_of_prime_at_row[i] = -1;
	row_of_green_at_column[i] = -1;
}

__global__ void step_4()
{
	__shared__ bool s_found;
	__shared__ bool s_goto_5;
	__shared__ bool s_repeat_kernel;
	volatile int *v_cover_row = cover_row;
	volatile int *v_cover_column = cover_column;

	const int i = threadIdx.x;
	const int b = blockIdx.x;
	// int limit; my__syncthreads_init(limit);

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

		for (int j = threadIdx.x; j < zeros_size_b[b]; j += blockDim.x)
		{
			int z = zeros[(b << log2_data_block_size) + j];
			int l = z & row_mask; // row
			int c = z >> log2_n;  // column
			int c1 = column_of_star_at_row[l];

			// for (int n = 0; n < 10; n++)	??
			// {

			if (!v_cover_column[c] && !v_cover_row[l])
			{
				s_found = true; // find uncovered zero
				s_repeat_kernel = true;
				column_of_prime_at_row[l] = c; // prime the uncovered zero

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
__global__ void step_5a()
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	int r_Z0, c_Z0;

	c_Z0 = column_of_prime_at_row[i];
	if (c_Z0 >= 0 && column_of_star_at_row[i] < 0) // if primed and not covered
	{
		row_of_green_at_column[c_Z0] = i; // mark the column as green

		while ((r_Z0 = row_of_star_at_column[c_Z0]) >= 0)
		{
			c_Z0 = column_of_prime_at_row[r_Z0];
			row_of_green_at_column[c_Z0] = r_Z0;
		}
	}
}

// Applies the alternating paths
__global__ void step_5b()
{
	int j = blockDim.x * blockIdx.x + threadIdx.x;

	int r_Z0, c_Z0, c_Z2;

	r_Z0 = row_of_green_at_column[j];

	if (r_Z0 >= 0 && row_of_star_at_column[j] < 0)
	{

		c_Z2 = column_of_star_at_row[r_Z0];

		column_of_star_at_row[r_Z0] = j;
		row_of_star_at_column[j] = r_Z0;

		while (c_Z2 >= 0)
		{
			r_Z0 = row_of_green_at_column[c_Z2]; // row of Z2
			c_Z0 = c_Z2;						 // col of Z2
			c_Z2 = column_of_star_at_row[r_Z0];	 // col of Z4

			// star Z2
			column_of_star_at_row[r_Z0] = c_Z0;
			row_of_star_at_column[c_Z0] = r_Z0;
		}
	}
}

// STEP 6
// Add the minimum uncovered value to every element of each covered
// row, and subtract it from every element of each uncovered column.
// Return to Step 4 without altering any stars, primes, or covered lines.

template <unsigned int blockSize>
__device__ void min_warp_reduce(volatile data *sdata, int tid)
{
	if (blockSize >= 64 && tid + 32 < blockSize)
		sdata[tid] = min(sdata[tid], sdata[tid + 32]);
	if (blockSize >= 32 && tid + 16 < blockSize)
		sdata[tid] = min(sdata[tid], sdata[tid + 16]);
	if (blockSize >= 16 && tid + 8 < blockSize)
		sdata[tid] = min(sdata[tid], sdata[tid + 8]);
	if (blockSize >= 8 && tid + 4 < blockSize)
		sdata[tid] = min(sdata[tid], sdata[tid + 4]);
	if (blockSize >= 4 && tid + 2 < blockSize)
		sdata[tid] = min(sdata[tid], sdata[tid + 2]);
	if (blockSize >= 2 && tid + 1 < blockSize)
		sdata[tid] = min(sdata[tid], sdata[tid + 1]);
}

template <unsigned int blockSize> // blockSize is the size of a block of threads
__device__ void min_reduce1(volatile data *g_idata, volatile data *g_odata, unsigned int n)
{
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * (blockSize * 2) + tid;
	unsigned int gridSize = blockSize * 2 * gridDim.x;
	sdata[tid] = MAX_DATA;

	while (i < n)
	{
		int i1 = i;
		int i2 = i + blockSize;
		int l1 = i1 & row_mask;
		int c1 = i1 >> log2_n;
		data g1;
		if (cover_row[l1] == 1 || cover_column[c1] == 1)
			g1 = MAX_DATA;
		else
			g1 = g_idata[i1];
		int l2 = i2 & row_mask;
		int c2 = i2 >> log2_n;
		data g2;
		if (cover_row[l2] == 1 || cover_column[c2] == 1)
			g2 = MAX_DATA;
		else
			g2 = g_idata[i2];
		sdata[tid] = min(sdata[tid], min(g1, g2));
		i += gridSize;
	}

	__syncthreads();
	if (blockSize >= 1024)
	{
		if (tid < 512)
		{
			sdata[tid] = min(sdata[tid], sdata[tid + 512]);
		}
		__syncthreads();
	}
	if (blockSize >= 512)
	{
		if (tid < 256)
		{
			sdata[tid] = min(sdata[tid], sdata[tid + 256]);
		}
		__syncthreads();
	}
	if (blockSize >= 256)
	{
		if (tid < 128)
		{
			sdata[tid] = min(sdata[tid], sdata[tid + 128]);
		}
		__syncthreads();
	}
	if (blockSize >= 128)
	{
		if (tid < 64)
		{
			sdata[tid] = min(sdata[tid], sdata[tid + 64]);
		}
		__syncthreads();
	}
	if (tid < 32)
		min_warp_reduce<blockSize>(sdata, tid);
	if (tid == 0)
		g_odata[blockIdx.x] = sdata[0];
}

template <unsigned int blockSize>
__device__ void min_reduce2(volatile data *g_idata, volatile data *g_odata, unsigned int n)
{
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * (blockSize * 2) + tid;

	sdata[tid] = min(g_idata[i], g_idata[i + blockSize]);

	__syncthreads();
	if (blockSize >= 1024)
	{
		if (tid < 512)
		{
			sdata[tid] = min(sdata[tid], sdata[tid + 512]);
		}
		__syncthreads();
	}
	if (blockSize >= 512)
	{
		if (tid < 256)
		{
			sdata[tid] = min(sdata[tid], sdata[tid + 256]);
		}
		__syncthreads();
	}
	if (blockSize >= 256)
	{
		if (tid < 128)
		{
			sdata[tid] = min(sdata[tid], sdata[tid + 128]);
		}
		__syncthreads();
	}
	if (blockSize >= 128)
	{
		if (tid < 64)
		{
			sdata[tid] = min(sdata[tid], sdata[tid + 64]);
		}
		__syncthreads();
	}
	if (tid < 32)
		min_warp_reduce<blockSize>(sdata, tid);
	if (tid == 0)
		g_odata[blockIdx.x] = sdata[0];
}

__global__ void step_6_init()
{
	if (threadIdx.x == 0)
		zeros_size = 0;
	zeros_size_b[threadIdx.x] = 0;
}

__global__ void step_6_add_sub_fused_compress_matrix()
{
	// STEP 6:
	/*STEP 6: Add the minimum uncovered value to every element of each covered
	row, and subtract it from every element of each uncovered column.
	Return to Step 4 without altering any stars, primes, or covered lines. */
	const int i = blockDim.x * blockIdx.x + threadIdx.x;
	const int l = i & row_mask;
	const int c = i >> log2_n;
	auto reg = slack[i];
	switch (cover_row[l] + cover_column[c])
	{
	case 2:
		reg += d_min_in_mat;
		slack[i] = reg;
		break;
	case 0:
		reg -= d_min_in_mat;
		slack[i] = reg;
		break;
	default:
		break;
	}

	// compress matrix
	if (near_zero(reg))
	{
		int b = i >> log2_data_block_size;
		int i0 = i & ~(data_block_size - 1); // == b << log2_data_block_size
		int j = atomicAdd(zeros_size_b + b, 1);
		zeros[i0 + j] = i;
	}
}

__global__ void min_reduce_kernel1()
{
	min_reduce1<n_threads_reduction>(slack, d_min_in_mat_vect, nrows * ncols);
}

__global__ void min_reduce_kernel2()
{
	min_reduce2<n_threads_reduction / 2>(d_min_in_mat_vect, &d_min_in_mat, n_blocks_reduction);
}

__device__ inline long long int d_get_globaltime(void)
{
	long long int ret;

	asm volatile("mov.u64 %0, %%globaltimer;"
				 : "=l"(ret));

	return ret;
}

// Returns the period in miliseconds
__device__ inline double d_get_timer_period(void)
{
	return 1.0e-6;
}

// -------------------------------------------------------------------------------------
// Host code
// -------------------------------------------------------------------------------------

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
	if (result != cudaSuccess)
	{
		printf("CUDA Runtime Error: %s\n",
			   cudaGetErrorString(result));
		assert(result == cudaSuccess);
	}
#endif
	return result;
};

typedef std::chrono::high_resolution_clock::rep hr_clock_rep;

inline hr_clock_rep get_globaltime(void)
{
	using namespace std::chrono;
	return high_resolution_clock::now().time_since_epoch().count();
}

// Returns the period in miliseconds
inline double get_timer_period(void)
{
	using namespace std::chrono;
	return 1000.0 * high_resolution_clock::period::num / high_resolution_clock::period::den;
}

#define declare_kernel(k)      \
	hr_clock_rep k##_time = 0; \
	int k##_runs = 0

#define call_kernel(k, n_blocks, n_threads) call_kernel_s(k, n_blocks, n_threads, 0ll)

#define call_kernel_s(k, n_blocks, n_threads, shared) \
	{                                                 \
		timer_start = dh_get_globaltime();            \
		k<<<n_blocks, n_threads, shared>>>();         \
		dh_checkCuda(cudaDeviceSynchronize());        \
		timer_stop = dh_get_globaltime();             \
		k##_time += timer_stop - timer_start;         \
		k##_runs++;                                   \
	}
// printf("Finished kernel " #k "(%d,%d,%lld)\n", n_blocks, n_threads, shared);			\
// fflush(0);											\

#define kernel_stats(k) \
	printf(#k "\t %g \t %d\n", dh_get_timer_period() * k##_time, k##_runs)

void Hungarian_Algorithm()
{
	hr_clock_rep timer_start, timer_stop;

	declare_kernel(init);
	declare_kernel(calc_min_in_rows);
	declare_kernel(step_1_row_sub);
	declare_kernel(calc_min_in_cols);
	declare_kernel(step_1_col_sub);
	declare_kernel(compress_matrix);
	declare_kernel(add_reduction);
	declare_kernel(step_2);
	declare_kernel(step_3ini);
	declare_kernel(step_3);
	declare_kernel(step_4_init);
	declare_kernel(step_4);
	declare_kernel(min_reduce_kernel1);
	declare_kernel(min_reduce_kernel2);
	declare_kernel(step_6_init);
	declare_kernel(step_6_add_sub_fused_compress_matrix);
	declare_kernel(step_5a);
	declare_kernel(step_5b);

	// Initialization
	call_kernel(init, n_blocks, n_threads);

	// Step 1 kernels
	call_kernel(calc_min_in_rows, n_blocks_reduction, n_threads_reduction);
	call_kernel(step_1_row_sub, n_blocks_full, n_threads_full);
	call_kernel(calc_min_in_cols, n_blocks_reduction, n_threads_reduction);
	call_kernel(step_1_col_sub, n_blocks_full, n_threads_full);

	// compress_matrix
	call_kernel(compress_matrix, n_blocks_full, n_threads_full);
	call_kernel(add_reduction, 1, n_blocks_step_4);

	// Step 2 kernels
	do
	{
		repeat_kernel = false;
		call_kernel(step_2, n_blocks_step_4, (n_blocks_step_4 > 1 || zeros_size > max_threads_per_block) ? max_threads_per_block : zeros_size);
		// If we have more than one block it means that we have 512 lines per block so 1024 threads should be adequate.
	} while (repeat_kernel);

	while (1)
	{ // repeat steps 3 to 6

		// Step 3 kernels
		call_kernel(step_3ini, n_blocks, n_threads);
		call_kernel(step_3, n_blocks, n_threads);

		if (n_matches >= ncols)
			break; // It's done

		// step 4_kernels
		call_kernel(step_4_init, n_blocks, n_threads);

		while (1) // repeat step 4 and 6
		{
			do
			{ // step 4 loop
				goto_5 = false;
				repeat_kernel = false;
				dh_checkCuda(cudaDeviceSynchronize());
				printf("Number of zeros %d\n", zeros_size);
				call_kernel(step_4, n_blocks_step_4, (n_blocks_step_4 > 1 || zeros_size > max_threads_per_block) ? max_threads_per_block : zeros_size);
				// If we have more than one block it means that we have 512 lines per block so 1024 threads should be adequate.

			} while (repeat_kernel && !goto_5);

			if (goto_5)
				break;

			// step 6_kernel
			call_kernel_s(min_reduce_kernel1, n_blocks_reduction, n_threads_reduction, n_threads_reduction * sizeof(data));
			call_kernel_s(min_reduce_kernel2, 1, n_blocks_reduction / 2, (n_blocks_reduction / 2) * sizeof(data));
			call_kernel(step_6_init, 1, n_blocks_step_4);
			call_kernel(step_6_add_sub_fused_compress_matrix, n_blocks_full, n_threads_full);

			// compress_matrix
			//  call_kernel(compress_matrix, n_blocks_full, n_threads_full);
			call_kernel(add_reduction, 1, n_blocks_step_4);

		} // repeat step 4 and 6

		call_kernel(step_5a, n_blocks, n_threads);
		call_kernel(step_5b, n_blocks, n_threads);

	} // repeat steps 3 to 6

	// total_time_stop = dh_get_globaltime();

	// printf("kernel \t time (ms) \t runs\n");

	kernel_stats(init);
	kernel_stats(calc_min_in_rows);
	kernel_stats(step_1_row_sub);
	kernel_stats(calc_min_in_cols);
	kernel_stats(step_1_col_sub);
	kernel_stats(compress_matrix);
	kernel_stats(add_reduction);
	kernel_stats(step_2);
	kernel_stats(step_3ini);
	kernel_stats(step_3);
	kernel_stats(step_4_init);
	kernel_stats(step_4);
	kernel_stats(min_reduce_kernel1);
	kernel_stats(min_reduce_kernel2);
	kernel_stats(step_6_add_sub_fused_compress_matrix);
	kernel_stats(step_6_init);
	kernel_stats(step_5a);
	kernel_stats(step_5b);

	// printf("Total time(ms) \t %g\n", dh_get_timer_period() * (total_time_stop - total_time_start));
}

int main(int argc, char **argv)
{

	cudaSetDevice(1);
	// Constant checks:
	check(n == (1 << log2_n), "Incorrect log2_n!");
	check(n_threads * n_blocks == n, "n_threads*n_blocks != n\n");
	// step 1
	check(n_blocks_reduction <= n, "Step 1: Should have several lines per block!");
	check(n % n_blocks_reduction == 0, "Step 1: Number of lines per block should be integer!");
	check((n_blocks_reduction * n_threads_reduction) % n == 0, "Step 1: The grid size must be a multiple of the line size!");
	check(n_threads_reduction * n_blocks_reduction <= n * n, "Step 1: The grid size is bigger than the matrix size!");
	// step 6
	check(n_threads_full * n_blocks_full <= n * n, "Step 6: The grid size is bigger than the matrix size!");
	check(columns_per_block_step_4 * n == (1 << log2_data_block_size), "Columns per block of step 4 is not a power of two!");

	default_random_engine generator(seed);
	uniform_int_distribution<int> distribution(0, range - 1);

	long long total_time = 0;
	for (int test = 0; test < n_tests; test++)
	{
		// printf("\n\n\n\ntest %d\n", test);
		// fflush(file);
		for (int c = 0; c < ncols; c++)
		{
			for (int r = 0; r < nrows; r++)
			{
				if (c < user_n && r < user_n)
				{
					// if (r % user_n == 0 && c >0)
					// 	printf("\n");
					double gen = distribution(generator);
					h_cost[c][r] = gen;
					// cout << gen << "\t";
				}
				else
				{
					if (c == r)
						h_cost[c][r] = 0;
					else
						h_cost[c][r] = MAX_DATA;
				}
			}
		}
		// printf("\n");

		// Copy vectors from host memory to device memory
		cudaMemcpyToSymbol(slack, h_cost, sizeof(data) * nrows * ncols); // symbol refers to the device memory hence "To" means from Host to Device

		// Invoke kernels
		typedef std::chrono::high_resolution_clock clock;

		cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 1024 * 1024 * 1024);

		auto start = clock::now();

		Hungarian_Algorithm();

		checkCuda(cudaDeviceSynchronize());

		auto elapsed = clock::now() - start;
		// fflush(file);

		// Copy assignments from Device to Host and calculate the total Cost
		cudaMemcpyFromSymbol(h_column_of_star_at_row, column_of_star_at_row, nrows * sizeof(int));

		double total_cost = 0;
		for (int r = 0; r < nrows; r++)
		{
			int c = h_column_of_star_at_row[r];
			if (c >= 0)
				total_cost += h_cost[c][r];
			// printf("r = %d, c = %d\n", r, c);
		}

		printf("Total cost: \t %f \n", total_cost);
		long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
		total_time += microseconds / n_tests;
	}
	cout << "Time taken: \t" << total_time / 1000.0f << " ms" << endl;
}
