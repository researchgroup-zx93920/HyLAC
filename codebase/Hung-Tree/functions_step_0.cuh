#pragma once
#include <cuda_runtime_api.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include "structures.h"
#include "helper_utils.cuh"

// Kernel for reducing the rows by subtracting row minimum from each row element.
__global__ void kernel_rowReduction(double *d_costs, double *d_row_duals, size_t N)
{
	int rowid = blockIdx.x * blockDim.x + threadIdx.x;
	double min = INF;

	if (rowid < N)
	{
		for (int colid = 0; colid < N; colid++)
		{
			double val = d_costs[rowid * N + colid];
			if (val < min)
			{
				min = val;
			}
		}

		d_row_duals[rowid] = min;

		//		for(int colid = 0; colid < N; colid++)
		//		{
		//			d_costs[rowid * N + colid] -= min;
		//		}
	}
}

// Kernel for reducing the column by subtracting column minimum from each column element.
__global__ void kernel_columnReduction(double *d_costs, double *d_row_duals, double *d_col_duals, size_t N)
{
	int colid = blockIdx.x * blockDim.x + threadIdx.x;
	double min = INF;

	if (colid < N)
	{
		for (int rowid = 0; rowid < N; rowid++)
		{
			double val = d_costs[rowid * N + colid] - d_row_duals[rowid];
			if (val < min)
			{
				min = val;
			}
		}

		d_col_duals[colid] = min;

		//		for(int rowid = 0; rowid < N; rowid++)
		//		{
		//			d_costs[rowid * N + colid] -= min;
		//		}
	}
}

void initialReduction(Matrix *d_costs, size_t N, unsigned int devid)
{

	dim3 blocks_per_grid;
	dim3 threads_per_block;
	int total_blocks = 0;

	calculateLinearDims(blocks_per_grid, threads_per_block, total_blocks, N);

	execKernel(kernel_rowReduction, blocks_per_grid, threads_per_block,
						 d_costs[devid].elements, d_costs[devid].row_duals, N);

	execKernel(kernel_columnReduction, blocks_per_grid, threads_per_block,
						 d_costs[devid].elements, d_costs[devid].row_duals, d_costs[devid].col_duals, N);
}