#pragma once
#include <cuda_runtime_api.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include "structures.h"
#include "helper_utils.cuh"

#define near_zero(cost) (cost > -EPSILON && cost < EPSILON)

// Kernel for calculating initial assignments.
__global__ void kernel_computeInitialAssignments(double *d_costs, double *d_row_duals, double *d_col_duals,
																								 int *d_row_assignments, int *d_col_assignments, int *d_row_lock, int *d_col_lock, size_t N)
{
	int colid = blockIdx.x * blockDim.x + threadIdx.x;

	if (colid < N)
	{
		for (int rowid = 0; rowid < N; rowid++)
		{
			if (d_col_lock[colid] == 1)
				break;

			double cost = d_costs[rowid * N + colid] - d_row_duals[rowid] - d_col_duals[colid];

			if (near_zero(cost))
			{
				if (atomicCAS(&d_row_lock[rowid], 0, 1) == 0)
				{
					d_row_assignments[rowid] = colid;
					d_col_assignments[colid] = rowid;
					d_col_lock[colid] = 1;
				}
			}
		}
	}
}

// Function for calculating initial assignments on individual cards and stitcing them together on host.
void computeInitialAssignments(Matrix *d_costs, Vertices *d_vertices_dev, size_t N, unsigned int devid)
{

	dim3 blocks_per_grid;
	dim3 threads_per_block;
	int total_blocks = 0;

	calculateLinearDims(blocks_per_grid, threads_per_block, total_blocks, N);

	cudaSafeCall(cudaMemset(d_vertices_dev[devid].row_assignments, -1, N * sizeof(int)), "Error in cudaMemset d_row_assignment");
	cudaSafeCall(cudaMemset(d_vertices_dev[devid].col_assignments, -1, N * sizeof(int)), "Error in cudaMemset d_col_assignment");

	int *d_row_lock, *d_col_lock;
	cudaSafeCall(cudaMalloc(&d_row_lock, N * sizeof(int)), "Error in cudaMalloc d_row_lock");
	cudaSafeCall(cudaMalloc(&d_col_lock, N * sizeof(int)), "Error in cudaMalloc d_col_lock");
	cudaSafeCall(cudaMemset(d_row_lock, 0, N * sizeof(int)), "Error in cudaMemset d_row_lock");
	cudaSafeCall(cudaMemset(d_col_lock, 0, N * sizeof(int)), "Error in cudaMemset d_col_lock");

	execKernel(kernel_computeInitialAssignments, blocks_per_grid, threads_per_block,
						 d_costs[devid].elements, d_costs[devid].row_duals, d_costs[devid].col_duals,
						 d_vertices_dev[devid].row_assignments, d_vertices_dev[devid].col_assignments,
						 d_row_lock, d_col_lock, N);

	cudaSafeCall(cudaFree(d_row_lock), "Error in cudaFree d_row_lock");
	cudaSafeCall(cudaFree(d_col_lock), "Error in cudaFree d_col_lock");
}