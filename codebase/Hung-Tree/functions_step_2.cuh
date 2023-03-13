#pragma once
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include "variables.h"
#include "structures.h"
#include "helper_utils.cuh"

// Kernel for populating the assignment arrays and cover arrays.
__global__ void kernel_computeRowCovers(int *d_row_assignments, int *d_row_covers, int row_count)
{
	int rowid = blockIdx.x * blockDim.x + threadIdx.x;

	// Copy the predicate matrix back to global memory
	if (rowid < row_count)
	{
		if (d_row_assignments[rowid] != -1)
		{
			d_row_covers[rowid] = 1;
		}
	}
}

// Kernel for initializing the row or column vertices, later used for recursive frontier update (in Step 3).
__global__ void kernel_rowInitialization(int *d_visited, int *d_row_assignments, int row_start, int row_count)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < row_count)
	{
		int assignment = d_row_assignments[id + row_start];
		d_visited[id] = (assignment == -1) ? ACTIVE : DORMANT;
	}
}

// Function for initializing all devices for execution of Step 2.
void initializeStep2(Vertices h_vertices, Vertices *d_vertices_dev, VertexData *d_row_data_dev, VertexData *d_col_data_dev, size_t N, unsigned int devid)
{
	cudaSetDevice(devID);

	int total_blocks = 0;
	dim3 blocks_per_grid;
	dim3 threads_per_block;

	size_t size = N;
	int start = 0;

	// Not needed to do it all the times!
	// cudaSafeCall(cudaMemcpy(h_vertices.row_assignments, d_vertices_dev[devid].row_assignments, N * sizeof(int), cudaMemcpyDeviceToHost), "Error in cudaMemcpy d_vertices_dev[devid].row_assignments");
	// cudaSafeCall(cudaMemcpy(h_vertices.col_assignments, d_vertices_dev[devid].col_assignments, N * sizeof(int), cudaMemcpyDeviceToHost), "Error in cudaMemcpy d_vertices_dev[devid].col_assignments");

	cudaSafeCall(cudaMemset(d_vertices_dev[devid].row_covers, 0, N * sizeof(int)), "Error in cudaMemset d_row_covers");
	cudaSafeCall(cudaMemset(d_vertices_dev[devid].col_covers, 0, N * sizeof(int)), "Error in cudaMemset d_col_covers");

	cudaSafeCall(cudaMemset(d_row_data_dev[devid].is_visited, DORMANT, N * sizeof(int)), "Error in cudaMemset d_row_data.is_visited");
	cudaSafeCall(cudaMemset(d_col_data_dev[devid].is_visited, DORMANT, N * sizeof(int)), "Error in cudaMemset d_col_data.is_visited"); // initialize "visited" array for columns. later used in BFS (Step 4).
	cudaSafeCall(cudaMemset(d_col_data_dev[devid].slack, INF, N * sizeof(double)), "Error in cudaMemset d_col_data.slack");

	cudaSafeCall(cudaMemset(d_row_data_dev[devid].parents, -1, N * sizeof(int)), "Error in cudaMemset d_row_data.parents");
	cudaSafeCall(cudaMemset(d_row_data_dev[devid].children, -1, N * sizeof(int)), "Error in cudaMemset d_row_data.children");
	cudaSafeCall(cudaMemset(d_col_data_dev[devid].parents, -1, N * sizeof(int)), "Error in cudaMemset d_col_data.parents");
	cudaSafeCall(cudaMemset(d_col_data_dev[devid].children, -1, N * sizeof(int)), "Error in cudaMemset d_col_data.children");

	calculateLinearDims(blocks_per_grid, threads_per_block, total_blocks, size);
	execKernel(kernel_rowInitialization, blocks_per_grid, threads_per_block,
						 d_row_data_dev[devid].is_visited, d_vertices_dev[devid].row_assignments, start, size);
}

// Function for finding row cover on individual devices.
int computeRowCovers(Vertices *d_vertices_dev, size_t N, unsigned int devid)
{

	dim3 blocks_per_grid;
	dim3 threads_per_block;
	int total_blocks = 0;

	calculateLinearDims(blocks_per_grid, threads_per_block, total_blocks, N);
	execKernel(kernel_computeRowCovers, blocks_per_grid, threads_per_block,
						 d_vertices_dev[devid].row_assignments, d_vertices_dev[devid].row_covers, N);

	thrust::device_ptr<int> ptr(d_vertices_dev[devid].row_covers);

	int cover_count = thrust::reduce(ptr, ptr + N);

	return cover_count;
}

// Function for copying row cover array back to each device.
void updateRowCovers(Vertices *d_vertices_dev, int *h_row_covers, size_t N, unsigned int devid)
{
	cudaSetDevice(devid);
	cudaSafeCall(cudaMemcpy(d_vertices_dev[devid].row_covers, h_row_covers, N * sizeof(int), cudaMemcpyHostToDevice), "Error in cudaMemcpy h_row_covers");
}