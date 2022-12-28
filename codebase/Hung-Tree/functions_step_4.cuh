#pragma once
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include "device_launch_parameters.h"
#include "structures.h"
#include "variables.h"
#include "helper_utils.cuh"

// Device function for augmenting the alternating path from unassigned column to unassigned row.
__device__ void __augment(int *d_row_assignments, int *d_col_assignments, int *d_row_children, int *d_col_children, int init_rowid)
{
	int cur_colid = -1;
	int cur_rowid = init_rowid;

	while (cur_rowid != -1)
	{
		cur_colid = d_row_children[cur_rowid];

		d_row_assignments[cur_rowid] = cur_colid;
		d_col_assignments[cur_colid] = cur_rowid;

		cur_rowid = d_col_children[cur_colid];
	}
}

// Device function for traversing an alternating path from unassigned row to unassigned column.
__device__ void __reverse_traversal(int *d_row_visited, int *d_row_children, int *d_col_children, int *d_row_parents, int *d_col_parents, int init_colid)
{
	int cur_colid = init_colid;
	int cur_rowid = -1;

	while (cur_colid != -1)
	{
		d_col_children[cur_colid] = cur_rowid;

		cur_rowid = d_col_parents[cur_colid];

		d_row_children[cur_rowid] = cur_colid;
		cur_colid = d_row_parents[cur_rowid];
	}
	d_row_visited[cur_rowid] = AUGMENT;
}

// Kernel for constructing the predicates for reverse pass or augmentation candidates.
__global__ void kernel_augmentPredicateConstruction(Predicates d_predicates, int *d_visited, int offset, size_t size)
{
	size_t id = blockIdx.x * blockDim.x + threadIdx.x;

	// Copy the matrix into shared memory.
	int visited = (id < size) ? d_visited[id + offset] : DORMANT;
	bool predicate = (visited == REVERSE || visited == AUGMENT);
	long addr = predicate ? 1 : 0;

	if (id < size)
	{
		d_predicates.predicates[id] = predicate;
		d_predicates.addresses[id] = addr;
	}
}

// Kernel for scattering the vertices based on the scatter addresses.
__global__ void kernel_augmentScatter(Array d_vertex_ids, Predicates d_predicates, int offset, size_t size)
{
	size_t id = blockIdx.x * blockDim.x + threadIdx.x;

	bool predicate = (id < size) ? d_predicates.predicates[id] : false;
	long compid = (predicate) ? d_predicates.addresses[id] : -1; // compaction id.

	if (id < size)
	{
		if (predicate)
			d_vertex_ids.elements[compid] = id + offset;
	}
}

// Kernel for executing the reverse pass of the maximum matching algorithm.
__global__ void kernel_reverseTraversal(Array d_col_vertices, VertexData d_row_data, VertexData d_col_data)
{
	size_t id = blockIdx.x * blockDim.x + threadIdx.x;
	int size = d_col_vertices.size;

	int colid = (id < size) ? d_col_vertices.elements[id] : -1;

	if (id < size)
	{
		__reverse_traversal(d_row_data.is_visited, d_row_data.children, d_col_data.children, d_row_data.parents, d_col_data.parents, colid);
	}
}

// Kernel for executing the augmentation pass of the maximum matching algorithm.
__global__ void kernel_augmentation(int *d_row_assignments, int *d_col_assignments, Array d_row_vertices, VertexData d_row_data, VertexData d_col_data)
{
	size_t id = blockIdx.x * blockDim.x + threadIdx.x;
	size_t size = d_row_vertices.size;

	int rowid = (id < size) ? d_row_vertices.elements[id] : -1;

	if (id < size)
	{
		__augment(d_row_assignments, d_col_assignments, d_row_data.children, d_col_data.children, rowid);
	}
}

// Function for executing reverse pass of the maximum matching.
void reversePass(VertexData *d_row_data_dev, VertexData *d_col_data_dev, size_t N, unsigned int devid)
{

	cudaSetDevice(devID);

	int total_blocks = 0;
	dim3 blocks_per_grid;
	dim3 threads_per_block;
	calculateLinearDims(blocks_per_grid, threads_per_block, total_blocks, N);

	Array d_col_ids_csr;
	Predicates d_col_predicates; // predicates for compacting the colids eligible for the reverse pass.

	d_col_predicates.size = N;
	cudaSafeCall(cudaMalloc((void **)(&d_col_predicates.predicates), d_col_predicates.size * sizeof(bool)), "Error in cudaMalloc d_col_predicates.predicates");
	cudaSafeCall(cudaMalloc((void **)(&d_col_predicates.addresses), d_col_predicates.size * sizeof(long)), "Error in cudaMalloc d_col_predicates.addresses");
	cudaSafeCall(cudaMemset(d_col_predicates.predicates, false, d_col_predicates.size * sizeof(bool)), "Error in cudaMemset d_col_predicates.predicates");
	cudaSafeCall(cudaMemset(d_col_predicates.addresses, 0, d_col_predicates.size * sizeof(long)), "Error in cudaMemset d_col_predicates.addresses");

	// compact the reverse pass row vertices.
	execKernel(kernel_augmentPredicateConstruction, blocks_per_grid, threads_per_block, d_col_predicates, d_col_data_dev[devid].is_visited, 0, N);

	thrust::device_ptr<long> ptr(d_col_predicates.addresses);
	d_col_ids_csr.size = thrust::reduce(ptr, ptr + d_col_predicates.size); // calculate total number of vertices.
	thrust::exclusive_scan(ptr, ptr + d_col_predicates.size, ptr);				 // exclusive scan for calculating the scatter addresses.

	if (d_col_ids_csr.size > 0)
	{
		int total_blocks_1 = 0;
		dim3 blocks_per_grid_1;
		dim3 threads_per_block_1;
		calculateLinearDims(blocks_per_grid_1, threads_per_block_1, total_blocks_1, d_col_ids_csr.size);

		cudaSafeCall(cudaMalloc((void **)(&d_col_ids_csr.elements), d_col_ids_csr.size * sizeof(int)), "Error in cudaMalloc d_col_ids_csr.elements");

		execKernel(kernel_augmentScatter, blocks_per_grid, threads_per_block, d_col_ids_csr, d_col_predicates, 0, N);

		execKernel(kernel_reverseTraversal, blocks_per_grid_1, threads_per_block_1, d_col_ids_csr, d_row_data_dev[devid], d_col_data_dev[devid]);

		cudaSafeCall(cudaFree(d_col_ids_csr.elements), "Error in cudaFree d_col_ids_csr.elements");
	}

	cudaSafeCall(cudaFree(d_col_predicates.predicates), "Error in cudaFree d_col_predicates.predicates");
	cudaSafeCall(cudaFree(d_col_predicates.addresses), "Error in cudaFree d_col_predicates.addresses");
}

// Function for executing augmentation pass of the maximum matching.
void augmentationPass(Vertices *d_vertices_dev, VertexData *d_row_data_dev, VertexData *d_col_data_dev, size_t N, unsigned int devid)
{

	cudaSetDevice(devID);

	int total_blocks = 0;
	dim3 blocks_per_grid;
	dim3 threads_per_block;
	calculateLinearDims(blocks_per_grid, threads_per_block, total_blocks, N);

	Array d_row_ids_csr;
	Predicates d_row_predicates; // predicates for compacting the colids eligible for the augmentation pass.

	d_row_predicates.size = N;
	cudaSafeCall(cudaMalloc((void **)(&d_row_predicates.predicates), d_row_predicates.size * sizeof(bool)), "Error in cudaMalloc d_row_predicates.predicates");
	cudaSafeCall(cudaMalloc((void **)(&d_row_predicates.addresses), d_row_predicates.size * sizeof(long)), "Error in cudaMalloc d_row_predicates.addresses");
	cudaSafeCall(cudaMemset(d_row_predicates.predicates, false, d_row_predicates.size * sizeof(bool)), "Error in cudaMemset d_row_predicates.predicates");
	cudaSafeCall(cudaMemset(d_row_predicates.addresses, 0, d_row_predicates.size * sizeof(long)), "Error in cudaMemset d_row_predicates.addresses");

	// compact the reverse pass row vertices.
	execKernel(kernel_augmentPredicateConstruction, blocks_per_grid, threads_per_block, d_row_predicates, d_row_data_dev[devid].is_visited, 0, N);

	thrust::device_ptr<long> ptr(d_row_predicates.addresses);
	d_row_ids_csr.size = thrust::reduce(ptr, ptr + d_row_predicates.size); // calculate total number of vertices.
	thrust::exclusive_scan(ptr, ptr + d_row_predicates.size, ptr);				 // exclusive scan for calculating the scatter addresses.

	if (d_row_ids_csr.size > 0)
	{
		int total_blocks_1 = 0;
		dim3 blocks_per_grid_1;
		dim3 threads_per_block_1;
		calculateLinearDims(blocks_per_grid_1, threads_per_block_1, total_blocks_1, d_row_ids_csr.size);

		cudaSafeCall(cudaMalloc((void **)(&d_row_ids_csr.elements), d_row_ids_csr.size * sizeof(int)), "Error in cudaMalloc d_row_ids_csr.elements");

		execKernel(kernel_augmentScatter, blocks_per_grid, threads_per_block, d_row_ids_csr, d_row_predicates, 0, N);

		execKernel(kernel_augmentation, blocks_per_grid_1, threads_per_block_1, d_vertices_dev[devid].row_assignments, d_vertices_dev[devid].col_assignments, d_row_ids_csr, d_row_data_dev[devid], d_col_data_dev[devid]);

		cudaSafeCall(cudaFree(d_row_ids_csr.elements), "Error in cudaFree d_row_ids_csr.elements");
	}

	cudaSafeCall(cudaFree(d_row_predicates.predicates), "Error in cudaFree d_row_predicates.predicates");
	cudaSafeCall(cudaFree(d_row_predicates.addresses), "Error in cudaFree d_row_predicates.addresses");
}
