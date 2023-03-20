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

__global__ void kernel_step3_init(int *d_vertex_ids, int row_count)
{
	size_t id = blockIdx.x * blockDim.x + threadIdx.x;

	if (id < row_count)
	{
		d_vertex_ids[id] = id;
	}
}

// Kernel for calculating predicates for row vertices.
__global__ void kernel_vertexPredicateConstructionCSR(Predicates d_vertex_predicates, Array d_vertices_csr_in, int *d_visited)
{

	size_t id = blockIdx.x * blockDim.x + threadIdx.x;
	size_t size = d_vertices_csr_in.size;

	int vertexid = (id < size) ? d_vertices_csr_in.elements[id] : -1;
	int visited = (id < size && vertexid != -1) ? d_visited[vertexid] : DORMANT;

	bool predicate = (visited == ACTIVE); // If vertex is not visited then it is added to frontier queue.
	long addr = predicate ? 1 : 0;

	if (id < size)
	{
		d_vertex_predicates.predicates[id] = predicate;
		d_vertex_predicates.addresses[id] = addr;
	}
}

// Kernel for scattering the vertices based on the scatter addresses.
__global__ void kernel_vertexScatterCSR(int *d_vertex_ids_csr, int *d_vertex_ids,
																				int *d_visited, Predicates d_vertex_predicates)
{
	size_t id = blockIdx.x * blockDim.x + threadIdx.x;
	size_t size = d_vertex_predicates.size;

	// Copy the matrix into shared memory.
	int vertexid = (id < size) ? d_vertex_ids[id] : -1;
	bool predicate = (id < size) ? d_vertex_predicates.predicates[id] : false;
	long compid = (predicate) ? d_vertex_predicates.addresses[id] : -1; // compaction id.

	if (id < size)
	{
		if (predicate)
		{
			d_vertex_ids_csr[compid] = vertexid;
			d_visited[id] = VISITED;
		}
	}
}

// Device function for traversing the neighbors from start pointer to end pointer and updating the covers.
// The function sets d_next to 4 if there are uncovered zeros, indicating the requirement of Step 4 execution.
__device__ void __traverse(Matrix d_costs, Vertices d_vertices, bool *d_flag,
													 int *d_row_parents, int *d_col_parents, int *d_row_visited,
													 int *d_col_visited, double *d_slacks, int *d_start_ptr, int *d_end_ptr, int colid, size_t N)
{
	int *ptr1 = d_start_ptr;

	while (ptr1 != d_end_ptr)
	{
		int rowid = *ptr1;

		double slack = d_costs.elements[rowid * N + colid] - d_costs.row_duals[rowid] - d_costs.col_duals[colid];

		int nxt_rowid = d_vertices.col_assignments[colid];

		if (rowid != nxt_rowid && d_vertices.col_covers[colid] == 0)
		{

			if (slack < d_slacks[colid])
			{

				d_slacks[colid] = slack;
				d_col_parents[colid] = rowid;
			}

			if (near_zero(d_slacks[colid]))
			{

				if (nxt_rowid != -1)
				{
					d_row_parents[nxt_rowid] = colid; // update parent info

					d_vertices.row_covers[nxt_rowid] = 0;
					d_vertices.col_covers[colid] = 1;

					if (d_row_visited[nxt_rowid] != VISITED)
						d_row_visited[nxt_rowid] = ACTIVE;
				}

				else
				{
					d_col_visited[colid] = REVERSE;
					*d_flag = true;
				}
			}
		}
		d_row_visited[rowid] = VISITED;
		ptr1++;
	}
}

// Kernel for finding the minimum zero cover.
__global__ void kernel_coverAndExpand(bool *d_flag, Array d_vertices_csr_in, Matrix d_costs,
																			Vertices d_vertices, VertexData d_row_data, VertexData d_col_data, int N)
{
	size_t id = blockIdx.x * blockDim.x + threadIdx.x;

	// Load values into local memory
	int in_size = d_vertices_csr_in.size;
	int *st_ptr = d_vertices_csr_in.elements;
	int *end_ptr = d_vertices_csr_in.elements + in_size;

	if (id < N)
	{
		__traverse(d_costs, d_vertices, d_flag, d_row_data.parents, d_col_data.parents,
							 d_row_data.is_visited, d_col_data.is_visited, d_col_data.slack, st_ptr, end_ptr, id, N);
	}
}

// Function for compacting row vertices. Used in Step 3 (minimum zero cover).
void compactRowVertices(Predicates &d_vertex_predicates, VertexData *d_row_data_dev,
												Array &d_vertices_csr_out, Array &d_vertices_csr_in, unsigned int devid)
{

	cudaSetDevice(devID);

	int total_blocks = 0;
	dim3 blocks_per_grid;
	dim3 threads_per_block;

	// Predicates d_vertex_predicates;

	// d_vertex_predicates.size = d_vertices_csr_in.size;

	// printf("size: %u\n", d_vertices_csr_in.size);

	cudaSafeCall(cudaMemset(d_vertex_predicates.predicates, false, d_vertex_predicates.size * sizeof(bool)),
							 "Error in cudaMemset d_vertex_predicates.predicates");
	cudaSafeCall(cudaMemset(d_vertex_predicates.addresses, 0, d_vertex_predicates.size * sizeof(long)),
							 "Error in cudaMemset d_vertex_predicates.addresses");

	calculateLinearDims(blocks_per_grid, threads_per_block, total_blocks, d_vertices_csr_in.size);
	execKernel(kernel_vertexPredicateConstructionCSR, blocks_per_grid, threads_per_block,
						 d_vertex_predicates, d_vertices_csr_in, d_row_data_dev[devid].is_visited);

	thrust::device_ptr<long> ptr(d_vertex_predicates.addresses);
	d_vertices_csr_out.size = thrust::reduce(ptr, ptr + d_vertex_predicates.size); // calculate total number of vertices.
	thrust::exclusive_scan(ptr, ptr + d_vertex_predicates.size, ptr);							 // exclusive scan for calculating the scatter addresses.
	cudaSafeCall(cudaDeviceSynchronize(), "Sync Error");

	if (d_vertices_csr_out.size > 0)
	{
		cudaSafeCall(cudaMalloc((void **)(&d_vertices_csr_out.elements), d_vertices_csr_out.size * sizeof(int)),
								 "Error in cudaMalloc d_vertices_csr_out.elements");

		execKernel(kernel_vertexScatterCSR, blocks_per_grid, threads_per_block,
							 d_vertices_csr_out.elements, d_vertices_csr_in.elements,
							 d_row_data_dev[0].is_visited, d_vertex_predicates);
	}
}

// Function for covering the zeros in uncovered rows and expanding the frontier.
void coverZeroAndExpand(Matrix *d_costs_dev, Vertices *d_vertices_dev, VertexData *d_row_data_dev, VertexData *d_col_data_dev,
												Array &d_vertices_csr_in, bool *goto_4, size_t N, unsigned int devid)
{
	cudaSetDevice(devID);

	int total_blocks = 0;
	dim3 blocks_per_grid;
	dim3 threads_per_block;

	calculateLinearDims(blocks_per_grid, threads_per_block, total_blocks, N);

	execKernel(kernel_coverAndExpand, blocks_per_grid, threads_per_block,
						 goto_4, d_vertices_csr_in, d_costs_dev[devid],
						 d_vertices_dev[devid], d_row_data_dev[devid], d_col_data_dev[devid], N);
}

// Function for executing recursive zero cover. Returns the next step (Step 4 or Step 5) depending on the presence of uncovered zeros.
void executeZeroCover(Matrix *d_costs_dev, Vertices *d_vertices_dev, VertexData *d_row_data_dev,
											Predicates d_vertex_predicates, Array &d_vertices_csr1,
											VertexData *d_col_data_dev, bool *goto_4, size_t N, unsigned int devid)
{

	Array d_vertices_csr2;

	int total_blocks = 0;
	dim3 blocks_per_grid;
	dim3 threads_per_block;

	int size = N;
	// int start = 0;

	d_vertices_csr1.size = size;

	calculateLinearDims(blocks_per_grid, threads_per_block, total_blocks, size);

	// sets each element to its index
	execKernel(kernel_step3_init, blocks_per_grid, threads_per_block, d_vertices_csr1.elements, d_vertices_csr1.size);

	// Performs BFS
	while (true)
	{
		compactRowVertices(d_vertex_predicates, d_row_data_dev, d_vertices_csr2, d_vertices_csr1, devid); // compact the current vertex frontier.

		if (d_vertices_csr2.size == 0)
			break;
		// traverse the frontier, cover zeros and expand.
		coverZeroAndExpand(d_costs_dev, d_vertices_dev, d_row_data_dev, d_col_data_dev, d_vertices_csr2, goto_4, N, devid);
		cudaSafeCall(cudaFree(d_vertices_csr2.elements), "Error in cudaFree d_vertices_csr2.elements");
	}
}
