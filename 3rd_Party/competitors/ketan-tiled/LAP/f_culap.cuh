#pragma once
#include <cuda.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "d_structs.h"
#include "f_cutils.cuh"

// Device function for traversing the neighbors from start pointer to end pointer and updating the covers.
// The function sets d_next to 4 if there are uncovered zeros, indicating the requirement of Step 4 execution.
__device__ void __traverse(Matrix d_costs, Vertices d_vertices, bool *d_flag, int *d_row_parents, int *d_col_parents, int *d_row_visited, int *d_col_visited, int *d_start_ptr, int *d_end_ptr, int spid, int colid, int SP, int N)
{
	int *ptr1 = d_start_ptr;

	while (ptr1 != d_end_ptr)
	{
		int rowid = *ptr1;

		int ROWID = spid * N + rowid;
		int COLID = spid * N + colid;

		double slack = d_costs.elements[spid * N * N + rowid * N + colid] - d_vertices.row_duals[ROWID] - d_vertices.col_duals[COLID];

		int nxt_rowid = d_vertices.col_assignments[COLID];
		int NXT_ROWID = spid * N + nxt_rowid;

		if (rowid != nxt_rowid && d_vertices.col_covers[COLID] == 0)
		{

			if (slack < d_vertices.col_slacks[COLID])
			{

				d_vertices.col_slacks[COLID] = slack;
				d_col_parents[COLID] = ROWID;
			}

			if (d_vertices.col_slacks[COLID] < EPSILON && d_vertices.col_slacks[COLID] > -EPSILON)
			{

				if (nxt_rowid != -1)
				{
					d_row_parents[NXT_ROWID] = COLID; // update parent info

					d_vertices.row_covers[NXT_ROWID] = 0;
					d_vertices.col_covers[COLID] = 1;

					if (d_row_visited[NXT_ROWID] != VISITED)
						d_row_visited[NXT_ROWID] = ACTIVE;
				}

				else
				{
					d_col_visited[COLID] = REVERSE;
					*d_flag = true;
				}
			}
		}
		d_row_visited[ROWID] = VISITED;
		ptr1++;
	}
}

// Device function for traversing an alternating path from unassigned row to unassigned column.
__device__ void __reverse_traversal(int *d_row_visited, int *d_row_children, int *d_col_children, int *d_row_parents, int *d_col_parents, int COLID)
{
	int cur_colid = COLID;
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

// Device function for augmenting the alternating path from unassigned column to unassigned row.
__device__ void __augment(int *d_row_assignments, int *d_col_assignments, int *d_row_children, int *d_col_children, int ROWID, int N)
{
	int cur_colid = -1;
	int cur_rowid = ROWID;

	while (cur_rowid != -1)
	{
		cur_colid = d_row_children[cur_rowid];

		d_row_assignments[cur_rowid] = cur_colid % N; // true colid
		d_col_assignments[cur_colid] = cur_rowid % N; // true rowid

		cur_rowid = d_col_children[cur_colid];
	}
}

// Kernel for reducing the rows by subtracting row minimum from each row element.
__global__ void kernel_rowReduction(double *d_costs, double *d_row_duals, double *d_col_duals, int SP, int N)
{

	int spid = blockIdx.y * blockDim.y + threadIdx.y;
	int rowid = blockIdx.x * blockDim.x + threadIdx.x;
	double min = INF;

	if (spid < SP && rowid < N)
	{

		for (int colid = 0; colid < N; colid++)
		{

			double slack = d_costs[spid * N * N + rowid * N + colid];

			if (slack < min)
			{
				min = slack;
			}
		}

		d_row_duals[spid * N + rowid] = min;
	}
}

// Kernel for reducing the column by subtracting column minimum from each column element.
__global__ void kernel_columnReduction(double *d_costs, double *d_row_duals, double *d_col_duals, int SP, int N)
{

	int spid = blockIdx.y * blockDim.y + threadIdx.y;
	int colid = blockIdx.x * blockDim.x + threadIdx.x;

	double min = INF;

	if (spid < SP && colid < N)
	{
		for (int rowid = 0; rowid < N; rowid++)
		{

			double cost = d_costs[spid * N * N + rowid * N + colid];
			double row_dual = d_row_duals[spid * N + rowid];

			double slack = cost - row_dual;

			if (slack < min)
			{
				min = slack;
			}
		}

		d_col_duals[spid * N + colid] = min;
	}
}

// This kernel is used to update the row duals and validate the optimality of solution.
__global__ void kernel_dynamicUpdate(int *d_row_assignments, int *d_col_assignments, double *d_row_duals, double *d_col_duals, double *d_costs, int SP, int N)
{

	int spid = blockIdx.y * blockDim.y + threadIdx.y;
	int rowid = blockIdx.x * blockDim.x + threadIdx.x;

	int ROWID = spid * N + rowid;

	double min = INF;

	if (spid < SP && rowid < N)
	{

		double row_dual = d_row_duals[ROWID];

		for (int colid = 0; colid < N; colid++)
		{

			double cost = d_costs[spid * N * N + rowid * N + colid];
			double col_dual = d_col_duals[spid * N + colid];
			double slack = cost - row_dual - col_dual;

			if (slack < min)
			{
				min = slack;
			}
		}

		// Update row duals.
		row_dual = (d_row_duals[ROWID] += min);

		////////////////////////////////////////////////////////////////////////////////////////
		// Validate optimality
		int colid = d_row_assignments[ROWID];

		if (colid != -1 && colid < N)
		{
			int COLID = spid * N + colid;

			double cost = d_costs[spid * N * N + rowid * N + colid];
			double col_dual = d_col_duals[COLID];
			double slack = cost - row_dual - col_dual;

			if (slack < -EPSILON || slack > EPSILON)
				d_row_assignments[ROWID] = -1;
			else
				d_col_assignments[COLID] = rowid;
		}
		////////////////////////////////////////////////////////////////////////////////////////
	}
}

////////////////////////////////////////////////////////////////////////////////////////

// Kernel for calculating initial assignments.
__global__ void kernel_computeInitialAssignments(double *d_costs, double *d_row_duals, double *d_col_duals, int *d_row_assignments, int *d_col_assignments, int *d_row_lock, int *d_col_lock, int SP, int N)
{

	int spid = blockIdx.y * blockDim.y + threadIdx.y;
	int colid = blockIdx.x * blockDim.x + threadIdx.x;

	int COLID = spid * N + colid;

	if (spid < SP && colid < N)
	{
		double col_dual = d_col_duals[COLID];

		for (int rowid = 0; rowid < N; rowid++)
		{

			int ROWID = spid * N + rowid;

			if (d_col_lock[COLID] == 1)
				break;

			double cost = d_costs[spid * N * N + rowid * N + colid];
			double row_dual = d_row_duals[ROWID];
			double slack = cost - row_dual - col_dual;

			if (slack > -EPSILON && slack < EPSILON)
			{
				if (atomicCAS(&d_row_lock[ROWID], 0, 1) == 0)
				{
					d_row_assignments[ROWID] = colid;
					d_col_assignments[COLID] = rowid;
					d_col_lock[COLID] = 1;
				}
			}
		}
	}
}

////////////////////////////////////////////////////////////////////////////////////////

// Kernel for populating the cover arrays and initializing alternating tree.
__global__ void kernel_computeRowCovers(int *d_row_assignments, int *d_row_covers, int *d_row_visited, int SP, int N)
{

	int spid = blockIdx.y * blockDim.y + threadIdx.y;
	int rowid = blockIdx.x * blockDim.x + threadIdx.x;

	int ROWID = spid * N + rowid;

	// Copy the predicate matrix back to global memory
	if (spid < SP && rowid < N)
	{
		if (d_row_assignments[ROWID] != -1)
		{
			d_row_covers[ROWID] = 1;
		}
		else
		{
			d_row_visited[ROWID] = ACTIVE;
		}
	}
}

////////////////////////////////////////////////////////////////////////////////////////

// Kernel for populating the predicate matrix for edges in row major format.
__global__ void kernel_rowPredicateConstructionCSR(Predicates d_row_predicates_csr, int *d_row_visited, int SP, int N)
{
	int spid = blockIdx.y * blockDim.y + threadIdx.y;
	int rowid = blockIdx.x * blockDim.x + threadIdx.x;

	int ROWID = spid * N + rowid;

	if (spid < SP && rowid < N)
	{

		int row_visited = d_row_visited[ROWID];

		bool predicate = (row_visited == ACTIVE);
		long addr = predicate ? 1 : 0;

		d_row_predicates_csr.predicates[ROWID] = predicate; // Copy the predicate matrix back to global memory
		d_row_predicates_csr.addresses[ROWID] = addr;
	}
}

// Kernel for scattering the edges based on the scatter addresses.
__global__ void kernel_rowScatterCSR(CompactEdges d_row_vertices_csr, Predicates d_row_predicates_csr, long M, int SP, int N)
{

	int spid = blockIdx.y * blockDim.y + threadIdx.y;
	int rowid = blockIdx.x * blockDim.x + threadIdx.x;

	int ROWID = spid * N + rowid;

	// Copy the matrix into shared memory

	if (spid < SP && rowid < N)
	{

		bool predicate = d_row_predicates_csr.predicates[ROWID];
		long compid = d_row_predicates_csr.addresses[ROWID];

		if (predicate)
		{
			d_row_vertices_csr.neighbors[compid] = rowid;
		}
		if (rowid == 0)
		{
			d_row_vertices_csr.ptrs[spid] = compid;
			d_row_vertices_csr.ptrs[SP] = M; // extra pointer for the total number of edges. necessary for calculating number of edges in each row.
		}
	}
}

// Kernel for finding the minimum zero cover.
__global__ void kernel_coverAndExpand(bool *d_flag, CompactEdges d_row_vertices_csr, Matrix d_costs, Vertices d_vertices, VertexData d_row_data, VertexData d_col_data, int SP, int N)
{
	int spid = blockIdx.y * blockDim.y + threadIdx.y;
	int colid = blockIdx.x * blockDim.x + threadIdx.x;

	// Load values into local memory

	if (spid < SP && colid < N)
	{
		int in_size = d_row_vertices_csr.ptrs[spid + 1] - d_row_vertices_csr.ptrs[spid];
		int nbr_start = d_row_vertices_csr.ptrs[spid];
		int *st_ptr = &d_row_vertices_csr.neighbors[nbr_start];
		int *end_ptr = &d_row_vertices_csr.neighbors[nbr_start + in_size];

		__traverse(d_costs, d_vertices, d_flag, d_row_data.parents, d_col_data.parents, d_row_data.is_visited, d_col_data.is_visited, st_ptr, end_ptr, spid, colid, SP, N);
	}
}

////////////////////////////////////////////////////////////////////////////////////////

// Kernel for constructing the predicates for reverse pass or augmentation candidates.
__global__ void kernel_augmentPredicateConstruction(Predicates d_predicates, int *d_visited, int size)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	// Copy the matrix into shared memory.

	if (id < size)
	{
		int visited = d_visited[id];
		bool predicate = (visited == REVERSE || visited == AUGMENT);
		long addr = predicate ? 1 : 0;

		d_predicates.predicates[id] = predicate;
		d_predicates.addresses[id] = addr;
	}
}

// Kernel for scattering the vertices based on the scatter addresses.
__global__ void kernel_augmentScatter(Array d_vertex_ids, Predicates d_predicates, int size)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if (id < size)
	{

		bool predicate = d_predicates.predicates[id];
		long compid = (predicate) ? d_predicates.addresses[id] : -1; // compaction id.

		if (predicate)
			d_vertex_ids.elements[compid] = id;
	}
}

// Kernel for executing the reverse pass of the maximum matching algorithm.
__global__ void kernel_reverseTraversal(Array d_col_vertices, VertexData d_row_data, VertexData d_col_data)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int size = d_col_vertices.size;

	if (id < size)
	{
		int COLID = d_col_vertices.elements[id];
		__reverse_traversal(d_row_data.is_visited, d_row_data.children, d_col_data.children, d_row_data.parents, d_col_data.parents, COLID);
	}
}

// Kernel for executing the augmentation pass of the maximum matching algorithm.
__global__ void kernel_augmentation(int *d_row_assignments, int *d_col_assignments, Array d_row_vertices, VertexData d_row_data, VertexData d_col_data, int N)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int size = d_row_vertices.size;

	if (id < size)
	{

		int ROWID = d_row_vertices.elements[id];

		__augment(d_row_assignments, d_col_assignments, d_row_data.children, d_col_data.children, ROWID, N);
	}
}

// Kernel for updating the dual values in Step 5.
__global__ void kernel_dualUpdate_1(double *d_sp_min, double *d_col_slacks, int *d_col_covers, int SP, int N)
{

	int spid = blockIdx.x * blockDim.x + threadIdx.x;

	if (spid < SP)
	{
		double min = INF;
		for (int colid = 0; colid < N; colid++)
		{
			int COLID = spid * N + colid;
			double slack = d_col_slacks[COLID];
			int col_cover = d_col_covers[COLID];

			if (col_cover == 0)
				if (slack < min)
					min = slack;
		}

		d_sp_min[spid] = min;
	}
}

// Kernel for updating the dual values in Step 5.
__global__ void kernel_dualUpdate_2(double *d_sp_min, double *d_row_duals, double *d_col_duals, double *d_col_slacks, int *d_row_covers, int *d_col_covers, int *d_row_visited, int *d_col_parents, int SP, int N)
{

	int spid = blockIdx.y * blockDim.y + threadIdx.y;
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	int ID = spid * N + id;

	if (spid < SP && id < N)
	{
		if (d_sp_min[spid] < INF3)
		{

			//			double theta = 0.5 * d_sp_min[spid];
			double theta = d_sp_min[spid];
			int row_cover = d_row_covers[ID];
			int col_cover = d_col_covers[ID];

			if (row_cover == 0) // Row vertex is reachable from source.
				d_row_duals[ID] += theta;

			//			else
			//				// Row vertex is unreachable from source.
			//				d_row_duals[ID] -= theta;

			if (col_cover == 1) // Col vertex is reachable from source.
				d_col_duals[ID] -= theta;

			else
			{
				// Col vertex is unreachable from source.
				//				d_col_duals[ID] += theta;

				d_col_slacks[ID] -= d_sp_min[spid];

				if (d_col_slacks[ID] > -EPSILON && d_col_slacks[ID] < EPSILON)
				{
					int PAR_ROWID = d_col_parents[ID];
					if (PAR_ROWID != -1)
						d_row_visited[PAR_ROWID] = ACTIVE;
				}
			}
		}
	}
}

// Kernel for calculating optimal objective function value using dual variables.
__global__ void kernel_calcObjVal(double *d_obj_val, double *d_row_duals, double *d_col_duals, int SP, int N)
{

	int spid = blockIdx.x * blockDim.x + threadIdx.x;

	if (spid < SP)
	{

		d_obj_val[spid] = 0;

		for (int i = 0; i < N; i++)
			d_obj_val[spid] += (d_row_duals[spid * N + i] + d_col_duals[spid * N + i]);
	}
}

// Kernel for calculating optimal objective function value using dual variables.
__global__ void kernel_calcObjVal2(double *d_obj_val, double *d_costs, int *d_row_assignments, int SP, int N)
{

	int spid = blockIdx.x * blockDim.x + threadIdx.x;

	if (spid < SP)
	{

		d_obj_val[spid] = 0;

		for (int i = 0; i < N; i++)
		{
			int j = d_row_assignments[spid * N + i];
			d_obj_val[spid] += d_costs[spid * N * N + i * N + j];
		}
	}
}

// This function is used to perform initial reduction.
void initialReduction(Matrix &d_costs, Vertices &d_vertices_dev, int SP, int N, unsigned int devid)
{

	cudaSetDevice(devid);

	dim3 blocks_per_grid;
	dim3 threads_per_block;
	int total_blocks = 0;

	calculateRectangularDims(blocks_per_grid, threads_per_block, total_blocks, N, SP);

	kernel_rowReduction<<<blocks_per_grid, threads_per_block>>>(d_costs.elements, d_vertices_dev.row_duals, d_vertices_dev.col_duals, SP, N); // Kernel execution.
	cudaSafeCall(cudaGetLastError(), "Error in kernel_rowReduction execution f_culap::initialReduction");
	kernel_columnReduction<<<blocks_per_grid, threads_per_block>>>(d_costs.elements, d_vertices_dev.row_duals, d_vertices_dev.col_duals, SP, N); // Kernel execution.
	cudaSafeCall(cudaGetLastError(), "Error in kernel_colReduction execution f_culap::initialReduction");
}

// This function is used to validate the optimality of the previous solution after cost update.
void dynamicUpdate(Matrix &d_costs, Vertices &d_vertices_dev, int SP, int N, unsigned int devid)
{

	cudaSetDevice(devid);

	dim3 blocks_per_grid;
	dim3 threads_per_block;
	int total_blocks = 0;

	calculateRectangularDims(blocks_per_grid, threads_per_block, total_blocks, N, SP);
	kernel_dynamicUpdate<<<blocks_per_grid, threads_per_block>>>(d_vertices_dev.row_assignments, d_vertices_dev.col_assignments, d_vertices_dev.row_duals, d_vertices_dev.col_duals, d_costs.elements, SP, N); // Kernel execution.
	cudaSafeCall(cudaGetLastError(), "Error in kernel_dynamicUpdate execution f_culap::dynamicUpdate");
}

////////////////////////////////////////////////////////////////////////////////////////

// Function for calculating initial assignments on individual cards and stitcing them together on host.
void computeInitialAssignments(Matrix &d_costs, Vertices &d_vertices_dev, int SP, int N, unsigned int devid)
{

	cudaSetDevice(devid);

	dim3 blocks_per_grid;
	dim3 threads_per_block;
	int total_blocks = 0;

	calculateLinearDims(blocks_per_grid, threads_per_block, total_blocks, N * SP);

	cudaSafeCall(cudaMemset(d_vertices_dev.row_assignments, -1, SP * N * sizeof(int)), "Error in cudaMemset f_culap::computeInitialAssignments::d_row_assignment");
	cudaSafeCall(cudaMemset(d_vertices_dev.col_assignments, -1, SP * N * sizeof(int)), "Error in cudaMemset f_culap::computeInitialAssignments::d_col_assignment");

	int *d_row_lock, *d_col_lock;
	cudaSafeCall(cudaMalloc((void **)&d_row_lock, SP * N * sizeof(int)), "Error in cudaMalloc f_culap::computeInitialAssignments::d_row_lock");
	cudaSafeCall(cudaMalloc((void **)&d_col_lock, SP * N * sizeof(int)), "Error in cudaMalloc f_culap::computeInitialAssignments::d_col_lock");
	cudaSafeCall(cudaMemset(d_row_lock, 0, SP * N * sizeof(int)), "Error in cudaMemset f_culap::computeInitialAssignments::d_row_lock");
	cudaSafeCall(cudaMemset(d_col_lock, 0, SP * N * sizeof(int)), "Error in cudaMemset f_culap::computeInitialAssignments::d_col_lock");

	calculateRectangularDims(blocks_per_grid, threads_per_block, total_blocks, N, SP);
	kernel_computeInitialAssignments<<<blocks_per_grid, threads_per_block>>>(d_costs.elements, d_vertices_dev.row_duals, d_vertices_dev.col_duals, d_vertices_dev.row_assignments, d_vertices_dev.col_assignments, d_row_lock, d_col_lock, SP, N); // Kernel execution.
	cudaSafeCall(cudaGetLastError(), "Error in kernel_computeInitialAssignments execution f_culap::computeInitialAssignments");

	cudaSafeCall(cudaFree(d_row_lock), "Error in cudaFree f_culap::computeInitialAssignments::d_row_lock");
	cudaSafeCall(cudaFree(d_col_lock), "Error in cudaFree f_culap::computeInitialAssignments::d_col_lock");
}

////////////////////////////////////////////////////////////////////////////////////////

// Function for finding row cover on individual devices.
int computeRowCovers(Vertices &d_vertices_dev, VertexData &d_row_data_dev, VertexData &d_col_data_dev, int SP, int N, unsigned int devid)
{

	cudaSetDevice(devid);

	dim3 blocks_per_grid;
	dim3 threads_per_block;
	int total_blocks = 0;

	cudaSafeCall(cudaMemset(d_vertices_dev.row_covers, 0, SP * N * sizeof(int)), "Error in cudaMemset f_culap::computeRowCovers::d_row_covers");
	cudaSafeCall(cudaMemset(d_vertices_dev.col_covers, 0, SP * N * sizeof(int)), "Error in cudaMemset f_culap::computeRowCovers::d_col_covers");

	calculateLinearDims(blocks_per_grid, threads_per_block, total_blocks, SP * N);
	kernel_memSet<<<blocks_per_grid, threads_per_block>>>(d_vertices_dev.col_slacks, INF, SP * N);

	cudaSafeCall(cudaMemset(d_row_data_dev.is_visited, DORMANT, SP * N * sizeof(int)), "Error in cudaMemset f_culap::computeRowCovers::d_row_data.is_visited");
	cudaSafeCall(cudaMemset(d_col_data_dev.is_visited, DORMANT, SP * N * sizeof(int)), "Error in cudaMemset f_culap::computeRowCovers::d_col_data.is_visited"); // initialize "visited" array for columns. later used in BFS (Step 4).
	cudaSafeCall(cudaMemset(d_row_data_dev.parents, -1, SP * N * sizeof(int)), "Error in cudaMemset f_culap::computeRowCovers::d_row_data.parents");
	cudaSafeCall(cudaMemset(d_row_data_dev.children, -1, SP * N * sizeof(int)), "Error in cudaMemset f_culap::computeRowCovers::d_row_data.children");
	cudaSafeCall(cudaMemset(d_col_data_dev.parents, -1, SP * N * sizeof(int)), "Error in cudaMemset f_culap::computeRowCovers::d_col_data.parents");
	cudaSafeCall(cudaMemset(d_col_data_dev.children, -1, SP * N * sizeof(int)), "Error in cudaMemset f_culap::computeRowCovers::d_col_data.children");

	calculateRectangularDims(blocks_per_grid, threads_per_block, total_blocks, N, SP);
	kernel_computeRowCovers<<<blocks_per_grid, threads_per_block>>>(d_vertices_dev.row_assignments, d_vertices_dev.row_covers, d_row_data_dev.is_visited, SP, N); // Kernel execution.
	cudaSafeCall(cudaGetLastError(), "Error in kernel_computeRowCovers execution f_culap::computeRowCovers");

	thrust::device_ptr<int> ptr(d_vertices_dev.row_covers);

	int cover_count = thrust::reduce(ptr, ptr + SP * N);

	return cover_count;
}

// Function for compacting the edges in row major format.
void compactRowVertices(CompactEdges &d_rows_csr_dev, VertexData &d_row_data_dev, long &M, int SP, int N, unsigned int devid)
{

	cudaSetDevice(devid);

	dim3 blocks_per_grid;
	dim3 threads_per_block;
	int total_blocks = 0;

	Predicates d_row_predicates_csr;

	d_row_predicates_csr.size = SP * N;
	cudaSafeCall(cudaMalloc((void **)(&d_row_predicates_csr.predicates), d_row_predicates_csr.size * sizeof(bool)), "Error in cudaMalloc f_culap::compactRowVertices::d_row_predicates_csr.predicates");
	cudaSafeCall(cudaMalloc((void **)(&d_row_predicates_csr.addresses), d_row_predicates_csr.size * sizeof(long)), "Error in cudaMalloc f_culap::compactRowVertices::d_row_predicates_csr.addresses");
	cudaSafeCall(cudaMemset(d_row_predicates_csr.predicates, false, d_row_predicates_csr.size * sizeof(bool)), "Error in cudaMemset f_culap::compactRowVertices::d_row_predicates_csr.predicates");
	cudaSafeCall(cudaMemset(d_row_predicates_csr.addresses, 0, d_row_predicates_csr.size * sizeof(long)), "Error in cudaMemset f_culap::compactRowVertices::d_row_predicates_csr.addresses");

	cudaSafeCall(cudaMalloc((void **)(&d_rows_csr_dev.ptrs), (SP + 1) * sizeof(long)), "Error in cudaMalloc f_culap::compactRowVertices::d_rows_csr_dev.ptrs");
	cudaSafeCall(cudaMemset(d_rows_csr_dev.ptrs, -1, (SP + 1) * sizeof(long)), "Error in cudaMemset f_culap::compactRowVertices::d_rows_csr_dev.ptrs");

	calculateRectangularDims(blocks_per_grid, threads_per_block, total_blocks, N, SP);
	kernel_rowPredicateConstructionCSR<<<blocks_per_grid, threads_per_block>>>(d_row_predicates_csr, d_row_data_dev.is_visited, SP, N); // construct predicate matrix for edges.
	cudaSafeCall(cudaGetLastError(), "Error in kernel_edgePredicateConstructionCSR execution f_culap::compactRowVertices");

	thrust::device_ptr<long> ptr(d_row_predicates_csr.addresses);
	M = thrust::reduce(ptr, ptr + d_row_predicates_csr.size);					 // calculate total number of edges.
	thrust::exclusive_scan(ptr, ptr + d_row_predicates_csr.size, ptr); // exclusive scan for calculating the scatter addresses.

	if (M > 0)
	{
		cudaSafeCall(cudaMalloc((void **)(&d_rows_csr_dev.neighbors), M * sizeof(int)), "Error in cudaMalloc f_culap::compactRowVertices::d_rows_csr_dev.neighbors");

		kernel_rowScatterCSR<<<blocks_per_grid, threads_per_block>>>(d_rows_csr_dev, d_row_predicates_csr, M, SP, N);
		cudaSafeCall(cudaGetLastError(), "Error in kernel_edgeScatterCSR execution f_culap::compactRowVertices");
	}

	cudaSafeCall(cudaFree(d_row_predicates_csr.predicates), "Error in cudaFree f_culap::compactRowVertices::d_row_predicates_csr.predicates");
	cudaSafeCall(cudaFree(d_row_predicates_csr.addresses), "Error in cudaFree f_culap::compactRowVertices::d_row_predicates_csr.addresses");
}

// Function for covering the zeros in uncovered rows and expanding the frontier.
void coverZeroAndExpand(Matrix &d_costs_dev, CompactEdges &d_rows_csr_dev, Vertices &d_vertices_dev, VertexData &d_row_data_dev, VertexData &d_col_data_dev, bool *h_flag, int SP, int N, unsigned int devid)
{
	cudaSetDevice(devid);

	int total_blocks = 0;
	dim3 blocks_per_grid;
	dim3 threads_per_block;

	calculateRectangularDims(blocks_per_grid, threads_per_block, total_blocks, N, SP);

	bool *d_flag;
	cudaSafeCall(cudaMalloc((void **)&d_flag, sizeof(bool)), "Error in cudaMalloc d_flag");
	cudaSafeCall(cudaMemcpy(d_flag, h_flag, sizeof(bool), cudaMemcpyHostToDevice), "Error in cudaMemcpy h_flag");

	kernel_coverAndExpand<<<blocks_per_grid, threads_per_block>>>(d_flag, d_rows_csr_dev, d_costs_dev, d_vertices_dev, d_row_data_dev, d_col_data_dev, SP, N);

	cudaSafeCall(cudaMemcpy(h_flag, d_flag, sizeof(bool), cudaMemcpyDeviceToHost), "Error in cudaMemcpy d_next");

	cudaSafeCall(cudaFree(d_flag), "Error in cudaFree d_next");
}

////////////////////////////////////////////////////////////////////////////////////////
// Function for executing recursive zero cover. Returns the next step (Step 4 or Step 5) depending on the presence of uncovered zeros.
void executeZeroCover(Matrix &d_costs_dev, Vertices &d_vertices_dev, VertexData &d_row_data_dev, VertexData &d_col_data_dev, bool *h_flag, int SP, int N, unsigned int devid)
{

	cudaSetDevice(devid);
	CompactEdges d_rows_csr_dev;

	while (true)
	{

		long M = 0;
		compactRowVertices(d_rows_csr_dev, d_row_data_dev, M, SP, N, devid); // compact the current vertex frontier.

		if (M > 0)
		{
			coverZeroAndExpand(d_costs_dev, d_rows_csr_dev, d_vertices_dev, d_row_data_dev, d_col_data_dev, h_flag, SP, N, devid);
			cudaSafeCall(cudaFree(d_rows_csr_dev.neighbors), "Error in cudaFree f_culap::executeZeroCover::d_edges_csr.neighbors");
			cudaSafeCall(cudaFree(d_rows_csr_dev.ptrs), "Error in cudaFree f_culap::executeZeroCover::d_edges_csr_dev.ptrs");
		}
		else
		{
			cudaSafeCall(cudaFree(d_rows_csr_dev.ptrs), "Error in cudaFree f_culap::executeZeroCover::d_edges_csr_dev.ptrs");
			break;
		}
	}
}

// Function for executing reverse pass of the maximum matching.
void reversePass(VertexData &d_row_data_dev, VertexData &d_col_data_dev, int SP, int N, unsigned int devid)
{

	cudaSetDevice(devid);

	int total_blocks = 0;
	dim3 blocks_per_grid;
	dim3 threads_per_block;
	calculateLinearDims(blocks_per_grid, threads_per_block, total_blocks, SP * N);

	Array d_col_ids_csr;
	Predicates d_col_predicates; // predicates for compacting the colids eligible for the reverse pass.

	d_col_predicates.size = SP * N;
	cudaSafeCall(cudaMalloc((void **)(&d_col_predicates.predicates), d_col_predicates.size * sizeof(bool)), "Error in cudaMalloc f_culap::reversePass::d_col_predicates.predicates");
	cudaSafeCall(cudaMalloc((void **)(&d_col_predicates.addresses), d_col_predicates.size * sizeof(long)), "Error in cudaMalloc f_culap::reversePass::d_col_predicates.addresses");
	cudaSafeCall(cudaMemset(d_col_predicates.predicates, false, d_col_predicates.size * sizeof(bool)), "Error in cudaMemset f_culap::reversePass::d_col_predicates.predicates");
	cudaSafeCall(cudaMemset(d_col_predicates.addresses, 0, d_col_predicates.size * sizeof(long)), "Error in cudaMemset f_culap::reversePass::d_col_predicates.addresses");

	// compact the reverse pass row vertices.
	kernel_augmentPredicateConstruction<<<blocks_per_grid, threads_per_block>>>(d_col_predicates, d_col_data_dev.is_visited, d_col_predicates.size);
	cudaSafeCall(cudaGetLastError(), "Error in kernel_augmentPredicateConstruction f_culap::reversePass");

	thrust::device_ptr<long> ptr(d_col_predicates.addresses);
	d_col_ids_csr.size = thrust::reduce(ptr, ptr + d_col_predicates.size); // calculate total number of vertices.
	thrust::exclusive_scan(ptr, ptr + d_col_predicates.size, ptr);				 // exclusive scan for calculating the scatter addresses.

	if (d_col_ids_csr.size > 0)
	{
		int total_blocks_1 = 0;
		dim3 blocks_per_grid_1;
		dim3 threads_per_block_1;
		calculateLinearDims(blocks_per_grid_1, threads_per_block_1, total_blocks_1, d_col_ids_csr.size);

		cudaSafeCall(cudaMalloc((void **)(&d_col_ids_csr.elements), d_col_ids_csr.size * sizeof(int)), "Error in cudaMalloc f_culap::reversePass::d_col_ids_csr.elements");

		kernel_augmentScatter<<<blocks_per_grid, threads_per_block>>>(d_col_ids_csr, d_col_predicates, d_col_predicates.size);
		cudaSafeCall(cudaGetLastError(), "Error in kernel_augmentScatter f_culap::reversePass");

		kernel_reverseTraversal<<<blocks_per_grid_1, threads_per_block_1>>>(d_col_ids_csr, d_row_data_dev, d_col_data_dev);
		cudaSafeCall(cudaGetLastError(), "Error in kernel_reverseTraversal f_culap::reversePass");

		cudaSafeCall(cudaFree(d_col_ids_csr.elements), "Error in cudaFree f_culap::reversePass::d_col_ids_csr.elements");
	}

	cudaSafeCall(cudaFree(d_col_predicates.predicates), "Error in cudaFree f_culap::reversePass::d_col_predicates.predicates");
	cudaSafeCall(cudaFree(d_col_predicates.addresses), "Error in cudaFree f_culap::reversePass::d_col_predicates.addresses");
}

// Function for executing augmentation pass of the maximum matching.
void augmentationPass(Vertices &d_vertices_dev, VertexData &d_row_data_dev, VertexData &d_col_data_dev, int SP, int N, unsigned int devid)
{

	cudaSetDevice(devid);

	int total_blocks = 0;
	dim3 blocks_per_grid;
	dim3 threads_per_block;
	calculateLinearDims(blocks_per_grid, threads_per_block, total_blocks, SP * N);

	Array d_row_ids_csr;
	Predicates d_row_predicates; // predicates for compacting the colids eligible for the augmentation pass.

	d_row_predicates.size = SP * N;
	cudaSafeCall(cudaMalloc((void **)(&d_row_predicates.predicates), d_row_predicates.size * sizeof(bool)), "Error in cudaMalloc f_culap::augmentationPass::d_row_predicates.predicates");
	cudaSafeCall(cudaMalloc((void **)(&d_row_predicates.addresses), d_row_predicates.size * sizeof(long)), "Error in cudaMalloc f_culap::augmentationPass::d_row_predicates.addresses");
	cudaSafeCall(cudaMemset(d_row_predicates.predicates, false, d_row_predicates.size * sizeof(bool)), "Error in cudaMemset f_culap::augmentationPass::d_row_predicates.predicates");
	cudaSafeCall(cudaMemset(d_row_predicates.addresses, 0, d_row_predicates.size * sizeof(long)), "Error in cudaMemset f_culap::augmentationPass::d_row_predicates.addresses");

	// compact the reverse pass row vertices.
	kernel_augmentPredicateConstruction<<<blocks_per_grid, threads_per_block>>>(d_row_predicates, d_row_data_dev.is_visited, d_row_predicates.size);
	cudaSafeCall(cudaGetLastError(), "Error in kernel_augmentPredicateConstruction f_culap::augmentationPass");

	thrust::device_ptr<long> ptr(d_row_predicates.addresses);
	d_row_ids_csr.size = thrust::reduce(ptr, ptr + d_row_predicates.size); // calculate total number of vertices.
	thrust::exclusive_scan(ptr, ptr + d_row_predicates.size, ptr);				 // exclusive scan for calculating the scatter addresses.

	if (d_row_ids_csr.size > 0)
	{
		int total_blocks_1 = 0;
		dim3 blocks_per_grid_1;
		dim3 threads_per_block_1;
		calculateLinearDims(blocks_per_grid_1, threads_per_block_1, total_blocks_1, d_row_ids_csr.size);

		cudaSafeCall(cudaMalloc((void **)(&d_row_ids_csr.elements), d_row_ids_csr.size * sizeof(int)), "Error in cudaMalloc f_culap::augmentationPass::d_row_ids_csr.elements");

		kernel_augmentScatter<<<blocks_per_grid, threads_per_block>>>(d_row_ids_csr, d_row_predicates, d_row_predicates.size);
		cudaSafeCall(cudaGetLastError(), "Error in kernel_augmentScatter f_culap::augmentationPass");

		kernel_augmentation<<<blocks_per_grid_1, threads_per_block_1>>>(d_vertices_dev.row_assignments, d_vertices_dev.col_assignments, d_row_ids_csr, d_row_data_dev, d_col_data_dev, N);
		cudaSafeCall(cudaGetLastError(), "Error in kernel_augmentation f_culap::augmentationPass");

		cudaSafeCall(cudaFree(d_row_ids_csr.elements), "Error in cudaFree f_culap::augmentationPass::d_row_ids_csr.elements");
	}

	cudaSafeCall(cudaFree(d_row_predicates.predicates), "Error in cudaFree f_culap::augmentationPass::d_row_predicates.predicates");
	cudaSafeCall(cudaFree(d_row_predicates.addresses), "Error in cudaFree f_culap::augmentationPass::d_row_predicates.addresses");
}

void dualUpdate(Vertices &d_vertices_dev, VertexData &d_row_data_dev, VertexData &d_col_data_dev, int SP, int N, unsigned int devid)
{

	dim3 blocks_per_grid;
	dim3 threads_per_block;
	int total_blocks;

	double *d_sp_min;
	cudaSafeCall(cudaMalloc((void **)(&d_sp_min), SP * sizeof(double)), "Error in cudaMalloc f_culap::dualUpdate::d_sp_min");

	calculateLinearDims(blocks_per_grid, threads_per_block, total_blocks, SP);
	kernel_dualUpdate_1<<<blocks_per_grid, threads_per_block>>>(d_sp_min, d_vertices_dev.col_slacks, d_vertices_dev.col_covers, SP, N);
	cudaSafeCall(cudaGetLastError(), "Error in kernel_augmentation f_culap::kernel_dualUpdate_1");

	calculateRectangularDims(blocks_per_grid, threads_per_block, total_blocks, N, SP);
	kernel_dualUpdate_2<<<blocks_per_grid, threads_per_block>>>(d_sp_min, d_vertices_dev.row_duals, d_vertices_dev.col_duals, d_vertices_dev.col_slacks, d_vertices_dev.row_covers, d_vertices_dev.col_covers, d_row_data_dev.is_visited, d_col_data_dev.parents, SP, N);
	cudaSafeCall(cudaGetLastError(), "Error in kernel_augmentation f_culap::kernel_dualUpdate_2");

	cudaSafeCall(cudaFree(d_sp_min), "Error in cudaFree f_culap::dualUpdate::d_sp_min");
}

////////////////////////////////////////////////////////////////////////////////////////

// Function for calculating optimal objective function value using dual variables.
void calcObjVal(double *d_obj_val, Vertices &d_vertices_dev, int SP, int N, unsigned int devid)
{

	cudaSetDevice(devid);

	dim3 blocks_per_grid;
	dim3 threads_per_block;
	int total_blocks = 0;

	calculateLinearDims(blocks_per_grid, threads_per_block, total_blocks, SP);

	kernel_calcObjVal<<<blocks_per_grid, threads_per_block>>>(d_obj_val, d_vertices_dev.row_duals, d_vertices_dev.col_duals, SP, N);
	cudaSafeCall(cudaGetLastError(), "Error in kernel_calcObjVal execution f_culap::calcObjVal");
}

// Function for calculating optimal objective function value using dual variables.
void calcObjVal2(double *d_obj_val, double *d_costs, Vertices &d_vertices_dev, int SP, int N, unsigned int devid)
{

	cudaSetDevice(devid);

	dim3 blocks_per_grid;
	dim3 threads_per_block;
	int total_blocks = 0;

	calculateLinearDims(blocks_per_grid, threads_per_block, total_blocks, SP);

	kernel_calcObjVal2<<<blocks_per_grid, threads_per_block>>>(d_obj_val, d_costs, d_vertices_dev.row_assignments, SP, N);
	cudaSafeCall(cudaGetLastError(), "Error in kernel_calcObjVal execution f_culap::calcObjVal");
}
