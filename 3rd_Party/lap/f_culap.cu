/*
 * f_culap.cu
 *
 *  Created on: Jul 29, 2015
 *      Author: date2
 */

#include "include/f_culap.h"

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

	calculateRectangularDims(blocks_per_grid, threads_per_block, total_blocks, N, SP);

	cudaSafeCall(cudaMemset(d_vertices_dev.row_assignments, -1, SP * N * sizeof(int)), "Error in cudaMemset f_culap::computeInitialAssignments::d_row_assignment");
	cudaSafeCall(cudaMemset(d_vertices_dev.col_assignments, -1, SP * N * sizeof(int)), "Error in cudaMemset f_culap::computeInitialAssignments::d_col_assignment");

	int *d_row_lock, *d_col_lock;
	cudaSafeCall(cudaMalloc((void **)&d_row_lock, SP * N * sizeof(int)), "Error in cudaMalloc f_culap::computeInitialAssignments::d_row_lock");
	cudaSafeCall(cudaMalloc((void **)&d_col_lock, SP * N * sizeof(int)), "Error in cudaMalloc f_culap::computeInitialAssignments::d_col_lock");
	cudaSafeCall(cudaMemset(d_row_lock, 0, SP * N * sizeof(int)), "Error in cudaMemset f_culap::computeInitialAssignments::d_row_lock");
	cudaSafeCall(cudaMemset(d_col_lock, 0, SP * N * sizeof(int)), "Error in cudaMemset f_culap::computeInitialAssignments::d_col_lock");

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

	calculateRectangularDims(blocks_per_grid, threads_per_block, total_blocks, N, SP);

	cudaSafeCall(cudaMemset(d_vertices_dev.row_covers, 0, SP * N * sizeof(int)), "Error in cudaMemset f_culap::computeRowCovers::d_row_covers");
	cudaSafeCall(cudaMemset(d_vertices_dev.col_covers, 0, SP * N * sizeof(int)), "Error in cudaMemset f_culap::computeRowCovers::d_col_covers");

	cudaSafeCall(cudaMemset(d_row_data_dev.is_visited, DORMANT, SP * N * sizeof(int)), "Error in cudaMemset f_culap::computeRowCovers::d_row_data.is_visited");
	cudaSafeCall(cudaMemset(d_col_data_dev.is_visited, DORMANT, SP * N * sizeof(int)), "Error in cudaMemset f_culap::computeRowCovers::d_col_data.is_visited"); // initialize "visited" array for columns. later used in BFS (Step 4).

	cudaSafeCall(cudaMemset(d_row_data_dev.parents, -1, SP * N * sizeof(int)), "Error in cudaMemset f_culap::computeRowCovers::d_row_data.parents");
	cudaSafeCall(cudaMemset(d_row_data_dev.children, -1, SP * N * sizeof(int)), "Error in cudaMemset f_culap::computeRowCovers::d_row_data.children");
	cudaSafeCall(cudaMemset(d_col_data_dev.parents, -1, SP * N * sizeof(int)), "Error in cudaMemset f_culap::computeRowCovers::d_col_data.parents");
	cudaSafeCall(cudaMemset(d_col_data_dev.children, -1, SP * N * sizeof(int)), "Error in cudaMemset f_culap::computeRowCovers::d_col_data.children");

	kernel_computeRowCovers<<<blocks_per_grid, threads_per_block>>>(d_vertices_dev.row_assignments, d_vertices_dev.row_covers, SP, N); // Kernel execution.
	cudaSafeCall(cudaGetLastError(), "Error in kernel_computeRowCovers execution f_culap::computeRowCovers");

	thrust::device_ptr<int> ptr(d_vertices_dev.row_covers);

	int cover_count = thrust::reduce(ptr, ptr + SP * N);

	return cover_count;
}

////////////////////////////////////////////////////////////////////////////////////////

// Function for compacting the edges in row major format.
void compactEdgesCSR(CompactEdges &d_edges_csr_dev, Matrix &d_costs_dev, Vertices &d_vertices_dev, long &M, int SP, int N, unsigned int devid)
{

	cudaSetDevice(devid);

	dim3 blocks_per_grid;
	dim3 threads_per_block;
	int total_blocks = 0;

	Predicates d_edge_predicates_csr;

	d_edge_predicates_csr.size = SP * N * N;
	cudaSafeCall(cudaMalloc((void **)(&d_edge_predicates_csr.predicates), d_edge_predicates_csr.size * sizeof(bool)), "Error in cudaMalloc f_culap::compactEdgesCSR::d_edge_predicates_csr.predicates");
	cudaSafeCall(cudaMalloc((void **)(&d_edge_predicates_csr.addresses), d_edge_predicates_csr.size * sizeof(long)), "Error in cudaMalloc f_culap::compactEdgesCSR::d_edge_predicates_csr.addresses");
	cudaSafeCall(cudaMemset(d_edge_predicates_csr.predicates, false, d_edge_predicates_csr.size * sizeof(bool)), "Error in cudaMemset f_culap::compactEdgesCSR::d_edge_predicates_csr.predicates");
	cudaSafeCall(cudaMemset(d_edge_predicates_csr.addresses, 0, d_edge_predicates_csr.size * sizeof(long)), "Error in cudaMemset f_culap::compactEdgesCSR::d_edge_predicates_csr.addresses");

	cudaSafeCall(cudaMalloc((void **)(&d_edges_csr_dev.ptrs), ((SP * N) + 1) * sizeof(long)), "Error in cudaMalloc f_culap::compactEdgesCSR::d_edges_csr.ptrs");
	cudaSafeCall(cudaMemset(d_edges_csr_dev.ptrs, -1, ((SP * N) + 1) * sizeof(long)), "Error in cudaMemset f_culap::compactEdgesCSR::d_edges_csr.ptrs");

	calculateCubicDims(blocks_per_grid, threads_per_block, total_blocks, N, N, SP);

	kernel_edgePredicateConstructionCSR<<<blocks_per_grid, threads_per_block>>>(d_edge_predicates_csr, d_costs_dev.elements, d_vertices_dev.row_duals, d_vertices_dev.col_duals, SP, N); // construct predicate matrix for edges.
	cudaSafeCall(cudaGetLastError(), "Error in kernel_edgePredicateConstructionCSR execution f_culap::compactEdgesCSR");

	thrust::device_ptr<long> ptr(d_edge_predicates_csr.addresses);
	M = thrust::reduce(ptr, ptr + d_edge_predicates_csr.size);			// calculate total number of edges.
	thrust::exclusive_scan(ptr, ptr + d_edge_predicates_csr.size, ptr); // exclusive scan for calculating the scatter addresses.

	cudaSafeCall(cudaMalloc((void **)(&d_edges_csr_dev.neighbors), M * sizeof(int)), "Error in cudaMalloc f_culap::compactEdgesCSR::d_edges_csr.neighbors");

	kernel_edgeScatterCSR<<<blocks_per_grid, threads_per_block>>>(d_edges_csr_dev, d_edge_predicates_csr, M, SP, N);
	cudaSafeCall(cudaGetLastError(), "Error in kernel_edgeScatterCSR execution f_culap::compactEdgesCSR");

	cudaSafeCall(cudaFree(d_edge_predicates_csr.predicates), "Error in cudaFree f_culap::compactEdgesCSR::d_edge_predicates_csr.predicates");
	cudaSafeCall(cudaFree(d_edge_predicates_csr.addresses), "Error in cudaFree f_culap::compactEdgesCSR::d_edge_predicates_csr.addresses");
}

// Function for executing recursive zero cover. Returns the next step (Step 4 or Step 5) depending on the presence of uncovered zeros.
void executeZeroCover(CompactEdges &d_edges_csr_dev, Vertices &d_vertices_dev, VertexData &d_row_data_dev, VertexData &d_col_data_dev, bool *h_flag, int SP, int N, unsigned int devid)
{

	cudaSetDevice(devid);

	Array d_vertices_csr1, d_vertices_csr2;

	int total_blocks = 0;
	dim3 blocks_per_grid;
	dim3 threads_per_block;

	d_vertices_csr1.size = SP * N;
	cudaSafeCall(cudaMalloc((void **)(&d_vertices_csr1.elements), d_vertices_csr1.size * sizeof(int)), "Error in cudaMalloc d_vertices_csr1.elements"); // compact vertices initialized to the row ids.

	calculateRectangularDims(blocks_per_grid, threads_per_block, total_blocks, N, SP);
	kernel_rowInitialization<<<blocks_per_grid, threads_per_block>>>(d_vertices_csr1.elements, d_row_data_dev.is_visited, d_vertices_dev.row_covers, d_edges_csr_dev.ptrs, SP, N);
	cudaSafeCall(cudaGetLastError(), "Error in kernel_rowInitialization execution f_culap::executeZeroCover");

	while (true)
	{
		compactRowVertices(d_row_data_dev, d_vertices_csr2, d_vertices_csr1, devid); // compact the current vertex frontier.
		cudaSafeCall(cudaFree(d_vertices_csr1.elements), "Error in cudaFree f_culap::executeZeroCover::d_vertices_csr1.elements");
		if (d_vertices_csr2.size == 0)
			break;

		coverZeroAndExpand(d_edges_csr_dev, d_vertices_dev, d_row_data_dev, d_col_data_dev, d_vertices_csr1, d_vertices_csr2, h_flag, N, devid); // traverse the frontier, cover zeros and expand.
		cudaSafeCall(cudaFree(d_vertices_csr2.elements), "Error in cudaFree f_culap::executeZeroCover::d_vertices_csr2.elements");
		if (d_vertices_csr1.size == 0)
			break;
	}
}

// Function for compacting row vertices. Used in Step 3 (minimum zero cover).
void compactRowVertices(VertexData &d_row_data_dev, Array &d_vertices_csr_out, Array &d_vertices_csr_in, unsigned int devid)
{

	cudaSetDevice(devid);

	int total_blocks = 0;
	dim3 blocks_per_grid;
	dim3 threads_per_block;

	Predicates d_vertex_predicates;

	d_vertex_predicates.size = d_vertices_csr_in.size;

	cudaSafeCall(cudaMalloc((void **)(&d_vertex_predicates.predicates), d_vertex_predicates.size * sizeof(bool)), "Error in cudaMalloc f_culap::compactRowVertices::d_vertex_predicates.predicates");
	cudaSafeCall(cudaMalloc((void **)(&d_vertex_predicates.addresses), d_vertex_predicates.size * sizeof(long)), "Error in cudaMalloc f_culap::compactRowVertices::d_vertex_predicates.addresses");
	cudaSafeCall(cudaMemset(d_vertex_predicates.predicates, false, d_vertex_predicates.size * sizeof(bool)), "Error in cudaMemset f_culap::compactRowVertices::d_vertex_predicates.predicates");
	cudaSafeCall(cudaMemset(d_vertex_predicates.addresses, 0, d_vertex_predicates.size * sizeof(long)), "Error in cudaMemset f_culap::compactRowVertices::d_vertex_predicates.addresses");

	calculateLinearDims(blocks_per_grid, threads_per_block, total_blocks, d_vertices_csr_in.size);

	kernel_vertexPredicateConstructionCSR<<<blocks_per_grid, threads_per_block>>>(d_vertex_predicates, d_vertices_csr_in, d_row_data_dev.is_visited);
	cudaSafeCall(cudaGetLastError(), "Error in kernel_vertexPredicateConstructionCSR f_culap::compactRowVertices");

	thrust::device_ptr<long> ptr(d_vertex_predicates.addresses);
	d_vertices_csr_out.size = thrust::reduce(ptr, ptr + d_vertex_predicates.size); // calculate total number of vertices.
	thrust::exclusive_scan(ptr, ptr + d_vertex_predicates.size, ptr);			   // exclusive scan for calculating the scatter addresses.

	if (d_vertices_csr_out.size > 0)
	{
		cudaSafeCall(cudaMalloc((void **)(&d_vertices_csr_out.elements), d_vertices_csr_out.size * sizeof(int)), "Error in cudaMalloc f_culap::compactRowVertices::d_vertices_csr_out.elements");

		kernel_vertexScatterCSR<<<blocks_per_grid, threads_per_block>>>(d_vertices_csr_out.elements, d_vertices_csr_in.elements, d_vertex_predicates);
		cudaSafeCall(cudaGetLastError(), "Error in kernel_vertexScatterCSR f_culap::compactRowVertices");
	}

	cudaSafeCall(cudaFree(d_vertex_predicates.predicates), "Error in cudaFree f_culap::compactRowVertices::d_vertex_predicates.predicates");
	cudaSafeCall(cudaFree(d_vertex_predicates.addresses), "Error in cudaFree f_culap::compactRowVertices::d_vertex_predicates.addresses");
}

// Function for covering the zeros in uncovered rows and expanding the frontier.
void coverZeroAndExpand(CompactEdges &d_edges_csr_dev, Vertices &d_vertices_dev, VertexData &d_row_data_dev, VertexData &d_col_data_dev, Array &d_vertices_csr_out, Array &d_vertices_csr_in, bool *h_flag, int N, unsigned int devid)
{

	cudaSetDevice(devid);

	int total_blocks = 0;
	dim3 blocks_per_grid;
	dim3 threads_per_block;

	bool *d_flag;
	cudaSafeCall(cudaMalloc((void **)&d_flag, sizeof(bool)), "Error in cudaMalloc f_culap::coverZeroAndExpand::d_flag");
	cudaSafeCall(cudaMemcpy(d_flag, h_flag, sizeof(bool), cudaMemcpyHostToDevice), "Error in cudaMemcpy f_culap::coverZeroAndExpand::h_flag");

	Predicates d_vertex_allocations;
	d_vertex_allocations.size = d_vertices_csr_in.size;
	cudaSafeCall(cudaMalloc((void **)&d_vertex_allocations.predicates, d_vertex_allocations.size * sizeof(bool)), "Error in cudaMalloc f_culap::coverZeroAndExpand::d_vertex_allocations.predicates");
	cudaSafeCall(cudaMalloc((void **)&d_vertex_allocations.addresses, d_vertex_allocations.size * sizeof(long)), "Error in cudaMalloc f_culap::coverZeroAndExpand::d_vertex_allocations.addresses");

	cudaSafeCall(cudaMemset(d_vertex_allocations.predicates, false, d_vertex_allocations.size * sizeof(bool)), "Error in cudaMemset f_culap::coverZeroAndExpand::d_vertex_allocations.predicates");
	cudaSafeCall(cudaMemset(d_vertex_allocations.addresses, 0, d_vertex_allocations.size * sizeof(long)), "Error in cudaMemset f_culap::coverZeroAndExpand::d_vertex_allocations.addresses");

	calculateLinearDims(blocks_per_grid, threads_per_block, total_blocks, d_vertices_csr_in.size);

	kernel_vertexAllocationConstructionCSR<<<blocks_per_grid, threads_per_block>>>(d_vertex_allocations, d_vertices_csr_in, d_edges_csr_dev.ptrs);
	cudaSafeCall(cudaGetLastError(), "Error in kernel_vertexAllocationConstructionCSR f_culap::coverZeroAndExpand");

	thrust::device_ptr<long> ptr(d_vertex_allocations.addresses);
	d_vertices_csr_out.size = thrust::reduce(ptr, ptr + d_vertex_allocations.size); // calculate total number of vertices.
	thrust::exclusive_scan(ptr, ptr + d_vertex_allocations.size, ptr);				// exclusive scan for calculating the scatter addresses.

	if (d_vertices_csr_out.size > 0)
	{
		cudaSafeCall(cudaMalloc((void **)&d_vertices_csr_out.elements, d_vertices_csr_out.size * sizeof(int)), "Error in cudaMalloc f_culap::coverZeroAndExpand::d_vertices_csr_out.elements");

		kernel_coverAndExpand<<<blocks_per_grid, threads_per_block>>>(d_flag, d_vertices_csr_out, d_vertices_csr_in, d_vertex_allocations, d_edges_csr_dev, d_vertices_dev, d_row_data_dev, d_col_data_dev, N);
		cudaSafeCall(cudaGetLastError(), "Error in kernel_coverAndExpand f_culap::coverZeroAndExpand");

		cudaSafeCall(cudaMemcpy(h_flag, d_flag, sizeof(bool), cudaMemcpyDeviceToHost), "Error in cudaMemcpy f_culap::coverZeroAndExpand::d_flag");
	}

	cudaSafeCall(cudaFree(d_flag), "Error in cudaFree f_culap::coverZeroAndExpand::d_next");
	cudaSafeCall(cudaFree(d_vertex_allocations.predicates), "Error in cudaFree f_culap::coverZeroAndExpand::d_vertex_allocations.predicates");
	cudaSafeCall(cudaFree(d_vertex_allocations.addresses), "Error in cudaFree f_culap::coverZeroAndExpand::d_vertex_allocations.addresses");
}

// Function for deleting the CSR matrix on each device.
void deleteCSR(CompactEdges &d_edges_csr_dev, unsigned int devid)
{

	cudaSetDevice(devid);

	cudaSafeCall(cudaFree(d_edges_csr_dev.neighbors), "Error in cudaFree f_culap::deleteCSR::d_edges_csr.neighbors");
	cudaSafeCall(cudaFree(d_edges_csr_dev.ptrs), "Error in cudaFree f_culap::deleteCSR::d_edges_csr_dev.ptrs");
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
	thrust::exclusive_scan(ptr, ptr + d_col_predicates.size, ptr);		   // exclusive scan for calculating the scatter addresses.

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
	thrust::exclusive_scan(ptr, ptr + d_row_predicates.size, ptr);		   // exclusive scan for calculating the scatter addresses.

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

// Function for computing uncovered minimum cost element on each device.
void computeUncoveredMinima(double *d_sp_min, Matrix &d_costs_dev, Vertices &d_vertices_dev, int SP, int N, unsigned int devid)
{

	cudaSetDevice(devid);

	dim3 blocks_per_grid;
	dim3 threads_per_block;
	int total_blocks;

	double *d_device_min;

	cudaSafeCall(cudaMalloc((void **)(&d_device_min), SP * N * sizeof(double)), "Error in cudaMalloc f_culap::computeUncoveredMinima::d_device_min");

	calculateRectangularDims(blocks_per_grid, threads_per_block, total_blocks, N, SP);
	kernel_computeUncoveredMinima1<<<blocks_per_grid, threads_per_block>>>(d_device_min, d_costs_dev.elements, d_vertices_dev.row_duals, d_vertices_dev.col_duals, d_vertices_dev.row_covers, d_vertices_dev.col_covers, SP, N);
	cudaSafeCall(cudaGetLastError(), "Error in kernel_computeUncoveredMinima1 execution f_culap::computeUncoveredMinima");

	calculateLinearDims(blocks_per_grid, threads_per_block, total_blocks, SP);
	kernel_computeUncoveredMinima2<<<blocks_per_grid, threads_per_block>>>(d_sp_min, d_device_min, SP, N);
	cudaSafeCall(cudaGetLastError(), "Error in kernel_computeUncoveredMinima2 execution f_culap::computeUncoveredMinima");

	cudaSafeCall(cudaFree(d_device_min), "Error in cudaFree f_culap::computeUncoveredMinima::d_device_min");
}

// Function for updating the dual variables.
void updateDuals(double *d_sp_min, Vertices &d_vertices_dev, int SP, int N, unsigned int devid)
{

	cudaSetDevice(devid);

	dim3 blocks_per_grid;
	dim3 threads_per_block;
	int total_blocks = 0;

	calculateRectangularDims(blocks_per_grid, threads_per_block, total_blocks, N, SP);

	kernel_dualUpdate<<<blocks_per_grid, threads_per_block>>>(d_sp_min, d_vertices_dev.row_duals, d_vertices_dev.col_duals, d_vertices_dev.row_covers, d_vertices_dev.col_covers, SP, N);
	cudaSafeCall(cudaGetLastError(), "Error in kernel_dualUpdate execution f_culap::updateDuals");
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

// Kernel for reducing the rows by subtracting row minimum from each row element.
__global__ void kernel_rowReduction(double *d_costs, double *d_row_duals, double *d_col_duals, int SP, int N)
{

	int spid = blockIdx.y * blockDim.y + threadIdx.y;
	int rowid = blockIdx.x * blockDim.x + threadIdx.x;
	double min = BIG_NUMBER;

	if (spid < SP && rowid < N)
	{

		for (int colid = 0; colid < N; colid++)
		{

			double slack = d_costs[spid * N * N + rowid * N + colid];

			if (slack <= min)
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

	double min = BIG_NUMBER;

	if (spid < SP && colid < N)
	{
		for (int rowid = 0; rowid < N; rowid++)
		{

			double cost = d_costs[spid * N * N + rowid * N + colid];
			double row_dual = d_row_duals[spid * N + rowid];

			double slack = cost - row_dual;

			if (slack <= min)
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

	double min = BIG_NUMBER;

	if (spid < SP && rowid < N)
	{

		double row_dual = d_row_duals[ROWID];

		for (int colid = 0; colid < N; colid++)
		{

			double cost = d_costs[spid * N * N + rowid * N + colid];
			double col_dual = d_col_duals[spid * N + colid];
			double slack = cost - row_dual - col_dual;

			if (slack <= min)
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

// Kernel for populating the cover arrays.
__global__ void kernel_computeRowCovers(int *d_row_assignments, int *d_row_covers, int SP, int N)
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
	}
}

////////////////////////////////////////////////////////////////////////////////////////

// Kernel for populating the predicate matrix for edges in row major format.
__global__ void kernel_edgePredicateConstructionCSR(Predicates d_edge_predicates_csr, double *d_costs, double *d_row_duals, double *d_col_duals, int SP, int N)
{
	int spid = blockIdx.z * blockDim.z + threadIdx.z;
	int rowid = blockIdx.y * blockDim.y + threadIdx.y;
	int colid = blockIdx.x * blockDim.x + threadIdx.x;

	int ID = spid * N * N + rowid * N + colid;
	int ROWID = spid * N + rowid;
	int COLID = spid * N + colid;

	if (spid < SP && rowid < N && colid < N)
	{

		double cost = d_costs[ID];
		double row_dual = d_row_duals[ROWID];
		double col_dual = d_col_duals[COLID];
		double slack = cost - row_dual - col_dual;

		bool predicate = (slack > -EPSILON && slack < EPSILON);
		long addr = predicate ? 1 : 0;

		d_edge_predicates_csr.predicates[ID] = predicate; // Copy the predicate matrix back to global memory
		d_edge_predicates_csr.addresses[ID] = addr;
	}
}

// Kernel for scattering the edges based on the scatter addresses.
__global__ void kernel_edgeScatterCSR(CompactEdges d_edges_csr, Predicates d_edge_predicates_csr, long M, int SP, int N)
{

	int spid = blockIdx.z * blockDim.z + threadIdx.z;
	int rowid = blockIdx.y * blockDim.y + threadIdx.y;
	int colid = blockIdx.x * blockDim.x + threadIdx.x;

	int ID = spid * N * N + rowid * N + colid;
	int ROWID = spid * N + rowid;

	// Copy the matrix into shared memory

	if (spid < SP && rowid < N && colid < N)
	{

		bool predicate = d_edge_predicates_csr.predicates[ID];
		long compid = d_edge_predicates_csr.addresses[ID];

		if (predicate)
		{
			d_edges_csr.neighbors[compid] = colid;
		}
		if (colid == 0)
		{
			d_edges_csr.ptrs[ROWID] = compid;
			d_edges_csr.ptrs[SP * N] = M; // extra pointer for the total number of edges. necessary for calculating number of edges in each row.
		}
	}
}

// Kernel for initializing the row or column vertices, later used for recursive frontier update (in Step 3).
__global__ void kernel_rowInitialization(int *d_vertex_ids, int *d_visited, int *d_row_covers, long *d_ptrs, int SP, int N)
{

	int spid = blockIdx.y * blockDim.y + threadIdx.y;
	int rowid = blockIdx.x * blockDim.x + threadIdx.x;

	int ROWID = spid * N + rowid;

	if (spid < SP && rowid < N)
	{

		int cover = d_row_covers[ROWID];
		long size = d_ptrs[ROWID + 1] - d_ptrs[ROWID];

		d_vertex_ids[ROWID] = ROWID;
		d_visited[ROWID] = (size == 0) ? VISITED : ((cover == 0) ? ACTIVE : DORMANT);
	}
}

// Kernel for calculating predicates for row vertices.
__global__ void kernel_vertexPredicateConstructionCSR(Predicates d_vertex_predicates, Array d_vertices_csr_in, int *d_visited)
{

	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int size = d_vertices_csr_in.size;

	// Copy the matrix into shared memory.
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
__global__ void kernel_vertexScatterCSR(int *d_vertex_ids_csr, int *d_vertex_ids, Predicates d_vertex_predicates)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int size = d_vertex_predicates.size;

	// Copy the matrix into shared memory.
	int vertexid = (id < size) ? d_vertex_ids[id] : -1;
	bool predicate = (id < size) ? d_vertex_predicates.predicates[id] : false;
	long compid = (predicate) ? d_vertex_predicates.addresses[id] : -1; // compaction id.

	if (id < size)
	{
		if (predicate)
			d_vertex_ids_csr[compid] = vertexid;
	}
}

// Kernel for calculating allocations for new frontier (new unvisited rows).
__global__ void kernel_vertexAllocationConstructionCSR(Predicates d_vertex_allocations, Array d_vertices_csr_in, long *d_ptrs)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	long size = d_vertices_csr_in.size;

	if (id < size)
	{

		int vertexid = d_vertices_csr_in.elements[id];
		long st_ptr = d_ptrs[vertexid];
		long end_ptr = d_ptrs[vertexid + 1];
		long allocation_size = end_ptr - st_ptr;
		bool predicate = (allocation_size > 0);

		d_vertex_allocations.predicates[id] = predicate;
		d_vertex_allocations.addresses[id] = allocation_size;
	}
}

// Kernel for finding the minimum zero cover.
__global__ void kernel_coverAndExpand(bool *d_flag, Array d_vertices_csr_out, Array d_vertices_csr_in, Predicates d_vertex_allocations, CompactEdges d_edges_csr, Vertices d_vertices, VertexData d_row_data, VertexData d_col_data, int N)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	int in_size = d_vertices_csr_in.size;
	int out_size = d_vertices_csr_out.size;

	if (id < in_size)
	{

		int vertexid = d_vertices_csr_in.elements[id];
		long alloc_id = d_vertex_allocations.addresses[id]; // allocation id.
		long alloc_id_nxt = (id < in_size - 1) ? d_vertex_allocations.addresses[id + 1] : out_size;
		long allocation_size = alloc_id_nxt - alloc_id;

		long st_ptr = d_edges_csr.ptrs[vertexid];
		int *allocation_start = &d_vertices_csr_out.elements[alloc_id];
		int *neighbor_start = &d_edges_csr.neighbors[st_ptr];
		int *neighbor_end = neighbor_start + allocation_size;

		__update_covers(d_vertices, d_flag, d_row_data.parents, d_col_data.parents, d_row_data.is_visited, d_col_data.is_visited, allocation_start, neighbor_start, neighbor_end, vertexid, N);
		d_row_data.is_visited[vertexid] = VISITED;
	}
}

////////////////////////////////////////////////////////////////////////////////////////

// Kernel for constructing the predicates for reverse pass or augmentation candidates.
__global__ void kernel_augmentPredicateConstruction(Predicates d_predicates, int *d_visited, int size)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	// Copy the matrix into shared memory
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

// Kernel for updating the dual reduced costs in Step 5, without using atomic functions.
__global__ void kernel_computeUncoveredMinima1(double *d_min_val, double *d_costs, double *d_row_duals, double *d_col_duals, int *d_row_covers, int *d_col_covers, int SP, int N)
{

	int spid = blockIdx.y * blockDim.y + threadIdx.y;
	int colid = blockIdx.x * blockDim.x + threadIdx.x;

	int COLID = spid * N + colid;

	double min = BIG_NUMBER;

	if (spid < SP && colid < N)
	{
		double col_dual = d_col_duals[COLID];
		int col_cover = d_col_covers[COLID];

		for (int rowid = 0; rowid < N; rowid++)
		{

			double cost = d_costs[spid * N * N + rowid * N + colid];
			double row_dual = d_row_duals[spid * N + rowid];
			int row_cover = d_row_covers[spid * N + rowid];

			double slack = cost - row_dual - col_dual;

			if (row_cover == 0 && col_cover == 0 && slack <= min)
				min = slack;
		}

		d_min_val[COLID] = min;
	}
}

// Kernel for updating the dual reduced costs in Step 5, without using atomic functions.
__global__ void kernel_computeUncoveredMinima2(double *d_sp_min, double *d_min_val, int SP, int N)
{

	int spid = blockIdx.x * blockDim.x + threadIdx.x;

	double min = BIG_NUMBER;

	if (spid < SP)
	{

		for (int colid = 0; colid < N; colid++)
		{

			double val = d_min_val[spid * N + colid];

			if (val <= min)
				min = val;
		}

		d_sp_min[spid] = min;
	}
}

// Kernel for updating the dual values in Step 5.
__global__ void kernel_dualUpdate(double *d_sp_min, double *d_row_duals, double *d_col_duals, int *d_row_covers, int *d_col_covers, int SP, int N)
{

	int spid = blockIdx.y * blockDim.y + threadIdx.y;
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	int ID = spid * N + id;

	if (spid < SP && id < N)
	{

		double theta = (double)d_sp_min[spid] / 2;
		int row_cover = d_row_covers[ID];
		int col_cover = d_col_covers[ID];

		if (row_cover == 0) // Row vertex is reachable from source.
			d_row_duals[ID] += theta;

		else
			// Row vertex is unreachable from source.
			d_row_duals[ID] -= theta;

		if (col_cover == 0) // Col vertex is unreachable from source.
			d_col_duals[ID] += theta;

		else
			// Col vertex is reachable from source.
			d_col_duals[ID] -= theta;
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

// Device function for traversing the neighbors from start pointer to end pointer and updating the covers.
// The function sets d_next to 4 if there are uncovered zeros, indicating the requirement of Step 4 execution.
__device__ void __update_covers(Vertices d_vertices, bool *d_flag, int *d_row_parents, int *d_col_parents, int *d_row_visited, int *d_col_visited, int *new_frontier, int *d_start_ptr, int *d_end_ptr, int ROWID, int N)
{
	int *ptr1 = d_start_ptr;
	int *ptr2 = new_frontier;

	int spid = ROWID / N;

	while (ptr1 != d_end_ptr)
	{
		int colid = *ptr1;
		int COLID = spid * N + colid;

		int _rowid = d_vertices.col_assignments[COLID];
		int _ROWID = spid * N + _rowid;

		if (ROWID != _ROWID && d_vertices.col_covers[COLID] == 0)
		{ // if the column is already covered, it should not be included in next frontier expansion.

			d_col_parents[COLID] = ROWID; // update parent info

			if (_rowid != -1)
			{

				d_row_parents[_ROWID] = COLID; // update parent info

				d_vertices.row_covers[_ROWID] = 0;
				d_vertices.col_covers[COLID] = 1;

				if (d_row_visited[_ROWID] == DORMANT)
					d_row_visited[_ROWID] = ACTIVE;
			}
			else
			{ // activate the column for maximum matching step (Step 4).

				d_col_visited[COLID] = REVERSE;
				*d_flag = true;
			}

			*ptr2 = _ROWID;
		}
		else
		{
			*ptr2 = -1;
		}
		ptr1++;
		ptr2++;
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
