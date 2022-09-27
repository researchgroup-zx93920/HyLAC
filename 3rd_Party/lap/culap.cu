/*
 * LinearAssignmentProblem.cpp
 *
 *  Created on: Oct 30, 2014
 *      Author: ketandate
 */

#include "include/culap.h"

CuLAP::CuLAP(int _size, int _spcount, int _devid, bool _is_dynamic)
{

	N = _size;
	N2 = N * N;

	SP = _spcount;
	devid = _devid;
	dynamic = _is_dynamic;

	M = 0;

	prevstep = 0;

	flag = false;

	initial_assignment_count = 0;

	stepcounts = new int[7];
	steptimes = new double[9];

	d_obj_val_dev = 0;
}

CuLAP::~CuLAP()
{

	delete[] stepcounts;
	delete[] steptimes;
}

// Helper function for initializing global variables and arrays on a single host.
void CuLAP::initializeDevice(void)
{

	cudaSetDevice(devid);

	//	cudaSafeCall(cudaMalloc((void**) (&d_vertices_dev.row_assignments), SP * N * sizeof(int)), "error in cudaMalloc CuLAP::initializeDevice::d_row_assignment");
	cudaSafeCall(cudaMalloc((void **)(&d_vertices_dev.col_assignments), SP * N * sizeof(int)), "error in cudaMalloc CuLAP::initializeDevice::d_col_assignment");
	cudaSafeCall(cudaMalloc((void **)(&d_vertices_dev.row_covers), SP * N * sizeof(int)), "error in cudaMalloc CuLAP::initializeDevice::d_row_covers");
	cudaSafeCall(cudaMalloc((void **)(&d_vertices_dev.col_covers), SP * N * sizeof(int)), "error in cudaMalloc CuLAP::initializeDevice::d_col_covers");

	//	cudaSafeCall(cudaMalloc((void**) (&d_vertices_dev.row_duals), SP * N * sizeof(int)), "error in cudaMalloc CuLAP::initializeDevice::d_row_duals");
	//	cudaSafeCall(cudaMalloc((void**) (&d_vertices_dev.col_duals), SP * N * sizeof(int)), "error in cudaMalloc CuLAP::initializeDevice::d_col_duals");

	//	cudaSafeCall(cudaMemset(d_vertices_dev.row_assignments, -1, SP * N * sizeof(int)), "Error in cudaMemset d_row_assignment");
	cudaSafeCall(cudaMemset(d_vertices_dev.col_assignments, -1, SP * N * sizeof(int)), "Error in cudaMemset CuLAP::initializeDevice::d_col_assignment");
	cudaSafeCall(cudaMemset(d_vertices_dev.row_covers, 0, SP * N * sizeof(int)), "Error in cudaMemset CuLAP::initializeDevice::d_row_covers");
	cudaSafeCall(cudaMemset(d_vertices_dev.col_covers, 0, SP * N * sizeof(int)), "Error in cudaMemset CuLAP::initializeDevice::d_col_covers");

	//	cudaSafeCall(cudaMemset(d_vertices_dev.row_duals, 0, SP * N * sizeof(double)), "Error in cudaMemset CuLAP::initializeDevice::d_row_duals");
	//	cudaSafeCall(cudaMemset(d_vertices_dev.col_duals, 0, SP * N * sizeof(double)), "Error in cudaMemset CuLAP::initializeDevice::d_col_duals");

	cudaSafeCall(cudaMalloc((void **)(&d_row_data_dev.is_visited), SP * N * sizeof(int)), "Error in cudaMalloc CuLAP::initializeDevice::d_row_data.is_visited");
	cudaSafeCall(cudaMalloc((void **)(&d_col_data_dev.is_visited), SP * N * sizeof(int)), "Error in cudaMalloc CuLAP::initializeDevice::d_col_data.is_visited");

	cudaSafeCall(cudaMalloc((void **)(&d_row_data_dev.parents), SP * N * sizeof(int)), "Error in cudaMalloc CuLAP::initializeDevice::d_row_data.parents");
	cudaSafeCall(cudaMalloc((void **)(&d_row_data_dev.children), SP * N * sizeof(int)), "Error in cudaMalloc CuLAP::initializeDevice::d_row_data.children");
	cudaSafeCall(cudaMalloc((void **)(&d_col_data_dev.parents), SP * N * sizeof(int)), "Error in cudaMalloc CuLAP::initializeDevice::d_col_data.parents");
	cudaSafeCall(cudaMalloc((void **)(&d_col_data_dev.children), SP * N * sizeof(int)), "Error in cudaMalloc CuLAP::initializeDevice::d_col_data.children");

	//	cudaSafeCall(cudaMalloc((void**) (&d_costs_dev.elements), SP * N * N * sizeof(double)), "error in cudaMalloc CuLAP::initializeDevice::d_costs");
}

// Helper function for finalizing global variables and arrays on a single host.
void CuLAP::finalizeDevice(void)
{

	cudaSetDevice(devid);

	//	cudaSafeCall(cudaFree(d_vertices_dev.row_assignments), "Error in cudaFree CuLAP::finalizeDevice::d_row_assignment");
	cudaSafeCall(cudaFree(d_vertices_dev.col_assignments), "Error in cudaFree CuLAP::finalizeDevice::d_col_assignment");
	cudaSafeCall(cudaFree(d_vertices_dev.row_covers), "Error in cudaFree CuLAP::finalizeDevice::d_row_covers");
	cudaSafeCall(cudaFree(d_vertices_dev.col_covers), "Error in cudaFree CuLAP::finalizeDevice::d_col_covers");

	//	cudaSafeCall(cudaFree(d_vertices_dev.row_duals), "Error in cudaFree CuLAP::finalizeDevice::d_row_duals");
	//	cudaSafeCall(cudaFree(d_vertices_dev.col_duals), "Error in cudaFree CuLAP::finalizeDevice::d_col_duals");

	cudaSafeCall(cudaFree(d_row_data_dev.is_visited), "Error in cudaFree CuLAP::finalizeDevice::d_row_data.is_visited");
	cudaSafeCall(cudaFree(d_col_data_dev.is_visited), "Error in cudaFree CuLAP::finalizeDevice::d_col_data.is_visited");

	cudaSafeCall(cudaFree(d_row_data_dev.parents), "Error in cudaFree CuLAP::finalizeDevice::d_row_data.parents");
	cudaSafeCall(cudaFree(d_row_data_dev.children), "Error in cudaFree CuLAP::finalizeDevice::d_row_data.children");
	cudaSafeCall(cudaFree(d_col_data_dev.parents), "Error in cudaFree CuLAP::finalizeDevice::d_col_data.parents");
	cudaSafeCall(cudaFree(d_col_data_dev.children), "Error in cudaFree CuLAP::finalizeDevice::d_col_data.children");

	//	cudaSafeCall(cudaFree(d_costs_dev.elements), "error in cudaFree CuLAP::finalizeDevice::d_costs");
}

// Executes Hungarian algorithm on the input cost matrix. Returns minimum cost.
int CuLAP::solve(double *d_cost_matrix, int *d_row_assignments, double *d_row_duals, double *d_col_duals, double *d_obj_val)
{

	d_costs_dev.elements = d_cost_matrix;
	d_vertices_dev.row_assignments = d_row_assignments;
	d_vertices_dev.row_duals = d_row_duals;
	d_vertices_dev.col_duals = d_col_duals;
	d_obj_val_dev = d_obj_val;

	initializeDevice();

	int step = 0;
	int total_count = 0;
	bool done = false;
	prevstep = -1;

	std::fill(stepcounts, stepcounts + 7, 0);
	std::fill(steptimes, steptimes + 9, 0);

	while (!done)
	{
		total_count++;
		switch (step)
		{
		case 0:
			stepcounts[0]++;
			step = hungarianStep0(true);
			break;
		case 1:
			stepcounts[1]++;
			step = hungarianStep1(true);
			break;
		case 2:
			stepcounts[2]++;
			step = hungarianStep2(true);
			break;
		case 3:
			stepcounts[3]++;
			step = hungarianStep3(true);
			break;
		case 4:
			stepcounts[4]++;
			step = hungarianStep4(true);
			break;
		case 5:
			stepcounts[5]++;
			step = hungarianStep5(true);
			break;
		case 6:
			stepcounts[6]++;
			step = hungarianStep6(true);
			break;
		case 100:
			done = true;
			break;
		}
	}

	finalizeDevice();

	return 0;
}

// Function for calculating initial zeros by subtracting row and column minima from each element.
int CuLAP::hungarianStep0(bool count_time)
{

	double start = omp_get_wtime();

	if (dynamic)
		dynamicUpdate(d_costs_dev, d_vertices_dev, SP, N, devid);
	else
		initialReduction(d_costs_dev, d_vertices_dev, SP, N, devid);

	int next = (dynamic) ? 2 : 1;

	double end = omp_get_wtime();

	if (count_time)
		steptimes[0] += (end - start);

	prevstep = 0;

	return next;
}

// Function for calculating initial zeros by subtracting row and column minima from each element.
int CuLAP::hungarianStep1(bool count_time)
{

	double start = omp_get_wtime();

	computeInitialAssignments(d_costs_dev, d_vertices_dev, SP, N, devid);

	double mid = omp_get_wtime();

	int next = 2;

	while (true)
	{

		initial_assignment_count = 0;

		if ((next = hungarianStep2(false)) == 6)
			break;

		if ((next = hungarianStep3(false)) == 5)
			break;

		hungarianStep4(false);
	}

	double end = omp_get_wtime();
	if (count_time)
	{
		steptimes[1] += (mid - start);
		steptimes[2] += (end - mid);
	}
	prevstep = 1;

	return next;
}

// Function for checking optimality and constructing predicates and covers.
int CuLAP::hungarianStep2(bool count_time)
{

	double start = omp_get_wtime();

	int cover_count = computeRowCovers(d_vertices_dev, d_row_data_dev, d_col_data_dev, SP, N, devid);

	if (initial_assignment_count == 0)
		initial_assignment_count = cover_count;

	int next = (cover_count == SP * N) ? 6 : 3;

	if (next == 6 && prevstep != 0)
	{
		deleteCSR(d_edges_csr_dev, 0);
	}

	double end = omp_get_wtime();

	if (count_time)
		steptimes[3] += (end - start);

	prevstep = 2;

	return next;
}

// Function for building alternating tree rooted at unassigned rows.
int CuLAP::hungarianStep3(bool count_time)
{

	double start = omp_get_wtime();

	if (!flag)
	{
		flag = true;
		compactEdgesCSR(d_edges_csr_dev, d_costs_dev, d_vertices_dev, M, SP, N, devid);
	}

	///////////////////////////////////////////////////////////////

	double mid = omp_get_wtime();

	///////////////////////////////////////////////////////////////

	int next;

	bool h_flag = false;

	executeZeroCover(d_edges_csr_dev, d_vertices_dev, d_row_data_dev, d_col_data_dev, &h_flag, SP, N, devid); // execute zero cover algorithm.

	next = h_flag ? 4 : 5;

	///////////////////////////////////////////////////////////////

	double end = omp_get_wtime();

	if (count_time)
	{
		steptimes[4] += (mid - start);
		steptimes[5] += (end - mid);
	}

	prevstep = 3;

	if (next == 5)
	{
		flag = false;
		deleteCSR(d_edges_csr_dev, devid);
	}

	return next;
}

// Function for augmenting the solution along multiple node-disjoint alternating trees.
int CuLAP::hungarianStep4(bool count_time)
{

	double start = omp_get_wtime();

	///////////////////////////////////////////////////////////////

	reversePass(d_row_data_dev, d_col_data_dev, SP, N, devid); // execute reverse pass of the maximum matching algorithm.

	augmentationPass(d_vertices_dev, d_row_data_dev, d_col_data_dev, SP, N, devid); // execute augmentation pass of the maximum matching algorithm.

	///////////////////////////////////////////////////////////////

	double end = omp_get_wtime();

	if (count_time)
		steptimes[6] += (end - start);

	prevstep = 4;

	return 2;
}

// Function for updating dual solution to introduce new zero-cost arcs.
int CuLAP::hungarianStep5(bool count_time)
{

	double start = omp_get_wtime();
	double *d_sp_min;

	cudaSafeCall(cudaMalloc((void **)&d_sp_min, SP * sizeof(double)), "Error in cudaMalloc CuLAP::hungarianStep5::d_sp_min");

	computeUncoveredMinima(d_sp_min, d_costs_dev, d_vertices_dev, SP, N, devid);

	updateDuals(d_sp_min, d_vertices_dev, SP, N, devid);

	cudaSafeCall(cudaFree(d_sp_min), "Error in cudaFree CuLAP::hungarianStep5::d_sp_min");

	double end = omp_get_wtime();

	if (count_time)
		steptimes[7] += (end - start);

	prevstep = 5;

	return 3;
}

int CuLAP::hungarianStep6(bool count_time)
{

	double start = omp_get_wtime();

	calcObjVal(d_obj_val_dev, d_vertices_dev, SP, N, devid);

	double end = omp_get_wtime();

	if (count_time)
		steptimes[8] += (end - start);

	prevstep = 6;

	return 100;
}
