/*
 * LinearAssignmentProblem.cpp
 *
 *  Created on: Oct 30, 2014
 *      Author: ketandat
 */

#include <omp.h>
#include "structures.h"
#include "variables.h"
#include "functions_step_0.cuh"
#include "functions_step_1.cuh"
#include "functions_step_2.cuh"
#include "functions_step_3.cuh"
#include "functions_step_4.cuh"
#include "functions_step_5.cuh"

class LinearAssignmentProblem
{

	size_t N;
	size_t N2;
	size_t M; // total number of zero cost edges on a single host.

	int initial_assignment_count;
	int *stepcounts;
	double *steptimes;

	int numdev;
	int prevstep;

	bool flag;

	double obj_val;

	Matrix h_costs, *d_costs_dev;
	Vertices h_vertices, *d_vertices_dev;
	CompactEdges *d_edges_csr_dev;
	VertexData *d_row_data_dev, *d_col_data_dev;

public:
	LinearAssignmentProblem(size_t _size, int *_stepcounts, int _numdev);
	~LinearAssignmentProblem();

	int solve(double *cost_matrix, double &obj_val);

	void getAssignments(int *_row_assignments);
	void getStepTimes(double *_steptimes) { memcpy(_steptimes, steptimes, 9 * sizeof(double)); };

private:
	void initializeDevice(int devid);
	void finalizeDev(int devid);

	int hungarianStep0(bool count_time);
	int hungarianStep1(bool count_time);
	int hungarianStep2(bool count_time);
	int hungarianStep3(bool count_time);
	int hungarianStep4(bool count_time);
	int hungarianStep5(bool count_time);
	int hungarianStep6(bool count_time);
};

LinearAssignmentProblem::LinearAssignmentProblem(size_t _size, int *_stepcounts, int _numdev)
{

	N = _size;
	N2 = N * N;
	M = 0;
	numdev = 1;

	prevstep = 0;

	obj_val = 0;

	flag = false;

	int device_count;
	cudaGetDeviceCount(&device_count);
	numdev = (numdev < device_count) ? numdev : device_count;

	initial_assignment_count = 0;

	stepcounts = _stepcounts;
	steptimes = new double[9];

	d_costs_dev = new Matrix[numdev];
	d_vertices_dev = new Vertices[numdev];
	d_edges_csr_dev = new CompactEdges[numdev];
	d_row_data_dev = new VertexData[numdev];
	d_col_data_dev = new VertexData[numdev];

	h_vertices.row_assignments = new int[N];
	h_vertices.col_assignments = new int[N];
}

LinearAssignmentProblem::~LinearAssignmentProblem()
{

	delete[] steptimes;

	delete[] h_vertices.row_assignments;
	delete[] h_vertices.col_assignments;

	delete[] d_costs_dev;
	delete[] d_vertices_dev;
	delete[] d_edges_csr_dev;
	delete[] d_row_data_dev;
	delete[] d_col_data_dev;
}

// Helper function for initializing global variables and arrays on a single host.
void LinearAssignmentProblem::initializeDevice(int devid)
{
	// std::cout << 'device ID ' << devid << std::endl;
	printf("device ID %u\n", devID);

	cudaSetDevice(devID);

	cudaSafeCall(cudaMalloc((void **)(&d_vertices_dev[devid].row_assignments), N * sizeof(int)), "error in cudaMalloc d_row_assignment");
	cudaSafeCall(cudaMalloc((void **)(&d_vertices_dev[devid].col_assignments), N * sizeof(int)), "error in cudaMalloc d_col_assignment");
	cudaSafeCall(cudaMalloc((void **)(&d_vertices_dev[devid].row_covers), N * sizeof(int)), "error in cudaMalloc d_row_covers");
	cudaSafeCall(cudaMalloc((void **)(&d_vertices_dev[devid].col_covers), N * sizeof(int)), "error in cudaMalloc d_col_covers");

	cudaSafeCall(cudaMemset(d_vertices_dev[devid].row_assignments, -1, N * sizeof(int)), "Error in cudaMemset d_row_assignment");
	cudaSafeCall(cudaMemset(d_vertices_dev[devid].col_assignments, -1, N * sizeof(int)), "Error in cudaMemset d_col_assignment");
	cudaSafeCall(cudaMemset(d_vertices_dev[devid].row_covers, 0, N * sizeof(int)), "Error in cudaMemset d_row_covers");
	cudaSafeCall(cudaMemset(d_vertices_dev[devid].col_covers, 0, N * sizeof(int)), "Error in cudaMemset d_col_covers");

	cudaSafeCall(cudaMalloc((void **)(&d_row_data_dev[devid].is_visited), N * sizeof(int)), "Error in cudaMalloc d_row_data.is_visited");
	cudaSafeCall(cudaMalloc((void **)(&d_col_data_dev[devid].is_visited), N * sizeof(int)), "Error in cudaMalloc d_col_data.is_visited");
	cudaSafeCall(cudaMalloc((void **)(&d_col_data_dev[devid].slack), N * sizeof(double)), "Error in cudaMalloc d_col_data.slack");

	cudaSafeCall(cudaMalloc((void **)(&d_row_data_dev[devid].parents), N * sizeof(int)), "Error in cudaMalloc d_row_data.parents");
	cudaSafeCall(cudaMalloc((void **)(&d_row_data_dev[devid].children), N * sizeof(int)), "Error in cudaMalloc d_row_data.children");
	cudaSafeCall(cudaMalloc((void **)(&d_col_data_dev[devid].parents), N * sizeof(int)), "Error in cudaMalloc d_col_data.parents");
	cudaSafeCall(cudaMalloc((void **)(&d_col_data_dev[devid].children), N * sizeof(int)), "Error in cudaMalloc d_col_data.children");

	cudaSafeCall(cudaMalloc((void **)(&d_costs_dev[devid].elements), N2 * sizeof(double)), "error in cudaMalloc d_costs");
	cudaSafeCall(cudaMalloc((void **)(&d_costs_dev[devid].row_duals), N * sizeof(double)), "error in cudaMalloc d_row_duals");
	cudaSafeCall(cudaMalloc((void **)(&d_costs_dev[devid].col_duals), N * sizeof(double)), "error in cudaMalloc d_col_duals");

	cudaSafeCall(cudaMemcpy(d_costs_dev[devid].elements, h_costs.elements, N2 * sizeof(double), cudaMemcpyHostToDevice), "Error in cudaMemcpy h_costs");

	cudaSafeCall(cudaMemset(d_costs_dev[devid].row_duals, 0, N * sizeof(double)), "Error in cudaMemset d_row_duals");
	cudaSafeCall(cudaMemset(d_costs_dev[devid].col_duals, 0, N * sizeof(double)), "Error in cudaMemset d_col_duals");
}

// Helper function for finalizing global variables and arrays on a single host.
void LinearAssignmentProblem::finalizeDev(int devid)
{

	cudaSetDevice(devID);

	cudaSafeCall(cudaFree(d_vertices_dev[devid].row_assignments), "Error in cudaFree d_row_assignment");
	cudaSafeCall(cudaFree(d_vertices_dev[devid].col_assignments), "Error in cudaFree d_col_assignment");
	cudaSafeCall(cudaFree(d_vertices_dev[devid].row_covers), "Error in cudaFree d_row_covers");
	cudaSafeCall(cudaFree(d_vertices_dev[devid].col_covers), "Error in cudaFree d_col_covers");

	cudaSafeCall(cudaFree(d_row_data_dev[devid].is_visited), "Error in cudaFree d_row_data.is_visited");
	cudaSafeCall(cudaFree(d_col_data_dev[devid].is_visited), "Error in cudaFree d_col_data.is_visited");
	cudaSafeCall(cudaFree(d_col_data_dev[devid].slack), "Error in cudaFree d_col_data.slack");

	cudaSafeCall(cudaFree(d_row_data_dev[devid].parents), "Error in cudaFree d_row_data.parents");
	cudaSafeCall(cudaFree(d_row_data_dev[devid].children), "Error in cudaFree d_row_data.children");
	cudaSafeCall(cudaFree(d_col_data_dev[devid].parents), "Error in cudaFree d_col_data.parents");
	cudaSafeCall(cudaFree(d_col_data_dev[devid].children), "Error in cudaFree d_col_data.children");

	cudaSafeCall(cudaFree(d_costs_dev[devid].elements), "error in cudaFree d_edges.costs");
	cudaSafeCall(cudaFree(d_costs_dev[devid].row_duals), "error in cudaFree d_edges.row_duals");
	cudaSafeCall(cudaFree(d_costs_dev[devid].col_duals), "error in cudaFree d_edges.col_duals");

	cudaDeviceReset();
}

// Executes Hungarian algorithm on the input cost matrix. Returns minimum cost.
int LinearAssignmentProblem::solve(double *_cost_matrix, double &_obj_val)
{

	h_costs.elements = _cost_matrix;

	initializeDevice(0);
	obj_val = 0;
	int step = 0;
	int total_count = 0;
	bool done = false;
	prevstep = -1;

	std::fill(stepcounts, stepcounts + 7, 0);
	std::fill(steptimes, steptimes + 9, 0);

	while (!done)
	{
		total_count++;
		//		std::cout << step << std::endl;
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

	finalizeDev(0);

	_obj_val = obj_val;

	return 0;
}

// Function for calculating initial zeros by subtracting row and column minima from each element.
int LinearAssignmentProblem::hungarianStep0(bool count_time)
{

	double start = omp_get_wtime();

	initialReduction(d_costs_dev, N, 0);

	double end = omp_get_wtime();

	if (count_time)
		steptimes[0] += (end - start);

	prevstep = 0;

	///////////////////////////////////////////////////////////////////
	// printDebugMatrix(d_costs_dev[0].elements, N, N, "costs");
	// printDebugArray(d_costs_dev[0].row_duals, N, "row duals", 0);
	// printDebugArray(d_costs_dev[0].col_duals, N, "col duals", 0);
	///////////////////////////////////////////////////////////////////

	return 1;
}

// Function for calculating initial zeros by subtracting row and column minima from each element.
int LinearAssignmentProblem::hungarianStep1(bool count_time)
{

	double start = omp_get_wtime();

	computeInitialAssignments(d_costs_dev, d_vertices_dev, N, 0);

	double mid = omp_get_wtime();

	int next = 2;

	///////////////////////////////////////////////////////////////////
	// printDebugArray(d_vertices_dev[0].row_assignments, N, "row ass", 0);
	// printDebugArray(d_vertices_dev[0].col_assignments, N, "col ass", 0);
	///////////////////////////////////////////////////////////////////

	//	while (true) {

	//		initial_assignment_count = 0;

	//		if ((next = hungarianStep2(false)) == 6)
	//			break;

	//		if ((next = hungarianStep3(false)) == 5)
	//			break;

	//		hungarianStep4(false);
	//	}

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
int LinearAssignmentProblem::hungarianStep2(bool count_time)
{

	double start = omp_get_wtime();

	initializeStep2(h_vertices, d_vertices_dev, d_row_data_dev, d_col_data_dev, N, 0);

	int cover_count = computeRowCovers(d_vertices_dev, N, 0);

	///////////////////////////////////////////////////////////////////
	// std::cout << "Covers: " << cover_count << std::endl;
	// printDebugArray(d_vertices_dev[0].row_covers, N, "row cover", 0);
	// printDebugArray(d_vertices_dev[0].col_covers, N, "col cover", 0);
	///////////////////////////////////////////////////////////////////

	if (initial_assignment_count == 0)
		initial_assignment_count = cover_count;

	int next = (cover_count == N) ? 6 : 3;

	double end = omp_get_wtime();

	if (count_time)
		steptimes[3] += (end - start);

	prevstep = 2;

	if (next == 6)
	{
		cudaSafeCall(cudaMemcpy(h_vertices.row_assignments, d_vertices_dev[0].row_assignments, N * sizeof(int), cudaMemcpyDeviceToHost),
								 "Error in cudaMemcpy d_vertices_dev[devid].row_assignments");
		cudaSafeCall(cudaMemcpy(h_vertices.col_assignments, d_vertices_dev[0].col_assignments, N * sizeof(int), cudaMemcpyDeviceToHost),
								 "Error in cudaMemcpy d_vertices_dev[devid].col_assignments");
	}

	return next;
}

// Function for building alternating tree rooted at unassigned rows.
int LinearAssignmentProblem::hungarianStep3(bool count_time)
{

	double start = omp_get_wtime();

	///////////////////////////////////////////////////////////////

	double mid = omp_get_wtime();

	///////////////////////////////////////////////////////////////

	int next;

	// bool goto_4 = false;
	bool *goto_4;
	cudaMallocManaged((void **)&goto_4, sizeof(bool));

	// execute zero cover algorithm.
	executeZeroCover(d_costs_dev, d_vertices_dev, d_row_data_dev, d_col_data_dev, goto_4, N, 0);
	cudaSafeCall(cudaDeviceSynchronize(), "Error in synchronizing device with host");

	next = (*goto_4) ? 4 : 5;

	///////////////////////////////////////////////////////////////

	double end = omp_get_wtime();

	if (count_time)
	{
		steptimes[4] += (mid - start);
		steptimes[5] += (end - mid);
	}

	///////////////////////////////////////////////////////////////////
	// printDebugArray(d_vertices_dev[0].row_covers, N, "row cover", 0);
	// printDebugArray(d_vertices_dev[0].col_covers, N, "col cover", 0);
	///////////////////////////////////////////////////////////////////

	prevstep = 3;
	cudaFree(goto_4);
	return next;
}

// Function for augmenting the solution along multiple node-disjoint alternating trees.
int LinearAssignmentProblem::hungarianStep4(bool count_time)
{

	double start = omp_get_wtime();

	///////////////////////////////////////////////////////////////

	reversePass(d_row_data_dev, d_col_data_dev, N, 0);											// execute reverse pass of the maximum matching algorithm.
	augmentationPass(d_vertices_dev, d_row_data_dev, d_col_data_dev, N, 0); // execute augmentation pass of the maximum matching algorithm.

	///////////////////////////////////////////////////////////////

	double end = omp_get_wtime();

	if (count_time)
		steptimes[6] += (end - start);

	prevstep = 4;

	return 2;
}

// Function for updating dual solution to introduce new zero-cost arcs.
int LinearAssignmentProblem::hungarianStep5(bool count_time)
{

	double start = omp_get_wtime();
	double h_device_min = 0;

	computeTheta(h_device_min, d_costs_dev, d_vertices_dev, d_row_data_dev, d_col_data_dev, N, 0);

	///////////////////////////////////////////////////////////////////
	// printDebugMatrix(d_costs_dev[0].elements, N, N, "costs");
	// printDebugArray(d_costs_dev[0].row_duals, N, "row duals", 0);
	// printDebugArray(d_costs_dev[0].col_duals, N, "col duals", 0);
	///////////////////////////////////////////////////////////////////

	double end = omp_get_wtime();

	if (count_time)
		steptimes[7] += (end - start);

	prevstep = 5;

	return 3;
}

int LinearAssignmentProblem::hungarianStep6(bool count_time)
{

	// consoleOut();

	double start = omp_get_wtime();

	obj_val = 0;

	for (int i = 0; i < N; i++)
	{
		int rowid = i;
		int colid = h_vertices.row_assignments[rowid];
		obj_val += h_costs.elements[i * N + colid];
	}

	//	printMemoryUsage ();
	//	printf("used = %f MB\n", memory/1024.0/1024.0);

	double end = omp_get_wtime();

	if (count_time)
		steptimes[8] += (end - start);

	prevstep = 6;

	return 100;
}
