#include <iostream>
#include <fstream>
#include <ctime>
#include <sstream>
#include <omp.h>
#include "structures.h"
#include "variables.h"
#include "helper_utils.cuh"
#include "LinearAssignmentProblem.cuh"
#include "cost_generator.h"

int main(int argc, char **argv)
{

	size_t size = atoi(argv[1]);
	double costrange = std::stod(argv[2]);
	// int devID = 0;
	devID = atoi(argv[3]);
	int repetitions = 1;
	if (argc >= 5)
		repetitions = atoi(argv[4]);

	// int multiplier = 1;

	// int init_assignments = 0;
	int stepcounts[7];
	// double steptimes[9];

	std::stringstream logpath;
	size_t problemsize = size;

	cudaSafeCall(cudaSetDevice(devID), "Error initializing device");
	// double *cost_matrix = new double[problemsize * problemsize];

	double *cost_matrix = generate_cost<double>(problemsize, costrange);

	for (int repeatID = 0; repeatID < repetitions; repeatID++)
	{

		double start = omp_get_wtime();
		double obj_val = 0;
		LinearAssignmentProblem lpx(problemsize, stepcounts, 1);
		lpx.solve(cost_matrix, obj_val);

		double end = omp_get_wtime();

		double total_time = (end - start);

		std::cout << "Size: " << problemsize << "\nrange: " << costrange << std::endl;
		std::cout << "Obj val: " << obj_val << "\tItn count: " << stepcounts[3] << "\nTotal time: " << total_time << " s" << std::endl;

		// printHostArray(stepcounts, 7, "step counts: ");
		// double *stimes = new double[9];
		// lpx.getStepTimes(stimes);
		// printHostArray(stimes, 9, "step times: ");
		// delete[] stimes;
	}

	delete[] cost_matrix;

	return 0;
}
