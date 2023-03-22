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
#include "timer.h"
#include <iomanip>

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
	double time;
	Timer t;
	double *cost_matrix = generate_cost<double>(problemsize, costrange);
	time = t.elapsed_and_reset();
	Log(info, "cost generation time %f s", time);

	for (int repeatID = 0; repeatID < repetitions; repeatID++)
	{

		double start = omp_get_wtime();
		double obj_val = 0;
		t.reset();
		LinearAssignmentProblem lpx(problemsize, stepcounts, 1);
		time = t.elapsed_and_reset();
		Log(info, "LAP object generation time %f s", time);

		lpx.solve(cost_matrix, obj_val);
		uint obj = (uint)obj_val;
		double end = omp_get_wtime();

		double total_time = (end - start);

		std::cout << "Size: " << problemsize << "\nrange: " << costrange << std::endl;
		std::cout << "Obj val: " << obj << "\nItn count: " << stepcounts[3] << "\nTotal time: " << total_time << " sec" << std::endl;

		// printHostArray(stepcounts, 7, "step counts: ");
		double *stimes = new double[9];
		lpx.getStepTimes(stimes);
		// printHostArray(stimes, 9, "step times: ");
		/*{
			using namespace std;
			cout << "S0:\t" << stepcounts[0] << "\t" << setprecision(2) << stimes[0] << endl;
			cout << "S1:\t" << stepcounts[1] << "\t" << setprecision(2) << stimes[1] + stimes[2] << endl;
			cout << "S2:\t" << stepcounts[2] << "\t" << setprecision(2) << stimes[3] << endl;
			cout << "S3:\t" << stepcounts[3] << "\t" << setprecision(2) << stimes[4] + stimes[5] << endl;
			cout << "S4:\t" << stepcounts[4] << "\t" << setprecision(2) << stimes[6] << endl;
			cout << "S5:\t" << stepcounts[5] << "\t" << setprecision(2) << stimes[7] << endl;
			// cout << "S6:\t" << stepcounts[6] << "\t" << stimes[8] << endl;
		}*/
		delete[] stimes;
		std::cout << "\n\n\n\n\n";
	}

	delete[] cost_matrix;

	return 0;
}
