#pragma once

#include <iostream>
#include <fstream>
#include <cuda.h>
#include <thrust/scan.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "d_vars.h"
#include <sstream>
#include <random>
#include "../include/Timer.h"
#include <math.h>

#define cudaSafeCall_new(ans, message)     \
	{                                        \
		gpuAssert1((ans), __FILE__, __LINE__); \
	}
inline void gpuAssert1(cudaError_t code, const char *file, int line, bool abort = false)
{
	cudaDeviceSynchronize();
	if (code != cudaSuccess)
	{
		// fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		printf("GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);

		// if (abort)
		exit(1);
	}
}

#define Hcheckpoint()                              \
	{                                                \
		printf("Reached %s:%u\n", __FILE__, __LINE__); \
	}

// Helper function for printing device errors.
void cudaSafeCall(cudaError_t error, const char *message)
{
	if (error != cudaSuccess)
	{
		std::cerr << "Error " << error << ": " << message << ": " << cudaGetErrorString(error) << std::endl;
		exit(-1);
	}
}

// void cudaSafeCall(cudaError_t error, const char *message) { gpuAssert(error, __FILE__, __LINE__); }
// inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = false)
// {

// 	if (code != cudaSuccess)
// 	{
// 		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
// 		if (abort)
// 			exit(1);
// 	}
// }

// Helper function for printing device memory info.
void printMemoryUsage(double memory)
{
	size_t free_byte;
	size_t total_byte;

	cudaSafeCall(cudaMemGetInfo(&free_byte, &total_byte), "Error in cudaMemGetInfo");

	double free_db = (double)free_byte;
	double total_db = (double)total_byte;
	double used_db = total_db - free_db;

	if (memory < used_db)
		memory = used_db;

	printf("used = %f MB, free = %f MB, total = %f MB\n", used_db / 1024.0 / 1024.0, free_db / 1024.0 / 1024.0, total_db / 1024.0 / 1024.0);
}

// Function for calculating grid and block dimensions from the given input size.
void calculateLinearDims(dim3 &blocks_per_grid, dim3 &threads_per_block, int &total_blocks, int size)
{
	threads_per_block.x = BLOCKDIMX * BLOCKDIMY;

	int value = size / threads_per_block.x;
	if (size % threads_per_block.x > 0)
		value++;

	total_blocks = value;
	blocks_per_grid.x = value;
}

// Function for calculating grid and block dimensions from the given input size for square grid.
void calculateSquareDims(dim3 &blocks_per_grid, dim3 &threads_per_block, int &total_blocks, int size)
{
	threads_per_block.x = BLOCKDIMX;
	threads_per_block.y = BLOCKDIMY;

	int sq_size = (int)ceil(sqrt(size));

	int valuex = (int)ceil((double)(sq_size) / BLOCKDIMX);
	int valuey = (int)ceil((double)(sq_size) / BLOCKDIMY);

	total_blocks = valuex * valuey;
	blocks_per_grid.x = valuex;
	blocks_per_grid.y = valuey;
}

// Function for calculating grid and block dimensions from the given input size for rectangular grid.
void calculateRectangularDims(dim3 &blocks_per_grid, dim3 &threads_per_block, int &total_blocks, int xsize, int ysize)
{

	threads_per_block.x = BLOCKDIMX;
	threads_per_block.y = BLOCKDIMY;

	int valuex = xsize / threads_per_block.x;
	if (xsize % threads_per_block.x > 0)
		valuex++;

	int valuey = ysize / threads_per_block.y;
	if (ysize % threads_per_block.y > 0)
		valuey++;

	total_blocks = valuex * valuey;
	blocks_per_grid.x = valuex;
	blocks_per_grid.y = valuey;
}

// Function for calculating grid and block dimensions from the given input size for cubic grid.
void calculateCubicDims(dim3 &blocks_per_grid, dim3 &threads_per_block, int &total_blocks, int xsize, int ysize, int zsize)
{

	threads_per_block.x = BLOCKDIMX;
	threads_per_block.y = BLOCKDIMY;
	threads_per_block.z = BLOCKDIMZ;

	int valuex = xsize / threads_per_block.x;
	if (xsize % threads_per_block.x > 0)
		valuex++;

	int valuey = ysize / threads_per_block.y;
	if (ysize % threads_per_block.y > 0)
		valuey++;

	int valuez = zsize / threads_per_block.z;
	if (zsize % threads_per_block.z > 0)
		valuez++;

	total_blocks = valuex * valuey * valuez;
	blocks_per_grid.x = valuex;
	blocks_per_grid.y = valuey;
	blocks_per_grid.z = valuez;
}

// Function for printing the output log.
void printLog(int prno, int repetition, int numprocs, int numdev, int costrange, long obj_val, int init_assignments, double total_time, int *stepcounts, double *steptimes, const char *logpath, int N)
{
	std::ofstream logfile(logpath, std::ios_base::app);

	logfile << prno << "\t" << numprocs << "\t" << numdev << "\t" << N << "\t[0, " << costrange << "]\t" << obj_val << "\t" << init_assignments << "\t" << stepcounts[0] << "\t" << stepcounts[1] << "\t" << stepcounts[2] << "\t" << stepcounts[3] << "\t" << stepcounts[4] << "\t" << stepcounts[5] << "\t" << stepcounts[6] << "\t" << steptimes[0] << "\t" << steptimes[1] << "\t" << steptimes[2] << "\t" << steptimes[3] << "\t" << steptimes[4] << "\t" << steptimes[5] << "\t" << steptimes[6] << "\t"
					<< steptimes[7] << "\t" << steptimes[8] << "\t" << total_time << std::endl;

	logfile.close();
}

// Function for sequential exclusive scan.
void exclusiveSumScan(int *array, int size)
{

	int sum = 0;
	int val = 0;

	for (int i = 0; i <= size; i++)
	{
		sum += val;
		val = array[i];
		array[i] = sum;
	}
}

// Function for sequential exclusive scan.
void exclusiveSumScan(long *array, int size)
{

	long sum = 0;
	long val = 0;

	for (int i = 0; i <= size; i++)
	{
		sum += val;
		val = array[i];
		array[i] = sum;
	}
}

void exclusiveSumScan(long *array, long size)
{

	long sum = 0;
	long val = 0;

	for (int i = 0; i <= size; i++)
	{
		sum += val;
		val = array[i];
		array[i] = sum;
	}
}

// Function for sequential exclusive scan.
void exclusiveSumScan(double *array, int size)
{

	double sum = 0;
	double val = 0;

	for (long i = 0; i <= size; i++)
	{
		sum += val;
		val = array[i];
		array[i] = sum;
	}
}

// Function for reducing an array (SUM operation)
int reduceSUM(int *array, int size)
{
	int val = 0;

	for (int i = 0; i < size; i++)
	{
		val += array[i];
	}

	return val;
}

// Function for reducing an array (SUM operation)
long reduceSUM(long *array, int size)
{
	long val = 0;

	for (int i = 0; i < size; i++)
	{
		val += array[i];
	}

	return val;
}

// Function for reducing an array (SUM operation)
double reduceSUM(double *d_array, int size, int devid)
{
	double val = 0;

	cudaSetDevice(devid);

	double *h_array = new double[size];

	// std::cout << name << " dev " << devid << std::endl;
	cudaMemcpy(h_array, d_array, size * sizeof(double), cudaMemcpyDeviceToHost);

	for (int i = 0; i < size; i++)
	{
		val += h_array[i];
	}

	return val;
}

// Function for reducing an array (SUM operation)
long reduceSUM(long *array, long size)
{
	long val = 0;

	for (long i = 0; i < size; i++)
	{
		val += array[i];
	}

	return val;
}
double reduceSUM(double *array, int size)
{
	double val = 0;

	for (int i = 0; i < size; i++)
	{
		val += array[i];
	}

	return val;
}
// Function for reducing an array (SUM operation)
long reduceSUM(int *array, long size)
{
	long val = 0;

	for (int i = 0; i < size; i++)
	{
		val += array[i];
	}

	return val;
}

// Function for reducing an array (SUM operation)
double reduceMIN(double *array, int size)
{
	double val = INF;

	for (int i = 0; i < size; i++)
	{
		if (array[i] <= val - EPSILON)
			val = array[i];
	}

	return val;
}

// Function for reducing an array (OR operation)
bool reduceOR(bool *array, int size)
{
	bool val = false;

	for (int i = 0; i < size; i++)
	{
		val = val || array[i];
	}

	return val;
}

void printDebugArray(int *d_array, int size, const char *name)
{

	int *h_array = new int[size];

	std::cout << name << std::endl;
	cudaMemcpy(h_array, d_array, size * sizeof(int), cudaMemcpyDeviceToHost);
	for (int i = 0; i < size; i++)
	{
		std::cout << h_array[i] << "\t";
	}
	std::cout << std::endl;

	delete[] h_array;
}

void printDebugArray(long *d_array, int size, const char *name)
{

	long *h_array = new long[size];

	std::cout << name << std::endl;
	cudaMemcpy(h_array, d_array, size * sizeof(long), cudaMemcpyDeviceToHost);
	for (int i = 0; i < size; i++)
	{
		std::cout << h_array[i] << "\t";
	}
	std::cout << std::endl;

	delete[] h_array;
}

void printDebugArrayDouble(double *d_array, int size, const char *name)
{

	double *h_array = new double[size];

	std::cout << name << std::endl;
	cudaMemcpy(h_array, d_array, size * sizeof(double), cudaMemcpyDeviceToHost);
	for (int i = 0; i < size; i++)
	{
		std::cout << h_array[i] << "\t";
	}
	std::cout << std::endl;

	delete[] h_array;
}

void printDebugMatrix(double *d_matrix, int rowsize, int colsize, const char *name)
{
	double *h_matrix = new double[rowsize * colsize];

	std::cout << name << std::endl;
	cudaMemcpy(h_matrix, d_matrix, rowsize * colsize * sizeof(double), cudaMemcpyDeviceToHost);
	for (int i = 0; i < rowsize; i++)
	{
		for (int j = 0; j < colsize; j++)
		{
			std::cout << h_matrix[i * colsize + j] << "\t";
		}
		std::cout << std::endl;
	}

	std::cout << std::endl;

	delete[] h_matrix;
}

void printDebugMatrix(int *d_matrix, int rowsize, int colsize, const char *name)
{
	int *h_matrix = new int[rowsize * colsize];

	std::cout << name << std::endl;
	cudaMemcpy(h_matrix, d_matrix, rowsize * colsize * sizeof(int), cudaMemcpyDeviceToHost);
	for (int i = 0; i < rowsize; i++)
	{
		for (int j = 0; j < colsize; j++)
		{
			std::cout << h_matrix[i * colsize + j] << "\t";
		}
		std::cout << std::endl;
	}

	std::cout << std::endl;

	delete[] h_matrix;
}

void printHostArray(int *h_array, int size, const char *name)
{
	std::cout << name << std::endl;

	for (int i = 0; i < size; i++)
	{
		std::cout << h_array[i] << "\t";
	}
	std::cout << std::endl;
}

void printHostArray(double *h_array, int size, const char *name)
{
	std::cout << name << std::endl;

	for (int i = 0; i < size; i++)
	{
		std::cout << h_array[i] << "\t";
	}
	std::cout << std::endl;
}

void printHostMatrix(double *h_matrix, int rowsize, int colsize, const char *name)
{

	std::cout << name << std::endl;
	for (int i = 0; i < rowsize; i++)
	{
		for (int j = 0; j < colsize; j++)
		{
			std::cout << h_matrix[i * colsize + j] << "\t";
		}
		std::cout << std::endl;
	}

	std::cout << std::endl;
}

void printHostMatrix(int *h_matrix, int rowsize, int colsize, const char *name)
{

	std::cout << name << std::endl;
	for (int i = 0; i < rowsize; i++)
	{
		for (int j = 0; j < colsize; j++)
		{
			std::cout << h_matrix[i * colsize + j] << "\t";
		}
		std::cout << std::endl;
	}

	std::cout << std::endl;
}

void printHostArray(long *h_array, int size, const char *name)
{
	std::cout << name << std::endl;

	for (int i = 0; i < size; i++)
	{
		std::cout << h_array[i] << "\t";
	}
	std::cout << std::endl;
}

// Function for reading specified input file.
void readFile(double *cost_matrix, const char *filename)
{
	std::string s = filename;
	std::ifstream myfile(s.c_str());

	if (!myfile)
	{
		std::cerr << "Error: input file not found: " << s.c_str() << std::endl;
		exit(-1);
	}

	while (myfile.is_open() && myfile.good())
	{
		int N, K = 0;
		myfile >> N;
		myfile >> K;

		long N2 = N * N * (K - 1);

		for (long i = 0; i < N2; i++)
		{
			myfile >> cost_matrix[i];
		}
	}

	myfile.close();
}

// This function splits "val" equally among the elements "array" of length equal to "size."
void split(int *array, int val, int size)
{

	int split_val = val / size;
	int overflow = val % size;

	std::fill(array, array + size, split_val);

	if (overflow > 0)
	{
		for (int i = 0; i < size; i++)
		{
			array[i]++;
			overflow--;
			if (overflow == 0)
				break;
		}
	}
}

// This function splits "val" equally among the elements "array" of length equal to "size."
void split(long *array, long val, int size)
{

	long split_val = val / size;
	long overflow = val % size;

	std::fill(array, array + size, split_val);

	if (overflow > 0)
	{
		for (int i = 0; i < size; i++)
		{
			array[i]++;
			overflow--;
			if (overflow == 0)
				break;
		}
	}
}

/*
 *	This is a test kernel.
 */
__global__ void kernel_memSet(double *array, double val, int size)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if (id < size)
	{
		array[id] = val;
	}
}

// Function for reading specified input file.

/*
void readFiley(double *cost_matrix, const char *filename)
{
	std::string s = filename;
	std::ifstream myfile(s.c_str());

	if (!myfile)
	{
		std::cerr << "Error: input file not found: " << s.c_str() << std::endl;
		exit(-1);
	}

	while (myfile.is_open() && myfile.good())
	{

		int N = 0;
		int K = 0;
		myfile >> N;
		myfile >> K;
		// long N2 = N * N * N * (K - 2);

		// for(int i =0;i<N2;i++){
		// 		// int id = N * N * N * (p) + (i* N * N) + (j * N) + k ;
		// 		myfile >> cost_matrix[i] ;
		// }

		for (int p = 0; p < K - 2; p++)
		{
			for (int i = 0; i < N; i++)
			{
				for (int j = 0; j < N; j++)
				{
					for (int k = 0; k < N; k++)
					{
						int id = N * N * N * (p) + (k * N * N) + (i * N) + j;
						myfile >> cost_matrix[id];
					}
				}
			}
		}
	}
	myfile.close();
}
*/

void printDebugArray(double *d_array, int size, const char *name, unsigned int devid)
{

	cudaSetDevice(devid);

	double *h_array = new double[size];

	std::cout << name << " dev " << devid << std::endl;
	cudaMemcpy(h_array, d_array, size * sizeof(double), cudaMemcpyDeviceToHost);
	for (int i = 0; i < size; i++)
	{
		std::cout << h_array[i] << "\t";
	}
	std::cout << std::endl;

	delete[] h_array;
}

/*
void printToFile(int *row_assignments, double LB, double UB, double time, int N, int K, int procsize, int numdev)
{

	std::stringstream ss;
	ss << "output_assignments_Samhita_K" << K << "_N" << n << "_S" << scorer << "_problem" << problem_number << "_proc" << procsize << "_dev" << numdev << ".txt";
	std::ofstream fileout(ss.str().c_str());

	for (int i = 0; i < N * K; i++)
	{
		fileout << row_assignments[i] % N << std::endl;
	}
	std::stringstream ss1;
	ss1 << "./Results/Results_K" << K << "_N" << n << "_S" << scorer << "_problem" << problem_number << "_proc" << procsize << "_dev" << numdev << ".txt";
	std::ofstream fileout1(ss1.str().c_str());

	fileout1 << "Lower bound : " << LB << std::endl;
	fileout1 << "Upper bound : " << UB << std::endl;
	fileout1 << "Gap : " << (std::abs(UB - LB) / std::abs(UB)) * 100 << "%" << std::endl;
	fileout1 << "Solution Time : " << time << std::endl;
	// fileout1<<"Iterations: "<<iter<<std::endl;
}

void printToFile2(double LB, double UB, int N, int K, int procsize, int numdev, double time)
{

	std::stringstream ss1;
	ss1 << "./Results/Results_K" << K << "_N" << n << "_S" << scorer << "_problem" << problem_number << "_proc" << procsize << "_dev" << numdev << ".txt";
	std::ofstream fileout1(ss1.str().c_str());

	fileout1 << "Procsize: " << procsize << std::endl;
	fileout1 << "Devices: " << numdev << std::endl;
	fileout1 << "Lower bound : " << LB << std::endl;
	fileout1 << "Upper bound : " << UB << std::endl;
	fileout1 << "Gap : " << (std::abs(UB - LB) / std::abs(UB)) * 100 << "%" << std::endl;
	fileout1 << "Time: " << time << std::endl;
}


void createProbGenData(int *cycle, unsigned long seed)
{
	srand(seed);

	for (int i = 0; i < K; i++)
	{
		for (int j = 0; j < n; j++)
		{
			cycle[i * n + j] = j;
		}

		for (int j = 0; j < n; j++)
		{
			int j1 = rand() % (n - j) + j;
			int temp = cycle[i * n + j1];
			cycle[i * n + j1] = cycle[i * n + j];
			cycle[i * n + j] = temp;
		}
	}
}

void gen_costs(double *cost_matrix, double *y_costs, int *cycle, unsigned long seed, int SP_x, int SP_y, std::size_t N, std::size_t K)
{

	// std::random_device rd;
	std::mt19937 gen(seed);
	double val = 0;
	double value = 0;
	//	int SP = K-2;
	float sigma10 = 0.3;
	float sigma20 = 0.2;
	std::cout << SP_x << std::endl;
	std::size_t p = 0;
	std::size_t i = 0;
	std::size_t j = 0;
#pragma acc parallel
	{
#pragma acc loop independent

		for (p = 0; p < SP_x - 1; p++)
		{
#pragma acc loop independent

			for (i = 0; i < N; i++)
			{
#pragma acc loop independent

				for (j = 0; j < N; j++)
				{

					value = 0;
					val = 0;

					value = std::abs(cycle[p * N + i] - cycle[(p + 1) * N + j]) - 1;
					std::normal_distribution<double> d(value, sigma10);
					val = d(gen);
					std::size_t index = (p * N * N) + (i * N) + j;
					cost_matrix[index] = val;
					//		std::cout<<cost_matrix[index]<<std::endl;
				}
			}
		}
	}

	std::cout << "gen x costs" << std::endl;
	val = 0;
	value = 0;
	// long count = 0;
	std::size_t p2 = 0;
	std::size_t i2 = 0;
	std::size_t j2 = 0;
	std::size_t k2 = 0;
#pragma acc parallel
	{
#pragma acc loop independent
		for (p2 = 0; p2 < SP_y - 1; p2++)
		{
#pragma acc loop independent
			for (i2 = 0; i2 < N; i2++)
			{
#pragma acc loop independent
				for (j2 = 0; j2 < N; j2++)
				{
#pragma acc loop independent
					for (k2 = 0; k2 < N; k2++)
					{

						value = 0;
						val = 0;
						value = cost_matrix[p2 * N * N + (i2 * N) + j2] + cost_matrix[(p2 + 1) * N * N + (j2 * N) + k2];
						std::normal_distribution<double> d(value, sigma20);
						// val = d(gen);
						std::size_t id = N * N * N * (p2) + (k2 * N * N) + (i2 * N) + j2;
						y_costs[id] = d(gen);
						//	std::cout<<y_costs[id]<<std::endl;
					}
				}
			}
		}
	}

	std::cout << "gen y costs" << std::endl;
	std::size_t i3 = 0;
	std::size_t j3 = 0;
	std::size_t k3 = 0;
#pragma acc parallel
	{
#pragma acc loop independent

		for (i3 = 0; i3 < N; i3++)
		{
#pragma acc loop independent

			for (j3 = 0; j3 < N; j3++)
			{
				std::size_t index = ((SP_x - 1) * N * N) + (i3 * N) + j3;
				cost_matrix[index] = 0;
				for (k3 = 0; k3 < N; k3++)
				{
					std::size_t id = N * N * N * (SP_y - 1) + (k3 * N * N) + (i3 * N) + j3;
					y_costs[id] = 0;
				}
			}
		}
	}
	std::cout << "gen last costs" << std::endl;
}

*/