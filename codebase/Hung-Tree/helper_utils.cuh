#pragma once

#include <iostream>
#include <fstream>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <thrust/scan.h>
#include <cmath>
#include "structures.h"
#include "variables.h"
#include "logger.cuh"

bool verbose = false;
#define cudaSafeCall(ans, message)                 \
	{                                                \
		gpuAssert((ans), message, __FILE__, __LINE__); \
	}

inline void gpuAssert(cudaError_t error, const char *message, const char *file, int line)
{

	if (error != cudaSuccess)
	{
		fprintf(stderr, "%s \n %s in %s at %d\n", cudaGetErrorString(error), message, file, line);
		exit(-1);
	}
}

#define execKernel(kernel, grid, block, ...)                                              \
	{                                                                                       \
		if (verbose)                                                                          \
			Log(info, "Launching %s with nblocks: %u, blockDim: %u", #kernel, grid.x, block.x); \
		kernel<<<grid, block>>>(__VA_ARGS__);                                                 \
		cudaSafeCall(cudaGetLastError(), "Error in kernel launch");                           \
		cudaSafeCall(cudaDeviceSynchronize(), "Error in Synchronization");                    \
	}

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
void calculateLinearDims(dim3 &blocks_per_grid, dim3 &threads_per_block, int &total_blocks, size_t size)
{
	threads_per_block.x = BLOCKDIMX * BLOCKDIMY;
	int value = (int)ceil((double)(size) / threads_per_block.x);
	total_blocks = value;
	blocks_per_grid.x = value;
}

// Function for calculating grid and block dimensions from the given input size for square grid.
void calculateSquareDims(dim3 &blocks_per_grid, dim3 &threads_per_block, int &total_blocks, size_t size)
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
void calculateRectangularDims(dim3 &blocks_per_grid, dim3 &threads_per_block, int &total_blocks, size_t xsize, size_t ysize)
{

	threads_per_block.x = BLOCKDIMX;
	threads_per_block.y = BLOCKDIMY;

	int valuex = (int)ceil((double)(xsize) / BLOCKDIMX);
	int valuey = (int)ceil((double)(ysize) / BLOCKDIMY);

	total_blocks = valuex * valuey;
	blocks_per_grid.x = valuex;
	blocks_per_grid.y = valuey;
}

// Function for printing the output log.
void printLog(int prno, int repetition, int numprocs, int numdev, int costrange, long obj_val, int init_assignments, double total_time, int *stepcounts, double *steptimes, const char *logpath, int N)
{
	std::ofstream logfile(logpath, std::ios_base::app);

	logfile << prno << "\t" << numprocs << "\t" << numdev << "\t" << N << "\t[0, " << costrange << "]\t" << obj_val << "\t" << init_assignments << "\t" << stepcounts[0] << "\t" << stepcounts[1] << "\t" << stepcounts[2] << "\t" << stepcounts[3] << "\t" << stepcounts[4] << "\t" << stepcounts[5] << "\t" << stepcounts[6] << "\t" << steptimes[0] << "\t" << steptimes[1] << "\t" << steptimes[2] << "\t"
					<< steptimes[3] << "\t" << steptimes[4] << "\t" << steptimes[5] << "\t" << steptimes[6] << "\t" << steptimes[7] << "\t" << steptimes[8] << "\t" << total_time << std::endl;

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

// Function for sequential exclusive scan.
void exclusiveSumScan(long *array, long size)
{

	long sum = 0;
	long val = 0;

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
long reduceSUM(long *array, long size)
{
	long val = 0;

	for (long i = 0; i < size; i++)
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

void printDebugArray(int *d_array, int size, const char *name, unsigned int devid)
{

	cudaSetDevice(devid);

	int *h_array = new int[size];

	std::cout << name << devid << std::endl;
	cudaMemcpyAsync(h_array, d_array, size * sizeof(int), cudaMemcpyDeviceToHost);
	for (int i = 0; i < size; i++)
	{
		std::cout << h_array[i] << ",";
	}
	std::cout << std::endl;

	delete[] h_array;
}

void printDebugArray(long *d_array, int size, const char *name, unsigned int devid)
{

	cudaSetDevice(devid);

	long *h_array = new long[size];

	std::cout << name << devid << std::endl;
	cudaMemcpyAsync(h_array, d_array, size * sizeof(long), cudaMemcpyDeviceToHost);
	for (int i = 0; i < size; i++)
	{
		std::cout << h_array[i] << ",";
	}
	std::cout << std::endl;

	delete[] h_array;
}

void printDebugArray(double *d_array, int size, const char *name, unsigned int devid)
{

	cudaSetDevice(devid);

	double *h_array = new double[size];

	std::cout << name << devid << std::endl;
	cudaMemcpyAsync(h_array, d_array, size * sizeof(double), cudaMemcpyDeviceToHost);
	for (int i = 0; i < size; i++)
	{
		std::cout << h_array[i] << ",";
	}
	std::cout << std::endl;

	delete[] h_array;
}

void printDebugMatrix(int *d_matrix, int rowsize, int colsize, const char *name)
{
	int *h_matrix = new int[rowsize * colsize];

	std::cout << name << std::endl;
	cudaMemcpy(h_matrix, d_matrix, rowsize * colsize * sizeof(int), cudaMemcpyDeviceToHost);
	for (int i = 0; i < rowsize; i++)
	{
		for (int j = 0; j < colsize - 1; j++)
		{
			std::cout << h_matrix[i * colsize + j] << ",";
		}
		std::cout << h_matrix[i * colsize + colsize - 1] << std::endl;
	}

	delete[] h_matrix;
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

void printHostArray(long *h_array, int size, const char *name)
{
	std::cout << name << std::endl;

	for (int i = 0; i < size; i++)
	{
		std::cout << h_array[i] << "\t";
	}
	std::cout << std::endl;
}

// Function for generating problem with uniformly distributed integer costs between [0, COSTRANGE].
// void generateProblem(double *cost_matrix, int N, int costrange)
// {

// 	long N2 = N * N;

// 	for (long i = 0; i < N2; i++)
// 	{
// 		int val = randomGenerator.IRandomX(0, costrange);
// 		cost_matrix[i] = val;
// 	}
// }

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
		int N = 0;
		myfile >> N;

		long N2 = N * N;

		for (long i = 0; i < N2; i++)
		{
			int val = 0;
			myfile >> val;

			cost_matrix[i] = val;
		}
	}

	myfile.close();
}
