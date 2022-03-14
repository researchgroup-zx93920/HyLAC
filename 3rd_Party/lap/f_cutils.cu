/*
 * f_cutils.cpp
 *
 *  Created on: Jul 11, 2015
 *      Author: date2
 */

#include "include/f_cutils.h"
#include <iomanip>


// Helper function for printing device errors.
void cudaSafeCall(cudaError_t error, const char *message) {
	if (error != cudaSuccess) {
		std::cerr << "Error " << error << ": " << message << ": " << cudaGetErrorString(error) << std::endl;
		std::cout << "Error was catched and handled by cudaSafeCall" << std::endl;
		exit(-1);
	}
}

void cudaSafeCall(cudaError_t error, const char *message, int line, const char* file) {
	if (error != cudaSuccess) {
		std::cout << message << " at line " << line << " in " << file << std::endl;
		std::cerr << "Error " << error << ": " << cudaGetErrorString(error) << std::endl;
		std::cout << "Error was catched and handled by cudaSafeCall" << std::endl;
		exit(-1);
	}
}

void Initialize_C_hat_Serial(double* C, double* C_hat, int N) {
	int N2 = N*N;
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			for (int t = 0; t < N; t++) {
				C_hat[N2*i + N*j + t] = C[N*i + j];
			}
		}
	}
}
void Initialize_C_hat_Parallel(double* C, double* C_hat, int N) {		//Update Later
	dim3 blocks_per_grid;
	dim3 threads_per_block;
	int total_blocks = 0;
	

	calculateRectangularDims(blocks_per_grid, threads_per_block, total_blocks, N, N);
	int N2 = N*N;
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			for (int t = 0; t < N; t++) {
				C_hat[N2*i + N*j + t] = C[N*i + j];
			}
		}
	}
}

// Helper function for printing device memory info.
void printMemoryUsage(double memory) {
	size_t free_byte;
	size_t total_byte;

	cudaSafeCall(cudaMemGetInfo(&free_byte, &total_byte), "Error in cudaMemGetInfo");

	double free_db = (double) free_byte;
	double total_db = (double) total_byte;
	double used_db = total_db - free_db;

	if (memory < used_db)
		memory = used_db;

	printf("used = %f MB, free = %f MB, total = %f MB\n", used_db / 1024.0 / 1024.0, free_db / 1024.0 / 1024.0, total_db / 1024.0 / 1024.0);
}

// Function for calculating grid and block dimensions from the given input size.
void calculateLinearDims(dim3 &blocks_per_grid, dim3 &threads_per_block, int &total_blocks, int size) {
	threads_per_block.x = BLOCKDIMX * BLOCKDIMY;
	threads_per_block.y = 1;
	threads_per_block.z = 1;
	int value = (int) ceil((double) (size) / threads_per_block.x);
	total_blocks = value;
	blocks_per_grid.x = value;
	blocks_per_grid.y = 1;
	blocks_per_grid.z = 1;

}

// Function for calculating grid and block dimensions from the given input size for square grid.
void calculateSquareDims(dim3 &blocks_per_grid, dim3 &threads_per_block, int &total_blocks, int size) {
	threads_per_block.x = BLOCKDIMX;
	threads_per_block.y = BLOCKDIMY;
	threads_per_block.z = 1;

	int sq_size = (int) ceil(sqrt(size));

	int valuex = (int) ceil((double) (sq_size) / BLOCKDIMX);
	int valuey = (int) ceil((double) (sq_size) / BLOCKDIMY);

	total_blocks = valuex * valuey;
	blocks_per_grid.x = valuex;
	blocks_per_grid.y = valuey;
	blocks_per_grid.z = 1;
}

// Function for calculating grid and block dimensions from the given input size for rectangular grid.
void calculateRectangularDims(dim3 &blocks_per_grid, dim3 &threads_per_block, int &total_blocks, int xsize, int ysize) {

	threads_per_block.x = BLOCKDIMX;
	threads_per_block.y = BLOCKDIMY;
	threads_per_block.z = 1;

	int valuex = (int) ceil((double) (xsize) / BLOCKDIMX);
	int valuey = (int) ceil((double) (ysize) / BLOCKDIMY);

	total_blocks = valuex * valuey;
	blocks_per_grid.x = valuex;
	blocks_per_grid.y = valuey;
	blocks_per_grid.z = 1;
}

// Function for calculating grid and block dimensions from the given input size for cubic grid.
void calculateCubicDims(dim3 &blocks_per_grid, dim3 &threads_per_block, int &total_blocks, int xsize, int ysize, int zsize) {

	threads_per_block.x = BLOCKDIMX;
	threads_per_block.y = BLOCKDIMY;
	threads_per_block.z = BLOCKDIMZ;

	int valuex = (int) ceil((double) (xsize) / BLOCKDIMX);
	int valuey = (int) ceil((double) (ysize) / BLOCKDIMY);
	int valuez = (int) ceil((double) (zsize) / BLOCKDIMZ);

	total_blocks = valuex * valuey * valuez;
	blocks_per_grid.x = valuex;
	blocks_per_grid.y = valuey;
	blocks_per_grid.z = valuez;
}

// Function for printing the output log.
void printLog(int prno, int repetition, int numprocs, int numdev, int costrange, long obj_val, int init_assignments, double total_time, int *stepcounts, double *steptimes, const char *logpath, int N) {
	std::ofstream logfile(logpath, std::ios_base::app);

	logfile << prno << "\t" << numprocs << "\t" << numdev << "\t" << N << "\t[0, " << costrange << "]\t" << obj_val << "\t" << init_assignments << "\t" << stepcounts[0] << "\t" << stepcounts[1] << "\t" << stepcounts[2] << "\t" << stepcounts[3] << "\t" << stepcounts[4] << "\t" << stepcounts[5] << "\t" << stepcounts[6] << "\t" << steptimes[0] << "\t" << steptimes[1] << "\t" << steptimes[2] << "\t" << steptimes[3] << "\t" << steptimes[4] << "\t" << steptimes[5] << "\t" << steptimes[6] << "\t"
			<< steptimes[7] << "\t" << steptimes[8] << "\t" << total_time << std::endl;

	logfile.close();
}

// Function for sequential exclusive scan.
void exclusiveSumScan(int *array, int size) {

	int sum = 0;
	int val = 0;

	for (int i = 0; i <= size; i++) {
		sum += val;
		val = array[i];
		array[i] = sum;
	}
}

// Function for sequential exclusive scan.
void exclusiveSumScan(long *array, int size) {

	long sum = 0;
	long val = 0;

	for (int i = 0; i <= size; i++) {
		sum += val;
		val = array[i];
		array[i] = sum;
	}
}

// Function for sequential exclusive scan.
void exclusiveSumScan(long *array, long size) {

	long sum = 0;
	long val = 0;

	for (long i = 0; i <= size; i++) {
		sum += val;
		val = array[i];
		array[i] = sum;
	}
}

// Function for reducing an array (SUM operation)
int reduceSUM(int *array, int size) {
	int val = 0;

	for (int i = 0; i < size; i++) {
		val += array[i];
	}

	return val;
}

// Function for reducing an array (SUM operation)
long reduceSUM(long *array, int size) {
	long val = 0;

	for (int i = 0; i < size; i++) {
		val += array[i];
	}

	return val;
}

// Function for reducing an array (SUM operation)
long reduceSUM(long *array, long size) {
	long val = 0;

	for (long i = 0; i < size; i++) {
		val += array[i];
	}

	return val;
}

// Function for reducing an array (SUM operation)
long reduceSUM(int *array, long size) {
	long val = 0;

	for (int i = 0; i < size; i++) {
		val += array[i];
	}

	return val;
}

// Function for reducing an array (SUM operation)
double reduceMIN(double *array, int size) {
	double val = BIG_NUMBER;

	for (int i = 0; i < size; i++) {
		if (array[i] <= val - EPSILON)
			val = array[i];
	}

	return val;
}

// Function for reducing an array (OR operation)
bool reduceOR(bool *array, int size) {
	bool val = false;

	for (int i = 0; i < size; i++) {
		val = val || array[i];
	}

	return val;
}

void printDebugArray(int *d_array, int size, const char *name) {

	int *h_array = new int[size];

	std::cout << name  << std::endl;
	cudaMemcpy(h_array, d_array, size * sizeof(int), cudaMemcpyDeviceToHost);
	for (int i = 0; i < size; i++) {
		std::cout << h_array[i] << "\t";
	}
	std::cout << std::endl;

	delete[] h_array;
}

void printDebugArray(long *d_array, int size, const char *name) {

	long *h_array = new long[size];

	std::cout << name  << std::endl;
	cudaMemcpy(h_array, d_array, size * sizeof(long), cudaMemcpyDeviceToHost);
	for (int i = 0; i < size; i++) {
		std::cout << h_array[i] << "\t";
	}
	std::cout << std::endl;

	delete[] h_array;
}


void printDebugArray(double *d_array, int size, const char *name) {

	double *h_array = new double[size];
	std::cout << std::fixed << std::setprecision(6);
	std::cout << name << std::endl;
	cudaMemcpy(h_array, d_array, size * sizeof(double), cudaMemcpyDeviceToHost);
	for (int i = 0; i < size; i++) {
		std::cout << h_array[i] << "\t";
	}
	std::cout << std::endl;

	delete[] h_array;
}

void printDebugArray_temp(double *d_array, int size, const char *name) {

	double *h_array = new double[size];

	std::cout << name << std::endl;
	cudaMemcpy(h_array, d_array, size * sizeof(double), cudaMemcpyDeviceToHost);
	for (int i = 0; i < size; i++) {
		std::cout << h_array[i]-42 << "\t";
	}
	std::cout << std::endl;

	delete[] h_array;
}

void printDebugMatrix(double *d_matrix, int rowsize, int colsize, const char *name) {
	double *h_matrix = new double[rowsize * colsize];
	
	std::cout << name << std::endl;
	cudaMemcpy(h_matrix, d_matrix, rowsize * colsize * sizeof(double), cudaMemcpyDeviceToHost);
	for (int i = 0; i < rowsize; i++) {
		for (int j = 0; j < colsize; j++) {
			std::cout << std::setprecision(4) << h_matrix[i * colsize + j] << "\t";
		}
		std::cout << std::endl;
	}

	std::cout << std::endl;

	delete[] h_matrix;
}

void printDebugMatrix(int* d_matrix, int rowsize, int colsize, const char *name) {
	int *h_matrix = new int[rowsize * colsize];

	std::cout << name << std::endl;
	cudaMemcpy(h_matrix, d_matrix, rowsize * colsize * sizeof(int), cudaMemcpyDeviceToHost);
	for (int i = 0; i < rowsize; i++) {
		for (int j = 0; j < colsize; j++) {
			std::cout << h_matrix[i * colsize + j] << "\t";
		}
		std::cout << std::endl;
	}

	std::cout << std::endl;

	delete[] h_matrix;
}

void printHostArray(int *h_array, int size, const char *name) {
	std::cout << name << std::endl;

	for (int i = 0; i < size; i++) {
		std::cout << h_array[i] << "\t";
	}
	std::cout << std::endl;
}

void printHostArray(double *h_array, int size, const char *name) {
	std::cout << name << std::endl;

	for (int i = 0; i < size; i++) {
		std::cout << h_array[i] << "\t";
	}
	std::cout << std::endl;
}

void printHostMatrix(double *h_matrix, int rowsize, int colsize, const char *name) {

	std::cout << name << std::endl;
	for (int i = 0; i < rowsize; i++) {
		for (int j = 0; j < colsize; j++) {
			std::cout << h_matrix[i * colsize + j] << "\t";
		}
		std::cout << std::endl;
	}

	std::cout << std::endl;

}

void printHostMatrix(int *h_matrix, int rowsize, int colsize, const char *name) {

	std::cout << name << std::endl;
	for (int i = 0; i < rowsize; i++) {
		for (int j = 0; j < colsize; j++) {
			std::cout << h_matrix[i * colsize + j] << "\t";
		}
		std::cout << std::endl;
	}

	std::cout << std::endl;

}

void printHostArray(long *h_array, int size, const char *name) {
	std::cout << name << std::endl;

	for (int i = 0; i < size; i++) {
		std::cout << h_array[i] << "\t";
	}
	std::cout << std::endl;
}

// Functions for reading specified input file.
double* read_normalcosts(double* C, int* Nad, const char *filepath) {
	std::string s = filepath;
	std::ifstream myfile(s.c_str());
	if (!myfile) {
		std::cerr << "Error: input file not found: " << s.c_str() << std::endl;
		exit(-1);
	}
	myfile >> Nad[0];
	int N = Nad[0];
	C = new double[N * N];
	for (int i = 0; i < N*N; i++) {
		myfile >> C[i];
	}
	myfile.close();
	return C;
}

double* read_euclideancosts(double* C, int* Nad, const char *filepath) {
	std::string s = filepath;
	std::ifstream myfile(s.c_str());
	if (!myfile) {
		std::cerr << "Error: input file not found: " << s.c_str() << std::endl;
		exit(-1);
	}
	myfile >> Nad[0];
	int N = Nad[0];
	C = new double[N * N];
	double *x_coord = new double[N];
	double *y_coord = new double[N];
	int extra;
	while (myfile.is_open() && myfile.good()) {

		for (int i = 0; i < N; i++) {
			//the first number of each row is not needed
			myfile >> extra;
			myfile >> x_coord[i];
			myfile >> y_coord[i];
		}


	}
	myfile.close();

	// Costs are calculated by Euclidean distance between points. Cost from i to j is the same as j to i, which is why we start j at i+1.
	// TSPLIB requires distances (costs) to be integers, so I have done that here as well.
	//        EUCLIDEAN DISTANCE

	for (int i = 0; i < N - 1; i++) {
		for (int j = i + 1; j < N; j++) {
			double xd = x_coord[i] - x_coord[j];
			double yd = y_coord[i] - y_coord[j];
			C[N * i + j] = int(sqrt(xd * xd + yd * yd) + 0.5);
			C[N * j + i] = C[N * i + j];
		}
		C[N  *i + i] = BIG_NUMBER;
	}
	C[N*N - 1] = BIG_NUMBER;
	delete[] x_coord;
	delete[] y_coord;
	return C;
}

double* read_geocosts(double* C, int* Nad, const char *filepath) {
	double RRR = 6378.388;
	double PI = 3.141592;
	int extra;
	double latitude1;
	double latitude2;
	double longitude1;
	double longitude2;
	int deg;
	double min;

	std::string s = filepath;
	std::ifstream myfile(s.c_str());

	if (!myfile) {
		std::cerr << "Error: input file not found: " << s.c_str() << std::endl;
		exit(-1);
	}
	myfile >> Nad[0];
	int N = Nad[0];
	C = new double[N * N];
	double *x_coord = new double[N];
	double *y_coord = new double[N];

	while (myfile.is_open() && myfile.good()) {



		for (int i = 0; i < N; i++) {
			//the first number of each row is not needed
			myfile >> extra;
			myfile >> x_coord[i];
			myfile >> y_coord[i];
		}


	}
	myfile.close();
	for (int i = 0; i < N - 1; i++) {
		for (int j = i + 1; j < N; j++) {
			deg = int(x_coord[i] + 0.5);
			min = x_coord[i] - deg;
			latitude1 = PI * (deg + 5.0 * min / 3.0) / 180.0;
			deg = int(y_coord[i] + 0.5);
			min = y_coord[i] - deg;
			longitude1 = PI * (deg + 5.0 * min / 3.0) / 180.0;

			deg = int(x_coord[j] + 0.5);
			min = x_coord[j] - deg;
			latitude2 = PI * (deg + 5.0 * min / 3.0) / 180.0;
			deg = int(y_coord[j] + 0.5);
			min = y_coord[j] - deg;
			longitude2 = PI * (deg + 5.0 * min / 3.0) / 180.0;

			double q1 = cos(longitude1 - longitude2);
			double q2 = cos(latitude1 - latitude2);
			double q3 = cos(latitude1 + latitude2);
			C[N * i + j] = (int)(RRR * acos(0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)) + 1.0);
			C[N * j + i] = (int)(RRR * acos(0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)) + 1.0);
		}
		C[N * i + i] = BIG_NUMBER;
	}
	C[N*N - 1] = BIG_NUMBER;

	delete[] x_coord;
	delete[] y_coord;
	return C;
}

// This function splits "val" equally among the elements "array" of length equal to "size."
void split(int *array, int val, int size) {

	int split_val = val / size;
	int overflow = val % size;

	std::fill(array, array + size, split_val);

	if (overflow > 0) {
		for (int i = 0; i < size; i++) {
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
__global__ void kernel_memSet(int *array, int val, int size)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if(id < size)
	{
		array[id] = val;
	}
}

/*
 *	This is a test kernel.
 */
__global__ void kernel_memSet(double *array, double val, int size)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if(id < size)
	{
		array[id] = val;
	}
}
