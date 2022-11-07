/*
 * f_cutils.cpp
 *
 *  Created on: Jul 11, 2015
 *      Author: date2
 */

#pragma once
#include <iostream>
#include <fstream>
#include <cuda.h>
#include <thrust/scan.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "d_vars.h"
//#include "d_structs.h"

#ifndef F_CUTILS_H_
#define F_CUTILS_H_

void cudaSafeCall(cudaError_t error, const char *message);
void cudaSafeCall(cudaError_t error, const char *message, int line, const char *file);
void printMemoryUsage(double memory);
void customcopy(double *d_multipliers, int N, double *d_U);
void readFile(const char *filename);
void calculateLinearDims(dim3 &blocks_per_grid, dim3 &threads_per_block, int &total_blocks, int size);
void calculateSquareDims(dim3 &blocks_per_grid, dim3 &threads_per_block, int &total_blocks, int size);
void calculateRectangularDims(dim3 &blocks_per_grid, dim3 &threads_per_block, int &total_blocks, int xsize, int ysize);
void calculateCubicDims(dim3 &blocks_per_grid, dim3 &threads_per_block, int &total_blocks, int xsize, int ysize, int zsize);
void printLog(int prno, int repetition, int numprocs, int numdev, int costrange, long obj_val, int init_assignments, double total_time, int *stepcounts, double *steptimes, const char *logpath, int N);
void exclusiveSumScan(int *array, int size);
void exclusiveSumScan(long *array, int size);
int reduceSUM(int *array, int size);
long reduceSUM(long *array, int size);
bool reduceOR(bool *array, int size);
double reduceMIN(double *array, int size);
void Initialize_C_hat_Serial(double *C, double *C_hat, int N);
void Initialize_C_hat_Parallel(double *C, double *C_hat, int N);
void printDebugArray(int *d_array, int size, const char *name);
void printDebugMatrix(double *d_matrix, int rowsize, int colsize, const char *name);
void printDebugMatrix(int *d_matrix, int rowsize, int colsize, const char *name);
void printHostArray(int *h_array, int size, const char *name);
void printHostArray(double *h_array, int size, const char *name);
void printDebugArray(long *d_array, int size, const char *name);
void printDebugArray(double *d_array, int size, const char *name);
void printDebugArray_temp(double *d_array, int size, const char *name);

void printHostArray(long *h_array, int size, const char *name);
void printHostMatrix(double *h_matrix, int rowsize, int colsize, const char *name);
void printHostMatrix(int *h_matrix, int rowsize, int colsize, const char *name);

double *read_normalcosts(double *C, int *Nad, const char *filepath);
double *read_euclideancosts(double *C, int *Nad, const char *filepath);
double *read_geocosts(double *C, int *Nad, const char *filepath);

void split(int *array, int val, int size);

__global__ void kernel_memSet(int *array, int val, int size);
__global__ void kernel_memSet(double *array, double val, int size);

#endif /* F_CUTILS_H_ */
