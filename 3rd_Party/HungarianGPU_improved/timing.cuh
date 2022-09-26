#pragma once

enum steps
{
	other,
	step1,
	step2,
	step3,
	step4,
	step5,
	step6,
	num_steps
};

struct times{
	unsigned long long tmp[num_steps];
	unsigned long long total[num_steps];
};

static __device__ void d_initializetimes(times *counters)
{
	__syncthreads();
	if (threadIdx.x == 0)
	{
		for (unsigned int i = 0; i < num_steps; ++i)
		{
			counters->total[i] = 0;
		}
	}
	__syncthreads();
}
static __host__ void h_initializetimes(times *counters)
{

	cudaDeviceSynchronize();
	for (unsigned int i = 0; i < num_steps; ++i)
	{
		counters->total[i] = 0;
	}
	cudaDeviceSynchronize();
}

static __device__ void d_startTime(steps counterName, times *counters)
{
	__syncthreads();
	if (threadIdx.x == 0)
		counters->tmp[counterName] = clock64();
	__syncthreads();
}
static __host__ void h_startTime(steps counterName, times *counters)
{
	cudaDeviceSynchronize();
	counters->tmp[counterName] = std::clock();
}

static __device__ void d_endTime(steps counterName, times *counters)
{
	__syncthreads();
	if (threadIdx.x == 0)
	{
		counters->total[counterName] += clock64() - counters->tmp[counterName];
	}
	__syncthreads();
}
static __host__ void h_endTime(steps counterName, times *counters)
{
	cudaDeviceSynchronize();
	counters->total[counterName] += std::clock() - counters->tmp[counterName];
	cudaDeviceSynchronize();
}

