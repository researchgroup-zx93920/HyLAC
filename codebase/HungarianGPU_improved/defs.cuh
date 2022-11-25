#pragma once
using namespace std;
// #define USE_TEST_MATRIX
#define DEBUG
#ifdef USE_TEST_MATRIX
const char *filepath;
#endif

// Used to make sure some constants are properly set
void check(bool val, const char *str)
{
	if (!val)
	{
		printf("Check failed: %s!\n", str);
		getchar();
		exit(-1);
	}
}

#define CUDA_RUNTIME(ans)                 \
	{                                       \
		gpuAssert((ans), __FILE__, __LINE__); \
	}
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = false)
{

	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);

		/*if (abort) */ exit(1);
	}
}

// __global__ void copy_to_heap(const int *input, int *output, size_t len)
// {
// 	size_t i = threadIdx.x + blockIdx.x * blockDim.x;
// 	if (i < len)
// 	{
// 		printf("%d\n", input[i]);
// 		output[i] = input[i];
// 	}
// }