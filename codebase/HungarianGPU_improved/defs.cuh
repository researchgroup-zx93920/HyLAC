#pragma once
using namespace std;
// #define USE_TEST_MATRIX

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
