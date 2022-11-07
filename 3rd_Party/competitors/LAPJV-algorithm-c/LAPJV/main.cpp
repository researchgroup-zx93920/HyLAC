#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <queue>
#include <float.h>
#include <ctime>
#include <map>
#include <algorithm>
#include <vector>
#include <cctype>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <random>
#include <stdlib.h>
#include <chrono>

#include "lap.h"

using namespace std;

int main(int argc, char **argv)
{

    const int seed = 45345;
    int N = atoi(argv[1]);
    double range = strtod(argv[2], nullptr);
    int n_tests = 1;
    int N2 = N * N;
    int devid = 0;
    int spcount = 1;
    printf("range: %f\n", range);
    double *C = new double[N2];
    long long total_time = 0;

    // C = read_normalcosts(C, &N, filepath);
    range *= N;
    printf("range: %f\n", range);

    default_random_engine generator(seed);
    uniform_int_distribution<int> distribution(0, range - 1);

    for (int i = 0; i < N; i++)
    {
        for (int k = 0; k < N; k++)
        {
            double gen = distribution(generator);
            // cout << gen << "\t";
            C[N * i + k] = gen;
        }
        // cout << endl;
    }

    // initializations for custom solver
    cost **costmatrix = new cost *[N];
    row *colsol = new row[N];
    col *rowsol = new col[N];
    cost *u = new cost[N];
    cost *v = new cost[N];

    for (int i = 0; i < N; i++)
        costmatrix[i] = &C[N * i];

    typedef std::chrono::high_resolution_clock clock;
    auto start = clock::now();
    cout << "jvc function invoked" << endl;
    cost totalCost = lap(N, costmatrix, rowsol, colsol, u, v); // Use lap algorithm to calculate the minimum total cost
    auto elapsed = clock::now() - start;

    cout << "Total cost:" << totalCost << endl;
    long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    total_time += microseconds / n_tests;

    delete[] costmatrix, colsol, rowsol, u, v;

    cout << "Time taken: \t" << total_time / 1000.0f << " ms" << endl;
}