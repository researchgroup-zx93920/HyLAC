#include <cuda.h>
#include <random>
#include <assert.h>
#include <iostream>
#include <cmath>
// #include <timing.cuh>
#include "../include/defs.cuh"
#include "../include/config.h"

#include "../LAP/hungarian_test.cuh"

int main(int argc, char **argv)
{
  Config config = parseArgs(argc, argv);
  printf("\033[0m");
  printf("Welcome ---------------------\n");
  printConfig(config);

  int user_n = config.user_n;
  // int dev = config.deviceId;

  typedef int data;
  // typedef double data;
  // typedef float data;
  double time;
  Timer t;

  data *h_costs = arrInit(config);

  time = t.elapsed();
  Log(debug, "cost generation time %f s", time);
  t.reset();

  Log(debug, "LAP object generated succesfully");
  hung_seq_sol_solve(h_costs, user_n);
  time = t.elapsed();
  Log(critical, "solve time %f s\n\n", time);

  delete[] h_costs;
}