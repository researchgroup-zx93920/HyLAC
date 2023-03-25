#include <cuda.h>
#include <random>
#include <assert.h>
#include <iostream>
#include <cmath>
// #include <timing.cuh>
#include "../include/defs.cuh"
#include "../include/config.h"
#include "../include/cost_generator.h"
#include "../LAP/Hung_lap.cuh"

int main(int argc, char **argv)
{
  Config config = parseArgs(argc, argv);
  printf("\033[0m");
  printf("Welcome ---------------------\n");
  printConfig(config);

  int seed = config.seed;
  int user_n = config.user_n;
  int dev = config.deviceId;

  typedef uint data;
  // typedef double data;
  // typedef float data;
  double time;
  Timer t;

  data *h_costs = generate_cost<data>(config, seed);

  time = t.elapsed();
  Log(debug, "cost generation time %f s", time);
  t.reset();
  LAP<data> *lap = new LAP<data>(h_costs, user_n, dev);
  Log(debug, "LAP object generated succesfully");
  lap->solve();
  time = t.elapsed();
  Log(critical, "solve time %f s\n\n", time);

  delete lap;
  delete[] h_costs;
}