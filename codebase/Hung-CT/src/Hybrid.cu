#include <cuda.h>
#include <random>
#include <assert.h>
#include <iostream>
#include <cmath>

#include "../include/defs.cuh"
#include "../include/config.h"
#include "../include/cost_generator.h"

#include "../LAP/HLAP.cuh"

int main(int argc, char **argv)
{
  Config config = parseArgs(argc, argv);
  printf("\033[0m");
  // printf("Welcome ---------------------\n");
  printConfig(config);

  size_t seed = config.seed;
  int size = config.size;
  int dev = config.deviceId;

  typedef uint data;
  // typedef float data;
  // typedef double data;

  double time;
  Timer t;
  data *h_costs = generate_cost<data>(config, seed);
  time = t.elapsed_and_reset();
  Log(debug, "cost generation time %f s", time);

  HLAP lpx = HLAP(h_costs, size, dev);
  time = t.elapsed_and_reset();
  Log(debug, "HLAP object generation time %f s", time);

  lpx.solve();
  time = t.elapsed_and_reset();
  Log(critical, "solve time %f s", time);
  std::cout << "\n\n\n\n\n";
  delete[] h_costs;
}