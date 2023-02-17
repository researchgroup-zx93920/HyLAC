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

  if (user_n > 512)
  {
    Log(critical, "implementation not ready yet, exiting...");
    exit(-1);
  }

  typedef int data;
  // typedef double data;
  // typedef float data;
  double time;
  Timer t;

  data *h_costs = generate_cost<data>(config, seed);

  time = t.elapsed();
  Log(debug, "cost generation time %f s", time);
  t.reset();
  CUDA_RUNTIME(cudaSetDevice(dev));
  BLAP<data> *lap = new BLAP<data>(h_costs, user_n, dev);
  time = t.elapsed();
  Log(debug, "LAP object generated succesfully in %f s", time);

  t.reset();
  lap->solve();
  time = t.elapsed();
  Log(critical, "solve time %f s\n\n", time);

  delete lap;
  memstatus("post deletion");
  delete[] h_costs;
}