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
  int nprob = config.tile;
  if (user_n > 512)
  {
    Log(critical, "implementation not ready yet, exiting...");
    exit(-1);
  }

  // typedef int data;
  // typedef double data;
  typedef float data;
  double time;
  Timer t;
  data *tcosts = new data[nprob * user_n * user_n];
  data *h_costs;
  for (int prob = 0; prob < nprob; prob++)
  {
    data *costs = generate_cost<data>(config, seed + user_n * prob);
    memcpy(&tcosts[prob * user_n * user_n], costs, user_n * user_n * sizeof(data));
    if (prob == 0)
      h_costs = costs;
    else
      delete[] costs;
  }

  time = t.elapsed();
  Log(debug, "cost generation time %f s", time);
  t.reset();
  CUDA_RUNTIME(cudaSetDevice(dev));
  data *d_tcosts;
  CUDA_RUNTIME(cudaMalloc((void **)&d_tcosts, nprob * user_n * user_n * sizeof(data)));
  CUDA_RUNTIME(cudaMemcpy(d_tcosts, tcosts, nprob * user_n * user_n * sizeof(data), cudaMemcpyDefault));

  /*BLAP<data> *lap = new BLAP<data>(h_costs, user_n, dev);
  time = t.elapsed();
  Log(debug, "BLAP object generated succesfully in %f s", time);
  t.reset();
  lap->solve();
  time = t.elapsed();
  Log(critical, "solve time %f s\n\n", time);
  delete lap;
  memstatus("post deletion");*/

  TLAP<data> *tlap = new TLAP<data>((uint)nprob, d_tcosts, user_n, dev);
  time = t.elapsed();
  Log(debug, "TLAP object generated succesfully in %f s", time);
  t.reset();
  tlap->solve();
  time = t.elapsed();
  Log(critical, "solve time %f s\n\n", time);
  delete tlap;

  delete[] h_costs;
}