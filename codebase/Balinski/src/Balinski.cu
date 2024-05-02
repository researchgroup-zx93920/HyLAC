#include <cuda.h>
#include <random>
#include <assert.h>
#include <iostream>
#include <cmath>
// #include <timing.cuh>
#include "../include/defs.cuh"
#include "../include/config.h"

#include "../LAP/Balinski.cuh"

int main(int argc, char **argv)
{
  Config config = parseArgs(argc, argv);
  printf("\033[0m");
  printf("Welcome ---------------------\n");
  printConfig(config);

  int user_n = config.user_n;
  // int dev = config.deviceId;

  // typedef int data;
  // typedef double data;
  typedef float data;
  double time;
  Timer t;

  data *h_costs = arrInit(config);

  time = t.elapsed();
  Log(debug, "cost generation time %f s", time);
  t.reset();

  Log(debug, "LAP object generated succesfully");
  int *uvrow = balinski_solve(h_costs, user_n);
  time = t.elapsed();
  Log(debug, "Initial solve time %f s", time);
  t.reset();


  random_device rd;
  mt19937 gen(rd());
  uniform_real_distribution<float> dis(-0.2f, 0.2f);

  float *NC = new float[user_n * user_n]; // Noise matrix

  for (int i=0; i<20; i++)
  {
    for (int j = 0; j < user_n * user_n; ++j)
      NC[j] = dis(gen);

    time = t.elapsed();
    Log(debug, "Noise generation time %f s", time);
    t.reset();

    uvrow = balinski_resolve(h_costs, user_n, uvrow, NC);
    
    time = t.elapsed();
    Log(critical, "Resolve time %f s\n\n", time);
    t.reset();

  }

  delete [] uvrow;
  delete [] NC;
  delete[] h_costs;
}