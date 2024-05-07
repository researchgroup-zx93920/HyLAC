#include <cuda.h>
#include <random>
#include <assert.h>
#include <iostream>
#include <cmath>
// #include <timing.cuh>
#include "../include/defs.cuh"
#include "../include/config.h"

#include "../LAP/Balinski.cuh"
#include "../LAP/Hungarian_init.cuh"

float* noise_matrix(float *NC, float density, float noise, int SIZE)
{
  bool* sparsity = new bool[SIZE * SIZE];
  random_device rd;
  mt19937 gen(rd());
  for(int i=0; i<SIZE*SIZE; i++)
    sparsity[i] = 0;

  int numElements = SIZE * SIZE;
  int numOnes = numElements * density;
  uniform_int_distribution<int> distribution(0, numElements - 1);
  for (int i = 0; i < numOnes; ++i)
  {
    int index = distribution(gen);
    sparsity[index] = 1;
  }
  uniform_real_distribution<float> disr(-noise, noise);
  for (int i=0; i<SIZE*SIZE; i++)
    NC[i] = disr(gen);
  for (int i=0; i<SIZE*SIZE; i++)
    NC[i] = NC[i]*sparsity[i];

  delete[] sparsity;
  return NC;

}

int main(int argc, char **argv)
{
  Config config = parseArgs(argc, argv);
  printf("\033[0m");
  printf("Welcome ---------------------\n");
  printConfig(config);

  int user_n = config.user_n;
  int precision = config.precision;
  int disp_C = config.disp_C;
  // int dev = config.deviceId;

  // typedef int data;
  // typedef double data;
  typedef float data;
  double time;
  Timer t;

  data *h_costs = arrInit(config);
  
  float *h_costs_copy = new float[user_n * user_n]; 

  for (int i=0; i<user_n * user_n; i++)
    h_costs_copy[i] = h_costs[i];

  time = t.elapsed();
  Log(debug, "Cost generation time %f s", time);
  t.reset();

  Log(debug, "LAP object generated successfully");
  cout<<"\n";

 /* 
  int *uvrow = balinski_solve(h_costs, user_n);
  time = t.elapsed();
  Log(critical, "Balinski Initial solve time %f s", time);
  t.reset();
  cout<<"\n";
  */

  int *uvrow = hung_seq_solve(h_costs, user_n);
  time = t.elapsed();
  Log(critical, "Hungarian Initial solve time %f s", time);
  t.reset();
  cout<<"\n";

  random_device rd;
  mt19937 gen(rd());

  uniform_real_distribution<float> dis(-0.05f, 0.05f);

  float *NC = new float[user_n * user_n]; // Noise matrix
  float n_start = 0.05f;
  float n_end = 0.2f;
  float n_step = 0.05f;


  for(float noise=n_start; noise<=n_end; noise+=n_step)
  {
    float density = 0.0;
    while (density <= 1.0)
    {
      cout<<"Noise range : "<<noise*100<<" %"<<endl;
      cout<<"Noise Density sparsity : "<<density*100<<" %"<<endl;
      NC = noise_matrix(NC, density, noise, user_n);
      if(disp_C==2)
      {
        for (int j=0; j<user_n ; j++)
        {
          for (int k=0; k<user_n; k++)
            cout<<NC[j*user_n + k]<<" ";
          cout<<endl;
        }
      }
      time = t.elapsed();
      Log(debug, "Noise generation time %f s", time);
      t.reset();

      uvrow = balinski_resolve(h_costs, user_n, uvrow, NC, precision);
    
      time = t.elapsed();
      Log(critical, "Balinski Resolve time %f s\n", time);
      t.reset();


      for (int i=0; i<user_n * user_n; i++)
        h_costs[i] = h_costs_copy[i];
      time = t.elapsed();
      Log(debug, "Cost matrix reinitialization time %f s\n", time);
      t.reset();

      uvrow = hung_seq_resolve(h_costs,user_n, NC, precision);
      time = t.elapsed();
      Log(critical, "Hungarian Resolve time %f s\n\n", time);
      t.reset();

      for (int i=0; i<user_n * user_n; i++)
        h_costs[i] = h_costs_copy[i];
      time = t.elapsed();
      Log(debug, "Cost matrix reinitialization time %f s\n", time);
      t.reset();

      density += 0.2;
    }
  }

  delete [] uvrow;
  delete [] NC;
  delete[] h_costs;
}