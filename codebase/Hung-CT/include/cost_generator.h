#pragma once
#include <random>
#include <omp.h>
#include <thread>
#include <fstream>
#include "config.h"
#include "timer.h"
#include "logger.cuh"
#include "defs.cuh"

using namespace std;

template <typename T>
T *generate_cost(Config config, const size_t seed = 45345)
{
  size_t size = config.size;
  size_t nrows = size;
  size_t ncols = size;
  double frac = config.frac;
  double range = frac * size;

  T *cost = new T[size * size];
  memset(cost, 0, size * size * sizeof(T));

  // use all available CPU threads for generating cost
  uint nthreads = min(size, (size_t)thread::hardware_concurrency() - 3); // remove 3 threads for OS and other tasks
  uint rows_per_thread = ceil((nrows * 1.0) / nthreads);
#pragma omp parallel for num_threads(nthreads)
  for (uint tid = 0; tid < nthreads; tid++)
  {
    uint first_row = tid * rows_per_thread;
    uint last_row = min(first_row + rows_per_thread, (uint)nrows);
    for (size_t r = first_row; r < last_row; r++)
    {
      default_random_engine generator(seed + r);
      generator.discard(1);
      uniform_int_distribution<T> distribution(0, range - 1);
      for (size_t c = 0; c < ncols; c++)
      {
        if (c < size && r < size)
        {
          double gen = distribution(generator);
          cost[size * r + c] = (T)gen;
        }
        else
        {
          if (c == r)
            cost[size * c + r] = 0;
          else
            cost[size * c + r] = UINT32_MAX;
        }
      }
    }
  }

  // ********* print cost array *********
  // for (uint i = 0; i < size; i++)
  // {
  //   for (uint j = 0; j < size; j++)
  //   {
  //     cout << cost[i * ncols + j] << " ";
  //   }
  //   cout << endl;
  // }

  // ********* write cost array to csv file *********
  // ofstream out("matrix_test.csv");
  // for (uint i = 0; i < size; i++)
  // {
  //   for (uint j = 0; j < size; j++)
  //   {
  //     out << cost[i * ncols + j] << ", ";
  //   }
  //   out << '\n';
  // }

  // ********* get frequency of all numbers *********
  // uint *freq = new uint[(uint)ceil(size * frac)];
  // memset(freq, 0, size * frac * sizeof(uint));
  // for (uint i = 0; i < size; i++)
  // {
  //   for (uint j = 0; j < size; j++)
  //   {
  //     freq[cost[i * ncols + j]]++;
  //   }
  // }
  // ofstream out("freq_test.csv");
  // for (size_t i = 0; i < size * frac; i++)
  // {
  //   out << freq[i] << ",\n";
  // }
  // delete[] freq;
  return cost;
}