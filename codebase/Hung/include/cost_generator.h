#pragma once
#include <random>
#include "config.h"
#include "Timer.h"
#include "logger.cuh"
#include "defs.cuh"

using namespace std;

template <typename T>
T *generate_cost(Config config, const int seed = 45345)
{
  size_t user_n = config.user_n;
  size_t nrows = user_n;
  size_t ncols = user_n;
  double frac = config.frac;
  double range = frac * user_n;

  T *cost = new T[user_n * user_n];
  memset(cost, 0, user_n * user_n * sizeof(T));
  default_random_engine generator(seed);
  uniform_int_distribution<int> distribution(0, range - 1);
  for (size_t c = 0; c < ncols; c++)
  {
    for (size_t r = 0; r < nrows; r++)
    {
      if (c < user_n && r < user_n)
      {
        // if (r % user_n == 0 && c >0)
        // 	printf("\n");
        double gen = distribution(generator);
        cost[user_n * c + r] = gen;
        // cout << gen << " ";
      }
      else
      {
        if (c == r)
          cost[user_n * c + r] = 0;
        else
          cost[user_n * c + r] = UINT32_MAX;
      }
    }
    // cout << endl;
  }

  return cost;
}