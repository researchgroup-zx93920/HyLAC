#pragma once
#include <stdio.h>
#include <string.h>
#include <unistd.h>

struct Config
{
  uint user_n;
  double frac;
  int deviceId;
  int seed;
  int mode;
  int precision;
  int disp_C;
};

static void usage()
{
  fprintf(stderr,
          "\nUsage: [options]\n"
          "\n-n <size of the problem>"
          "\n-f range-fracrtion"
          "\n-d <deviceId"
          "\n-s <seed-value>"
          "\n-m <mode> (0: Generate, 1: test1, 2: test2)"
          "\n-p <precision> (decimal precision)"
          "\n-c <disp_C> (Display 0: No, 1: cost matrix, 2: cost+noise matrix)"
          "\n");
}

static void printConfig(Config config)
{
  printf("  size: %u\n", config.user_n);
  printf("  frac: %f\n", config.frac);
  printf("  Device: %u\n", config.deviceId);
  printf("  seed value: %d\n", config.seed);
  printf("  mode: %d\n", config.mode);
  printf("  precision: %d\n", config.precision);
  printf("  disp_C: %d\n", config.disp_C);
}

static Config parseArgs(int argc, char **argv)
{
  Config config;
  config.user_n = 4096;
  config.frac = 1.0;
  config.deviceId = 0;
  config.seed = 45345;
  config.mode = 0;
  config.precision = 1;
  config.disp_C = 0;

  int opt;
  while ((opt = getopt(argc, argv, "n:f:d:s:h:m:p:c:")) >= 0)
  {
    switch (opt)
    {
    case 'n':
      config.user_n = atoi(optarg);
      break;
    case 'f':
      config.frac = std::stod(optarg);
    case 'd':
      config.deviceId = atoi(optarg);
      break;
    case 's':
      config.seed = atoi(optarg);
      break;
    case 'm':
      config.mode = atoi(optarg);
      break;
    case 'p':
      config.precision = atoi(optarg);
      break;
    case 'c':
      config.disp_C = atoi(optarg);
      break;
    case 'h':
      usage();
      exit(0);
      break;
    default:
      fprintf(stderr, "\nUnrecognized option!\n");
      usage();
      exit(0);
    }
  }
  return config;
}
