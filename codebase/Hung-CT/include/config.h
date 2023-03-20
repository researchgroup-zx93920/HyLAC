#pragma once
#include <stdio.h>
#include <string.h>
#include <unistd.h>

struct Config
{
  uint size;
  double frac;
  int deviceId;
  size_t seed;
};

static void usage()
{
  fprintf(stderr,
          "\nUsage: [options]\n"
          "\n-n <size of the problem>"
          "\n-f range-fracrtion"
          "\n-d <deviceId"
          "\n-s <seed-value>"
          "\n");
}

static void printConfig(Config config)
{
  printf("  size: %u\n", config.size);
  printf("  frac: %f\n", config.frac);
  printf("  Device: %u\n", config.deviceId);
  printf("  seed value: %llu\n", config.seed);
}

static Config parseArgs(int argc, char **argv)
{
  Config config;
  config.size = 4096;
  config.frac = 1.0;
  config.deviceId = 0;
  config.seed = 45345;

  int opt;
  while ((opt = getopt(argc, argv, "n:f:d:s:h:")) >= 0)
  {
    switch (opt)
    {
    case 'n':
      config.size = atoi(optarg);
      break;
    case 'f':
      config.frac = std::stod(optarg);
    case 'd':
      config.deviceId = atoi(optarg);
      break;
    case 's':
      config.seed = std::stoull(optarg);
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
