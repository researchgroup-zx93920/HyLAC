#pragma once

#define __DEBUG__
#define __DEBUG__D true
#define MAX_DATA INT_MAX
#define epsilon 1e-6

typedef unsigned long long int uint64;
typedef unsigned int uint;

enum LogPriorityEnum
{
  critical,
  warn,
  error,
  info,
  debug,
  none
};
