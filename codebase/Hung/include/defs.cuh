#pragma once

#define __DEBUG__
#define MAX_DATA INT_MAX
#define eps 1e-6

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
