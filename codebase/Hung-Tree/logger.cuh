#pragma once
// #include "defs.cuh"
// #include "../LAP/utils.cuh"
#include <string>
#define __DEBUG__

enum LogPriorityEnum
{
  critical,
  warn,
  error,
  info,
  debug,
  none
};

template <bool NEWLINE = true, typename... Args>
void Log(LogPriorityEnum l, const char *f, Args... args)
{

  bool print = true;
#ifdef __DEBUG__
  if (l == debug)
  {
    print = false;
  }
#endif // __DEBUG__

  if (print)
  {
    // Line Color Set
    switch (l)
    {
    case critical:
      printf("\033[1;31m"); // Set the text to the color red.
      break;
    case warn:
      printf("\033[1;33m"); // Set the text to the color red.
      break;
    case error:
      printf("\033[1;31m"); // Set the text to the color red.
      break;
    case info:
      printf("\033[1;32m"); // Set the text to the color red.
      break;
    case debug:
      printf("\033[1;34m"); // Set the text to the color red.
      break;
    default:
      printf("\033[0m"); // Resets the text to default color.
      break;
    }

    time_t rawtime;
    struct tm *timeinfo;
    time(&rawtime);
    timeinfo = localtime(&rawtime);
    printf("[%02d:%02d:%02d] ", timeinfo->tm_hour, timeinfo->tm_min,
           timeinfo->tm_sec);

    printf(f, args...);

    if (NEWLINE)
      printf("\n");

    printf("\033[0m");
  }
}

// #define Log(l_, f_, ...)printf((f_), __VA_ARGS__);

// template <typename data = int>
// void printDebugArray(const data *array, size_t len, std::string name = NULL)
// {
//   using namespace std;
//   data *temp = new data[len];
//   CUDA_RUNTIME(cudaMemcpy(temp, array, len * sizeof(data), cudaMemcpyDefault));

//   if (name != "NULL")
//   {
//     if (len < 1)
//       Log(debug, "%s", name.c_str());
//     else
//       Log<false>(debug, "%s: ", name.c_str());
//   }
//   for (size_t i = 0; i < len - 1; i++)
//   {
//     cout << temp[i] << ',';
//   }
//   cout << temp[len - 1] << '.' << endl;
//   delete[] temp;
// }