#pragma once

template <typename T = uint>
struct VertexData
{
  int *parents;
  int *children;
  int *is_visited;
  T *slack;
};

struct Predicates
{
  long size;
  bool *predicates;
  long *addresses;
  long *out_addresses;
};

#define DORMANT 0
#define ACTIVE 1
#define VISITED 2
#define REVERSE 3
#define AUGMENT 4
#define MODIFIED 5
// #define EPSILON 0.00001

enum algEnum
{
  CLASSICAL,
  TREE,
  BOTH
};