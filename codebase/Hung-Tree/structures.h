#pragma once

struct Array
{
	long size;
	int *elements;
};

struct Matrix
{
	int rowsize;
	int colsize;
	double *elements;
	double *row_duals;
	double *col_duals;
};

struct Vertices
{
	int *row_assignments;
	int *col_assignments;
	int *row_covers;
	int *col_covers;
};

struct CompactEdges
{
	int *neighbors;
	long *ptrs;
};

struct Predicates
{
	long size;
	bool *predicates;
	long *addresses;
};

struct VertexData
{
	int *parents;
	int *children;
	int *is_visited;
	double *slack;
};

int devID = 0;
int nmatches = 0;