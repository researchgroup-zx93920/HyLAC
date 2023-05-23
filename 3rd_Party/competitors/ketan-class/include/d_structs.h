/*
 * d_structs.h
 *
 *  Created on: Mar 31, 2013
 *      Author: ketandat
 */
#pragma once

#ifndef D_STRUCTS_H_
#define D_STRUCTS_H_



struct Node {
	//bool *disallowed;
	int *forced;	//Device variable (stored in tiled form)
	double *row_duals;		//Device variable (stored in tiled form)
	double *col_duals;		//Device variable (stored in tiled form)
	int *row_assignments;	//x-variables (stored in tiled form)
	double LB;			//Host variable
	double UB;			//Host variable
	int level;			//Host variable
	//double *C_hat;	//Warmstart_costs
	double* costs;		//Device variable (stored in tiled form)
	int branch_id;		//Host variable
	double *U;		//Device variable (stored in tiled form);
};

struct Variable {
	int id;
	int spid;
	int local_rowid; // rowid and colid in the corresponding suproblem
	int local_colid;
};

struct YVar{
	int yindex1;
	int yindex2;
};

struct ZVar{
	int zindex1;
	int zindex2;
	int zindex3;
	int zindex4;
	int zindex5;
	int zindex6;
};

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
};

struct Vertices
{
	int *row_assignments;
	int *col_assignments;
	int *row_covers;
	int *col_covers;
	double *row_duals;
	double *col_duals;
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
};

struct LAPData {
	double *obj_val;
	int *row_assignments;
	double *row_duals;
	double *col_duals;
};

struct Assignments
{
	int *row_assignments;
	int *col_assignments;
};

struct SubProbMap
{
	int size;
	int *dim1;
	int *dim2;
	int *procId;
	int *devId;
};


struct YSubProbMap
{
	int size;
	int *dim1;
	int *dim2;
	int *dim3;
	int *procId;
	int *devId;
};
struct ZSubProbMap
{
	int size;
	int *dim1;
	int *dim2;
	int *dim3;
	int *procId;
	int *devId;
};

struct SubProbDim
{
	int *dim1;
	int *dim2;
};

struct YSubProbDim
{
	int *dim1;
	// int *dim2;
	// int *dim3;
};
struct ZSubProbDim
{
	int *dim1;
	// int *dim2;
	// int *dim3;
};

struct CostChange
{
	int *dims3;
	int *nodes;
	unsigned long long *conId;
	double *Theta;
	int *state;
};


struct Status {

	int *assignments;
};

struct Objective {
	double *obj;
};
enum SPtype { X, Y, Z };

#endif /* D_STRUCTS_H_ */
