/*
 * d_structs.h
 *
 *  Created on: Mar 31, 2013
 *      Author: ketandat
 */

#ifndef D_STRUCTS_H_
#define D_STRUCTS_H_

struct Variable
{
	int id;
	int spid;
	int local_rowid; // rowid and colid in the corresponding suproblem
	int local_colid;
};

struct YVar
{
	int yindex1;
	int yindex2;
};

struct ZVar
{
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
	double *col_slacks;
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

struct LAPData
{
	double *obj_val;
	int *row_assignments;
	double *row_duals;
	double *col_duals;
	double *sum_potentials;
	double *row_min_slack;
};

struct BBXNode
{
	int ID;
	int DEPTH;
	//	int SYM; // 0 if symmetry is invalid. 1 if symmetry is valid.
	double LB;
	int *x_sol; // permutation array of size n.
							//	double *x_cost; // cost array of size n * n.
};

struct UB_RANK
{
	double ub;
	int rank;
};

struct ExchPair
{
	double obj;
	int fac_i;
	int fac_j;
};

struct Facility
{
	int id;
	int non_zero_count;
	double existing_interaction;
	double total_interaction;
	double weighted_score;
};

struct WarmstartData
{
	bool is_active;
	int y_lap_total, z_lap_total, extra_y_lap_count, extra_z_lap_count, y_sp_grpsize, z_sp_grpsize;
	double *x_costs, *y_costs_local, *z_costs_local;
	double *x_row_duals, *x_col_duals, *y_row_duals, *y_col_duals, *z_row_duals, *z_col_duals;
	int *y_lapid_map, *y_inverse_lapid_map, *z_lapid_map, *z_inverse_lapid_map;
	int *y_sp_proc_split, *y_sp_dev_split, *z_sp_proc_split, *z_sp_dev_split;
	int *x_row_assignments, *y_row_assignments, *z_row_assignments;
};

class CompareBBXNode
{
public:
	bool operator()(BBXNode &a, BBXNode &b)
	{
		return a.LB > b.LB;
	}
};

class CompareExchPairs
{
public:
	bool operator()(ExchPair &a, ExchPair &b)
	{
		return a.obj > b.obj;
	}
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

struct CostChange
{
	int *dims3;
	int *nodes;
	unsigned long long *conId;
	double *Theta;
	int *state;
};

struct Objective
{
	double *obj;
};
enum SPtype
{
	X,
	Y,
	Z
};

#endif /* D_STRUCTS_H_ */
