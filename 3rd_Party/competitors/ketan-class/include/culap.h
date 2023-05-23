/*
 * culap.h
 *
 *  Created on: Oct 30, 2014
 *      Author: ketandat
 */
#pragma once

#include <omp.h>
#include "f_culap.h"
#include "d_structs.h"
#include "d_vars.h"
#include "math.h"

#ifndef CUALP_H_
#define CUALP_H_

class CuLAP {

	int N;
	long N2;
	long M; // total number of zero cost edges on a single host.
	int SP;
	int devid;

	int initial_assignment_count;
	int *stepcounts;
	double *steptimes;

	int prevstep;

	bool flag;

	bool dynamic;

	double *d_obj_val_dev;

	Matrix d_costs_dev;
	Vertices d_vertices_dev;
	CompactEdges d_edges_csr_dev;
	VertexData d_row_data_dev, d_col_data_dev;

public:
	CuLAP(int _size, int _spcount, int _devid, bool _is_dynamic);
	virtual ~CuLAP();

	int solve(double *d_cost_matrix, int *d_row_assignments, double *d_row_duals, double *d_col_duals, double *d_obj_val);

	void getAssignments(int *_row_assignments);
	void getStepTimes(double *_steptimes);
	void getStepCounts(int *_stepcounts);

private:

	void initializeDevice(void);
	void finalizeDevice(void);

	int hungarianStep0(bool count_time);
	int hungarianStep1(bool count_time);
	int hungarianStep2(bool count_time);
	int hungarianStep3(bool count_time);
	int hungarianStep4(bool count_time);
	int hungarianStep5(bool count_time);
	int hungarianStep6(bool count_time);


};

#endif /* CUALP_H_ */
