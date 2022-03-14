/*
 * d_vars.h
 *
 *  Created on: Mar 24, 2013
 *      Author: ketandat
 */

#pragma once

#ifndef D_VARS_H_
#define D_VARS_H_

#define MAX_GRIDSIZE 65535
//#define INF 100000001
#define BIG_NUMBER 100000001
#define EPSILON 0.0001

#define SEED 24071987

#define BLOCKDIMX 8
#define BLOCKDIMY 8
#define BLOCKDIMZ 8

#define DORMANT 0
#define ACTIVE 1
#define VISITED 2
#define REVERSE 3
#define AUGMENT 4
#define MODIFIED 5

#define LAMBDA_Y 0.5

#define LAMBDA_Z_0_0 0.35
#define LAMBDA_Z_0_1 0.5
#define LAMBDA_Z_0_2 0.9

#define LAMBDA_Z_1_0 0.166
#define LAMBDA_Z_1_1 0.166
#define LAMBDA_Z_1_2 0.167
#define LAMBDA_Z_1_3 0.167
#define LAMBDA_Z_1_4 0.167
#define LAMBDA_Z_1_5 0.167

#define LAMBDA_Z_2 0.2

#endif /* D_VARS_H_ */
