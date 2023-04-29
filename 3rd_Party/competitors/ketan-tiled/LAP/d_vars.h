/*
 * d_vars.h
 *
 *  Created on: Mar 24, 2013
 *      Author: ketandat
 */

#include <cfloat>
#include "d_structs.h"
#include <cstddef>

#ifndef D_VARS_H_
#define D_VARS_H_


#define MAX_GRIDSIZE 65535

#define INF FLT_MAX
#define INF2 100001
#define INF3 ((float) INF / 10)
#define INF4 FLT_MAX
#define EPSILON 0.000001

#define PROBLEMSIZE 5000
#define COSTRANGE 5000
#define PROBLEMCOUNT 1
//#define MAX_SUBPROB 2500
//#define MAX_SUBPROB_OVER 501
#define REPETITIONS 10
#define DEVICE_COUNT 2
#define INPUTFILE "input1.txt"

#define SEED 020202

#define INT_OPT_GAP 1 // used for controlling minimum termination gap
#define DYN_GAP_UPDATE

#define BLOCKDIMX 16
#define BLOCKDIMY 8
#define BLOCKDIMZ 4


// #define BLOCKDIMX 16
// #define BLOCKDIMY 16
// #define BLOCKDIMZ 1

#define DORMANT 0
#define ACTIVE 1
#define VISITED 2
#define REVERSE 3
#define AUGMENT 4
#define MODIFIED 5

#define LAMBDA_X 1
#define LAMBDA_Y 1
#define LAMBDA_YY 0.666666
#define LAMBDA_XY 1
#define LAMBDA_Z 0.666666
#define LAMBDA_LB 1

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

#define LAMBDA_W_2 0.04166

#define TWO_PHASE true // Enable/disable two-phase method.

#define FAST_DA false // Enable/disable fast dual ascent.

#define ENABLE_DFS true

#define ENABLE_WS true

#define SYMMETRY true // Enable/disable symmetry elimination

#define STARTING_DEPTH 3 // Depth at which BFS/DFS should start. DA is not performed at upper levels.

#define DELTA 5 // branching depth

#define TERMINATION_WINDOW_SIZE 25 // size of termination window

#define BOUND_IMPROVEMENT_CRITERION 1

#define GAP_IMPROVEMENT_CRITERION 0.00015

#define WATCHDOG_TIME_LIMIT 300 // iteration time limit (seconds) before checking for interrupt
#define WATCHDOG_NODE_LIMIT 10

#define INITIAL_CHKPOINT 14400 // time for first checkpoint (minutes)

#define CHKPOINT_INTERVAL 60 //  time duration (minutes) between two checkpoints

#define SIGMA 1 // fraction of DFS stack admitted for redistribution

#define MAX_BB_NODECOUNT 10000 // Number of nodes allowed to be processed by a processor group. Used for offsetting the B&B NODEID

#define Y_SLACK_PERCENTAGE 1

#define X_SLACK_PERCENTAGE 1

//#define DA_UB // if this is defined then the UB is updated after each DA iteration using a simple upper bounding scheme (if it is better)

//#define THREE_BODY_INTERACTION

//#define ADAPTIVE /* Three stage adaptive: Fast-1P, then Fast-2P, then Slow-1P (1/3rd itn) */

#define DA_ONLY

#define KNOWN_UB

#define VERBOSE

//#define DFS_TEST

#define HEURISTIC_ORDERING

#define HEURISTIC_ORDERING_RULE 2

#define ANNEAL

#define ANNEAL_TEMP 100

#define ANNEAL_TEMP_FRACTION 0.04 // the annealing temperature is set as a fraction of best upper bound

#define ANNEAL_DELTA 0.99

#define MAX_REDIST_FRACTION 0.25

//#define VALID_CUT

#define QAP_COST


#define NORMAL 'N'
#define PRIME 'P'
#define STAR 'S'
#define ZERO 'Z'

#define OLD 0
#define CHANGED 1
#define NEW 2
// #define Y_SLACK_PERCENTAGE1 0.8
// #define Y_SLACK_PERCENTAGE2 0.2
// #define Y_SLACK_PERCENTAGE3 0.33
#define MAX_MEMORY_CLUSTER 2560000000


extern int devcount;
extern std::size_t n;
extern std::size_t K;
extern int n_factor;

extern int sub_prob_count;
extern int *proc_sub_prob_count;
extern int *proc_iterations;
extern int **proc_iter_sub_prob_count;
extern int **dev_sub_prob_count;
extern int **dev_iterations;
extern int ***dev_iter_sub_prob_count;
extern int ***n_dev_iter_ptr;
extern int **n_proc_iter_ptr;
extern int *n_global_ptr;
extern SubProbMap *spMap;
extern double Epsilon;
extern int MAX_SUBPROB;
extern int MAX_SUBPROB_OVER;
extern int MAX_SUBPROB_OVER_Y;
extern int MAX_MEMORY_HAL;
extern int *prob_gen_cycle;

extern unsigned int *spInverseMap;
// extern int*** global_dev_iter_sub_prob_count;
// extern int*** global_dev_y_iter_sub_prob_count;

//Samhita 03/15/19

extern int y_sub_prob_count;
extern int *proc_y_sub_prob_count;
extern int *proc_y_iterations;
extern int **proc_y_iter_sub_prob_count;
extern int **dev_y_sub_prob_count;
extern int **dev_y_iterations;
extern int ***dev_y_iter_sub_prob_count;
extern int ***n_y_dev_iter_ptr;
extern int **n_y_proc_iter_ptr;
extern int *n_y_global_ptr;
extern YSubProbMap *y_spMap;
extern unsigned int *y_spInverseMap;
extern int * y_sp_dev_split;
extern int * y_sp_proc_split;

extern int scorer;
extern int problem_number;

extern double *LB;
extern double *UB;
extern double proc_obj_val;
extern double proc_UB;
extern double global_obj_val;
extern double global_UB;
extern int *total_proc_iterations;
extern int **total_proc_iter_count;
extern int **total_y_proc_iter_count;
// End
//#define TSP_COST


#endif /* D_VARS_H_ */
