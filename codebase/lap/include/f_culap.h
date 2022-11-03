/*
 * f_culap.h
 *
 *  Created on: Jul 29, 2015
 *      Author: date2
 */
#pragma once

#include <cuda.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "d_structs.h"
#include "d_vars.h"
#include "f_cutils.h"

#ifdef __INTELLISENSE__
void __syncthreads();
void atomicAdd(int *a, int b);
#endif

#ifndef F_CULAP_H_
#define F_CULAP_H_

void initialReduction(Matrix &d_costs, Vertices &d_vertices_dev, int SP, int N, unsigned int devid);
void dynamicUpdate(Matrix &d_costs, Vertices &d_vertices_dev, int SP, int N, unsigned int devid);

void computeInitialAssignments(Matrix &d_costs, Vertices &d_vertices_dev, int SP, int N, unsigned int devid);

int computeRowCovers(Vertices &d_vertices_dev, VertexData &d_row_data_dev, VertexData &d_col_data_dev, int SP, int N, unsigned int devid);

void compactEdgesCSR(CompactEdges &d_edges_csr_dev, Matrix &d_costs_dev, Vertices &d_vertices_dev, long &M, int SP, int N, unsigned int devid);
void deleteCSR(CompactEdges &d_edges_csr_dev, unsigned int devid);
void executeZeroCover(CompactEdges &d_edges_csr_dev, Vertices &d_vertices_dev, VertexData &d_row_data_dev, VertexData &d_col_data_dev, bool *h_flag, int SP, int N, unsigned int devid);
void compactRowVertices(VertexData &d_row_data_dev, Array &d_vertices_csr_out, Array &d_vertices_csr_in, unsigned int devid);
void coverZeroAndExpand(CompactEdges &d_edges_csr_dev, Vertices &d_vertices_dev, VertexData &d_row_data_dev, VertexData &d_col_data_dev, Array &d_vertices_csr_out, Array &d_vertices_csr_in, bool *h_flag, int N, unsigned int devid);

void reversePass(VertexData &d_row_data_dev, VertexData &d_col_data_dev, int SP, int N, unsigned int devid);
void augmentationPass(Vertices &d_vertices_dev, VertexData &d_row_data_dev, VertexData &d_col_data_dev, int SP, int N, unsigned int devid);

void computeUncoveredMinima(double *d_sp_min, Matrix &d_costs_dev, Vertices &d_vertices_dev, int SP, int N, unsigned int devid);
void updateDuals(double *d_sp_min, Vertices &d_vertices_dev, int SP, int N, unsigned int devid);

void calcObjVal(double *d_obj_val, Vertices &d_vertices_dev, int SP, int N, unsigned int devid);

__global__ void kernel_rowReduction(double *d_costs, double *d_row_duals, double *d_col_duals, int SP, int N);
__global__ void kernel_columnReduction(double *d_costs, double *d_row_duals, double *d_col_duals, int SP, int N);
__global__ void kernel_dynamicUpdate(int *d_row_assignments, int *d_col_assignments, double *d_row_duals, double *d_col_duals, double *d_costs, int SP, int N);

__global__ void kernel_computeInitialAssignments(double *d_costs, double *d_row_duals, double *d_col_duals, int *d_row_assignments, int *d_col_assignments, int *d_row_lock, int *d_col_lock, int SP, int N);

__global__ void kernel_computeRowCovers(int *d_row_assignments, int *d_row_covers, int SP, int N);

__global__ void kernel_edgePredicateConstructionCSR(Predicates d_edge_predicates_csr, double *d_costs, double *d_row_duals, double *d_col_duals, int SP, int N);
__global__ void kernel_edgeScatterCSR(CompactEdges d_edges_csr, Predicates d_edge_predicates_csr, long M, int SP, int N);
__global__ void kernel_rowInitialization(int *d_vertex_ids, int *d_visited, int *d_row_covers, long *d_ptrs, int SP, int N);
__global__ void kernel_vertexPredicateConstructionCSR(Predicates d_vertex_predicates, Array d_vertices_csr_in, int *d_visited);
__global__ void kernel_vertexScatterCSR(int *d_vertex_ids_csr, int *d_vertex_ids, Predicates d_vertex_predicates);
__global__ void kernel_vertexAllocationConstructionCSR(Predicates d_vertex_allocations, Array d_vertices_csr_in, long *d_ptrs);
__global__ void kernel_coverAndExpand(bool *d_flag, Array d_vertices_csr_out, Array d_vertices_csr_in, Predicates d_vertex_allocations, CompactEdges d_edges_csr, Vertices d_vertices, VertexData d_row_data, VertexData d_col_data, int N);

__global__ void kernel_augmentPredicateConstruction(Predicates d_predicates, int *d_visited, int size);
__global__ void kernel_augmentScatter(Array d_vertex_ids, Predicates d_predicates, int size);
__global__ void kernel_reverseTraversal(Array d_col_vertices, VertexData d_row_data, VertexData d_col_data);
__global__ void kernel_augmentation(int *d_row_assignments, int *d_col_assignments, Array d_row_vertices, VertexData d_row_data, VertexData d_col_data, int N);

__global__ void kernel_computeUncoveredMinima1(double *d_min_val, double *d_costs, double *d_row_duals, double *d_col_duals, int *d_row_covers, int *d_col_covers, int SP, int N);
__global__ void kernel_computeUncoveredMinima2(double *d_sp_min, double *d_min_val, int SP, int N);
__global__ void kernel_dualUpdate(double *d_sp_min, double *d_row_duals, double *d_col_duals, int *d_row_covers, int *d_col_covers, int SP, int N);

__global__ void kernel_calcObjVal(double *d_obj_val, double *d_row_duals, double *d_col_duals, int SP, int N);

__device__ void __update_covers(Vertices d_vertices, bool *d_flag, int *d_row_parents, int *d_col_parents, int *d_row_visited, int *d_col_visited, int *new_frontier, int *d_start_ptr, int *d_end_ptr, int ROWID, int N);

__device__ void __reverse_traversal(int *d_row_visited, int *d_row_children, int *d_col_children, int *d_row_parents, int *d_col_parents, int COLID);
__device__ void __augment(int *d_row_assignments, int *d_col_assignments, int *d_row_children, int *d_col_children, int ROWID, int N);

#endif /* F_CULAP_H_ */
