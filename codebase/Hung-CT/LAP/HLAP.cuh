#pragma once
#include "S1.cuh"
#include "S2.cuh"
#include "S3.cuh"
#include "S456_classical.cuh"
#include "S4_tree.cuh"
#include "S5_tree.cuh"
#include "S6_tree.cuh"
#include "structures.h"

#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>

template <typename data>
class HLAP
{
private:
  int devID;
  // int ndev = 1;
  size_t psize, psize2;
  data *h_costs, *d_costs;
  data *objective;

  const uint cpbs4 = 512;
  uint nb4, nbr, dbs, l2dbs;

  // All device variables (To be added to a handle later)
  double *row_duals, *col_duals;
  data *slack;
  data *min_mat, *min_vect;
  int *row_ass, *col_ass, *row_cover, *col_cover;
  size_t *zeros, *zeros_size_b;
  int *row_visited, *col_visited;
  void *cub_storage = NULL;

  // device variables for tree-Hungarian
  int *vertices_csr1;
  // int *row_ass, *col_ass, *row_cover, *col_cover; //Part of Vertices in Tree code
  VertexData<data> row_data, col_data;
  Predicates vertex_predicates;
  double *row_duals_tree, *col_duals_tree;

  bool *goto5_tree;

  size_t b1 = 0, b2 = 0, b3 = 0, b4 = 0;

  int counter;

public:
  HLAP(data *cost, size_t size, int dev = 0) : h_costs(cost), psize(size), devID(dev)
  {
    CUDA_RUNTIME(cudaSetDevice(devID));
    psize2 = psize * psize;
    CUDA_RUNTIME(cudaMemcpyToSymbol(SIZE, &psize, sizeof(SIZE)));
    CUDA_RUNTIME(cudaMemcpyToSymbol(SIZE2, &psize2, sizeof(SIZE)));

    nb4 = max((uint)ceil((psize * 1.0) / cpbs4), 1);
    nbr = min(psize, 255UL);
    dbs = cpbs4 * pow(2, ceil(log2(psize)));
    l2dbs = (uint)log2(dbs);

    // Log(debug, " nb4: %u\n nbr: %u\n dbs: %u\n l2dbs %u\n", nb4, nbr, dbs, l2dbs);

    CUDA_RUNTIME(cudaMemcpyToSymbol(NB4, &nb4, sizeof(NB4)));
    CUDA_RUNTIME(cudaMemcpyToSymbol(NBR, &nbr, sizeof(NBR)));
    CUDA_RUNTIME(cudaMemcpyToSymbol(DBS, &dbs, sizeof(DBS)));
    CUDA_RUNTIME(cudaMemcpyToSymbol(L2DBS, &l2dbs, sizeof(L2DBS)));

    counter = 0;
  }

  ~HLAP()
  {
    DeAllocate();
  }
  void solve()
  {
    Allocate();
    // needed for cub reduce

    CUDA_RUNTIME(cub::DeviceReduce::Reduce(cub_storage, b1, min_vect, min_mat, nbr, cub::Min(), MAX_DATA));
    CUDA_RUNTIME(cub::DeviceReduce::Sum(cub_storage, b2, zeros_size_b, &zeros_size, (int)nb4));
    CUDA_RUNTIME(cub::DeviceReduce::Sum(cub_storage, b3, vertex_predicates.addresses, (long *)nullptr, (int)psize));
    CUDA_RUNTIME(cub::DeviceScan::ExclusiveSum(cub_storage, b4, vertex_predicates.addresses, (long *)nullptr, (int)psize));
    size_t greatest = max(b1, b2);
    greatest = max(greatest, b3);
    greatest = max(greatest, b4);
    CUDA_RUNTIME(cudaMalloc(&cub_storage, greatest));

    S1();
    // S2();
    computeInitialAssignments();

    nmatch_cur = 0, nmatch_old = 0;
    CUDA_RUNTIME(cudaDeviceSynchronize());
    S3();
    Log(info, "nmatches# %d", nmatch_cur);
    bool first = true;
    while (nmatch_cur < psize)
    {

      if (false)
        S456_classical();
      else
      {
        if (first)
        {
          CtoT();
          first = false;
          Log(info, "Switched to Tree after %d matches", nmatch_cur);
        }
        S456_tree();
      }
      S3();
      // Log(info, "nmatches# %d", nmatch_cur);
    }
    CUDA_RUNTIME(cudaFree(cub_storage));

    *objective = 0;
    uint gridDim = (uint)ceil((psize * 1.0) / BLOCK_DIMX);
    execKernel(get_obj, gridDim, BLOCK_DIMX, devID, false,
               row_ass, d_costs, objective);
    printf("Obj val: %u\n", (uint)*objective);
  }

private:
  void Allocate()
  {
    size_t N = psize, N2 = psize2;
    CUDA_RUNTIME(cudaMalloc((void **)&d_costs, N2 * sizeof(data)));
    CUDA_RUNTIME(cudaMemcpy(d_costs, h_costs, N2 * sizeof(data), cudaMemcpyDefault));

    CUDA_RUNTIME(cudaMalloc((void **)&row_duals, N * sizeof(double)));
    CUDA_RUNTIME(cudaMalloc((void **)&col_duals, N * sizeof(double)));
    // CUDA_RUNTIME(cudaMalloc((void **)&slack, N2 * sizeof(data)));

    // CUDA_RUNTIME(cudaMalloc((void **)&zeros, N2 * sizeof(size_t)));
    // CUDA_RUNTIME(cudaMalloc((void **)&zeros_size_b, nb4 * sizeof(size_t)));

    CUDA_RUNTIME(cudaMalloc((void **)&row_ass, N * sizeof(int)));
    CUDA_RUNTIME(cudaMalloc((void **)&col_ass, N * sizeof(int)));
    CUDA_RUNTIME(cudaMalloc((void **)&row_cover, N * sizeof(int)));
    CUDA_RUNTIME(cudaMalloc((void **)&col_cover, N * sizeof(int)));

    // CUDA_RUNTIME(cudaMalloc((void **)&min_vect, nbr * sizeof(data)));
    // CUDA_RUNTIME(cudaMalloc((void **)&min_mat, 1 * sizeof(data)));

    CUDA_RUNTIME(cudaMalloc((void **)&row_visited, N * sizeof(int)));
    CUDA_RUNTIME(cudaMalloc((void **)&col_visited, N * sizeof(int)));

    CUDA_RUNTIME(cudaMallocManaged((void **)&objective, 1 * sizeof(data)));
    CUDA_RUNTIME(cudaMallocManaged((void **)&goto5_tree, 1 * sizeof(bool)));
  }
  void DeAllocate(algEnum alg = CLASSICAL)
  {
    // if (alg == CLASSICAL || alg = BOTH)
    /*{
      CUDA_RUNTIME(cudaFree(zeros));
      CUDA_RUNTIME(cudaFree(zeros_size_b));
      CUDA_RUNTIME(cudaFree(min_vect));
      CUDA_RUNTIME(cudaFree(min_mat));
      CUDA_RUNTIME(cudaFree(slack));
    }*/

    // if (alg == TREE || alg == BOTH)
    {

      CUDA_RUNTIME(cudaFree(row_cover));
      CUDA_RUNTIME(cudaFree(col_cover));
      CUDA_RUNTIME(cudaFree(row_visited));
      CUDA_RUNTIME(cudaFree(col_visited));
      CUDA_RUNTIME(cudaFree(row_data.is_visited));
      CUDA_RUNTIME(cudaFree(row_data.parents));
      CUDA_RUNTIME(cudaFree(row_data.children));

      CUDA_RUNTIME(cudaFree(col_data.is_visited));
      CUDA_RUNTIME(cudaFree(col_data.parents));
      CUDA_RUNTIME(cudaFree(col_data.children));
      CUDA_RUNTIME(cudaFree(col_data.slack));

      CUDA_RUNTIME(cudaFree(vertex_predicates.predicates));
      CUDA_RUNTIME(cudaFree(vertex_predicates.addresses));
      CUDA_RUNTIME(cudaFree(vertices_csr1));
    }
    CUDA_RUNTIME(cudaFree(objective));
    CUDA_RUNTIME(cudaFree(d_costs));
    CUDA_RUNTIME(cudaFree(row_duals));
    CUDA_RUNTIME(cudaFree(col_duals));

    CUDA_RUNTIME(cudaFree(row_ass));
    CUDA_RUNTIME(cudaFree(col_ass));
    CUDA_RUNTIME(cudaFree(goto5_tree));
    CUDA_RUNTIME(cudaDeviceReset());
  }
  void S1() // Row and column reduction
  {

    // row_reduce
    execKernel(row_reduce, psize, BLOCK_DIMX, devID, false,
               d_costs, row_duals, slack);

    // column reduce
    {
      execKernel(col_min, psize, BLOCK_DIMX, devID, false,
                 d_costs, row_duals, col_duals); // uncoalesced
      // execKernel(col_sub, psize, BLOCK_DIMX, devID, false,
      //            slack, col_duals);
    }
  }
  void S2() // Compress and cover zeros (makes the zeros matrix)
  {
    uint gridDim = (uint)ceil(psize * 1.0 / BLOCK_DIMX);
    execKernel(init, gridDim, BLOCK_DIMX, devID, false,
               row_ass, col_ass, row_cover, col_cover);
    CUDA_RUNTIME(cudaMemset(zeros_size_b, 0, nb4 * sizeof(size_t)));

    gridDim = (uint)ceil(psize2 * 1.0 / BLOCK_DIMX);
    execKernel(compress_matrix, gridDim, BLOCK_DIMX, devID, false,
               zeros, zeros_size_b, slack);

    CUDA_RUNTIME(cub::DeviceReduce::Sum(cub_storage, b2, zeros_size_b,
                                        &zeros_size, (int)nb4));

    do
    {
      repeat_kernel = false;
      uint blockDim = (nb4 > 1 || zeros_size > BLOCK_DIMX) ? BLOCK_DIMX : zeros_size;
      execKernel(step2, nb4, blockDim, devID, true,
                 zeros, zeros_size_b, row_cover, col_cover, row_ass, col_ass);
    } while (repeat_kernel);
  }
  void S3() // get match count (read from row_ass and write to col_cover_)
  {
    CUDA_RUNTIME(cudaMemset(row_cover, 0, psize * sizeof(int)));
    CUDA_RUNTIME(cudaMemset(col_cover, 0, psize * sizeof(int)));
    uint gridDim = (uint)ceil(psize * 1.0 / BLOCK_DIMX);
    nmatch_old = nmatch_cur;
    nmatch_cur = 0;
    CUDA_RUNTIME(cudaDeviceSynchronize());
    execKernel(step3, gridDim, BLOCK_DIMX, devID, false, row_ass, col_cover); // read from row_ass and write to col_cover
  }
  void S6() // Classical step 6
  {
    execKernel((min_reduce_kernel1<data, BLOCK_DIMX>),
               nbr, BLOCK_DIMX, devID, false,
               slack, min_vect, row_cover, col_cover);

    // finding minimum with cub
    CUDA_RUNTIME(cub::DeviceReduce::Reduce(cub_storage, b1, min_vect, min_mat,
                                           nbr, cub::Min(), MAX_DATA));

    zeros_size = 0;
    CUDA_RUNTIME(cudaMemset(zeros_size_b, 0, nb4));
    uint gridDim = (uint)ceil(psize * 1.0 / BLOCK_DIMX);
    execKernel(S6_DualUpdate, gridDim, BLOCK_DIMX, devID, false, // Dual update for step6
               row_cover, col_cover, min_mat, row_duals, col_duals);
    gridDim = (uint)ceil(psize2 * 1.0 / BLOCK_DIMX);
    execKernel(S6_update, gridDim, BLOCK_DIMX, devID, false,
               slack, row_cover, col_cover, min_mat, zeros, zeros_size_b);
    CUDA_RUNTIME(cub::DeviceReduce::Sum(cub_storage, b2, zeros_size_b,
                                        &zeros_size, nb4));
  }

  void S456_classical() // Classical Version
  {
    uint gridDim = (uint)ceil(psize * 1.0 / BLOCK_DIMX);
    execKernel(S4_init, gridDim, BLOCK_DIMX, devID, false,
               col_visited, row_visited);
    while (1)
    {
      do
      {
        goto_5 = false;
        repeat_kernel = false;
        CUDA_RUNTIME(cudaDeviceSynchronize());
        uint blockDim = (nb4 > 1 || zeros_size > BLOCK_DIMX) ? BLOCK_DIMX : zeros_size;
        execKernel(S4, nb4, blockDim, devID, false,
                   row_cover, col_cover, col_visited,
                   zeros, zeros_size_b, col_ass);
      } while (repeat_kernel && !goto_5);
      if (goto_5)
        break;
      S6();
    }
    execKernel(S5a, gridDim, BLOCK_DIMX, devID, false,
               col_visited, row_visited, row_ass, col_ass);
    execKernel(S5b, gridDim, BLOCK_DIMX, devID, false,
               row_visited, row_ass, col_ass);
  }

  void CtoT()
  {
    // CUDA_RUNTIME(cudaFree(zeros));
    // CUDA_RUNTIME(cudaFree(zeros_size_b));
    // CUDA_RUNTIME(cudaFree(min_vect));
    // CUDA_RUNTIME(cudaFree(min_mat));
    // CUDA_RUNTIME(cudaFree(slack));

    const size_t N = psize;
    CUDA_RUNTIME(cudaMalloc((void **)&row_data.is_visited, N * sizeof(int)));
    CUDA_RUNTIME(cudaMalloc((void **)&row_data.parents, N * sizeof(int)));
    CUDA_RUNTIME(cudaMalloc((void **)&row_data.children, N * sizeof(int)));

    // CUDA_RUNTIME(cudaMalloc((void **)&row_duals_tree, N * sizeof(double)));
    // CUDA_RUNTIME(cudaMalloc((void **)&col_duals_tree, N * sizeof(double)));
    row_duals_tree = row_duals;
    col_duals_tree = col_duals;
    CUDA_RUNTIME(cudaMalloc((void **)&col_data.is_visited, N * sizeof(int)));
    CUDA_RUNTIME(cudaMalloc((void **)&col_data.parents, N * sizeof(int)));
    CUDA_RUNTIME(cudaMalloc((void **)&col_data.children, N * sizeof(int)));
    CUDA_RUNTIME(cudaMalloc((void **)&col_data.slack, N * sizeof(data)));

    CUDA_RUNTIME(cudaMalloc((void **)&vertex_predicates.predicates, N * sizeof(bool)));
    CUDA_RUNTIME(cudaMalloc((void **)&vertex_predicates.addresses, N * sizeof(long)));
    CUDA_RUNTIME(cudaMalloc((void **)&vertices_csr1, N * sizeof(int)));

    // uint gridDim = (uint)ceil(psize * 1.0 / BLOCK_DIMX); // Linear Grid dimension

    // execKernel(transfer_duals<data>, gridDim, BLOCK_DIMX, devID, false,
    //            row_duals, col_duals, row_duals_tree, col_duals_tree);
    size_t total = 0, free = 0;
    cudaMemGetInfo(&free, &total);
    Log(warn, "Occupied %f GB", ((total - free) * 1.0) / (1024 * 1024 * 1024));
  }
  void S456_tree() // Tree Version
  {
    *goto5_tree = false;
    uint gridDim = (uint)ceil(psize * 1.0 / BLOCK_DIMX); // Linear Grid dimension

    execKernel((tree::Initialization<data>), gridDim, BLOCK_DIMX, devID, false,
               row_ass,
               row_cover, col_cover,
               row_data, col_data);
    while (true)
    {
      // S4
      // sets each element to its index
      execKernel(tree::S4_init, gridDim, BLOCK_DIMX, devID, false, vertices_csr1);

      int *vertices_csr2;
      do
      {
        // compact Row vertices
        CUDA_RUNTIME(cudaMemset(vertex_predicates.predicates, false, psize * sizeof(bool)));
        CUDA_RUNTIME(cudaMemset(vertex_predicates.addresses, 0, psize * sizeof(long)));

        execKernel(vertexPredicateConstructionCSR, gridDim, BLOCK_DIMX, devID, false,
                   vertex_predicates, vertices_csr1, row_data.is_visited);

        // CUDA_RUNTIME(cub::DeviceReduce::Sum(cub_storage, b3, vertex_predicates.addresses, &csr2_size, (int)psize));
        // CUDA_RUNTIME(cub::DeviceScan::ExclusiveSum(cub_storage, b4, vertex_predicates.addresses, vertex_predicates.addresses, (int)psize));
        // CUDA_RUNTIME(cudaDeviceSynchronize());

        thrust::device_ptr<long> ptr(vertex_predicates.addresses);
        csr2_size = thrust::reduce(ptr, ptr + psize);  // calculate total number of vertices.
        thrust::exclusive_scan(ptr, ptr + psize, ptr); // exclusive scan for calculating the scatter addresses.
        CUDA_RUNTIME(cudaDeviceSynchronize());

        if (csr2_size > 0)
        {
          CUDA_RUNTIME(cudaMalloc((void **)&vertices_csr2, csr2_size * sizeof(int)));
          execKernel(vertexScatterCSR, gridDim, BLOCK_DIMX, devID, false,
                     vertices_csr2, vertices_csr1, row_data.is_visited, vertex_predicates);
        }
        else
          break;

        // Traverse the frontier, cover zeros and expand.
        // -- Most time consuming function
        execKernel((coverAndExpand<data>), gridDim, BLOCK_DIMX, devID, false,
                   goto5_tree,
                   vertices_csr2, csr2_size,
                   d_costs, row_duals_tree, col_duals_tree,
                   row_ass, col_ass, row_cover, col_cover,
                   row_data, col_data);

        CUDA_RUNTIME(cudaFree(vertices_csr2));
      } while (!*goto5_tree);
      if (*goto5_tree)
        break;

      // S6 Update dual solution --done on host
      data *temp = new data[psize];
      int *temp2 = new int[psize];
      CUDA_RUNTIME(cudaMemcpy(temp, col_data.slack, psize * sizeof(data), cudaMemcpyDeviceToHost));
      CUDA_RUNTIME(cudaMemcpy(temp2, col_cover, psize * sizeof(int), cudaMemcpyDeviceToHost));

      double theta = UINT32_MAX;
      for (int j = 0; j < psize; j++)
      {
        double slack = temp[j];
        if (temp2[j] == 0)
        {
          theta = (double)slack < theta ? slack : theta;
        }
      }
      theta /= 2;
      execKernel((tree::dualUpdate<data>), gridDim, BLOCK_DIMX, devID, false,
                 theta, row_duals_tree, col_duals_tree, col_data.slack, row_cover, col_cover,
                 col_data.parents, row_data.is_visited);

      delete[] temp;
      delete[] temp2;

      // exit(-1);
    }

    // S5
    //  Function for augmenting the solution along multiple node-disjoint alternating trees.
    // reverse pass
    int *col_id_csr;
    Predicates col_predicates;

    col_predicates.size = psize;
    CUDA_RUNTIME(cudaMalloc((void **)(&col_predicates.predicates), psize * sizeof(bool)));
    CUDA_RUNTIME(cudaMalloc((void **)(&col_predicates.addresses), psize * sizeof(long)));
    CUDA_RUNTIME(cudaMemset(col_predicates.predicates, false, psize * sizeof(bool)));
    CUDA_RUNTIME(cudaMemset(col_predicates.addresses, 0, psize * sizeof(long)));

    execKernel(augmentPredicateConstruction, gridDim, BLOCK_DIMX, devID, false,
               col_predicates, col_data.is_visited);

    // CUDA_RUNTIME(cub::DeviceReduce::Sum(cub_storage, b3, col_predicates.addresses, &col_id_size, (int)psize));                    // calculate total number of vertices.
    // CUDA_RUNTIME(cub::DeviceScan::ExclusiveSum(cub_storage, b4, col_predicates.addresses, col_predicates.addresses, (int)psize)); // exclusive scan for calculating the scatter addresses.
    // CUDA_RUNTIME(cudaDeviceSynchronize());
    {
      thrust::device_ptr<long> ptr(col_predicates.addresses);
      col_id_size = thrust::reduce(ptr, ptr + col_predicates.size); // calculate total number of vertices.
      thrust::exclusive_scan(ptr, ptr + col_predicates.size, ptr);  // exclusive scan for calculating the scatter addresses.
      CUDA_RUNTIME(cudaDeviceSynchronize());
    }
    if (col_id_size > 0)
    {
      uint local_gridDim = ceil((col_id_size * 1.0) / BLOCK_DIMX);
      CUDA_RUNTIME(cudaMalloc((void **)&col_id_csr, col_id_size * sizeof(int)));
      execKernel(augmentScatter, gridDim, BLOCK_DIMX, devID, false,
                 col_id_csr, col_predicates);
      execKernel(reverseTraversal<data>, local_gridDim, BLOCK_DIMX, devID, false,
                 col_id_csr, row_data, col_data, col_id_size);
      CUDA_RUNTIME(cudaFree(col_id_csr));
    }
    CUDA_RUNTIME(cudaFree(col_predicates.predicates));
    CUDA_RUNTIME(cudaFree(col_predicates.addresses));
    // augmentation pass

    int *row_id_csr;
    Predicates row_predicates;
    row_predicates.size = psize;
    CUDA_RUNTIME(cudaMalloc((void **)(&row_predicates.predicates), psize * sizeof(bool)));
    CUDA_RUNTIME(cudaMalloc((void **)(&row_predicates.addresses), psize * sizeof(long)));
    CUDA_RUNTIME(cudaMemset(row_predicates.predicates, false, psize * sizeof(bool)));
    CUDA_RUNTIME(cudaMemset(row_predicates.addresses, 0, psize * sizeof(long)));

    execKernel(augmentPredicateConstruction, gridDim, BLOCK_DIMX, devID, false,
               row_predicates, row_data.is_visited);
    // CUDA_RUNTIME(cub::DeviceReduce::Sum(cub_storage, b3, row_predicates.addresses, &row_id_size, (int)psize)); // calculate total number of vertices.
    // CUDA_RUNTIME(cub::DeviceScan::ExclusiveSum(cub_storage, b4, row_predicates.addresses, row_predicates.addresses, (int)psize));
    // CUDA_RUNTIME(cudaDeviceSynchronize());
    {
      thrust::device_ptr<long> ptr(row_predicates.addresses);
      row_id_size = thrust::reduce(ptr, ptr + row_predicates.size); // calculate total number of vertices.
      thrust::exclusive_scan(ptr, ptr + row_predicates.size, ptr);  // exclusive scan for calculating the scatter addresses.
      CUDA_RUNTIME(cudaDeviceSynchronize());
    }
    if (row_id_size > 0)
    {
      uint local_gridDim = ceil((row_id_size * 1.0) / BLOCK_DIMX);
      CUDA_RUNTIME(cudaMalloc((void **)&row_id_csr, row_id_size * sizeof(int)));
      execKernel(augmentScatter, gridDim, BLOCK_DIMX, devID, false,
                 row_id_csr, row_predicates);
      execKernel(augment, local_gridDim, BLOCK_DIMX, devID, false,
                 row_ass, col_ass, row_id_csr, row_data, col_data, row_id_size);
      CUDA_RUNTIME(cudaFree(row_id_csr));
    }
    CUDA_RUNTIME(cudaFree(row_predicates.predicates));
    CUDA_RUNTIME(cudaFree(row_predicates.addresses));
    // return; -- null return
  }

  void
  interrupt()
  {
    counter++;
    if (counter > 5)
      exit(-1);
  }

  void initialReduction()
  {
    uint gridDim = (uint)ceil(psize * 1.0 / BLOCK_DIMX); // Linear Grid dimension
    execKernel(row_reduction, gridDim, BLOCK_DIMX, devID, false,
               d_costs, row_duals);
  }
  void computeInitialAssignments()
  {
    uint gridDim = (uint)ceil(psize * 1.0 / BLOCK_DIMX); // Linear Grid dimension
    CUDA_RUNTIME(cudaMemset(row_ass, -1, psize * sizeof(int)));
    CUDA_RUNTIME(cudaMemset(col_ass, -1, psize * sizeof(int)));

    int *d_row_lock, *d_col_lock;
    CUDA_RUNTIME(cudaMalloc(&d_row_lock, psize * sizeof(int)));
    CUDA_RUNTIME(cudaMalloc(&d_col_lock, psize * sizeof(int)));
    CUDA_RUNTIME(cudaMemset(d_row_lock, 0, psize * sizeof(int)));
    CUDA_RUNTIME(cudaMemset(d_col_lock, 0, psize * sizeof(int)));

    execKernel(initial_assignments, gridDim, BLOCK_DIMX, devID, false,
               d_costs, row_ass, col_ass, d_row_lock, d_col_lock,
               row_duals, col_duals);

    CUDA_RUNTIME(cudaFree(d_row_lock));
    CUDA_RUNTIME(cudaFree(d_col_lock));
  }
};