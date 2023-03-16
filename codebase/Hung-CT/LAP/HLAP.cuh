#pragma once
#include "S1.cuh"
#include "S2.cuh"
#include "S3.cuh"
#include "S456_classical.cuh"
#include "S4_tree.cuh"
#include "S5_tree.cuh"
#include "S6_tree.cuh"
#include "structures.h"

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
  data *row_duals, *col_duals, *slack;
  data *min_mat, *min_vect;
  int *row_ass, *col_ass, *row_cover, *col_cover;
  size_t *zeros, *zeros_size_b;
  int *row_green, *col_prime;
  void *cub_storage = NULL;

  // device variables for tree-Hungarian
  int *vertices_csr1;
  // int *row_ass, *col_ass, *row_cover, *col_cover; //Part of Vertices in Tree code
  VertexData<data> row_data, col_data;
  Predicates vertex_predicates;
  double *row_duals_tree, *col_duals_tree;

  size_t b1 = 0, b2 = 0, b3 = 0;

  int counter;

public:
  HLAP(data *cost, size_t size, int dev = 0) : h_costs(cost), psize(size), devID(dev)
  {
    CUDA_RUNTIME(cudaSetDevice(devID));
    psize2 = psize * psize;
    CUDA_RUNTIME(cudaMemcpyToSymbol(SIZE, &psize, sizeof(SIZE)));
    CUDA_RUNTIME(cudaMemcpyToSymbol(SIZE2, &psize2, sizeof(SIZE)));

    nb4 = max((uint)ceil((psize * 1.0) / cpbs4), 1);
    nbr = min(psize, 256UL);
    dbs = cpbs4 * pow(2, ceil(log2(psize)));
    l2dbs = (uint)log2(dbs);

    Log(debug, " nb4: %u\n nbr: %u\n dbs: %u\n l2dbs %u\n", nb4, nbr, dbs, l2dbs);

    CUDA_RUNTIME(cudaMemcpyToSymbol(NB4, &nb4, sizeof(NB4)));
    CUDA_RUNTIME(cudaMemcpyToSymbol(NBR, &nbr, sizeof(NBR)));
    CUDA_RUNTIME(cudaMemcpyToSymbol(DBS, &dbs, sizeof(DBS)));
    CUDA_RUNTIME(cudaMemcpyToSymbol(L2DBS, &l2dbs, sizeof(L2DBS)));

    counter = 0;
  }

  ~HLAP()
  {
    Log(debug, "Destructor called!");
    DeAllocate();
  }
  void solve()
  {
    Allocate();

    S1();
    S2();

    // needed for cub reduce

    CUDA_RUNTIME(cub::DeviceReduce::Reduce(cub_storage, b1, min_vect, min_mat, nbr, cub::Min(), MAX_DATA));
    CUDA_RUNTIME(cub::DeviceReduce::Sum(cub_storage, b2, zeros_size_b, &zeros_size, nb4));
    CUDA_RUNTIME(cudaMalloc(&cub_storage, max(b1, b2)));

    nmatch_cur = 0, nmatch_old = 0;
    CUDA_RUNTIME(cudaDeviceSynchronize());
    S3();
    Log(debug, "nmatches# %d", nmatch_cur);
    bool first = true;
    while (nmatch_cur < psize)
    {
      printDeviceArray(row_ass, psize, "row assignments");
      printDeviceArray(col_ass, psize, "col assignments");
      // if (nmatch_cur - nmatch_old > 1)
      // S456_classical();
      // else
      // {
      if (first)
      {
        CtoT();
        first = false;
      }
      S456_tree();
      // }
      S3();
      Log(debug, "nmatches# %d", nmatch_cur);
      // interrupt();
    }
    CUDA_RUNTIME(cudaFree(cub_storage));

    // printDeviceMatrix<int>(row_ass, 1, psize, "row assignments");
    *objective = 0;
    uint gridDim = (uint)ceil((psize * 1.0) / BLOCK_DIMX);
    execKernel(get_obj, gridDim, BLOCK_DIMX, devID, true,
               row_ass, d_costs, objective);
    printf("Obj val: %u\n", *objective);
  }

private:
  void Allocate()
  {
    size_t N = psize, N2 = psize2;
    CUDA_RUNTIME(cudaMalloc((void **)&d_costs, N2 * sizeof(data)));
    CUDA_RUNTIME(cudaMemcpy(d_costs, h_costs, N2 * sizeof(data), cudaMemcpyDefault));

    CUDA_RUNTIME(cudaMalloc((void **)&row_duals, N * sizeof(data)));
    CUDA_RUNTIME(cudaMalloc((void **)&col_duals, N * sizeof(data)));
    CUDA_RUNTIME(cudaMalloc((void **)&slack, N2 * sizeof(data)));

    CUDA_RUNTIME(cudaMalloc((void **)&zeros, N2 * sizeof(size_t)));
    CUDA_RUNTIME(cudaMalloc((void **)&zeros_size_b, nb4 * sizeof(size_t)));

    CUDA_RUNTIME(cudaMalloc((void **)&row_ass, N * sizeof(int)));
    CUDA_RUNTIME(cudaMalloc((void **)&col_ass, N * sizeof(int)));
    CUDA_RUNTIME(cudaMalloc((void **)&row_cover, N * sizeof(int)));
    CUDA_RUNTIME(cudaMalloc((void **)&col_cover, N * sizeof(int)));

    CUDA_RUNTIME(cudaMalloc((void **)&min_vect, nbr * sizeof(data)));
    CUDA_RUNTIME(cudaMalloc((void **)&min_mat, 1 * sizeof(data)));

    CUDA_RUNTIME(cudaMalloc((void **)&row_green, N * sizeof(int)));
    CUDA_RUNTIME(cudaMalloc((void **)&col_prime, N * sizeof(int)));

    CUDA_RUNTIME(cudaMallocManaged((void **)&objective, 1 * sizeof(data)));
  }
  void DeAllocate()
  {
    CUDA_RUNTIME(cudaFree(d_costs));
    CUDA_RUNTIME(cudaFree(row_duals));
    CUDA_RUNTIME(cudaFree(col_duals));
    CUDA_RUNTIME(cudaFree(slack));

    CUDA_RUNTIME(cudaFree(zeros));
    CUDA_RUNTIME(cudaFree(zeros_size_b));

    CUDA_RUNTIME(cudaFree(row_ass));
    CUDA_RUNTIME(cudaFree(col_ass));
    CUDA_RUNTIME(cudaFree(row_cover));
    CUDA_RUNTIME(cudaFree(col_cover));

    CUDA_RUNTIME(cudaFree(min_vect));
    CUDA_RUNTIME(cudaFree(min_mat));

    CUDA_RUNTIME(cudaFree(row_green));
    CUDA_RUNTIME(cudaFree(col_prime));
    CUDA_RUNTIME(cudaFree(objective));
  }
  void S1() // Row and column reduction
  {

    // row_reduce
    execKernel(row_reduce, psize, BLOCK_DIMX, devID, true,
               d_costs, row_duals, slack);
    // column reduce
    {
      execKernel(col_min, psize, BLOCK_DIMX, devID, true,
                 slack, col_duals); // uncoalesced
      execKernel(col_sub, psize, BLOCK_DIMX, devID, true,
                 slack, col_duals);
    }
  }
  void S2() // Compress and cover zeros (makes the zeros matrix)
  {
    uint gridDim = (uint)ceil(psize * 1.0 / BLOCK_DIMX);
    execKernel(init, gridDim, BLOCK_DIMX, devID, true,
               row_ass, col_ass, row_cover, col_cover);
    CUDA_RUNTIME(cudaMemset(zeros_size_b, 0, nb4 * sizeof(size_t)));

    gridDim = (uint)ceil(psize2 * 1.0 / BLOCK_DIMX);
    execKernel(compress_matrix, gridDim, BLOCK_DIMX, devID, true,
               zeros, zeros_size_b, slack);
    zeros_size = thrust::reduce(thrust::device, zeros_size_b, zeros_size_b + nb4);
    printDeviceArray(zeros_size_b, nb4, "zeros array");
    // Log(debug, "Zeros size: %d", zeros_size);

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
    execKernel(step3, gridDim, BLOCK_DIMX, devID, true, row_ass, col_cover, row_cover); // read from row_ass and write to col_cover
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
    uint gridDim = (uint)ceil(psize2 * 1.0 / BLOCK_DIMX);
    execKernel(S6_update, gridDim, BLOCK_DIMX, devID, true,
               slack, row_cover, col_cover, min_mat, zeros, zeros_size_b);
    CUDA_RUNTIME(cub::DeviceReduce::Sum(cub_storage, b2, zeros_size_b,
                                        &zeros_size, nb4));
  }

  void S456_classical() // Classical Version
  {
    uint gridDim = (uint)ceil(psize * 1.0 / BLOCK_DIMX);
    execKernel(S4_init, gridDim, BLOCK_DIMX, devID, true,
               col_prime, row_green);
    while (1)
    {
      do
      {
        goto_5 = false;
        repeat_kernel = false;
        CUDA_RUNTIME(cudaDeviceSynchronize());
        uint blockDim = (nb4 > 1 || zeros_size > BLOCK_DIMX) ? BLOCK_DIMX : zeros_size;
        execKernel(S4, nb4, blockDim, devID, true,
                   row_cover, col_cover, col_prime,
                   zeros, zeros_size_b, col_ass);
      } while (repeat_kernel && !goto_5);
      if (goto_5)
        break;
      S6();
    }
    execKernel(S5a, gridDim, BLOCK_DIMX, devID, true,
               col_prime, row_green, row_ass, col_ass);
    execKernel(S5b, gridDim, BLOCK_DIMX, devID, true,
               row_green, row_ass, col_ass);
  }

  void CtoT()
  {
    const size_t N = psize;
    CUDA_RUNTIME(cudaMalloc((void **)&row_data.is_visited, N * sizeof(int)));
    CUDA_RUNTIME(cudaMalloc((void **)&row_data.parents, N * sizeof(int)));
    CUDA_RUNTIME(cudaMalloc((void **)&row_data.children, N * sizeof(int)));

    CUDA_RUNTIME(cudaMalloc((void **)&row_duals_tree, N * sizeof(double)));
    CUDA_RUNTIME(cudaMalloc((void **)&col_duals_tree, N * sizeof(double)));

    CUDA_RUNTIME(cudaMalloc((void **)&col_data.is_visited, N * sizeof(int)));
    CUDA_RUNTIME(cudaMalloc((void **)&col_data.parents, N * sizeof(int)));
    CUDA_RUNTIME(cudaMalloc((void **)&col_data.children, N * sizeof(int)));
    CUDA_RUNTIME(cudaMalloc((void **)&col_data.slack, N * sizeof(data)));

    CUDA_RUNTIME(cudaMalloc((void **)&vertex_predicates.predicates, N * sizeof(bool)));
    CUDA_RUNTIME(cudaMalloc((void **)&vertex_predicates.addresses, N * sizeof(long)));
    CUDA_RUNTIME(cudaMalloc((void **)&vertices_csr1, N * sizeof(int)));

    uint gridDim = (uint)ceil(psize * 1.0 / BLOCK_DIMX); // Linear Grid dimension

    execKernel(transfer_duals<data>, gridDim, BLOCK_DIMX, devID, true,
               row_duals, col_duals, row_duals_tree, col_duals_tree);
  }
  void S456_tree() // Tree Version
  {
    goto_5 = false;
    uint gridDim = (uint)ceil(psize * 1.0 / BLOCK_DIMX); // Linear Grid dimension

    execKernel((tree::Initialization<data>), gridDim, BLOCK_DIMX, devID, true,
               row_data.is_visited, row_ass,
               row_cover, col_cover,
               row_data, col_data);
    while (true)
    {
      // S4
      // sets each element to its index
      execKernel(tree::S4_init, gridDim, BLOCK_DIMX, devID, true, vertices_csr1);
      printDeviceArray(vertices_csr1, psize, "csr1: ");
      int *vertices_csr2;
      size_t csr2_size;
      do
      {
        // compact Row vertices
        CUDA_RUNTIME(cudaMemset(vertex_predicates.predicates, false, psize * sizeof(bool)));
        CUDA_RUNTIME(cudaMemset(vertex_predicates.addresses, 0, psize * sizeof(long)));

        execKernel(vertexPredicateConstructionCSR, gridDim, BLOCK_DIMX, devID, true,
                   vertex_predicates, vertices_csr1, row_data.is_visited);
        printDeviceArray<long>(vertex_predicates.addresses, psize, "Predicates");
        thrust::device_ptr<long> startPtr(vertex_predicates.addresses);
        // thrust::device_ptr<long> endPtr(&vertex_predicates.addresses[psize]);
        csr2_size = thrust::reduce(startPtr, startPtr + psize); // calculate total number of vertices.
        // exclusive scan for calculating the scatter addresses.
        thrust::exclusive_scan(startPtr, startPtr + psize, startPtr);
        CUDA_RUNTIME(cudaDeviceSynchronize());

        // Log(debug, "csr2 size: %lu", csr2_size);
        if (csr2_size > 0)
        {
          CUDA_RUNTIME(cudaMalloc((void **)&vertices_csr2, csr2_size * sizeof(int)));
          execKernel(vertexScatterCSR, gridDim, BLOCK_DIMX, devID, true,
                     vertices_csr2, vertices_csr1, row_data.is_visited, vertex_predicates);
          printDeviceArray<int>(vertices_csr2, csr2_size, "Post-Scatter");
        }
        else
        {
          printDeviceArray<int>(vertices_csr2, csr2_size, "Post-Scatter");
          break;
        }
        // printDeviceMatrix<data>(d_costs, psize, psize, "Costs");
        printDeviceArray<double>(row_duals_tree, psize, "row duals");
        printDeviceArray<double>(col_duals_tree, psize, "col duals");
        printDeviceArray<int>(row_data.is_visited, psize, "row visited");
        printDeviceArray<data>(col_data.slack, psize, "col slack");
        printDeviceArray<int>(row_data.parents, psize, "row parents");
        printDeviceArray<int>(col_data.parents, psize, "col parents");
        printDeviceArray<int>(row_cover, psize, "row cover");
        printDeviceArray<int>(col_cover, psize, "col cover");
        // Traverse the frontier, cover zeros and expand.
        // -- Most time consuming function
        execKernel((coverAndExpand<data>), gridDim, BLOCK_DIMX, devID, true,
                   vertices_csr2, csr2_size,
                   d_costs, row_duals_tree, col_duals_tree,
                   row_ass, col_ass, row_cover, col_cover,
                   row_data, col_data);
        printDeviceArray<int>(row_data.parents, psize, "row parents");
        printDeviceArray<int>(col_data.parents, psize, "col parents");
        printDeviceArray<int>(row_data.is_visited, psize, "row visited");
        printDeviceArray<int>(col_data.is_visited, psize, "col visited");
        printDeviceArray<data>(col_data.slack, psize, "col slack");
        printDeviceArray<int>(row_cover, psize, "row cover");
        printDeviceArray<int>(col_cover, psize, "col cover");

        CUDA_RUNTIME(cudaFree(vertices_csr2));
      } while (!goto_5);
      if (goto_5)
        break;
      // else

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
      Log(debug, "theta: %f", theta);
      execKernel((tree::dualUpdate<data>), gridDim, BLOCK_DIMX, devID, true,
                 theta, row_duals_tree, col_duals_tree, col_data.slack, row_cover, col_cover,
                 col_data.parents, row_data.is_visited);

      delete[] temp;
      delete[] temp2;
      printDeviceArray<double>(row_duals_tree, psize, "row dual");
      printDeviceArray<double>(col_duals_tree, psize, "col dual");
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

    execKernel(augmentPredicateConstruction, gridDim, BLOCK_DIMX, devID, true,
               col_predicates, col_data.is_visited);

    thrust::device_ptr<long> ptr(col_predicates.addresses);
    size_t col_id_size = thrust::reduce(ptr, ptr + col_predicates.size); // calculate total number of vertices.
    thrust::exclusive_scan(ptr, ptr + col_predicates.size, ptr);         // exclusive scan for calculating the scatter addresses.
    if (col_id_size > 0)
    {
      uint local_gridDim = ceil((col_id_size * 1.0) / BLOCK_DIMX);
      CUDA_RUNTIME(cudaMalloc((void **)&col_id_csr, col_id_size * sizeof(int)));
      execKernel(augmentScatter, gridDim, BLOCK_DIMX, devID, true,
                 col_id_csr, col_predicates);
      // exit(0);
      execKernel(reverseTraversal<data>, local_gridDim, BLOCK_DIMX, devID, true,
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

    execKernel(augmentPredicateConstruction, gridDim, BLOCK_DIMX, devID, true,
               row_predicates, row_data.is_visited);
    ptr = thrust::device_ptr<long>(row_predicates.addresses);
    size_t row_id_size = thrust::reduce(ptr, ptr + row_predicates.size); // calculate total number of vertices.
    thrust::exclusive_scan(ptr, ptr + row_predicates.size, ptr);         // exclusive scan for calculating the scatter addresses.

    if (row_id_size > 0)
    {
      uint local_gridDim = ceil((row_id_size * 1.0) / BLOCK_DIMX);
      CUDA_RUNTIME(cudaMalloc((void **)&row_id_csr, row_id_size * sizeof(int)));
      execKernel(augmentScatter, gridDim, BLOCK_DIMX, devID, true,
                 row_id_csr, row_predicates);
      execKernel(augment, local_gridDim, BLOCK_DIMX, devID, true,
                 row_ass, col_ass, row_id_csr, row_data, col_data, row_id_size);
      CUDA_RUNTIME(cudaFree(row_id_csr));
    }
    CUDA_RUNTIME(cudaFree(row_predicates.predicates));
    CUDA_RUNTIME(cudaFree(row_predicates.addresses));
    // return; -- null return
  }

  void interrupt()
  {
    counter++;
    if (counter > 5)
      exit(-1);
  }
};