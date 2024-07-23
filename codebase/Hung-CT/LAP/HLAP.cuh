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

  const uint n_threads = (uint)min(psize, 64UL);
  const uint n_threads_full = (uint)min(psize, 512UL);
  const uint n_threads_reduction = 256;
  size_t n_blocks, n_blocks_full;

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

  size_t b1 = 0, b2 = 0, b3 = 0, b4 = 0;

  int counter;

public:
  HLAP(data *cost, size_t size, int dev = 0) : h_costs(cost), psize(size), devID(dev)
  {
    CUDA_RUNTIME(cudaSetDevice(devID));
    psize2 = psize * psize;
    CUDA_RUNTIME(cudaMemcpyToSymbol(SIZE, &psize, sizeof(SIZE)));
    CUDA_RUNTIME(cudaMemcpyToSymbol(SIZE2, &psize2, sizeof(SIZE)));
    n_blocks = (size_t)ceil((psize * 1.0) / n_threads);
    n_blocks_full = (size_t)ceil((psize2 * 1.0) / n_threads_full);

    nb4 = max((uint)ceil((psize * 1.0) / cpbs4), 1);
    nbr = min(psize, 256UL);
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
    S2();
    // computeInitialAssignments();
    std::vector<int> match_trend;
    nmatch_cur = 0, nmatch_old = 0;
    CUDA_RUNTIME(cudaDeviceSynchronize());
    S3();
    match_trend.push_back(nmatch_cur - nmatch_old);
    bool first = true;
    // Log(info, "initial matches: %d", nmatch_cur);
    g_pre_DA_counts = 0;
    g_post_DA_counts = 0;
    g_DA_counts = 0;
    while (nmatch_cur < psize)
    {
      Timer t1;
      // if (decision(match_trend) && first)
      // if (nmatch_cur < 52000)
      if (true) // Always perform classical
        S456_classical();
      else
      {
        if (first)
        {
          CtoT();
          first = false;
          // Log(info, "Switched to Tree after %d matches", nmatch_cur);
        }
        S456_tree();
      }
      S3();
      match_trend.push_back(nmatch_cur - nmatch_old);
      double elap = t1.elapsed();
      // Log(info, "matches: %d, delta_t: %f", nmatch_cur, elap);
      // Log(info, "nmatches# %d", nmatch_cur);
    }
    CUDA_RUNTIME(cudaFree(cub_storage));

    *objective = 0;
    d_costs = slack;
    CUDA_RUNTIME(cudaMemcpy(d_costs, h_costs, psize2 * sizeof(data), cudaMemcpyHostToDevice));
    uint gridDim = (uint)ceil((psize * 1.0) / BLOCK_DIMX);
    execKernel(get_obj, gridDim, BLOCK_DIMX, devID, false,
               row_ass, d_costs, objective);
    printf("Obj val: %u\n", (uint)*objective);
    // Log(info, "Pre %u, post %u, DA %u, Total %u\n", g_pre_DA_counts, g_post_DA_counts, g_DA_counts, g_pre_DA_counts + g_post_DA_counts);
  }

private:
  void Allocate()
  {
    size_t N = psize, N2 = psize2;
    // CUDA_RUNTIME(cudaMalloc((void **)&d_costs, N2 * sizeof(data)));
    // CUDA_RUNTIME(cudaMemcpy(d_costs, h_costs, N2 * sizeof(data), cudaMemcpyDefault));

    CUDA_RUNTIME(cudaMalloc((void **)&row_duals, N * sizeof(double)));
    CUDA_RUNTIME(cudaMalloc((void **)&col_duals, N * sizeof(double)));
    CUDA_RUNTIME(cudaMalloc((void **)&slack, N2 * sizeof(data)));
    CUDA_RUNTIME(cudaMemcpy(slack, h_costs, N2 * sizeof(data), cudaMemcpyDefault));

    CUDA_RUNTIME(cudaMalloc((void **)&zeros, N2 * sizeof(size_t)));
    CUDA_RUNTIME(cudaMalloc((void **)&zeros_size_b, nb4 * sizeof(size_t)));

    CUDA_RUNTIME(cudaMalloc((void **)&row_ass, N * sizeof(int)));
    CUDA_RUNTIME(cudaMalloc((void **)&col_ass, N * sizeof(int)));
    CUDA_RUNTIME(cudaMalloc((void **)&row_cover, N * sizeof(int)));
    CUDA_RUNTIME(cudaMalloc((void **)&col_cover, N * sizeof(int)));

    CUDA_RUNTIME(cudaMalloc((void **)&min_vect, nbr * sizeof(data)));
    CUDA_RUNTIME(cudaMallocManaged((void **)&min_mat, 1 * sizeof(data)));

    CUDA_RUNTIME(cudaMalloc((void **)&row_visited, N * sizeof(int)));
    CUDA_RUNTIME(cudaMalloc((void **)&col_visited, N * sizeof(int)));

    CUDA_RUNTIME(cudaMallocManaged((void **)&objective, 1 * sizeof(data)));
    size_t total = 0, free = 0;
    cudaMemGetInfo(&free, &total);
    Log(warn, "Occupied %f GB", ((total - free) * 1.0) / (1024 * 1024 * 1024));
  }
  void DeAllocate(algEnum alg = CLASSICAL)
  {
    // if (alg == CLASSICAL || alg = BOTH)
    {
      CUDA_RUNTIME(cudaFree(zeros_size_b));
      CUDA_RUNTIME(cudaFree(min_vect));
      CUDA_RUNTIME(cudaFree(min_mat));
    }

    // if (alg == TREE || alg == BOTH)

    CUDA_RUNTIME(cudaFree(zeros));
    CUDA_RUNTIME(cudaFree(row_cover));
    CUDA_RUNTIME(cudaFree(col_cover));
    CUDA_RUNTIME(cudaFree(row_visited));
    CUDA_RUNTIME(cudaFree(col_visited));
    /*{
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
     }*/
    CUDA_RUNTIME(cudaFree(objective));
    CUDA_RUNTIME(cudaFree(row_duals));
    CUDA_RUNTIME(cudaFree(col_duals));

    CUDA_RUNTIME(cudaFree(row_ass));
    CUDA_RUNTIME(cudaFree(col_ass));
    CUDA_RUNTIME(cudaFree(d_costs));
    CUDA_RUNTIME(cudaDeviceReset());
  }
  void S1() // Row and column reduction
  {

    // row_reduce
    execKernel(row_reduce, psize, BLOCK_DIMX, devID, false,
               row_duals, slack);

    // column reduce
    {
      execKernel(col_min, psize, BLOCK_DIMX, devID, false,
                 slack, col_duals); // uncoalesced
      execKernel(col_sub, psize, BLOCK_DIMX, devID, false,
                 slack, col_duals);
    }
  }
  void S2() // Compress and cover zeros (makes the zeros matrix)
  {
    uint gridDim = (uint)ceil(psize * 1.0 / BLOCK_DIMX);
    execKernel(init, gridDim, BLOCK_DIMX, devID, false,
               row_ass, col_ass, row_cover, col_cover);
    CUDA_RUNTIME(cudaMemset(zeros_size_b, 0, nb4 * sizeof(size_t)));

    // gridDim = (uint)ceil(psize2 * 1.0 / BLOCK_DIMX);
    execKernel((compress_matrix<data>), n_blocks_full, n_threads_full, devID, false,
               zeros, zeros_size_b, slack);

    CUDA_RUNTIME(cub::DeviceReduce::Sum(cub_storage, b2, zeros_size_b,
                                        &zeros_size, (int)nb4));

    // double est_range = (double)(psize * 1.0) / zeros_size;
    // printf("Est %f\n", est_range);
    // exit(0);
    // cover zeros
    do
    {
      repeat_kernel = false;
      uint temp_blockDim = (nb4 > 1 || zeros_size > 1024) ? 1024 : zeros_size;
      execKernel(step2, nb4, temp_blockDim, devID, false,
                 zeros, zeros_size_b, row_cover, col_cover, row_ass, col_ass);
    } while (repeat_kernel);
  }
  void S3() // get match count (read from row_ass and write to col_cover_)
  {
    // S3 init
    CUDA_RUNTIME(cudaMemset(row_cover, 0, psize * sizeof(int)));
    CUDA_RUNTIME(cudaMemset(col_cover, 0, psize * sizeof(int)));

    uint gridDim = (uint)ceil(psize * 1.0 / BLOCK_DIMX);
    nmatch_old = nmatch_cur;
    nmatch_cur = 0;
    CUDA_RUNTIME(cudaDeviceSynchronize());
    execKernel(step3, n_blocks, n_threads, devID, false, row_ass, col_cover); // read from row_ass and write to col_cover
    // Log(critical, "# matches %u", nmatch_cur);
  }
  void S6() // Classical step 6
  {
    execKernel((min_reduce_kernel1<data, 256>),
               nbr, n_threads_reduction, devID, false,
               slack, min_vect, row_cover, col_cover);

    // finding minimum with cub
    CUDA_RUNTIME(cub::DeviceReduce::Reduce(cub_storage, b1, min_vect, min_mat,
                                           nbr, cub::Min(), MAX_DATA));
    if (!passes_sanity_test(min_mat))
      exit(-1);

    // S6_init
    zeros_size = 0;
    CUDA_RUNTIME(cudaMemset(zeros_size_b, 0, nb4 * sizeof(size_t)));

    uint gridDim = (uint)ceil(psize * 1.0 / BLOCK_DIMX);
    execKernel(S6_DualUpdate, gridDim, BLOCK_DIMX, devID, false, // Dual update for step6
               row_cover, col_cover, min_mat, row_duals, col_duals);

    execKernel(S6_update, n_blocks_full, n_threads_full, devID, false,
               slack, row_cover, col_cover, min_mat, zeros, zeros_size_b);
    // printDeviceArray<size_t>(zeros_size_b, nb4, "zeros size array");
    CUDA_RUNTIME(cub::DeviceReduce::Sum(cub_storage, b2, zeros_size_b,
                                        &zeros_size, nb4));
  }

  void S456_classical() // Classical Version
  {
    // uint gridDim = (uint)ceil(psize * 1.0 / BLOCK_DIMX);

    execKernel(classical::S4_init, n_blocks, n_threads, devID, false,
               col_visited, row_visited);
    while (1)
    {
      do
      {
        goto_5 = false;
        repeat_kernel = false;
        CUDA_RUNTIME(cudaDeviceSynchronize());
        uint temp_blockDim = (nb4 > 1 || zeros_size > 1024) ? 1024 : zeros_size;
        execKernel(S4, nb4, temp_blockDim, devID, false,
                   row_cover, col_cover, col_visited,
                   zeros, zeros_size_b, col_ass);
      } while (repeat_kernel && !goto_5);
      if (goto_5)
        break;
      S6();
    }
    execKernel(S5a, n_blocks, n_threads, devID, false,
               col_visited, row_visited, row_ass, col_ass);
    execKernel(S5b, n_blocks, n_threads, devID, false,
               row_visited, row_ass, col_ass);
  }

  void CtoT()
  {
    // CUDA_RUNTIME(cudaFree(zeros_size_b));
    // CUDA_RUNTIME(cudaFree(min_vect));
    // CUDA_RUNTIME(cudaFree(min_mat));

    // CUDA_RUNTIME(cudaFree(zeros)); //reuse for tree code
    // CUDA_RUNTIME(cudaFree(slack)); //reuse this memory for tree code
    const size_t N = psize, N2 = psize2;
    d_costs = slack;
    CUDA_RUNTIME(cudaMemcpy(d_costs, h_costs, N2 * sizeof(data), cudaMemcpyHostToDevice));

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

    goto_5 = false;
    uint gridDim = (uint)ceil(psize * 1.0 / BLOCK_DIMX); // Linear Grid dimension

    execKernel((tree::Initialization<data>), gridDim, BLOCK_DIMX, devID, false,
               row_ass,
               row_cover, col_cover,
               row_data, col_data);
    uint pre_DA_counts = 0;
    uint post_DA_counts = 0;
    uint DA_counts = 0;
    while (true)
    {
      // S4
      // sets each element to its index
      execKernel(tree::S4_init, gridDim, BLOCK_DIMX, devID, false, vertices_csr1);

      int *vertices_csr2 = (int *)zeros;
      // long csr2_size;
      while (true)
      {
        // compact Row vertices
        CUDA_RUNTIME(cudaMemset(vertex_predicates.predicates, false, psize * sizeof(bool)));
        CUDA_RUNTIME(cudaMemset(vertex_predicates.addresses, 0, psize * sizeof(long)));

        execKernel(vertexPredicateConstructionCSR, gridDim, BLOCK_DIMX, devID, false,
                   vertex_predicates, vertices_csr1, row_data.is_visited);

        CUDA_RUNTIME(cub::DeviceReduce::Sum(cub_storage, b3, vertex_predicates.addresses, &csr2_size, (int)psize));
        CUDA_RUNTIME(cub::DeviceScan::ExclusiveSum(cub_storage, b4, vertex_predicates.addresses, vertex_predicates.addresses, (int)psize));
        CUDA_RUNTIME(cudaDeviceSynchronize());

        if (csr2_size > 0)
        {
          execKernel(vertexScatterCSR, gridDim, BLOCK_DIMX, devID, false,
                     vertices_csr2, vertices_csr1, row_data.is_visited, vertex_predicates);
        }
        else
          break;

        // Traverse the frontier, cover zeros and expand.
        // -- Most time consuming function
        execKernel((coverAndExpand<data>), gridDim, BLOCK_DIMX, devID, false,
                   vertices_csr2, csr2_size,
                   slack, row_duals_tree, col_duals_tree,
                   row_ass, col_ass, row_cover, col_cover,
                   row_data, col_data);
        if (DA_counts == 0)
          pre_DA_counts++;
        else
          post_DA_counts++;
      }
      if (goto_5)
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
      DA_counts++;
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

    CUDA_RUNTIME(cub::DeviceReduce::Sum(cub_storage, b3, col_predicates.addresses, &col_id_size, (int)psize));                    // calculate total number of vertices.
    CUDA_RUNTIME(cub::DeviceScan::ExclusiveSum(cub_storage, b4, col_predicates.addresses, col_predicates.addresses, (int)psize)); // exclusive scan for calculating the scatter addresses.
    CUDA_RUNTIME(cudaDeviceSynchronize());
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
    CUDA_RUNTIME(cub::DeviceReduce::Sum(cub_storage, b3, row_predicates.addresses, &row_id_size, (int)psize)); // calculate total number of vertices.
    CUDA_RUNTIME(cub::DeviceScan::ExclusiveSum(cub_storage, b4, row_predicates.addresses, row_predicates.addresses, (int)psize));
    CUDA_RUNTIME(cudaDeviceSynchronize());
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

    // Log(info, "Pre DA: %u Post DA: %u DA: %u\n", pre_DA_counts, post_DA_counts, DA_counts);
    g_pre_DA_counts += pre_DA_counts;
    g_post_DA_counts += post_DA_counts;
    g_DA_counts += DA_counts;
  }

  void interrupt()
  {
    counter++;
    if (counter > 5)
      exit(-1);
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
               slack, row_ass, col_ass, d_row_lock, d_col_lock);

    CUDA_RUNTIME(cudaFree(d_row_lock));
    CUDA_RUNTIME(cudaFree(d_col_lock));
  }

  bool passes_sanity_test(data *d_min)
  {
    data temp;
    CUDA_RUNTIME(cudaMemcpy(&temp, d_min, 1 * sizeof(data), cudaMemcpyDeviceToHost));
    if (temp <= 0)
    {
      Log(critical, "minimum element in matrix is non positive => infinite loop condition !!!");
      Log(critical, "%d", temp);
      return false;
    }
    else
      return true;
  }

  bool decision(std::vector<int> match_trend)
  {
    // S1
    return (match_trend.back() > 1);

    // S2, S3
    // static int count;
    // if (match_trend.back() <= 1)
    // {
    //   count++;
    // }
    // return count <= 5; // S2
    // return count <= 10; // S3

    // S4
    // Log(critical, "match trend size %u", match_trend.size());
    // if (match_trend.size() > 5)
    // {
    //   int last5 = 0;
    //   for (int i = 0; i < 5; i++)
    //   {
    //     last5 += match_trend.at(match_trend.size() - i - 1);
    //     // Log(info, "total: %d", total);
    //   }
    //   return last5 >= 10; // S4
    //   // return total >= 30; // S5
    // }
    // else
    //   return true;
  }
};