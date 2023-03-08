#pragma once
#include "S1.cuh"
#include "S2.cuh"
#include "S3.cuh"
#include "S456_classical.cuh"

template <typename data>
class HLAP
{
private:
  int devID;
  // int ndev = 1;
  size_t psize, psize2;
  data *h_costs, *d_costs;

  const uint cpbs4 = 512;
  uint nb4, nbr, dbs, l2dbs;

  // All device variables (To be added to a handle later)
  data *row_duals, *col_duals, *slack;
  data *min_mat, *min_vect;

  int *row_ass, *col_ass, *row_cover, *col_cover;

  size_t *zeros, *zeros_size_b;
  int *row_green, *col_prime;
  void *cub_storage = NULL;
  size_t b1 = 0, b2 = 0;

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
    while (nmatch_cur < psize)
    {
      // if (nmatch_cur - nmatch_old > 1)
      S456_classical();
      // else
      //   S456_tree();

      S3();
    }
    CUDA_RUNTIME(cudaFree(cub_storage));
    printDebugArray(col_ass, psize, "col assignments");
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

    CUDA_RUNTIME(cudaMalloc((void **)&row_green, psize * sizeof(int)));
    CUDA_RUNTIME(cudaMalloc((void **)&col_prime, psize * sizeof(int)));
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
  }
  void S1() // Row and column reduction
  {

    // row_reduce
    execKernel(row_reduce, psize, BLOCK_DIMX, devID, false,
               d_costs, row_duals, slack);
    // column reduce
    {
      execKernel(col_min, psize, BLOCK_DIMX, devID, false,
                 slack, col_duals); // uncoalesced
      execKernel(col_sub, psize, BLOCK_DIMX, devID, false,
                 slack, col_duals);
    }
  }
  void S2() // Compress and cover zeros
  {
    uint gridDim = (uint)ceil(psize * 1.0 / BLOCK_DIMX);
    execKernel(init, gridDim, BLOCK_DIMX, devID, true,
               row_ass, col_ass, row_cover, col_cover);
    CUDA_RUNTIME(cudaMemset(zeros_size_b, 0, nb4 * sizeof(size_t)));

    gridDim = (uint)ceil(psize2 * 1.0 / BLOCK_DIMX);
    execKernel(compress_matrix, gridDim, BLOCK_DIMX, devID, true,
               zeros, zeros_size_b, slack);
    zeros_size = thrust::reduce(thrust::device, zeros_size_b, zeros_size_b + nb4);
    printDebugArray(zeros_size_b, nb4, "zeros array");
    Log(debug, "Zeros size: %d", zeros_size);

    do
    {
      repeat_kernel = false;
      uint blockDim = (nb4 > 1 || zeros_size > BLOCK_DIMX) ? BLOCK_DIMX : zeros_size;
      execKernel(step2, nb4, blockDim, devID, true,
                 zeros, zeros_size_b, row_cover, col_cover, row_ass, col_ass);
    } while (repeat_kernel);
    printDebugArray(row_ass, psize, "row assignments");
    printDebugArray(col_ass, psize, "col assignments");
  }
  void S3() // get match count
  {
    CUDA_RUNTIME(cudaMemset(row_cover, 0, psize * sizeof(int)));
    CUDA_RUNTIME(cudaMemset(col_cover, 0, psize * sizeof(int)));
    uint gridDim = (uint)ceil(psize * 1.0 / BLOCK_DIMX);
    nmatch_old = nmatch_cur;
    nmatch_cur = 0;
    CUDA_RUNTIME(cudaDeviceSynchronize());
    execKernel(step3, gridDim, BLOCK_DIMX, devID, true, row_ass, col_cover);
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
};