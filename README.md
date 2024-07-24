# **HyLAC** - **Hy**brid **L**inear **A**ssignment solver in **CUDA**

<!-- Insert index with links to individual headings -->

## Highlights

- The fastest Linear Assignment Problem (LAP) solver that uses GPUs.
- Improved implementation of classical and tree variants of Hungarian algorithm.
- Hybrid approach that switches between classical and tree variants based on sparsity and dynamic strategies.
- Useful for Solving a stream of small LAPs (22.59× faster than existing solution!).

## Description

HyLAC is a hybrid linear assignment solver in CUDA that uses a combination of classical and tree variants of the Hungarian algorithm. It is designed to solve the Linear Assignment Problem (LAP) efficiently on GPUs. The hybrid approach switches between classical and tree variants based on sparsity and dynamic strategies. The Hybrid approach ensures performance for sparce as well as dense problem instances.HyLAC achieves a speedup of up to 6.14× over existing state-of-the-art GPU implementations when run on the same hardware

HyLAC also presents a stream-solver that solves a list of LAPs using GPU thread blocks. The stream-solver is 22.59× faster than the existing solution.

We also provide the solver at different granularity levels:

- **Fine-grained solver**: Solves a single LAP with multiple thread blocks (global kernel launch).
- **Coarse-grained solver**: Solves a list of small LAPs, each solved with a single thread block (stream-solver).
- **Block-LAP**: Solves a single LAP with a single thread block (block kernel launch or a device function).
- **Warp-LAP**: Solves a single LAP with a single warp (warp kernel launch or a device function) _[In development]_.

## Navigate

[Installation](#installation) \
[Usage](#usage) \
[Example code](#example-code) \
[Using inside existing workflow](#using-inside-existing-workflow) \
[Reference](#reference)

## Installation

### Hardware and software requirements

- NVIDIA GPU with CUDA compute capability 6.0 or higher
- CUDA Toolkit 10.0 or higher (Need to add cub separately if CUDA toolkit version < 11.0)
- NVCC compiler with C++11 support
- Linux environment (should work on Windows, but not tested; WSL should work)

### Compiling the code

- Clone the repository
- Go to the codebase directory
- In each directory, run `make clean all` to compile the code with the makefile provided in that directory. The default architecture version used in each makefile is 80. Change it according to your GPU architecture. \
  Do not know the architecture of your GPU? Check [here](https://developer.nvidia.com/cuda-gpus).

## Usage

#### Case 1: Solving single LAP (Fine-grained solver) [Hung-Hybrid branch]

For solving with the classical solver: go to `codebase/Hung/` \
For solving with the Hybrid solver: go to `codebase/Hung-CT/`

```
Usage: [options]
-n <size of the problem> [required]
-f <range-fraction> [default: 1]   (%This is used to control the sparsity of the problem)
-d <deviceId [default: 0]
-s <seed-value> [default: 45345]

% For stream solver
-t size of the stream [required]
```

## Example code

### Solving a single LAP with the classical solver:

#### Terminal commands

```
cd codebase/Hung/
make clean all
./build/exe/src/Hung.cu.exe -n 1000 -f 10 -d 0
```

#### Sample output

```
Welcome ---------------------
  size: 10000
  frac: 1.000000
  Device: 0
  seed value: 45345
[22:54:18] cost generation time 0.067608 s
[22:54:18] LAP object generated succesfully
Obj val: 18543
[22:54:18] solve time 0.448983 s
```

## Using inside an existing workflow

The library has been made into a class. for the classical solver, use the LAP class: can generate new object instance using the constructor `LAP<data> *lap = new LAP<data>(h_costs, user_n, dev)` and solve the LAP using `lap->solve()`. h_costs is the cost matrix (on host memory), user_n is the size of the problem, and dev is the device ID.

### Reference

Samiran Kawtikwar and Rakesh Nagi. 2024. HyLAC: Hybrid linear assignment solver in CUDA. Journal of Parallel and Distributed Computing 187, (May 2024), 104838. https://doi.org/10.1016/j.jpdc.2024.104838

### Keywords

Hungarian algorithm, Linear assignment problem, GPU-accelerated graph algorithms, High-performance computing, Parallel algorithm.
