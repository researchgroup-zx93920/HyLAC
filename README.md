# **HyLAC** - **Hy**brid **L**inear **A**ssignment solver in **CUDA**

<!-- Insert a bullet list of highlights -->

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

## Usage

## Keywords

Hungarian algorithm, Linear assignment problem, GPU-accelerated graph algorithms, High-performance computing, Parallel algorithm.

### Reference

Samiran Kawtikwar and Rakesh Nagi. 2024. HyLAC: Hybrid linear assignment solver in CUDA. Journal of Parallel and Distributed Computing 187, (May 2024), 104838. https://doi.org/10.1016/j.jpdc.2024.104838
