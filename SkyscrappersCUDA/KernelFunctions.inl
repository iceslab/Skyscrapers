#ifndef __INCLUDED_KERNEL_FUNCTIONS_INL__
#define __INCLUDED_KERNEL_FUNCTIONS_INL__

#include "CUDAUtilities.cuh"
#include "ParallelSolver.cuh"


CUDA_GLOBAL void parallelBoardSolving(cuda::solver::kernelInputT d_solvers,
                                      cuda::solver::kernelOutputT d_outputBoards,
                                      cuda::solver::kernelOutputSizesT d_outputBoardsSizes)
{
    // It denotes thread index and array index
    const auto idx = threadIdx.x;
    d_outputBoardsSizes[idx] =
        d_solvers[idx].solve(d_outputBoards + idx * CUDA_MAX_RESULTS_PER_THREAD, idx);
}
#endif // !__INCLUDED_KERNEL_FUNCTIONS_INL__
