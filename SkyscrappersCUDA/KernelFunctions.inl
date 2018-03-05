#ifndef __INCLUDED_KERNEL_FUNCTIONS_INL__
#define __INCLUDED_KERNEL_FUNCTIONS_INL__

#include "CUDAUtilities.cuh"
#include "ParallelSolver.cuh"
#include "Stack.cuh"

CUDA_GLOBAL void parallelSolvingBase(cuda::solver::kernelInputT d_solvers,
                                     cuda::solver::kernelOutputT d_outputBoards,
                                     cuda::solver::kernelOutputSizesT d_outputBoardsSizes)
{
    // It denotes thread index and array index
    const auto idx = threadIdx.x;
    d_outputBoardsSizes[idx] =
        d_solvers[idx].backTrackingBase(d_outputBoards + idx * CUDA_MAX_RESULTS_PER_THREAD,
                                        idx);
}

CUDA_GLOBAL void parallelSolvingAOSStack(cuda::solver::kernelInputT d_solvers,
                                         cuda::solver::kernelOutputT d_outputBoards,
                                         cuda::solver::kernelOutputSizesT d_outputBoardsSizes,
                                         cuda::solver::stackAOST* stack)
{
    // It denotes thread index and array index
    const auto idx = threadIdx.x;
    const auto threads = blockDim.x;
    d_outputBoardsSizes[idx] =
        d_solvers[idx].backTrackingAOSStack(d_outputBoards + idx * CUDA_MAX_RESULTS_PER_THREAD,
                                            stack,
                                            idx,
                                            threads);
}

CUDA_GLOBAL void parallelSolvingSOAStack(cuda::solver::kernelInputT d_solvers,
                                         cuda::solver::kernelOutputT d_outputBoards,
                                         cuda::solver::kernelOutputSizesT d_outputBoardsSizes,
                                         cuda::solver::stackSOAT* stack)
{
    // It denotes thread index and array index
    const auto idx = threadIdx.x;
    const auto threads = blockDim.x;
    d_outputBoardsSizes[idx] =
        d_solvers[idx].backTrackingSOAStack(d_outputBoards + idx * CUDA_MAX_RESULTS_PER_THREAD,
                                            stack,
                                            idx,
                                            threads);
}


#endif // !__INCLUDED_KERNEL_FUNCTIONS_INL__
