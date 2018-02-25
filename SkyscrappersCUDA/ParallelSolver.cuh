#ifndef __INCLUDED_PARALLEL_SOLVER_CUH__
#define __INCLUDED_PARALLEL_SOLVER_CUH__

#include <vector>
#include "../Board.h"
#include "SequentialSolver.cuh"

namespace cuda
{
    namespace solver
    {
        typedef solver::SequentialSolver* kernelInputT;
        typedef cuda::Board* kernelOutputT;
        typedef size_t* kernelOutputSizesT;

        // Generates solvers for boards till given tree level (count)
        // Solvers count is then placed in count variable
        CUDA_HOST kernelInputT prepareSolvers(const std::vector<board::Board> & boards,
                                              std::vector<SequentialSolver> & h_solvers,
                                              size_t & count);
        CUDA_HOST kernelOutputT prepareResultArray(size_t solversCount);
        CUDA_HOST kernelOutputSizesT prepareResultArraySizes(size_t solversCount);
        CUDA_HOST stackPtrT prepareStack(size_t boardSize, size_t solversCount);

        CUDA_HOST kernelOutputT prepareHostResultArray(size_t solversCount);
        CUDA_HOST kernelOutputSizesT prepareHostResultArraySizes(size_t solversCount);

        // Complementary function to free solver array
        CUDA_HOST void freeSolvers(kernelInputT & d_solvers);
        // Complementary function to free results array
        CUDA_HOST void freeResultArray(kernelOutputT & d_outputBoards);
        // Complementary function to free results array sizes
        CUDA_HOST void freeResultArraySizes(kernelOutputSizesT & d_outputBoardsSizes);
        // Complementary function to free stack
        CUDA_HOST void freeStack(stackPtrT & d_stack);

        // Complementary function to free results array
        CUDA_HOST void freeHostResultArray(kernelOutputT & h_outputBoards);
        // Complementary function to free results array sizes
        CUDA_HOST void freeHostResultArraySizes(kernelOutputSizesT & h_outputBoardsSizes);

        CUDA_HOST void copyResultsArray(kernelOutputT h_outputBoards,
                                        kernelOutputT d_outputBoards,
                                        size_t solversCount);
        CUDA_HOST void copyResultsArraySizes(kernelOutputSizesT h_outputBoardsSizes,
                                             kernelOutputSizesT d_outputBoardsSizes,
                                             size_t solversCount);

        CUDA_HOST bool verifyAllocation(kernelInputT & d_solvers,
                                        kernelOutputT & d_outputBoards,
                                        kernelOutputSizesT & d_outputBoardsSizes,
                                        stackPtrT & d_stack);

        /*CUDA_GLOBAL void parallelBoardSolving(kernelInputT d_solvers,
                                              kernelOutputT d_outputBoards,
                                              kernelOutputSizesT d_outputBoardsSizes,
                                              stackT d_stack);*/
    }
}

#endif // !__INCLUDED_PARALLEL_SOLVER_CUH__
