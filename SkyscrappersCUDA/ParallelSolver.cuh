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

        //CUDA_HOST std::vector<board::Board> generateBoards(const board::Board & board, const size_t stopLevel);
        //CUDA_HOST void generateBoards(const board::Board & board,
        //                              size_t stopLevel,
        //                              std::vector<board::Board> & retVal,
        //                              size_t level = 0,
        //                              size_t row = 0,
        //                              size_t column = 0);

        // Generates solvers for boards till given tree level (count)
        // Solvers count is then placed in count variable
        CUDA_HOST kernelInputT prepareSolvers(const std::vector<board::Board> & boards, size_t & count);
        CUDA_HOST kernelOutputT prepareResultArray(size_t solversCount);
        CUDA_HOST kernelOutputSizesT prepareResultArraySizes(size_t solversCount);
        CUDA_HOST stackT prepareStack(size_t boardSize, size_t solversCount);

        // Complementary function to free solver array
        CUDA_HOST void freeSolvers(kernelInputT & d_solvers);
        // Complementary function to free results array
        CUDA_HOST void freeResultArray(kernelOutputT & d_outputBoards);
        // Complementary function to free results array sizes
        CUDA_HOST void freeResultArraySizes(kernelOutputSizesT & d_outputBoardsSizes);
        // Complementary function to free stack
        CUDA_HOST void freeStack(stackT & d_stack);

        CUDA_HOST bool verifyAllocation(kernelInputT & d_solvers,
                                        kernelOutputT & d_outputBoards,
                                        kernelOutputSizesT & d_outputBoardsSizes,
                                        stackT & d_stack);

        CUDA_GLOBAL void parallelBoardSolving(const kernelInputT & d_solvers,
                                              kernelOutputT & d_outputBoards,
                                              kernelOutputSizesT & d_outputBoardsSizes,
                                              stackT & d_stack);
    }
}

#endif // !__INCLUDED_PARALLEL_SOLVER_CUH__
