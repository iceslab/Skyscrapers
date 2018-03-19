#ifndef __INCLUDED_PARALLEL_SOLVER_CUH__
#define __INCLUDED_PARALLEL_SOLVER_CUH__

#include <vector>
#include "../Skyscrappers/Board.h"
#include "SequentialSolver.cuh"

namespace cuda
{
    namespace solver
    {
        typedef solver::SequentialSolver* kernelInputT;
        typedef cuda::Board* kernelOutputT;
        typedef size_t* kernelOutputSizesT;
        typedef struct
        {
            // Result boards count
            cuda::uint32T resultsCount = 0;
            // Stack
            cuda::uint32T* stack = nullptr;
            // Stack rows
            cuda::uint32T* stackRows = nullptr;
            // Stack columns
            cuda::uint32T* stackColumns = nullptr;
            // Current valid stack frames
            cuda::uint32T stackSize = 0;
            // Used for row result from getNextFreeCell()
            cuda::uint32T rowRef = 0;
            // Used for column result from getNextFreeCell()
            cuda::uint32T columnRef = 0;
            // Stack size limit
            cuda::uint32T stackEntrySize = 0;
            // Building index
            cuda::uint32T buildingIdx = 0;
        } threadLocalsT;

        // Generates solvers for boards till given tree level (count)
        // Solvers count is then placed in count variable
        CUDA_HOST kernelInputT prepareSolvers(const std::vector<board::Board> & boards,
                                              std::vector<SequentialSolver> & h_solvers,
                                              size_t & count);
        CUDA_HOST kernelOutputT prepareResultArray(std::vector<cuda::Board> & h_boards,
                                                   size_t solversCount,
                                                   size_t boardSize);
        CUDA_HOST kernelOutputSizesT prepareResultArraySizes(size_t solversCount);
        CUDA_HOST threadLocalsT* prepareThreadLocals(size_t solversCount);
        CUDA_HOST uint32T* prepareScatterArray(size_t solversCount);

        CUDA_HOST kernelOutputT prepareHostResultArray(size_t solversCount);
        CUDA_HOST kernelOutputSizesT prepareHostResultArraySizes(size_t solversCount);

        // Complementary function to free solver array
        CUDA_HOST void freeSolvers(kernelInputT & d_solvers);
        // Complementary function to free results array
        CUDA_HOST void freeResultArray(kernelOutputT & d_outputBoards);
        // Complementary function to free results array sizes
        CUDA_HOST void freeResultArraySizes(kernelOutputSizesT & d_outputBoardsSizes);
        // Complementary function to free thread locals structures
        CUDA_HOST void freeThreadLocals(threadLocalsT* & d_threadLocals);
        // Complementary function to free scatter array
        CUDA_HOST void freeScatterArray(uint32T* & d_scatterArray);

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
                                        kernelOutputSizesT & d_outputBoardsSizes);
    }
}

#endif // !__INCLUDED_PARALLEL_SOLVER_CUH__
