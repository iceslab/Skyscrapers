#ifndef __INCLUDED_PARALLEL_SOLVER_CUH__
#define __INCLUDED_PARALLEL_SOLVER_CUH__

#include <vector>
#include "../Skyscrappers/Board.h"
#include "SequentialSolver.cuh"
#include "XGetopt.h"

extern CUDA_CONSTANT cuda::boardFieldT constantMemoryPtr[];

namespace cuda
{
    namespace solver
    {
        typedef solver::SequentialSolver* kernelInputT;
        typedef cuda::Board* kernelOutputT;
        typedef size_t* kernelOutputSizesT;
        typedef uint32T kernelOutputSizeT;
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
        CUDA_HOST cuda::uint32T* prepareGeneratedSolversCount(cuda::uint32T generatedSolversCount);
        CUDA_HOST kernelInputT prepareSolvers(const std::vector<board::Board> & boards,
                                              std::vector<SequentialSolver> & h_solvers,
                                              size_t & count);
        CUDA_HOST cuda::uint32T* prepareSolversTaken();
        CUDA_HOST kernelOutputT prepareResultArray(std::vector<cuda::Board> & h_boards,
                                                   size_t solversCount,
                                                   size_t boardSize);
        CUDA_HOST uint32T* prepareResultArraySize();
        CUDA_HOST threadLocalsT* prepareThreadLocals(size_t solversCount);
        CUDA_HOST uint32T* prepareScatterArray(size_t solversCount);
        CUDA_HOST cudaEventsDeviceT* prepareCudaEventDevice(const std::vector<cudaEventsDeviceT> & h_events);
        CUDA_HOST void* prepareStack(SolversEnableE solverType,
                                     size_t generatedSolversCount,
                                     size_t cellsCount);
        CUDA_HOST void prepareConstantMemory(const board::Board & board);

        CUDA_HOST kernelOutputT prepareHostResultArray();

        // Complementary function to free generated solvers variable
        CUDA_HOST void freeGeneratedSolversCount(cuda::uint32T* d_generatedSolversCount);
        // Complementary function to free solver array
        CUDA_HOST void freeSolvers(kernelInputT & d_solvers);
        // Complementary function to free solver taken counter
        CUDA_HOST void freeSolversTaken(cuda::uint32T* & d_solversTaken);
        // Complementary function to free results array
        CUDA_HOST void freeResultArray(kernelOutputT & d_outputBoards);
        // Complementary function to free results array sizes
        CUDA_HOST void freeResultArraySize(uint32T* & d_outputBoardsSize);
        // Complementary function to free thread locals structures
        CUDA_HOST void freeThreadLocals(threadLocalsT* & d_threadLocals);
        // Complementary function to free scatter array
        CUDA_HOST void freeScatterArray(uint32T* & d_scatterArray);
        // Complementary function to free cudaEventDevice array
        CUDA_HOST void freeCudaEventDevice(cudaEventsDeviceT* & d_timers);
        // Complementary function to free stack
        CUDA_HOST void freeStack(void* & d_stack);

        // Complementary function to free results array
        CUDA_HOST void freeHostResultArray(kernelOutputT & h_outputBoards);

        CUDA_HOST void copyResultsArray(kernelOutputT h_outputBoards,
                                        kernelOutputT d_outputBoards,
                                        size_t solversCount);
        CUDA_HOST void copyResultsArraySize(uint32T* h_outputBoardsSize,
                                            uint32T* d_outputBoardsSize);
        CUDA_HOST void copyCudaEventDevice(std::vector<cudaEventsDeviceT> & h_timers,
                                           cudaEventsDeviceT* & d_timers);

        CUDA_HOST bool verifyAllocation(kernelInputT & d_solvers,
                                        kernelOutputT & d_outputBoards,
                                        uint32T* & d_outputBoardsSizes);
        CUDA_HOST int getSharedMemorySize(SolversEnableE solverType);
    }
}

#endif // !__INCLUDED_PARALLEL_SOLVER_CUH__
