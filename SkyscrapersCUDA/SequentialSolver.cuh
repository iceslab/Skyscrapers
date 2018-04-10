#ifndef __INCLUDED_SEQUENTIAL_SOLVER_CUH__
#define __INCLUDED_SEQUENTIAL_SOLVER_CUH__

#include "Solver.cuh"
#include "BitManipulation.cuh"
#include "Stack.cuh"

#define CUDA_SIZE_T_MAX (size_t(~0))
#define CUDA_LAST_CELL_PAIR (rowAndColumnPairT(CUDA_UINT32T_MAX, CUDA_UINT32T_MAX))
#define CUDA_MAX_RESULTS (cuda::uint32T(20))

#define BT_WITH_STACK

namespace cuda
{
    namespace solver
    {
        class SequentialSolver :
            public Solver
        {
        public:
            SequentialSolver(const board::Board& board);
            CUDA_HOST_DEVICE SequentialSolver(const cuda::Board& board,
                                              void * constantMemoryPtr = nullptr,
                                              void * sharedMemoryPtr = nullptr);
            CUDA_HOST_DEVICE ~SequentialSolver();

            /// Backtracking
            CUDA_DEVICE void backTrackingBase(cuda::Board* resultArray,
                                              uint32T* allResultsCount,
                                              uint32T threadIdx,
                                              cuda::cudaEventsDeviceT & timers);
            CUDA_DEVICE void backTrackingIncrementalStack(cuda::Board* resultArray,
                                                          uint32T* allResultsCount,
                                                          uint32T threadIdx,
                                                          cuda::cudaEventsDeviceT & timers);
            CUDA_DEVICE void backTrackingAOSStack(cuda::Board * resultArray,
                                                  uint32T* allResultsCount,
                                                  stackAOST * stack,
                                                  const uint32T threadIdx,
                                                  const uint32T threadsCount,
                                                  cuda::cudaEventsDeviceT & timers);
            CUDA_DEVICE void backTrackingSOAStack(cuda::Board* resultArray,
                                                  uint32T* allResultsCount,
                                                  stackSOAT* stack,
                                                  const uint32T threadIdx,
                                                  const uint32T threadsCount,
                                                  cuda::cudaEventsDeviceT & timers);

            CUDA_DEVICE void getNextFreeCell(uint32T row,
                                             uint32T column,
                                             uint32T & rowOut,
                                             uint32T & columnOut) const;

            static CUDA_DEVICE bool isCellValid(uint32T row, uint32T column);

            CUDA_DEVICE const cuda::Board & getBoard() const;
            static CUDA_HOST_DEVICE uint32T getStackFrameNumber(uint32T stackSize,
                                                                const uint32T threadId,
                                                                const uint32T threadsCount);

        };
    }
}
#endif // !__INCLUDED_SEQUENTIAL_SOLVER_CUH__
