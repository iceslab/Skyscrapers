#ifndef __INCLUDED_SEQUENTIAL_SOLVER_CUH__
#define __INCLUDED_SEQUENTIAL_SOLVER_CUH__

#include "Solver.cuh"
#include "BitManipulation.cuh"
#include "Stack.cuh"

#define CUDA_SIZE_T_MAX (size_t(~0))
#define CUDA_UINT32_T_MAX (cuda::uint32T(~0))
#define CUDA_LAST_CELL_PAIR (rowAndColumnPairT(CUDA_UINT32_T_MAX, CUDA_UINT32_T_MAX))
#define CUDA_MAX_RESULTS_PER_THREAD (cuda::uint32T(5))

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
            ~SequentialSolver() = default;

            /// Backtracking
            CUDA_DEVICE uint32T backTrackingBase(cuda::Board* resultArray, uint32T threadIdx);
            CUDA_DEVICE uint32T backTrackingAOSStack(cuda::Board * resultArray,
                                                     stackAOST * stack,
                                                     const uint32T threadIdx,
                                                     const uint32T threadsCount);
            CUDA_DEVICE uint32T backTrackingSOAStack(cuda::Board* resultArray,
                                                     stackSOAT* stack,
                                                     const uint32T threadIdx,
                                                     const uint32T threadsCount);

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
