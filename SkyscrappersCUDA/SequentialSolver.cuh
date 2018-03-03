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

            CUDA_DEVICE uint32T solve(cuda::Board* resultArray, uint32T threadIdx);

            /// Backtracking
            CUDA_DEVICE uint32T backTrackingBase(cuda::Board* resultArray, uint32T threadIdx);
            CUDA_DEVICE uint32T backTrackingAOSStack(cuda::Board* resultArray, uint32T threadIdx, stackAOST* stack);
            CUDA_DEVICE uint32T backTrackingSOAStack(cuda::Board* resultArray, uint32T threadIdx, stackSOAT* stack);

            CUDA_DEVICE void getNextFreeCell(uint32T row,
                                             uint32T column,
                                             uint32T & rowOut,
                                             uint32T & columnOut) const;

            static CUDA_DEVICE bool isCellValid(uint32T row, uint32T column);

            CUDA_DEVICE const cuda::Board & getBoard() const;
        };
    }
}
#endif // !__INCLUDED_SEQUENTIAL_SOLVER_CUH__
