#ifndef __INCLUDED_SEQUENTIAL_SOLVER_CUH__
#define __INCLUDED_SEQUENTIAL_SOLVER_CUH__

#include "Solver.cuh"
#include "BitManipulation.cuh"

#define CUDA_SIZE_T_MAX (size_t(~0))
#define CUDA_LAST_CELL_PAIR (rowAndColumnPairT(CUDA_SIZE_T_MAX, CUDA_SIZE_T_MAX))


#define BT_WITH_STACK

namespace cuda
{
    namespace solver
    {
        constexpr const size_t maxResultsPerThread = 5;
        constexpr const size_t maxStackEntrySize = 64;

        class SequentialSolver :
            public Solver
        {
        public:
            SequentialSolver(const board::Board& board);
            ~SequentialSolver() = default;

            CUDA_DEVICE size_t solve(cuda::Board* resultArray, size_t threadIdx);

            /// Backtracking
#ifndef BT_WITH_STACK
            void backTracking(std::vector<cuda::Board> & retVal, size_t level = 0, size_t row = 0, size_t column = 0);
#else
            CUDA_DEVICE size_t backTrackingWithStack(cuda::Board* resultArray, size_t threadIdx);
#endif // !BT_WITH_STACK
            CUDA_DEVICE void getNextFreeCell(size_t row,
                                             size_t column,
                                             size_t & rowOut,
                                             size_t & columnOut) const;

            static CUDA_DEVICE bool isCellValid(size_t row, size_t column);

            CUDA_DEVICE const cuda::Board & getBoard() const;
        };
    }
}
#endif // !__INCLUDED_SEQUENTIAL_SOLVER_CUH__
