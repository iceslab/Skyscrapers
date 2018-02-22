#ifndef __INCLUDED_SEQUENTIAL_SOLVER_CUH__
#define __INCLUDED_SEQUENTIAL_SOLVER_CUH__

#include "Solver.cuh"
#include "StackEntry.cuh"
#include "Pair.cuh"

#define BT_WITH_STACK

namespace cuda
{
    namespace solver
    {
        constexpr const size_t maxResultsPerThread = 5;
        constexpr const size_t maxStackEntrySize = 64;

        typedef cuda::Pair<size_t, size_t> rowAndColumnPairT;
        typedef StackEntry<maxStackEntrySize> stackEntryT;
        typedef cuda::Pair<stackEntryT, rowAndColumnPairT> stackPairT;
        typedef stackPairT* stackT;

        class SequentialSolver :
            public Solver
        {
        public:
            SequentialSolver(const board::Board& board);
            ~SequentialSolver() = default;

            CUDA_DEVICE size_t solve(cuda::Board* resultArray, stackT stack);

        protected:
            // Max value for cell
            const size_t maxVal;
            const rowAndColumnPairT lastCellPair;

            /// Backtracking
#ifndef BT_WITH_STACK
            void backTracking(std::vector<cuda::Board> & retVal, size_t level = 0, size_t row = 0, size_t column = 0);
#else
            CUDA_DEVICE size_t backTrackingWithStack(cuda::Board* resultArray, stackT stack);
#endif // !BT_WITH_STACK
            CUDA_DEVICE rowAndColumnPairT getNextFreeCell(size_t row, size_t column) const;
        };
    }
}
#endif // !__INCLUDED_SEQUENTIAL_SOLVER_CUH__
