#ifndef __INCLUDED_SOLVER_CUH__
#define __INCLUDED_SOLVER_CUH__
#include "../Board.h"
#include "Board.cuh"

namespace cuda
{
    namespace solver
    {
        class Solver
        {
        public:
            Solver(const board::Board& board);
            Solver(board::Board&& board);
            ~Solver() = default;
        protected:
            cuda::Board board;
        };
    }
}
#endif // !__INCLUDED_SOLVER_CUH__
