#ifndef __INCLUDED_SOLVER_CUH__
#define __INCLUDED_SOLVER_CUH__
#include "../Skyscrappers/Board.h"
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
            Solver(const cuda::Board& board);
            ~Solver() = default;
        protected:
            cuda::Board board;
        };
    }
}
#endif // !__INCLUDED_SOLVER_CUH__
