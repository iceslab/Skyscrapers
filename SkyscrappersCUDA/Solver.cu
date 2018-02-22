#include "Solver.cuh"

namespace cuda
{
    namespace solver
    {
        Solver::Solver(const board::Board & board) : board(board)
        {
            // Nothing to do
        }

        solver::Solver::Solver(board::Board && board) : board(board)
        {
            // Nothing to do
        }
    }
}
