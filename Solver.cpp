#include "Solver.h"

using namespace solver;

Solver::Solver(const board::Board & board) : board(board)
{
    // Nothing to do
}

solver::Solver::Solver(board::Board && board) : board(board)
{
    // Nothing to do
}

void solver::Solver::printResults(const std::vector<board::Board>& results)
{
    for (const auto & board : results)
    {
        board.print();
    }
}
