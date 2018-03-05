#include "Solver.h"

namespace solver
{

    Solver::Solver(const board::Board & board) : board(board)
    {
        // Nothing to do
    }

    Solver::Solver(board::Board && board) : board(board)
    {
        // Nothing to do
    }

    void Solver::printResults(const std::vector<board::Board>& results)
    {
        if (results.empty())
        {
            std::cout << "No results" << std::endl;
        }

        for (const auto & board : results)
        {
            board.print();
        }
    }
}
