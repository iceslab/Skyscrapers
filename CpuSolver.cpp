#include "CpuSolver.h"

using namespace solver;

CpuSolver::CpuSolver(const board::Board & board) : Solver(board), constraints(board.size())
{
    this->board.fill(board::boardFieldT());
}

void CpuSolver::solve()
{
    for (size_t row = 0; row < board.size(); row++)
    {
        for (size_t column = 0; column < board.size(); column++)
        {
            findEdgeConstraints(row, column);
        }
    }
}

void solver::CpuSolver::print() const
{
    constraints.print();
}

bool CpuSolver::findEdgeConstraints(size_t row, size_t column)
{
    auto rowEdge = board.whichEdgeRow(row);
    auto columnEdge = board.whichEdgeColumn(column);
        
    // There is only one building seen from row
    if (board.getVisibleBuildings(rowEdge, column) == 1)
    {
        constraints[row][column].insert(board.size());
        return true;
    }

    // There is only one building seen from column
    if (board.getVisibleBuildings(columnEdge, row) == 1)
    {
        constraints[row][column].insert(board.size());
        return true;
    }

    return false;
}

