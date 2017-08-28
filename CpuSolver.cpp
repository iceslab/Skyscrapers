#include "CpuSolver.h"

using namespace solver;

CpuSolver::CpuSolver(const board::Board & board) : Solver(board), constraints(board.size())
{
    this->board.fill(board::boardFieldT());
}

void CpuSolver::solve()
{
    auto findEdgeConstraintsFn =
        std::bind(&CpuSolver::findEdgeConstraints,
                  this,
                  std::placeholders::_1,
                  std::placeholders::_2);

    auto setSatisfiedConstraintsFn =
        std::bind(&CpuSolver::setSatisfiedConstraints,
                  this,
                  std::placeholders::_1,
                  std::placeholders::_2);

    board.forEach(findEdgeConstraintsFn);
    board.forEach(setSatisfiedConstraintsFn);
}

void solver::CpuSolver::print() const
{
    constraints.print();
    board.print();
}

void CpuSolver::findEdgeConstraints(size_t row, size_t column)
{
    auto rowEdge = board.whichEdgeRow(row);
    auto columnEdge = board.whichEdgeColumn(column);

    // There is only one building seen from row or column
    if (rowEdge != matrix::NONE && board.hints[rowEdge][column] == 1 ||
        columnEdge != matrix::NONE && board.hints[columnEdge][row] == 1)
    {
        constraints[row][column].insert(board.size());
    }

    // All buildings are visible
    if (board.hints[matrix::TOP][column] == board.size())
    {
        auto columns = constraints.getColumn(column);
        auto it = columns.begin();
        for (board::boardFieldT i = 1; i <= board.size(); i++, it++)
        {
            it->get().insert(i);
        }
    }
    else if (board.hints[matrix::BOTTOM][column] == board.size())
    {
        auto columns = constraints.getColumn(column);
        auto it = columns.rbegin();
        for (board::boardFieldT i = 1; i <= board.size(); i++, it++)
        {
            it->get().insert(i);
        }
    }
    else if (board.hints[matrix::LEFT][row] == board.size())
    {
        auto& rows = constraints.getRow(row);
        auto it = rows.begin();
        for (board::boardFieldT i = 1; i <= board.size(); i++, it++)
        {
            it->insert(i);
        }
    }
    else if (board.hints[matrix::RIGHT][row] == board.size())
    {
        auto rows = constraints.getRow(row);
        auto it = rows.rbegin();
        for (board::boardFieldT i = 1; i <= board.size(); i++, it++)
        {
            it->insert(i);
        }
    }
}

void solver::CpuSolver::findPhase2Constraints(size_t row, size_t column)
{
    auto highestInRow = board.locateHighestInRow(row);
    auto highestInColumn = board.locateHighestInColumn(column);
    auto distanceInRow = column > highestInRow ? column - highestInRow : highestInRow - column;
    auto distanceInColumn = row > highestInColumn ? row - highestInColumn : highestInColumn - row;


    if (highestInRow < board.size())
    {

    }
}

void solver::CpuSolver::setSatisfiedConstraints(size_t row, size_t column)
{
    // There is only constraint
    if (constraints[row][column].size() == 1)
    {
        board[row][column] = *constraints[row][column].begin();
    }
}

