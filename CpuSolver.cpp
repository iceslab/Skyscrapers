#include "CpuSolver.h"

using namespace solver;

CpuSolver::CpuSolver(const board::Board & board) : Solver(board), constraints(board.size())
{
    this->board.fill(board::boardFieldT());
}

void CpuSolver::solve()
{
    ASSERT_VERBOSE(board.size() > 0,
                   "Board size must be greater than 0. Got: %lu",
                   board.size());
    auto startingTechniques = [this](size_t row, size_t column)->void
    {
        findCluesOfOne(row, column);
        findCluesOfN(row, column);
        setSatisfiedConstraints(row, column);
    };

    auto basicTechniques = [this](size_t row, size_t column)->void
    {
        findHighSkyscrapers1(row, column);
        // TODO: add more basic techniques methods to call
    };

    auto setConstraints = [this](size_t row, size_t column)->void
    {
        setSatisfiedConstraints(row, column);
    };

    board.forEachCell(startingTechniques);
    board.forEachVector(basicTechniques);
    board.forEachCell(setConstraints);
}

void solver::CpuSolver::print() const
{
    constraints.print();
    board.print();
}

bool solver::CpuSolver::setConstraint(size_t row, size_t column, board::boardFieldT value, bool conditionally)
{
    if (conditionally)
        return setConstraintConditionally(row, column, value);
    return setConstraintUnconditionally(row, column, value);
}

bool solver::CpuSolver::setConstraintConditionally(size_t row, size_t column, board::boardFieldT value)
{
    auto rowVec = board.getRow(row);
    auto columnVec = board.getColumn(column);
    auto valueElementsInRow = std::count(rowVec.begin(), rowVec.end(), value);
    auto valueElementsInColumn = std::count(columnVec.begin(), columnVec.end(), value);
    ASSERT(valueElementsInRow <= 1 && valueElementsInColumn <= 1);

    if (valueElementsInRow == 0 && valueElementsInColumn == 0)
    {
        constraints[row][column].insert(value);
        return true;
    }
    return false;
}

bool solver::CpuSolver::setConstraintUnconditionally(size_t row, size_t column, board::boardFieldT value)
{
    constraints[row][column].insert(value);
    return true;
}

void solver::CpuSolver::findCluesOfOne(size_t row, size_t column)
{
    auto rowEdge = board.whichEdgeRow(row);
    auto columnEdge = board.whichEdgeColumn(column);

    // There is only one building seen from row or column
    if (rowEdge != matrix::NONE && board.hints[rowEdge][column] == 1 ||
        columnEdge != matrix::NONE && board.hints[columnEdge][row] == 1)
    {
        constraints[row][column].insert(board.size());
    }
}

void solver::CpuSolver::findCluesOfN(size_t row, size_t column)
{
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

void solver::CpuSolver::findHighSkyscrapers1(size_t row, size_t column)
{
    const auto highest = board.size();
    const auto secondHighest = highest - 1;
    const auto leftHint = board.hints[matrix::LEFT][row];
    const auto rightHint = board.hints[matrix::RIGHT][row];
    const auto topHint = board.hints[matrix::TOP][column];
    const auto bottomHint = board.hints[matrix::BOTTOM][column];

    const auto firstIdx = 0;
    const auto secondIdx = board.size() > 1 ? firstIdx + 1 : firstIdx;
    const auto lastIdx = board.size() - 1;
    // Just to be safe and not get MAX_UINT
    const auto secondLastIdx = lastIdx > 0 ? lastIdx - 1 : lastIdx;

    if (leftHint == secondHighest)
    {
        setConstraint(row, secondLastIdx, highest);
        setConstraint(row, lastIdx, highest);
    }
    else if (rightHint == secondHighest)
    {
        setConstraint(row, firstIdx, highest);
        setConstraint(row, secondIdx, highest);
    }
    else if (topHint == secondHighest)
    {
        setConstraint(secondLastIdx, column, highest);
        setConstraint(lastIdx, column, highest);
    }
    else if (bottomHint == secondHighest)
    {
        setConstraint(firstIdx, column, highest);
        setConstraint(secondIdx, column, highest);
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

