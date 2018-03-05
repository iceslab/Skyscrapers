#include "SequentialSolver.h"

using namespace solver;
const rowAndColumnPairT SequentialSolver::lastCellPair = std::make_pair(std::numeric_limits<size_t>::max(),
                                                                        std::numeric_limits<size_t>::max());

SequentialSolver::SequentialSolver(const board::Board & board) :
    Solver(board),
    continueBackTracking(nullptr),
    constraints(board.size())
{
    this->board.fill(board::boardFieldT());
}

solver::SequentialSolver::SequentialSolver(board::Board && board) :
    Solver(board),
    continueBackTracking(nullptr),
    constraints(board.size())
{
    // Nothing to do
}

std::vector<board::Board> SequentialSolver::solve()
{
    ASSERT_VERBOSE(board.size() > 0,
                   "Board size must be greater than 0. Got: %zu",
                   board.size());
    auto startingTechniques = [this](size_t row, size_t column)->void
    {
        findCluesOfOne(row, column);
        findCluesOfN(row, column);
        setSatisfiedConstraints(row, column);
    };

    auto basicTechniquesVector = [this](size_t row, size_t column)->void
    {
        findHighSkyscrapers1(row, column);
        // TODO: add more basic techniques methods to call
    };

    auto basicTechniquesCell = [this](size_t row, size_t column)->void
    {
        findHighSkyscrapers2(row, column);
    };

    auto setConstraints = [this](size_t row, size_t column)->void
    {
        setSatisfiedConstraints(row, column);
    };

    //board.forEachCell(startingTechniquesCell);
    //board.forEachVector(basicTechniquesVector);
    //board.forEachCell(setConstraints);
    //board.forEachCell(basicTechniquesCell);

#ifdef BT_WITH_STACK
    auto retVal = backTrackingWithStack();
#else
    auto freeCell = rowAndColumnPairT(0, 0);
    if (board.getCell(0, 0) != 0)
    {
        freeCell = getNextFreeCell(0, 0);
    }

    std::vector<board::Board> retVal;
    backTracking(retVal, 0, freeCell.first, freeCell.second);
#endif // BT_WITH_STACK

    if (retVal.empty())
    {
        const auto fileName = "no_solution.txt";
        DEBUG_PRINTLN_VERBOSE_WARNING("Couldn't find any solutions. Saving...");
        board.saveToFile(fileName);
        DEBUG_PRINTLN_VERBOSE_WARNING("Saved as \"%s\"", fileName);
    }

    return retVal;
}

void solver::SequentialSolver::print() const
{
    constraints.print();
    board.print();
}

bool solver::SequentialSolver::checkIfLatinSquare() const
{
    return board.checkIfLatinSquare();
}

bool solver::SequentialSolver::checkValidityWithHints() const
{
    return board.checkValidityWithHints();
}

void solver::SequentialSolver::setContinueBackTrackingPointer(continueBoolPtrT ptr)
{
    continueBackTracking = ptr;
}

void solver::SequentialSolver::setContinueBackTracking(bool value)
{
    if (continueBackTracking != nullptr)
    {
        *continueBackTracking = value;
    }
}

bool solver::SequentialSolver::getContinueBackTracking() const
{
    if (continueBackTracking != nullptr)
    {
        return continueBackTracking;
    }

    return true;
}

bool solver::SequentialSolver::setConstraint(size_t row, size_t column, board::boardFieldT value, bool conditionally)
{
    if (conditionally)
        return setConstraintConditionally(row, column, value);
    return setConstraintUnconditionally(row, column, value);
}

bool solver::SequentialSolver::setConstraintConditionally(size_t row, size_t column, board::boardFieldT value)
{
    auto retVal = board.isBuildingPlaceable(row, column, value);
    if (retVal)
    {
        constraints[row][column].insert(value);
    }

    return retVal;
}

bool solver::SequentialSolver::setConstraintUnconditionally(size_t row, size_t column, board::boardFieldT value)
{
    constraints[row][column].insert(value);
    return true;
}

void solver::SequentialSolver::findCluesOfOne(size_t row, size_t column)
{
    auto rowEdge = board.whichEdgeRow(row);
    auto columnEdge = board.whichEdgeColumn(column);

    // There is only one building seen from row or column
    if (rowEdge != matrix::NONE && board.hints[rowEdge][column] == 1 ||
        columnEdge != matrix::NONE && board.hints[columnEdge][row] == 1)
    {
        constraints[row][column].insert(static_cast<board::boardFieldT>(board.size()));
    }
}

void solver::SequentialSolver::findCluesOfN(size_t row, size_t column)
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

void solver::SequentialSolver::findHighSkyscrapers1(size_t row, size_t column)
{
    const auto highest = static_cast<board::boardFieldT>(board.size());
    const auto secondHighest = static_cast<board::boardFieldT>(highest - 1);

    const auto leftHint = board.hints[matrix::LEFT][row];
    const auto rightHint = board.hints[matrix::RIGHT][row];
    const auto topHint = board.hints[matrix::TOP][column];
    const auto bottomHint = board.hints[matrix::BOTTOM][column];

    const auto firstIdx = 0;
    const auto secondIdx = board.size() > 1 ? firstIdx + 1 : firstIdx;
    const auto lastIdx = board.size() - 1;
    const auto secondLastIdx = lastIdx > 0 ? lastIdx - 1 : lastIdx; // Just to be safe and not get MAX_UINT

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

void solver::SequentialSolver::findHighSkyscrapers2(size_t row, size_t column)
{
    const auto secondHighest = static_cast<board::boardFieldT>(board.size() - 1);
    if (board.isBuildingPlaceable(row, column, secondHighest))
    {
        const auto leftHint = board.hints[matrix::LEFT][row];
        const auto rightHint = board.hints[matrix::RIGHT][row];
        const auto topHint = board.hints[matrix::TOP][column];
        const auto bottomHint = board.hints[matrix::BOTTOM][column];

        const auto leftCount = board.getVisibleBuildingsIf(matrix::LEFT, row, secondHighest, column);
        const auto rightCount = board.getVisibleBuildingsIf(matrix::RIGHT, row, secondHighest, column);
        const auto topCount = board.getVisibleBuildingsIf(matrix::TOP, column, secondHighest, row);
        const auto bottomCount = board.getVisibleBuildingsIf(matrix::BOTTOM, column, secondHighest, row);

        if (leftCount == leftHint &&
            rightCount == rightHint &&
            topCount == topHint &&
            bottomCount == bottomHint)
        {
            setConstraint(row, column, secondHighest, false);
        }
    }
}

void solver::SequentialSolver::findPhase2Constraints(size_t row, size_t column)
{
    auto highestInRow = board.locateHighestInRow(row);
    auto highestInColumn = board.locateHighestInColumn(column);
    auto distanceInRow = column > highestInRow ? column - highestInRow : highestInRow - column;
    auto distanceInColumn = row > highestInColumn ? row - highestInColumn : highestInColumn - row;

    UNREFERENCED_PARAMETER(distanceInRow);
    UNREFERENCED_PARAMETER(distanceInColumn);

    if (highestInRow < board.size())
    {

    }
}

void solver::SequentialSolver::setSatisfiedConstraints(size_t row, size_t column)
{
    // There is only constraint
    if (constraints[row][column].size() == 1)
    {
        board.setCell(row, column, *constraints[row][column].begin());
    }
}

void solver::SequentialSolver::backTracking(std::vector<board::Board> & retVal, size_t level, size_t row, size_t column)
{
    const auto treeRowSize = board.size();

    if (!getContinueBackTracking())
    {
        return;
    }

    // Check if it is last cell
    const auto cellPair = getNextFreeCell(row, column);
    if (cellPair == lastCellPair)
    {
        DEBUG_PRINTLN_VERBOSE_INFO("Last cell");
        for (size_t i = 0; i < treeRowSize; i++)
        {
            const auto consideredBuilding = static_cast<board::boardFieldT>(i + 1);

            if (board.isBuildingPlaceable(row, column, consideredBuilding))
            {
                board.setCell(row, column, consideredBuilding);
                if (getContinueBackTracking() && board.isBoardPartiallyValid(row, column))
                {
                    DEBUG_PRINTLN_VERBOSE_INFO("Found result");
                    DEBUG_CALL(board.print());
                    setContinueBackTracking(false);
                    retVal.emplace_back(board);
                }
                board.clearCell(row, column);
            }
        }
    }
    else
    {
        for (size_t i = 0; i < treeRowSize; i++)
        {
            const auto consideredBuilding = static_cast<board::boardFieldT>(i + 1);
            if (board.isBuildingPlaceable(row, column, consideredBuilding))
            {
                board.setCell(row, column, consideredBuilding);
                if (getContinueBackTracking() && board.isBoardPartiallyValid(row, column))
                {
                    backTracking(retVal, level + 1, cellPair.first, cellPair.second);
                }

                board.clearCell(row, column);
            }
        }
    }
}

std::vector<board::Board> solver::SequentialSolver::backTrackingWithStack()
{
    std::vector<board::Board> retVal;
    stackT stack;
    stack.reserve(board.size());

    auto initialCellPair = rowAndColumnPairT(0, 0);
    if (board.getCell(0, 0) != 0)
    {
        initialCellPair = getNextFreeCell(0, 0);
    }

    const auto stackEntrySize = board.size();
    stack.emplace_back(stackEntrySize, initialCellPair);

    do
    {
        auto & entry = stack.back().first;
        auto & stackCell = stack.back().second;

        const auto & row = stackCell.first;
        const auto & column = stackCell.second;

        auto idx = entry.firstZero();
        if (idx != StackEntry::badIndex)
        {
            entry.setBit(idx);

            const auto consideredBuilding = static_cast<board::boardFieldT>(idx + 1);
            if (board.isBuildingPlaceable(row, column, consideredBuilding))
            {
                board.setCell(row, column, consideredBuilding);
                if (board.isBoardPartiallyValid(row, column))
                {
                    const auto nextCellPair = getNextFreeCell(row, column);
                    if (nextCellPair == lastCellPair)
                    {
                        DEBUG_PRINTLN_VERBOSE_INFO("Found result");
                        DEBUG_CALL(board.print());
                        retVal.emplace_back(board);
                        board.clearCell(row, column);
                    }
                    else
                    {
                        stack.emplace_back(stackEntrySize, nextCellPair);
                    }
                }
                else
                {
                    board.clearCell(row, column);
                }
            }
        }
        else
        {
            board.clearCell(row, column);
            stack.pop_back();
            if (!stack.empty())
            {
                const auto & newStackCell = stack.back().second;
                board.clearCell(newStackCell.first, newStackCell.second);
            }
        }

    } while (!stack.empty());

    return retVal;
}

rowAndColumnPairT solver::SequentialSolver::getNextFreeCell(size_t row, size_t column) const
{
    const auto maxSize = board.size();

    // Search till free cell is found
    do
    {
        // Next column
        if (column < maxSize - 1)
        {
            column++;
        }
        // Next row
        else if (column >= maxSize - 1)
        {
            column = 0;
            row++;
        }
    } while (row < maxSize && board.getCell(row, column) != 0);

    // If row is too big return max values
    if (row >= maxSize)
    {
        DEBUG_PRINTLN_VERBOSE_INFO("Returning max values for pair");
        const auto maxVal = std::numeric_limits<size_t>::max();
        row = maxVal;
        column = maxVal;
    }

    return rowAndColumnPairT(row, column);
}
