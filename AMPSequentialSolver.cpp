#include "AMPSequentialSolver.h"


namespace AMP
{
    AMPSequentialSolver::AMPSequentialSolver(const AMPBoard & board) __CPU_ONLY: board(board)
    {
        // Nothing to do
    }

    AMPSequentialSolver::~AMPSequentialSolver() __CPU_ONLY
    {
        // Nothing to do
    }

    std::vector<AMPBoard> AMPSequentialSolver::solve() __GPU
    {
        auto retVal = backTrackingWithStack(board.size());

        if (retVal.empty())
        {
            DEBUG_PRINTLN_VERBOSE_WARNING("Couldn't find any solutions");
        }

        return retVal;
    }

    std::vector<AMPBoard> AMPSequentialSolver::backTrackingWithStack(const size_t boardSize) __GPU
    {
        std::vector<AMPBoard> retVal();
        stackT stack(boardSize);

        auto initialCellPair = rowAndColumnPairT(0, 0);
        if (board.getCell(0, 0) != 0)
        {
            initialCellPair = getNextFreeCell(0, 0, boardSize);
        }

        const auto stackEntrySize = board.size();
        stack.emplaceBack(stackEntrySize, initialCellPair);

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

                const auto consideredBuilding = idx + 1;
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

    rowAndColumnPairT AMPSequentialSolver::getNextFreeCell(size_t row, size_t column, const size_t boardSize) const __GPU
    {
        // Search till free cell is found
        do
        {
            // Next column
            if (column < boardSize - 1)
            {
                column++;
            }
            // Next row
            else if (column >= boardSize - 1)
            {
                column = 0;
                row++;
            }
        } while (row < boardSize && board.getCell(row, column) != 0);

        rowAndColumnPairT retVal;

        // If row is too big return max values
        if (row >= boardSize)
        {
            DEBUG_PRINTLN_VERBOSE_INFO("Returning max values for pair");
            retVal.first = ~0;
            retVal.second = ~0;
        }

        return retVal;
    }
}
