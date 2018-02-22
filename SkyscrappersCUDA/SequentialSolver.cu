#include "SequentialSolver.cuh"

namespace cuda
{
    namespace solver
    {
        SequentialSolver::SequentialSolver(const board::Board & board) :
            Solver(board),
            maxVal(std::numeric_limits<size_t>::max()),
            lastCellPair(rowAndColumnPairT(maxVal, maxVal))
        {
            this->board.clear();
        }

        CUDA_DEVICE size_t SequentialSolver::solve(cuda::Board* resultArray, stackT stack)
        {

#ifdef BT_WITH_STACK
            auto retVal = backTrackingWithStack(resultArray, stack);
#else
            auto freeCell = rowAndColumnPairT(0, 0);
            if (board.getCell(0, 0) != 0)
            {
                freeCell = getNextFreeCell(0, 0);
            }

            std::vector<board::Board> retVal;
            backTracking(retVal, 0, freeCell.first, freeCell.second);
#endif // BT_WITH_STACK

            return retVal;
        }

#ifndef BT_WITH_STACK
        void solver::SequentialSolver::backTracking(std::vector<board::Board> & retVal, size_t level, size_t row, size_t column)
        {
            const auto treeRowSize = board.size();

            // Check if it is last cell
            const auto cellPair = getNextFreeCell(row, column);
            if (cellPair == lastCellPair)
            {
                DEBUG_PRINTLN_VERBOSE_INFO("Last cell");
                for (size_t i = 0; i < treeRowSize; i++)
                {
                    const auto consideredBuilding = i + 1;

                    if (board.isBuildingPlaceable(row, column, consideredBuilding))
                    {
                        board.setCell(row, column, consideredBuilding);
                        if (board.isBoardPartiallyValid(row, column))
                        {
                            DEBUG_PRINTLN_VERBOSE_INFO("Found result");
                            DEBUG_CALL(board.print());
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
                    const auto consideredBuilding = i + 1;
                    if (board.isBuildingPlaceable(row, column, consideredBuilding))
                    {
                        board.setCell(row, column, consideredBuilding);
                        if (board.isBoardPartiallyValid(row, column))
                        {
                            backTracking(retVal, level + 1, cellPair.first, cellPair.second);
                        }

                        board.clearCell(row, column);
                    }
                }
            }
        }
#else
        CUDA_DEVICE size_t SequentialSolver::backTrackingWithStack(cuda::Board* resultArray, stackT stack)
        {
            size_t resultsCount = 0;
            size_t stackSize = 0;

            rowAndColumnPairT initialCellPair(0, 0);
            if (board.getCell(0, 0) != 0)
            {
                initialCellPair = getNextFreeCell(0, 0);
            }

            auto stackEntrySize = board.getSize();
            stack[stackSize].first.clearAll();
            stack[stackSize++].second = initialCellPair;
            do
            {
                auto & entry = stack[stackSize - 1].first;
                auto & stackCell = stack[stackSize - 1].second;

                const auto & row = stackCell.first;
                const auto & column = stackCell.second;

                auto idx = entry.firstZero();
                if (idx != entry.badIndex)
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
                                if (resultsCount < maxResultsPerThread)
                                {
                                    memcpy(resultArray + resultsCount++, &board, sizeof(board));
                                }
                                else
                                {
                                    // Found a result, but it doesn't fit inside array
                                }
                                board.clearCell(row, column);
                            }
                            else
                            {
                                stack[stackSize].first.clearAll();
                                stack[stackSize++].second = nextCellPair;
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
                    --stackSize;
                    if (stackSize > 0)
                    {
                        const auto & newStackCell = stack[stackSize - 1].second;
                        board.clearCell(newStackCell.first, newStackCell.second);
                    }
                }

            } while (stackSize > 0);

            return resultsCount;
        }
#endif // !BT_WITH_STACK

        CUDA_DEVICE rowAndColumnPairT SequentialSolver::getNextFreeCell(size_t row, size_t column) const
        {
            const auto maxSize = board.getSize();

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
                row = maxVal;
                column = maxVal;
            }

            return rowAndColumnPairT(row, column);
        }
    }
}
