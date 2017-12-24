#include "ParallelSolver.h"


namespace solver
{
    ParallelSolver::ParallelSolver(const board::Board & board) : SequentialSolver(board)
    {
        // Nothing to do
    }

    std::vector<board::Board> ParallelSolver::generateBoards(const size_t stopLevel)
    {
        std::vector<board::Board> retVal;
        generateBoards(stopLevel, retVal);
        return retVal;
    }

    void ParallelSolver::generateBoards(size_t stopLevel,
                                        std::vector<board::Board> & retVal,
                                        size_t level,
                                        size_t row,
                                        size_t column)
    {
        ASSERT(stopLevel > 0 && stopLevel <= board.size());
        DEBUG_CALL(std::cout << "level: " << level << " row: " << row << " column: " << column << "\n";);
        DEBUG_CALL(board.print());
        const auto treeRowSize = board.size();

        // Check if it is last cell
        const auto cellPair = getNextFreeCell(row, column);
        if (level == stopLevel || cellPair == lastCellPair)
        {
            retVal.emplace_back(board);
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
                        generateBoards(stopLevel, retVal, level + 1, cellPair.first, cellPair.second);
                    }

                    board.clearCell(row, column);
                }
            }
        }
    }
}