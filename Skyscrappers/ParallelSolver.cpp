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

    std::vector<board::Board> ParallelSolver::generateNBoards(const size_t desiredBoards)
    {
        std::vector<board::Board> retVal;
        std::vector<board::Board> currentLevel;
        const auto boardCells = board.getSize() * board.getSize();
        const auto theoreticalStopLevel = static_cast<size_t>(
            std::floor(std::log(desiredBoards) / std::log(board.size())));

        //printf("Theoretical stop level for %zu boards: %zu\n", desiredBoards, theoreticalStopLevel);

        auto currentStopLevel = theoreticalStopLevel;
        do
        {
            retVal = std::move(currentLevel);
            generateBoards(currentStopLevel++, currentLevel);
        } while (currentLevel.size() <= desiredBoards && currentStopLevel <= boardCells);

        //printf("Final stop level with %zu boards: %zu\n", retVal.size(), currentStopLevel);
        
        return retVal;
    }

    void ParallelSolver::generateBoards(size_t stopLevel,
                                        std::vector<board::Board> & retVal,
                                        size_t level,
                                        size_t row,
                                        size_t column)
    {
        ASSERT(stopLevel <= board.size());
        //DEBUG_CALL(std::cout << "level: " << level << " row: " << row << " column: " << column << "\n";);
        //DEBUG_CALL(board.print());
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
                const auto consideredBuilding = static_cast<board::boardFieldT>(i + 1);
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