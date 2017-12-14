#include "ParallelCpuSolver.h"

namespace solver
{
    ParallelCpuSolver::ParallelCpuSolver(const board::Board & board) : CpuSolver(board)
    {
        // Nothing to do
    }

    ParallelCpuSolver::~ParallelCpuSolver()
    {
        // Nothing to do
    }

    std::vector<board::Board> ParallelCpuSolver::solve()
    {
        std::vector<board::Board> retVal;
        const auto solvers = prepareSolvers(1);
        std::vector<std::future<std::vector<board::Board>>> results;
        results.reserve(solvers.size());

        for (auto& solver : solvers)
        {
            results.emplace_back(std::async(std::launch::async, &CpuSolver::solve, solver));
        }

        for (auto& result : results)
        {
            const auto resultVector = result.get();
            retVal.insert(retVal.end(), resultVector.begin(), resultVector.end());
        }

        return retVal;
    }

    std::vector<CpuSolver> ParallelCpuSolver::prepareSolvers(const size_t count)
    {
        auto boards = generateBoards(count);
        std::vector<CpuSolver> retVal;
        retVal.reserve(boards.size());

        for (const auto & el : boards)
        {
            retVal.emplace_back(std::move(el));
        }
            
        return retVal;
    }

    std::vector<board::Board> ParallelCpuSolver::generateBoards(const size_t stopLevel)
    {
        std::vector<board::Board> retVal;
        generateBoards(stopLevel, retVal);
        return retVal;
    }

    void ParallelCpuSolver::generateBoards(size_t stopLevel,
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
