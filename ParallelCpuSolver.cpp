#include "ParallelCpuSolver.h"

namespace solver
{
    ParallelCpuSolver::ParallelCpuSolver(const board::Board & board) : ParallelSolver(board)
    {
        // Nothing to do
    }

    std::vector<board::Board> ParallelCpuSolver::solve(const size_t stopLevel)
    {
        std::vector<board::Board> retVal;
        continueBoolT continueBT = true;
        UNREFERENCED_PARAMETER(continueBT);
        auto solvers = prepareSolvers(stopLevel);
        retVal.reserve(solvers.size());
        std::vector<std::future<std::vector<board::Board>>> results;
        results.reserve(solvers.size());

        for (auto& solver : solvers)
        {
            //solver.setContinueBackTrackingPointer(&continueBT);
            results.emplace_back(std::async(std::launch::async, &SequentialSolver::solve, solver));
        }

        for (auto& result : results)
        {
            const auto resultVector = result.get();
            retVal.insert(retVal.end(), resultVector.begin(), resultVector.end());
        }

        return retVal;
    }

    std::vector<SequentialSolver> ParallelCpuSolver::prepareSolvers(const size_t count)
    {
        auto boards = generateBoards(count);
        std::vector<SequentialSolver> retVal;
        retVal.reserve(boards.size());

        for (auto & el : boards)
        {
            retVal.emplace_back(std::move(el));
        }
            
        return retVal;
    }
}
