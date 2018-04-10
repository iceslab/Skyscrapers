#include "ParallelCpuSolver.h"

namespace solver
{
    ParallelCpuSolver::ParallelCpuSolver(const board::Board & board) : ParallelSolver(board)
    {
        // Nothing to do
    }

    std::vector<board::Board> ParallelCpuSolver::solve(const size_t stopLevel,
                                                       double & initMilliseconds,
                                                       double & generationMilliseconds,
                                                       double & threadsLaunchMilliseconds,
                                                       double & threadsSyncMilliseconds)
    {
        Timer time;
        Timer timeInit;
        Timer timeGeneration;

        timeInit.start();
        std::vector<board::Board> retVal;
        continueBoolT continueBT = true;
        UNREFERENCED_PARAMETER(continueBT);
        timeGeneration.start();
        auto solvers = prepareSolvers(stopLevel);
        generationMilliseconds = timeGeneration.stop(Resolution::MILLISECONDS);
        printf("Generated %zu solvers\n", solvers.size());
        fflush(stdout);
        fflush(stderr);
        retVal.reserve(solvers.size());
        std::vector<std::future<std::vector<board::Board>>> results;
        results.reserve(solvers.size());
        initMilliseconds = timeInit.stop(Resolution::MILLISECONDS);

        time.start();
        for (auto& solver : solvers)
        {
            //solver.setContinueBackTrackingPointer(&continueBT);
            results.emplace_back(std::async(std::launch::async, &SequentialSolver::solve, solver));
        }
        threadsLaunchMilliseconds = time.stop(Resolution::MILLISECONDS);

        time.start();
        for (auto& result : results)
        {
            const auto resultVector = result.get();
            retVal.insert(retVal.end(), resultVector.begin(), resultVector.end());
        }
        threadsSyncMilliseconds = time.stop(Resolution::MILLISECONDS);

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
