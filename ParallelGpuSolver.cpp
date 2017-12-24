#include "ParallelGpuSolver.h"

namespace solver
{

    ParallelGpuSolver::ParallelGpuSolver(const board::Board & board) : ParallelSolver(board)
    {
        // Nothing to do
    }

    std::vector<board::Board> ParallelGpuSolver::solve()
    {
        std::vector<board::Board> retVal;
        auto solvers = prepareSolvers(1);
        retVal.reserve(solvers.size());
        std::vector<std::vector<board::Board>> results;
        results.reserve(solvers.size());

        //Concurrency::array_view<SequentialSolver, 1> solversView(solvers.size(), solvers.data());
        //Concurrency::array_view<std::vector<board::Board>, 1> resultsView(results.size(), results.data());


        //Concurrency::parallel_for_each(
        //    solversView.extent,
        //    [=](Concurrency::index<1> idx) restrict(amp)
        //{
        //    solversView[idx].checkIfLatinSquare();
        //});

        // Concatenating vectors
        for (const auto& result : results)
        {
            retVal.insert(retVal.end(), result.begin(), result.end());
        }

        return retVal;
    }

    std::vector<SequentialSolver> ParallelGpuSolver::prepareSolvers(const size_t count)
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
