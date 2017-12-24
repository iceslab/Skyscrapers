#pragma once
#include "ParallelSolver.h"
#include "AMPUtilities.h"

namespace solver
{
    class ParallelGpuSolver : public ParallelSolver
    {
    public:
        ParallelGpuSolver(const board::Board& board);
        ~ParallelGpuSolver() = default;

        std::vector<board::Board> solve();
    protected:
        std::vector<SequentialSolver> prepareSolvers(const size_t count);
    };
}
