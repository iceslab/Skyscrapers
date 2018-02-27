#pragma once
#include "ParallelSolver.h"
#include <thread>
#include <future>

namespace solver
{
    class ParallelCpuSolver : public ParallelSolver
    {
    public:
        ParallelCpuSolver(const board::Board& board);
        ~ParallelCpuSolver() = default;

        std::vector<board::Board> solve(const size_t stopLevel);
    protected:
        std::vector<SequentialSolver> prepareSolvers(const size_t count);
    };
}
