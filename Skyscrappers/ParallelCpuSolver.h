#pragma once
#include "ParallelSolver.h"
#include "../Utilities/Timer.h"
#include <thread>
#include <future>

namespace solver
{
    class ParallelCpuSolver : public ParallelSolver
    {
    public:
        ParallelCpuSolver(const board::Board& board);
        ~ParallelCpuSolver() = default;

        std::vector<board::Board> solve(const size_t stopLevel,
                                        double & initMilliseconds,
                                        double & generationMilliseconds,
                                        double & threadsLaunchMilliseconds,
                                        double & threadsSyncMilliseconds);
    protected:
        std::vector<SequentialSolver> prepareSolvers(const size_t count);
    };
}
