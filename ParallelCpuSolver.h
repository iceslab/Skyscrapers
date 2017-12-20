#pragma once
#include "SequentialSolver.h"
#include <thread>
#include <future>

namespace solver
{
    class ParallelCpuSolver : public SequentialSolver
    {
    public:
        ParallelCpuSolver(const board::Board& board);
        ~ParallelCpuSolver();

        std::vector<board::Board> solve();
    protected:
        std::vector<SequentialSolver> prepareSolvers(const size_t count);
        std::vector<board::Board> generateBoards(const size_t stopLevel);
        void generateBoards(size_t stopLevel,
                            std::vector<board::Board> & retVal,
                            size_t level = 0,
                            size_t row = 0,
                            size_t column = 0);
    };
}
