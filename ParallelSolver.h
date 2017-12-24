#pragma once
#include "SequentialSolver.h"

namespace solver
{
    class ParallelSolver : public SequentialSolver
    {
    public:
        ParallelSolver(const board::Board& board);
        virtual ~ParallelSolver() = default;

    protected:
        std::vector<board::Board> generateBoards(const size_t stopLevel);
        void generateBoards(size_t stopLevel,
                            std::vector<board::Board> & retVal,
                            size_t level = 0,
                            size_t row = 0,
                            size_t column = 0);
    };
}
