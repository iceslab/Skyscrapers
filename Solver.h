#pragma once
#include "Board.h"
namespace solver
{
    class Solver
    {
    public:
        Solver(const board::Board& board);
        ~Solver() = default;
    private:
        board::Board board;
    };
}
