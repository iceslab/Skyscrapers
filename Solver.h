#pragma once
#include "Board.h"
namespace solver
{
    class Solver
    {
    public:
        Solver(const board::Board& board);
        ~Solver() = default;

        virtual void solve() = 0;
    protected:
        board::Board board;
    };
}
