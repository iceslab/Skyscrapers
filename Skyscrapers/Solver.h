#pragma once
#include "Board.h"
namespace solver
{
    class Solver
    {
    public:
        Solver(const board::Board& board);
        Solver(board::Board&& board);
        ~Solver() = default;

        virtual std::vector<board::Board> solve() = 0;
        static void printResults(const std::vector<board::Board> & results);
    protected:
        board::Board board;
    };
}
