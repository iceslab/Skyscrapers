#pragma once
#include "Solver.h"
namespace solver
{
    typedef matrix::SquareMatrix<std::set<board::boardFieldT>> constraintMatrixT;
    class CpuSolver :
        public Solver
    {
    public:
        CpuSolver(const board::Board& board);
        ~CpuSolver() = default;

    private:
        constraintMatrixT constraintMatrix;
    };
}