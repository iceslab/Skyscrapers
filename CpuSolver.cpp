#include "CpuSolver.h"

using namespace solver;

CpuSolver::CpuSolver(const board::Board & board) : Solver(board), constraintMatrix(board.size())
{
}

