#include "ParalellCpuSolver.h"

namespace solver
{
    ParalellCpuSolver::ParalellCpuSolver(const board::Board & board) : CpuSolver(board)
    {
        // Nothing to do
    }

    ParalellCpuSolver::~ParalellCpuSolver()
    {
        // Nothing to do
    }

    void ParalellCpuSolver::solve()
    {}

    std::vector<CpuSolver> ParalellCpuSolver::prepareSolvers(const size_t count)
    {

        return std::vector<CpuSolver>();
    }
}
