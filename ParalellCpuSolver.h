#pragma once
#include "CpuSolver.h"

namespace solver
{
    class ParalellCpuSolver : public CpuSolver
    {
    public:
        ParalellCpuSolver(const board::Board& board);
        ~ParalellCpuSolver();

        void solve();
    protected:
        std::vector<CpuSolver> prepareSolvers(const size_t count);
    };
}
