#pragma once
#include "Solver.h"
#include "Constraints.h"

namespace solver
{
    class CpuSolver :
        public Solver
    {
    public:
        CpuSolver(const board::Board& board);
        void solve();
        ~CpuSolver() = default;

        // Output
        
        void print() const;


    private:

        // Collects data about available solutions for given field
        constraints::Constraints constraints;

        // Looks for constraints in given cell
        bool findEdgeConstraints(size_t row, size_t column);
    };
}