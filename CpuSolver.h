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

        /// Finding techniques

        // One visible building
        void findCluesOfOne(size_t row, size_t column);
        // All buildings are visible
        void findCluesOfN(size_t row, size_t column);

        // Looks for only one visible building after initial search
        void findPhase2Constraints(size_t row, size_t column);

        /// Setters

        // Sets fields if finds that there is only constraint
        void setSatisfiedConstraints(size_t row, size_t column);
    };
}