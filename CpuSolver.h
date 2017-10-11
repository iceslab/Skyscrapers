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

        /// Setters
        bool setConstraint(size_t row, size_t column, board::boardFieldT value, bool conditionally = true);
        bool setConstraintConditionally(size_t row, size_t column, board::boardFieldT value);
        bool setConstraintUnconditionally(size_t row, size_t column, board::boardFieldT value);

        /// Finding techniques
        /// Starting techniques

        // One visible building
        void findCluesOfOne(size_t row, size_t column);
        // All buildings are visible
        void findCluesOfN(size_t row, size_t column);

        /// Basic techniques

        // Looks for cell to place highest skyscraper (1st technique)
        void findHighSkyscrapers1(size_t row, size_t column);
        // Looks for cell to place highest skyscraper (2nd technique)
        void findHighSkyscrapers2(size_t row, size_t column);
        // Looks for unique sequences of skyscrapers
        void findUniqueSequences(size_t row, size_t column);
        // Looks ...
        void findUsingNoRepeatRule(size_t row, size_t column);

        /// Advanced techniques

        // Looks for only one visible building after initial search
        void findPhase2Constraints(size_t row, size_t column);

        /// Setters

        // Sets fields if finds that there is only constraint
        void setSatisfiedConstraints(size_t row, size_t column);
    };
}