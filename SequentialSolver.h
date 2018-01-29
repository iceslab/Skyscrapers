#pragma once
#include "Solver.h"
#include "Constraints.h"
#include <atomic>
#include "StackEntry.h"

#define BT_WITH_STACK

namespace solver
{
    typedef bool continueBoolT;
    typedef continueBoolT* continueBoolPtrT;

    typedef std::pair<size_t, size_t> rowAndColumnPairT;
    typedef std::pair<StackEntry, rowAndColumnPairT> stackEntryT;
    typedef std::vector<stackEntryT> stackT;

    class SequentialSolver :
        public Solver
    {
    public:
        SequentialSolver(const board::Board& board);
        SequentialSolver(board::Board&& board);
        ~SequentialSolver() = default;

        std::vector<board::Board> solve();
        std::vector<board::Board> ampSolve();

        void print() const;
        // Checks if board is latin square
        bool checkIfLatinSquare() const;
        // Checks validity of board in terms of hints 
        bool checkValidityWithHints() const;

        void setContinueBackTrackingPointer(continueBoolPtrT ptr);
        void setContinueBackTracking(bool value);
        bool getContinueBackTracking() const;
    protected:
        continueBoolPtrT continueBackTracking;

        // Collects data about available solutions for given field
        constraints::Constraints constraints;

        static const rowAndColumnPairT lastCellPair;

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

        /// Backtracking
        void backTracking(std::vector<board::Board> & retVal, size_t level = 0, size_t row = 0, size_t column = 0);
        std::vector<board::Board> backTrackingWithStack();
        rowAndColumnPairT getNextFreeCell(size_t row, size_t column) const;
    };
}