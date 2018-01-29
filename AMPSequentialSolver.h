#pragma once
#include "asserts.h"
#include <amp.h>
#include "AMPBoard.h"
#include "Stack.h"
#include "Pair.h"

namespace AMP
{
    typedef Pair<size_t, size_t> rowAndColumnPairT;
    typedef Stack::stackEntryT stackEntryT;
    typedef Stack stackT;

    class AMPSequentialSolver
    {
    public:
        AMPSequentialSolver(const AMPBoard& board) __CPU_ONLY;
        ~AMPSequentialSolver() __CPU_ONLY;

        Concurrency::array<AMPBoard, 1> solve() __GPU;

        void print() const;
        // Checks if board is latin square
        bool checkIfLatinSquare() const;
        // Checks validity of board in terms of hints 
        bool checkValidityWithHints() const;
    protected:
        AMPBoard board;

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

        /// Backtracking
        Concurrency::array<AMPBoard, 1> backTrackingWithStack(const size_t boardSize) __GPU;
        rowAndColumnPairT getNextFreeCell(size_t row, size_t column, const size_t boardSize) const __GPU;
    };
}
