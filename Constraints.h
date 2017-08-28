#pragma once
#include <set>
#include "SquareMatrix.h"
#include "Board.h"

namespace constraints
{
    typedef std::set<board::boardFieldT> constraintsFieldT;
    class Constraints : public matrix::SquareMatrix<constraintsFieldT>
    {
    public:
        Constraints(const board::boardFieldT boardSize);
        Constraints(const board::Board& board);
        ~Constraints() = default;

        // Output

        void print() const;
    };
}

