#include "Constraints.h"
using namespace constraints;

Constraints::Constraints(const board::boardFieldT boardSize) :
    matrix::SquareMatrix<constraintsFieldT>(boardSize)
{
    // Nothing to do
}

Constraints::Constraints(const board::Board & board) :
    Constraints(board.size())
{
    // Nothing to do
}

void Constraints::print() const
{
    std::ostream_iterator<board::boardFieldT> field_it(std::cout, ", ");
    auto& emptyField = " , ";

    // Whole board
    for (auto& row : *this)
    {
        for (auto& field : row)
        {
            // Contents of set
            std::cout << "[";
            std::copy(field.begin(), field.end(), field_it);

            // Print empty fields
            for (auto i = field.size(); i < size() - 1; i++)
            {
                std::cout << emptyField;
            }

            std::cout << "] ";
        }
        std::cout << std::endl;
    }
}
