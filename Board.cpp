#include "Board.h"

using namespace board;

const std::array<matrix::SideE, 4> Board::validSides =
{
    matrix::TOP,
    matrix::RIGHT,
    matrix::BOTTOM,
    matrix::LEFT
};

Board::Board(const boardFieldT boardSize) :
    matrix::SquareMatrix<boardFieldT>(boardSize)
{
    // Resize hints
    for (auto& h : hints)
    {
        h.resize(boardSize);
    }
}

void Board::generate()
{
    generate(size());
}

void Board::generate(const boardFieldT boardSize)
{
    resize(boardSize);

    // Generate uniformly distributed latin square
    EfficientIncidenceCube eic(boardSize);
    eic.shuffle();

    // Copy contents
    for (size_t x = 0; x < boardSize; x++)
    {
        for (size_t y = 0; y < boardSize; y++)
        {
            // Latin square is indexed from 0 to boardSize, it is needed to add 1
            (*this)[x][y] = eic.plusOneZCoordOf(x, y) + 1;
        }
    }

    // Fill hints for TOP, RIGHT, BOTTOM and LEFT
    for (size_t i = 0; i < size(); i++)
    {
        for (auto& side : validSides)
        {
            hints[side][i] = getVisibleBuildings(side, i);
        }
    }
}

bool Board::operator==(const Board & other) const
{

    if (!std::equal(this->begin(), this->end(), other.begin()))
    {
        return false;
    }

    for (size_t i = 0; i < hintSize; i++)
    {
        if (hints[i] != other.hints[i])
            return false;
    }

    return true;
}

bool Board::operator!=(const Board & other) const
{
    return !(*this == other);
}

bool Board::checkIfLatinSquare() const
{
    for (size_t i = 0; i < size(); i++)
    {
        std::vector<bool> rowChecker(size(), false);
        std::vector<bool> columnChecker(size(), false);
        for (size_t j = 0; j < size(); j++)
        {
            // Check if current fields values were present before
            auto rowField = (*this)[i][j] - 1;
            auto columnField = (*this)[j][i] - 1;
            if (rowChecker[rowField] || columnChecker[columnField])
            {
                // If yes, board is not latin square
                return false;
            }
            else
            {
                // Else, mark them as present and continue
                rowChecker[rowField] = true;
                columnChecker[columnField] = true;
            }
        }
    }

    return true;
}

bool Board::checkValidityWithHints() const
{
    if (!checkIfLatinSquare())
    {
        return false;
    }

    for (size_t i = 0; i < size(); i++)
    {
        for (auto& enumVal : validSides)
        {
            if (hints[enumVal][i] != getVisibleBuildings(enumVal, i))
            {
                return false;
            }
        }
    }

    return true;
}

void Board::print() const
{
    std::ostream_iterator<boardFieldT> field_it(std::cout, " ");
    std::string space = " ";

    // Free field to align columns
    std::cout << "  ";
    // Top hints
    std::copy(hints[matrix::TOP].begin(), hints[matrix::TOP].end(), field_it);
    std::cout << std::endl;

    // Whole board
    for (size_t rowIdx = 0; rowIdx < size(); rowIdx++)
    {
        // Left hint field
        std::copy(hints[matrix::LEFT].begin() + rowIdx, hints[matrix::LEFT].begin() + rowIdx + 1, field_it);

        // Board fields
        std::copy((*this)[rowIdx].begin(), (*this)[rowIdx].end(), field_it);

        // Right hint field
        std::copy(hints[matrix::RIGHT].begin() + rowIdx, hints[matrix::RIGHT].begin() + rowIdx + 1, field_it);
        std::cout << std::endl;
    }

    // Free field to align columns
    std::cout << "  ";
    // Bottom hints
    std::copy(hints[matrix::BOTTOM].begin(), hints[matrix::BOTTOM].end(), field_it);
    std::cout << std::endl;
}

void Board::resize(const boardFieldT boardSize)
{
    if (boardSize == size())
        return;

    // Resize rows count
    (*this).resize(boardSize);
    for (auto& row : (*this))
    {
        // Resize rows
        row.resize(boardSize);
    }

    for (auto& h : hints)
    {
        // Resize hints
        h.resize(boardSize);
    }
}

boardFieldT Board::getVisibleBuildings(matrix::SideE side, size_t rowOrColumn) const
{
    ASSERT_VERBOSE(rowOrColumn < size(),
                   "%u < %u",
                   rowOrColumn, size());

    boardFieldT retVal = 0;
    auto& row = getRow(rowOrColumn);
    auto column = getColumn(rowOrColumn);
    switch (side)
    {
        case matrix::TOP:
            retVal = countVisibility(column.begin(), column.end());
            break;
        case matrix::RIGHT:
            retVal = countVisibility(row.rbegin(), row.rend());
            break;
        case matrix::BOTTOM:
            retVal = countVisibility(column.rbegin(), column.rend());
            break;
        case matrix::LEFT:
            retVal = countVisibility(row.begin(), row.end());
            break;
        default:
            // Nothing to do
            break;
    }

    return retVal;
}

boardFieldT board::Board::locateHighestInRow(size_t rowIdx) const
{
    ASSERT_VERBOSE(rowIdx < size(),
                   "%u < %u",
                   rowIdx, size());

    auto& row = getRow(rowIdx);
    return std::find(row.begin(), row.end(), size()) - row.begin();
}

boardFieldT board::Board::locateHighestInColumn(size_t columnIdx) const
{
    ASSERT_VERBOSE(columnIdx < size(),
                   "%u < %u",
                   columnIdx, size());

    auto column = getColumn(columnIdx);
    return std::find(column.begin(), column.end(), size()) - column.begin();
}



