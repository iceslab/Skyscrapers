#include "Board.h"

using namespace board;

const std::array<HintsSideE, 4> Board::hintsArray = { TOP, RIGHT, BOTTOM, LEFT};

Board::Board(const boardFieldT boardSize) : matrix::SquareMatrix<boardFieldT>(boardSize)
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
        hints[TOP][i] = getVisibleBuildings(TOP, i);
        hints[RIGHT][i] = getVisibleBuildings(RIGHT, i);
        hints[BOTTOM][i] = getVisibleBuildings(BOTTOM, i);
        hints[LEFT][i] = getVisibleBuildings(LEFT, i);
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
        for (auto& enumVal : hintsArray)
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
    std::ostream_iterator<std::string> space_it(std::cout, " ");
    std::ostream_iterator<boardFieldT> field_it(std::cout, " ");
    std::string space = " ";

    // Free field to align columns
    std::cout << "  ";
    // Top hints
    std::copy(hints[TOP].begin(), hints[TOP].end(), field_it);
    std::cout << std::endl;

    // Whole board
    for (size_t rowIdx = 0; rowIdx < size(); rowIdx++)
    {
        // Left hint field
        std::copy(hints[LEFT].begin() + rowIdx, hints[LEFT].begin() + rowIdx + 1, field_it);

        // Board fields
        std::copy((*this)[rowIdx].begin(), (*this)[rowIdx].end(), field_it);

        // Right hint field
        std::copy(hints[RIGHT].begin() + rowIdx, hints[RIGHT].begin() + rowIdx + 1, field_it);
        std::cout << std::endl;
    }

    // Free field to align columns
    std::cout << "  ";
    // Bottom hints
    std::copy(hints[BOTTOM].begin(), hints[BOTTOM].end(), field_it);
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

void Board::fillWithZeros()
{
    for (auto& row : (*this))
    {
        std::fill(row.begin(), row.end(), boardFieldT());
    }
}

boardFieldT Board::getVisibleBuildings(HintsSideE side, size_t rowOrColumn) const
{
    boardFieldT retVal = 0;
    auto& row = getRow(rowOrColumn);
    auto& column = getColumn(rowOrColumn);
    switch (side)
    {
        case TOP:
            retVal = countVisibility(column.begin(), column.end());
            break;
        case RIGHT:
            retVal = countVisibility(row.rbegin(), row.rend());
            break;
        case BOTTOM:
            retVal = countVisibility(column.rbegin(), column.rend());
            break;
        case LEFT:
            retVal = countVisibility(row.begin(), row.end());
            break;
        default:

            break;
    }

    return retVal;
}


