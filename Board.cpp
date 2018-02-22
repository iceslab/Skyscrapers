#include "Board.h"

using namespace board;

const std::array<matrix::SideE, 4> Board::validSides =
{
    matrix::TOP,
    matrix::RIGHT,
    matrix::BOTTOM,
    matrix::LEFT
};

Board::Board(const size_t boardSize) :
    matrix::SquareMatrix<boardFieldT>(boardSize),
    setRows(boardSize, std::vector<bool>(boardSize, false)),
    setColumns(boardSize, std::vector<bool>(boardSize, false))
{
    // Resize hints
    for (auto& h : hints)
    {
        h.resize(boardSize);
    }
}

board::Board::Board(const std::string & path)
{
    readFromFile(path);
}

board::Board::Board(std::ifstream & stream)
{
    readFromFile(stream);
}

bool board::Board::saveToFile(const std::string & path) const
{
    std::ofstream ofs(path);
    return saveToFile(ofs);
}

bool board::Board::saveToFile(std::ofstream & stream) const
{
    return SquareMatrix<boardFieldT>::saveToFile(stream);
}

bool board::Board::readFromFile(const std::string & path)
{
    std::ifstream ifs(path);
    return readFromFile(ifs);
}

bool board::Board::readFromFile(std::ifstream & stream)
{
    auto retVal = SquareMatrix<boardFieldT>::readFromFile(stream);
    if (retVal == true)
    {
        // Resize hints
        for (auto& h : hints)
        {
            h.resize(size());
        }

        setRows = memoizedSetValuesT(size(), std::vector<bool>(size(), false));
        setColumns = memoizedSetValuesT(size(), std::vector<bool>(size(), false));
    }
    return retVal;
}

void Board::generate()
{
    generate(size());
}

void Board::generate(const size_t boardSize)
{
    resize(boardSize);

    // Generate uniformly distributed latin square
    EfficientIncidenceCube eic(static_cast<int>(boardSize));
    eic.shuffle();

    // Copy contents
    for (size_t x = 0; x < boardSize; x++)
    {
        for (size_t y = 0; y < boardSize; y++)
        {
            // Latin square is indexed from 0 to boardSize, it is needed to add 1
            (*this)[x][y] = static_cast<boardFieldT>(eic.plusOneZCoordOf(x, y) + 1);
        }
    }

    calculateHints();
}

void board::Board::calculateHints()
{
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

            // Board is not filled properly so it's not latin square
            if ((*this)[i][j] == 0 || (*this)[j][i] == 0)
            {
                return false;
            }

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

size_t board::Board::size() const
{
    return SquareMatrix<boardFieldT>::size();
}

void board::Board::fill(const boardFieldT & value)
{
    SquareMatrix<boardFieldT>::fill(value);
}

matrix::SideE board::Board::whichEdgeRow(size_t row) const
{
    return SquareMatrix<boardFieldT>::whichEdgeRow(row);
}

matrix::SideE board::Board::whichEdgeColumn(size_t column) const
{
    return SquareMatrix<boardFieldT>::whichEdgeColumn(column);
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

void Board::resize(const size_t boardSize)
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
                   "%zu < %zu",
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

boardFieldT Board::getVisibleBuildingsIf(matrix::SideE side, size_t rowOrColumn, boardFieldT value, size_t index) const
{
    ASSERT_VERBOSE(rowOrColumn < size(),
                   "%zu < %zu",
                   rowOrColumn, size());

    boardFieldT retVal = 0;
    auto row = getRow(rowOrColumn);
    auto column = getColumn(rowOrColumn);
    row[index] = value;
    column[index] = value;
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

bool board::Board::isBuildingPlaceable(size_t row, size_t column, boardFieldT building)
{
    ASSERT(row < size());
    ASSERT(column < size());
    ASSERT(building <= size() && building > 0);

    if ((*this)[row][column] != 0)
        return false;
#ifdef ENABLE_MEMOIZATION
    return setRows[row][building - 1] == 0 && setColumns[column][building - 1] == 0;
#else
    auto rowVec = getRow(row);
    auto columnVec = getColumn(column);
    auto valueElementsInRow = std::count(rowVec.begin(), rowVec.end(), building);
    auto valueElementsInColumn = std::count(columnVec.begin(), columnVec.end(), building);

    ASSERT(valueElementsInRow <= 1 && valueElementsInColumn <= 1);

    return valueElementsInRow == 0 && valueElementsInColumn == 0;
#endif // ENABLE_MEMOIZATION
}

bool board::Board::isBoardPartiallyValid(size_t row, size_t column)
{
    ASSERT(row < size());
    ASSERT(column < size());

    const auto rowEdge = whichEdgeRow(row);
    const auto columnEdge = whichEdgeColumn(column);

    const auto leftVisible = getVisibleBuildings(matrix::LEFT, row);
    const auto& leftHints = hints[matrix::LEFT][row];
    const auto topVisible = getVisibleBuildings(matrix::TOP, column);
    const auto& topHints = hints[matrix::TOP][column];

    auto retVal = (leftVisible <= leftHints) && (topVisible <= topHints);

    if (columnEdge == matrix::RIGHT)
    {
        const auto rightVisible = getVisibleBuildings(matrix::RIGHT, row);
        const auto& rightHints = hints[matrix::RIGHT][row];
        retVal = retVal && (leftVisible == leftHints) && (rightVisible == rightHints);
    }

    if (rowEdge == matrix::BOTTOM)
    {
        const auto bottomVisible = getVisibleBuildings(matrix::BOTTOM, column);
        const auto& bottomHints = hints[matrix::BOTTOM][column];
        retVal = retVal && (topVisible == topHints) && (bottomVisible == bottomHints);
    }

    return retVal;
}

boardFieldT board::Board::locateHighestInRow(size_t rowIdx) const
{
    ASSERT_VERBOSE(rowIdx < size(),
                   "%zu < %zu",
                   rowIdx, size());

    auto& row = getRow(rowIdx);
    return static_cast<boardFieldT>(std::find(row.begin(), row.end(), size()) - row.begin());
}

boardFieldT board::Board::locateHighestInColumn(size_t columnIdx) const
{
    ASSERT_VERBOSE(columnIdx < size(),
                   "%zu < %zu",
                   columnIdx, size());

    auto column = getColumn(columnIdx);
    return static_cast<boardFieldT>(std::find(column.begin(), column.end(), size()) - column.begin());
}

void board::Board::setCell(size_t row, size_t column, boardFieldT value)
{
    const auto currentValue = getCell(row, column);
    // Cell is clear
    if (currentValue == 0)
    {
        // Set that this value is set in rows and columns
        SquareMatrix<boardFieldT>::setCell(row, column, value);
    }
    // Cell has other value than already set
    else if (currentValue != value)
    {
        setRows[row][currentValue - 1] = false;
        setColumns[column][currentValue - 1] = false;
        SquareMatrix<boardFieldT>::setCell(row, column, value);
    }

    if (value != 0 && value != currentValue)
    {
        setRows[row][value - 1] = true;
        setColumns[column][value - 1] = true;
    }
}

void board::Board::clearCell(size_t row, size_t column)
{
    setCell(row, column, 0);
}

boardFieldT board::Board::getCell(size_t row, size_t column) const
{
    return SquareMatrix<boardFieldT>::getCell(row, column);
}

