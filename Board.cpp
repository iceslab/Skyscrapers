#include "Board.h"

using namespace board;

Board::Board(const boardFieldT boardSize) : board(boardSize, rowT(boardSize))
{
    // Resize hints
    for (auto& h : hints)
    {
        h.resize(boardSize);
    }
}

void Board::generate()
{
    generate(board.size());
}

void Board::generate(const boardFieldT boardSize)
{
    resize(boardSize);
    fillWithZeros();

    // Random device init
    std::random_device r;
    std::default_random_engine r_engine(r());

    // Fill first row with values from 1 to board size
    auto& firstRowR = board.front();
    for (size_t i = 0; i < firstRowR.size(); i++)
    {
        firstRowR[i] = i + 1;
    }

    // Randomize first row
    std::shuffle(firstRowR.begin(), firstRowR.end(), r_engine);

    // Prepare column sets for finding available values
    std::vector<columnSetT> columnSets(getSize());

    for (size_t i = 0; i < getSize(); i++)
    {
        columnSets[i] = columnSetT(firstRowR.begin(), firstRowR.end());
        columnSets[i].erase(firstRowR[i]);
    }

    // For each row...
    for (size_t rowIdx = 1; rowIdx < getSize(); rowIdx++)
    {
        // This is full, because on every iteration of loop, new row is chosen
        rowSetT rowSet(firstRowR.begin(), firstRowR.end());

        // ... and each column...
        for (size_t columnIdx = 0; columnIdx < getSize();)
        {
            if (board[rowIdx][columnIdx] != 0)
            {
                columnIdx++;
                continue;
            }

            auto& columnSetR = columnSets[columnIdx];
            std::vector<setIntersectionT> setIntersections(getSize() - columnIdx);

            // ... find which values are available for their intersection
            for (size_t i = columnIdx; i < getSize(); i++)
            {
                std::set_intersection(rowSet.begin(),
                                      rowSet.end(),
                                      columnSets[i].begin(),
                                      columnSets[i].end(),
                                      std::back_inserter(setIntersections[i - columnIdx]));
            }

            // Debug prints, braces only for wrapping
            {
                DEBUG_PRINTLN_VERBOSE("rowIdx: %lu, colIdx: %lu", rowIdx, columnIdx);
                DEBUG_CALL(print());
                DEBUG_PRINTLN("");
                DEBUG_PRINT("rowSet: ");
                for (auto& r : rowSet)
                    DEBUG_PRINT("%lu ", r);
                DEBUG_PRINTLN("");

                for (size_t i = 0; i < columnSets.size(); i++)
                {
                    DEBUG_PRINT("columnSets[%lu]: ", i);
                    for (auto& c : columnSets[i])
                        DEBUG_PRINT("%lu ", c);
                    DEBUG_PRINTLN("");
                }

                for (size_t i = 0; i < setIntersections.size(); i++)
                {
                    DEBUG_PRINT("setIntersections[%lu]: ", i);
                    for (auto& si : setIntersections[i])
                        DEBUG_PRINT("%lu ", si);
                    DEBUG_PRINTLN("");
                }
            }

            auto setIntersectionR = setIntersections.front();

            ASSERT_VERBOSE(setIntersectionR.empty() == false, "setIntersectionR.size(): %lu", setIntersectionR.size());
            // Randomly choose one of the values
            std::shuffle(setIntersectionR.begin(), setIntersectionR.end(), r_engine);

            boardFieldT value = setIntersectionR.front();
            for (auto& si : setIntersections)
            {
                auto& it = std::find(si.begin(), si.end(), value);
                if (it != si.end())
                {
                    si.erase(it);
                }
            }

            // Debug prints, braces only for wrapping
            {
                DEBUG_PRINTLN("After erasure");
                for (size_t i = 0; i < setIntersections.size(); i++)
                {
                    DEBUG_PRINT("setIntersections[%lu]: ", i);
                    for (auto& si : setIntersections[i])
                        DEBUG_PRINT("%lu ", si);
                    DEBUG_PRINTLN("");
                }
            }

            size_t modifiedColumnIdx = columnIdx;
            auto& itBegin = setIntersections.rbegin();
            auto& itEnd = setIntersections.rend();
            auto it = std::find_if(itBegin, itEnd, [](auto& s) { return s.size() == 1; });

            for (auto it = itBegin; it != itEnd; it++)
            {
                auto index = getSize() - std::abs(it - itBegin) - 1;
                DEBUG_PRINTLN("index: %lu", index);
                if (it->size() == 1 && board[rowIdx][index] == 0)
                {
                    modifiedColumnIdx = index;
                    value = it->front();
                    break;
                }
            }

            // Update row and column sets and board itself
            rowSet.erase(value);
            columnSets[modifiedColumnIdx].erase(value);
            board[rowIdx][modifiedColumnIdx] = value;

            DEBUG_PRINTLN("modifiedColumnIdx: %lu\nvalue: %lu\n", modifiedColumnIdx, value);
            DEBUG_CALL(print());
            DEBUG_PRINTLN("");

            if (modifiedColumnIdx == columnIdx)
            {
                columnIdx++;
            }
        }
    }

    // Fill hints for TOP, RIGHT, BOTTOM and LEFT
    for (size_t i = 0; i < getSize(); i++)
    {
        hints[TOP][i] = getVisibleBuildings(TOP, i);
        hints[RIGHT][i] = getVisibleBuildings(RIGHT, i);
        hints[BOTTOM][i] = getVisibleBuildings(BOTTOM, i);
        hints[LEFT][i] = getVisibleBuildings(LEFT, i);
    }
}

bool Board::operator==(const Board & other) const
{
    if (board != other.board)
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

size_t Board::getSize() const
{
    return board.size();
}

const rowT & Board::getRow(size_t index) const
{
    return board[index];
}

rowT & Board::getRow(size_t index)
{
    return board[index];
}

columnT Board::getColumn(size_t index)
{
    columnT column;
    column.reserve(getSize());

    for (auto& row : board)
    {
        column.push_back(row[index]);
    }

    return column;
}

bool Board::checkValidity() const
{
    return false;
}

bool Board::checkValidityWithHints() const
{
    return false;
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
    for (size_t rowIdx = 0; rowIdx < getSize(); rowIdx++)
    {
        // Left hint field
        std::copy(hints[LEFT].begin() + rowIdx, hints[LEFT].begin() + rowIdx + 1, field_it);

        // Board fields
        std::copy(board[rowIdx].begin(), board[rowIdx].end(), field_it);

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
    if (boardSize == board.size())
        return;

    // Resize rows count
    board.resize(boardSize);
    for (auto& row : board)
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
    for (auto& row : board)
    {
        std::fill(row.begin(), row.end(), boardFieldT());
    }
}

boardFieldT Board::getVisibleBuildings(HintsSide side, size_t rowOrColumn)
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


