#include "AMPBoard.h"

namespace AMP
{
    AMPBoard::AMPBoard(const board::Board & board) __CPU_ONLY : AMPBoard(board.size())
    {
        for (size_t row = 0; row < size(); row++)
        {
            for (size_t column = 0; column < size(); column++)
            {
                setCell(row, column, board.getCell(row, column));
            }
        }
    }

    AMPBoard::AMPBoard(const AMPBoard & board) __CPU_ONLY : AMPBoard(board.size())
    {
        for (size_t row = 0; row < size(); row++)
        {
            for (size_t column = 0; column < size(); column++)
            {
                setCell(row, column, board.getCell(row, column));
            }
        }
    }

    AMPBoard::AMPBoard(const boardFieldT boardSize) __CPU_ONLY :
    AMPSquareMatrix<boardFieldT>(boardSize),
        hints(boardSize)
    {
        // Nothing to do
    }

    AMPBoard::~AMPBoard() __CPU_ONLY
    {
        // Nothing to do
    }

    boardFieldT AMPBoard::getVisibleBuildings(SideE side, size_t rowOrColumn) const __GPU
    {
        return boardFieldT();
    }

    boardFieldT AMPBoard::getVisibleBuildingsIf(SideE side, size_t rowOrColumn, boardFieldT value, size_t index) const __GPU
    {
        return boardFieldT();
    }

    bool AMPBoard::isBuildingPlaceable(size_t row, size_t column, boardFieldT building) __GPU
    {
        return false;
    }

    bool AMPBoard::isBoardPartiallyValid(size_t row, size_t column) __GPU
    {
        return false;
    }

    void AMPBoard::setCell(size_t row, size_t column, boardFieldT value) __GPU
    {
        AMPSquareMatrix<boardFieldT>::setCell(row, column, value);
    }

    void AMPBoard::clearCell(size_t row, size_t column) __GPU
    {
        setCell(row, column, 0);
    }

    boardFieldT AMPBoard::getCell(size_t row, size_t column) __GPU
    {
        return AMPSquareMatrix<boardFieldT>::getCell(row, column);
    }

    const boardFieldT AMPBoard::getCell(size_t row, size_t column) const __GPU
    {
        return AMPSquareMatrix<boardFieldT>::getCell(row, column);
    }

    size_t AMPBoard::size() const __GPU
    {
        return AMPSquareMatrix<boardFieldT>::size();
    }

    void AMPBoard::fill(const boardFieldT & value) __GPU
    {
        AMPSquareMatrix<boardFieldT>::fill(value);
    }

    SideE AMPBoard::whichEdgeRow(size_t row) const __GPU
    {
        return AMPSquareMatrix<boardFieldT>::whichEdgeRow(row);
    }

    SideE AMPBoard::whichEdgeColumn(size_t column) const __GPU
    {
        return AMPSquareMatrix<boardFieldT>::whichEdgeColumn(column);
    }

    board::Board AMPBoard::toBoard() const __CPU_ONLY
    {
        board::Board retVal(size());

        for (size_t row = 0; row < size(); row++)
        {
            for (size_t column = 0; column < size(); column++)
            {
                retVal.setCell(row, column, getCell(row, column));
            }
        }
        return retVal;
    }
}
