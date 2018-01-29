#pragma once
#include "AMPSquareMatrix.h"
#include "Board.h"
#define AMP_BOARD_HINT_SIZE 4U

namespace AMP
{
    typedef uint32_t boardFieldT;

    typedef Concurrency::array<boardFieldT, 1> hintT;
    typedef Concurrency::array<boardFieldT, AMP_BOARD_HINT_SIZE> hintsArrayT;
    typedef Concurrency::array<boardFieldT, 1> rowT;
    typedef Concurrency::array_view<boardFieldT, 1> columnConstT;
    typedef Concurrency::array_view<boardFieldT, 1> columnT;
    typedef uint32_t memoizedSetValuesT;

    class AMPBoard : protected AMPSquareMatrix<boardFieldT>
    {
    public:
        AMPBoard(const board::Board & board) __CPU_ONLY;
        AMPBoard(const AMPBoard & board) __CPU_ONLY;
        AMPBoard(const boardFieldT boardSize) __CPU_ONLY;
        ~AMPBoard() __CPU_ONLY;

        hintsArrayT hints;

        /// Hints manipulators
        // Gets visible buildings from given side and for given row or column
        boardFieldT getVisibleBuildings(SideE side, size_t rowOrColumn) const __GPU;
        // Gets visible buildings from given side and for given row or column assuming value at index
        boardFieldT getVisibleBuildingsIf(SideE side, size_t rowOrColumn, boardFieldT value, size_t index) const __GPU;
        // Returns if building can be placed in cell in terms of already placed buildings
        bool isBuildingPlaceable(size_t row, size_t column, boardFieldT building) __GPU;
        // Returns if board is valid up tu given row and column in terms of already placed buildings and hints
        bool isBoardPartiallyValid(size_t row, size_t column) __GPU;

        /// Accessors

        // Sets cell to specific value only when it doesn't violate Latin square rules
        void setCell(size_t row, size_t column, boardFieldT value) __GPU;
        void clearCell(size_t row, size_t column) __GPU;

        boardFieldT getCell(size_t row, size_t column) __GPU;
        const boardFieldT getCell(size_t row, size_t column) const __GPU;

        size_t size() const __GPU;
        void fill(const boardFieldT & value) __GPU;

        SideE whichEdgeRow(size_t row) const __GPU;
        SideE whichEdgeColumn(size_t column) const __GPU;

        /// Output
        board::Board toBoard() const __CPU_ONLY;
    private:
        // Contains which values are set in each row and column
        memoizedSetValuesT setRows;
        memoizedSetValuesT setColumns;
    };
}
