#ifndef __INCLUDED_BOARD_H__
#define __INCLUDED_BOARD_H__

#include "CUDAUtilities.cuh"
#include "../Utilities/asserts.h"
#include "SquareMatrix.cuh"
#include <vector>

#define CUDA_SIZE_T_MAX (size_t(~0))

// Forward declaration
namespace board
{
    class Board;
}

#define ENABLE_MEMOIZATION

namespace cuda
{
    // Typedefs for easier typing
    typedef unsigned int boardFieldT;
    typedef boardFieldT* boardT; // Originally double pointer
    typedef boardFieldT* hintT;
    typedef boardFieldT* rowT;
    typedef const boardFieldT* columnConstT;
    typedef boardFieldT* columnT;
    typedef boardFieldT* setIntersectionT;
    typedef bool memoizedSetValuesTypeT;
    typedef memoizedSetValuesTypeT* memoizedSetValuesT; // Originally double pointer


    class Board : public SquareMatrix<boardFieldT>
    {
    public:
        static constexpr size_t hintsSize = 4;
        hintT hints[hintsSize];

        CUDA_HOST Board(const size_t boardSize);
        CUDA_DEVICE Board(const size_t boardSize,
                          void* constantMemoryPtr,
                          void* sharedMemoryPtr);
        CUDA_HOST Board(const Board & board);
        CUDA_DEVICE Board(const Board & board,
                          void* constantMemoryPtr,
                          void* sharedMemoryPtr);
        CUDA_HOST Board(const board::Board & board);
        CUDA_HOST_DEVICE ~Board();

        /// Generators

        // Calculates hints
        //void calculateHints();

        /// Operators
        CUDA_DEVICE bool operator==(const Board &other) const;
        CUDA_DEVICE bool operator!=(const Board &other) const;

        Board & operator=(const Board & board) = default;
        Board & operator=(Board && board) = default;

        /// Hints manipulators
        // Gets visible buildings from given side and for given row or column
        CUDA_DEVICE boardFieldT getVisibleBuildings(SideE side, size_t rowOrColumn) const;
        // Returns if building can be placed in cell in terms of already placed buildings
        CUDA_DEVICE bool isBuildingPlaceable(size_t row, size_t column, boardFieldT building);
        // Returns if board is valid up tu given row and column in terms of already placed buildings and hints
        CUDA_DEVICE bool isBoardPartiallyValid(size_t row, size_t column);

        /// Accessors

        // Sets cell to specific value only when it doesn't violate Latin square rules
        CUDA_DEVICE void setCell(size_t row, size_t column, boardFieldT value);
        CUDA_DEVICE void clearCell(size_t row, size_t column);

        CUDA_DEVICE boardFieldT getCell(size_t row, size_t column) const;

        CUDA_HOST_DEVICE size_t getSize() const;
        CUDA_HOST_DEVICE size_t getCellsCount() const;
        CUDA_HOST_DEVICE size_t getBoardMemoryUsage() const;
        static CUDA_HOST_DEVICE size_t getBoardMemoryUsage(const size_t boardSize);

        CUDA_DEVICE void fill(const boardFieldT & value);
        CUDA_HOST void clear();

        CUDA_HOST_DEVICE SideE whichEdgeRow(size_t row) const;
        CUDA_HOST_DEVICE SideE whichEdgeColumn(size_t column) const;

        CUDA_DEVICE void print(size_t threadIdx = CUDA_SIZE_T_MAX) const;

        CUDA_HOST std::vector<boardFieldT> getHostVector();

        // Method assumes equal sizes and proper allocations of existing fields
        CUDA_DEVICE void copyInto(cuda::Board & board);
        /// Output
        //void print() const;
    private:
        // Contains which values are set in each row and column
        memoizedSetValuesT setRows;
        memoizedSetValuesT setColumns;

        // Counts visible buildings in specified row
        CUDA_DEVICE size_t countRowVisibility(size_t row) const;
        // Counts visible buildings in specified column
        CUDA_DEVICE size_t countColumnVisibility(size_t column) const;
        // Counts (in reverse) visible buildings in specified row
        CUDA_DEVICE size_t reverseCountRowVisibility(size_t row) const;
        // Counts (in reverse) visible buildings in specified column
        CUDA_DEVICE size_t reverseCountColumnVisibility(size_t column) const;

        //void resize(const boardFieldT boardSize);
    };
}

#endif // !__INCLUDED_BOARD_H__
