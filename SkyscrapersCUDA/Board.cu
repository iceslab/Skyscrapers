#include "Board.cuh"
#include "../Skyscrapers/Board.h"

namespace cuda
{
    CUDA_HOST Board::Board(const size_t boardSize) :
        SquareMatrix<boardFieldT>(boardSize)
    {
        //printf("cuda::Board host size constructor...\n");
        if (boardSize > 0)
        {
            // Alloc and memset setRows
            cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&setRows),
                                         boardSize * boardSize * sizeof(memoizedSetValuesT));
            if (err != cudaSuccess)
            {
                CUDA_PRINT_ERROR("Failed allocation setRows", err);
                setRows = nullptr;
            }
            else
            {
                err = cudaMemset(reinterpret_cast<void*>(setRows),
                                 0,
                                 boardSize * boardSize * sizeof(memoizedSetValuesT));
                if (err != cudaSuccess)
                {
                    CUDA_PRINT_ERROR("Failed memset setRows", err);
                }
            }

            // Alloc and memset setColumns
            err = cudaMalloc(reinterpret_cast<void**>(&setColumns),
                             boardSize * boardSize * sizeof(memoizedSetValuesT));
            if (err != cudaSuccess)
            {
                CUDA_PRINT_ERROR("Failed allocation setColumns", err);
                setColumns = nullptr;
            }
            else
            {
                err = cudaMemset(reinterpret_cast<void*>(setColumns),
                                 0,
                                 boardSize * boardSize * sizeof(memoizedSetValuesT));
                if (err != cudaSuccess)
                {
                    CUDA_PRINT_ERROR("Failed memset setColumns", err);
                }
            }

            // Alloc and memset hints
            for (size_t side = 0; side < hintsSize; side++)
            {
                err = cudaMalloc(reinterpret_cast<void**>(&hints[side]),
                                 boardSize * sizeof(boardFieldT));
                if (err != cudaSuccess)
                {
                    CUDA_PRINT_ERROR("Failed allocation hints[side]", err);
                    hints[side] = nullptr;
                }
                else
                {
                    err = cudaMemset(reinterpret_cast<void*>(hints[side]),
                                     0,
                                     boardSize * sizeof(boardFieldT));
                    if (err != cudaSuccess)
                    {
                        CUDA_PRINT_ERROR("Failed memset hints[side]", err);
                    }
                }
            }
        }
        //printf("cuda::Board size constructor\n");
    }

    CUDA_HOST Board::Board(const size_t boardSize,
                           void* constantMemoryPtr,
                           void* sharedMemoryPtr) :
        SquareMatrix<boardFieldT>(boardSize, sharedMemoryPtr)
    {
        //printf("cuda::Board device size constructor...\n");
        if (boardSize > 0)
        {
            auto boardCells = getCellsCount();
            auto remainingMemoryPtr = reinterpret_cast<memoizedSetValuesT>(d_data + boardCells);

            // Setting proper pointers
            setRows = remainingMemoryPtr;
            setColumns = remainingMemoryPtr + boardCells;

            // Prepare hints (assuming that constant memory is already initialized)
            hintT shmemHintsPtr = reinterpret_cast<hintT>(constantMemoryPtr);
            for (size_t side = 0; side < hintsSize; side++)
            {
                hints[side] = shmemHintsPtr;
                shmemHintsPtr += boardSize;
            }
        }
        //printf("cuda::Board device size constructor\n");
    }

    CUDA_HOST Board::Board(const Board & board) :
        Board(board.getSize())
    {
        //printf("cuda::Board host copy constructor...\n");
        const auto boardSize = board.getSize();
        const auto boardCells = board.getCellsCount();

        cudaError_t err = cudaMemcpy(d_data,
                                     board.d_data,
                                     sizeof(*d_data) * boardCells,
                                     cudaMemcpyDeviceToDevice);
        if (err != cudaSuccess)
        {
            CUDA_PRINT_ERROR("Failed memcpy board", err);
        }
        else
        {
            // Copy setRows
            err = cudaMemcpy(setRows,
                             board.setRows,
                             sizeof(*setRows) * boardCells,
                             cudaMemcpyDeviceToDevice);

            if (err != cudaSuccess)
            {
                CUDA_PRINT_ERROR("Failed memcpy setRows", err);
            }
            else
            {
                // Copy setColumns
                err = cudaMemcpy(setColumns,
                                 board.setColumns,
                                 sizeof(*setColumns) * boardCells,
                                 cudaMemcpyDeviceToDevice);
                if (err != cudaSuccess)
                {
                    CUDA_PRINT_ERROR("Failed memcpy setColumns", err);
                }
                else
                {
                    // Copy hints
                    for (size_t side = 0; side < hintsSize; side++)
                    {
                        err = cudaMemcpy(hints[side],
                                         board.hints[side],
                                         boardSize * sizeof(boardFieldT),
                                         cudaMemcpyDeviceToDevice);
                        if (err != cudaSuccess)
                        {
                            CUDA_PRINT_ERROR("Failed memcpy hints[side]", err);
                        }
                    }
                }
            }
        }
        //printf("cuda::Board host copy constructor\n");
    }

    CUDA_DEVICE Board::Board(const Board & board,
                             void * constantMemoryPtr,
                             void * sharedMemoryPtr) :
        Board(board.getSize(), constantMemoryPtr, sharedMemoryPtr)
    {
        //printf("cuda::Board device copy constructor...\n");
        const auto boardSize = board.getSize();
        const auto boardCells = board.getCellsCount();

        // Copy data
        memcpy(d_data,
               board.d_data,
               getMatrixMemoryUsage());

        // Copy set rows
        memcpy(setRows,
               board.setRows,
               boardCells * sizeof(*setRows));

        // Copy set columns
        memcpy(setColumns,
               board.setColumns,
               boardCells * sizeof(*setColumns));
        //printf("cuda::Board device copy constructor\n");
    }

    CUDA_HOST Board::Board(const board::Board & board) : Board(board.size())
    {
        const auto boardSize = board.size();
        cudaError_t err = cudaSuccess;
        for (size_t row = 0; row < boardSize; row++)
        {
            const auto h_data = board.getRow(row).data();
            err = cudaMemcpy(d_data + row * boardSize,
                             h_data,
                             sizeof(*d_data) * boardSize,
                             cudaMemcpyHostToDevice);
            if (err != cudaSuccess)
            {
                CUDA_PRINT_ERROR("Failed memcpy row", err);
            }
        }

        // Copy hints
        for (size_t side = 0; side < hintsSize; side++)
        {
            err = cudaMemcpy(hints[side],
                             board.hints[side].data(),
                             boardSize * sizeof(boardFieldT),
                             cudaMemcpyHostToDevice);
            if (err != cudaSuccess)
            {
                CUDA_PRINT_ERROR("Failed memcpy hints[side]", err);
                hints[side] = nullptr;
            }
        }

        // Set rows and columns copy
        auto setRowsCopy = reinterpret_cast<memoizedSetValuesT>(
            std::malloc(boardSize * boardSize * sizeof(memoizedSetValuesT)));
        auto setColumnsCopy = reinterpret_cast<memoizedSetValuesT>(
            std::malloc(boardSize * boardSize * sizeof(memoizedSetValuesT)));
        const auto & boardSetRows = board.getSetRows();
        const auto & boardSetColumns = board.getSetColumns();

        for (size_t i = 0; i < boardSize; i++)
        {
            for (size_t buildingIdx = 0; buildingIdx < boardSize; buildingIdx++)
            {
                auto & copyRowElement = setRowsCopy[i * boardSize + buildingIdx];
                auto & copyColumnElement = setColumnsCopy[i * boardSize + buildingIdx];
                const auto & boardRowElement = boardSetRows[i][buildingIdx];
                const auto & boardColumnElement = boardSetColumns[i][buildingIdx];

                copyRowElement = boardRowElement;
                copyColumnElement = boardColumnElement;
            }
        }

        err = cudaMemcpy(setRows,
                         setRowsCopy,
                         boardSize * boardSize * sizeof(memoizedSetValuesT),
                         cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            CUDA_PRINT_ERROR("Failed memcpy setRows", err);
        }

        err = cudaMemcpy(setColumns,
                         setColumnsCopy,
                         boardSize * boardSize * sizeof(memoizedSetValuesT),
                         cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            CUDA_PRINT_ERROR("Failed memcpy setColumns", err);
        }
    }

    CUDA_HOST_DEVICE Board::~Board()
    {
        if (usesSharedMemory == false)
        {
            cudaFree(setRows);
            cudaFree(setColumns);
            for (size_t side = 0; side < hintsSize; side++)
            {
                cudaFree(hints[side]);
                hints[side] = nullptr;
            }
        }
        else
        {
            // Only nullify  pointers
            for (size_t side = 0; side < hintsSize; side++)
            {
                hints[side] = nullptr;
            }
        }

        setRows = nullptr;
        setColumns = nullptr;
    }

    CUDA_DEVICE bool Board::operator==(const Board & other) const
    {
        for (size_t row = 0; row < getSize(); row++)
        {
            for (size_t column = 0; column < getSize(); column++)
            {
                if (getCell(row, column) != other.getCell(row, column))
                {
                    return false;
                }
            }
        }

        for (size_t i = 0; i < hintsSize; i++)
        {
            if (hints[i] != other.hints[i])
            {
                return false;
            }
        }

        return true;
    }

    CUDA_DEVICE bool Board::operator!=(const Board & other) const
    {
        return !(*this == other);
    }

    CUDA_HOST_DEVICE size_t Board::getSize() const
    {
        return SquareMatrix<boardFieldT>::getSize();
    }

    CUDA_HOST_DEVICE size_t Board::getCellsCount() const
    {
        return SquareMatrix<boardFieldT>::getCellsCount();
    }

    CUDA_HOST_DEVICE size_t Board::getBoardMemoryUsage() const
    {
        return getBoardMemoryUsage(getSize());
    }

    CUDA_HOST_DEVICE size_t Board::getBoardMemoryUsage(const size_t boardSize)
    {
        const auto boardCells = boardSize * boardSize;
        // Underlying matrix + fields for set rows and columns
        return SquareMatrix<boardFieldT>::getMatrixMemoryUsage(boardCells) +
            2 * boardCells * sizeof(memoizedSetValuesTypeT);
    }

    CUDA_DEVICE void Board::fill(const boardFieldT & value)
    {
        SquareMatrix<boardFieldT>::fill(value);
    }

    CUDA_HOST void Board::clear()
    {
        SquareMatrix<boardFieldT>::clear();
    }

    CUDA_HOST_DEVICE SideE Board::whichEdgeRow(size_t row) const
    {
        return SquareMatrix<boardFieldT>::whichEdgeRow(row);
    }

    CUDA_HOST_DEVICE SideE Board::whichEdgeColumn(size_t column) const
    {
        return SquareMatrix<boardFieldT>::whichEdgeColumn(column);
    }

    CUDA_DEVICE void Board::print(size_t threadIdx) const
    {
        const char* formatSpace = "%llu:   ";
        const char* formatNum = "%llu: ";

        if (threadIdx == CUDA_SIZE_T_MAX)
        {
            formatSpace = "  ";
            formatNum = "";
        }

        // Free field to align columns
        printf(formatSpace, threadIdx);

        // Top hints
        for (size_t i = 0; i < getSize(); i++)
        {
            printf("%llu ", hints[matrix::TOP][i]);
        }
        printf("\n");

        // Whole board
        for (size_t rowIdx = 0; rowIdx < getSize(); rowIdx++)
        {
            // Thread idx
            printf(formatNum, threadIdx);

            // Left hint field
            printf("%llu ", hints[matrix::LEFT][rowIdx]);

            // Board fields
            for (size_t i = 0; i < getSize(); i++)
            {
                printf("%llu ", getCell(rowIdx, i));
            }

            // Right hint field
            printf("%llu ", hints[matrix::RIGHT][rowIdx]);
            printf("\n");
        }

        // Free field to align columns
        printf(formatSpace, threadIdx);
        // Bottom hints
        for (size_t i = 0; i < getSize(); i++)
        {
            printf("%llu ", hints[matrix::BOTTOM][i]);
        }
        printf("\n");
    }

    CUDA_HOST std::vector<boardFieldT> Board::getHostVector()
    {
        const auto boardElementsCount = getSize() * getSize();
        std::vector<boardFieldT> h_retVal(boardElementsCount, 0);

        cudaError_t err = cudaMemcpy(h_retVal.data(),
                                     d_data,
                                     boardElementsCount * sizeof(boardFieldT),
                                     cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
        {
            CUDA_PRINT_ERROR("Failed memcpy", err);
        }

        return h_retVal;
    }

    CUDA_DEVICE void Board::copyInto(cuda::Board & board)
    {
        const auto boardSize = board.getSize();
        memcpy(board.d_data,
               d_data,
               sizeof(*d_data) * boardSize * boardSize);

        // Copy hints
        for (size_t side = 0; side < hintsSize; side++)
        {
            memcpy(board.hints[side],
                   hints[side],
                   boardSize * sizeof(boardFieldT));
        }
    }

    CUDA_DEVICE size_t Board::countRowVisibility(size_t row) const
    {
        size_t retVal = 1;
        size_t currentMax = 0;
        for (size_t column = 0; column < getSize(); column++)
        {
            const auto value = getCell(row, column);
            if (value == getSize())
                break;

            if (currentMax < value)
            {
                currentMax = value;
                retVal++;
            }
        }

        return retVal;
    }

    CUDA_DEVICE size_t Board::countColumnVisibility(size_t column) const
    {
        size_t retVal = 1;
        size_t currentMax = 0;
        for (size_t row = 0; row < getSize(); row++)
        {
            const auto value = getCell(row, column);
            if (value == getSize())
                break;

            if (currentMax < value)
            {
                currentMax = value;
                retVal++;
            }
        }

        return retVal;
    }

    CUDA_DEVICE size_t Board::reverseCountRowVisibility(size_t row) const
    {
        size_t retVal = 1;
        size_t currentMax = 0;
        for (size_t columnIt = 0; columnIt < getSize(); columnIt++)
        {
            const auto column = getSize() - columnIt;
            const auto value = getCell(row, column);
            if (value == getSize())
                break;

            if (currentMax < value)
            {
                currentMax = value;
                retVal++;
            }
        }

        return retVal;
    }

    CUDA_DEVICE size_t Board::reverseCountColumnVisibility(size_t column) const
    {
        size_t retVal = 1;
        size_t currentMax = 0;
        for (size_t rowIt = 0; rowIt < getSize(); rowIt++)
        {
            const auto row = getSize() - rowIt;
            const auto value = getCell(row, column);
            if (value == getSize())
                break;

            if (currentMax < value)
            {
                currentMax = value;
                retVal++;
            }
        }

        return retVal;
    }

    CUDA_DEVICE boardFieldT Board::getVisibleBuildings(SideE side, size_t rowOrColumn) const
    {
        boardFieldT retVal = 0;
        switch (side)
        {
        case TOP:
            retVal = countColumnVisibility(rowOrColumn);
            break;
        case RIGHT:
            retVal = reverseCountRowVisibility(rowOrColumn);
            break;
        case BOTTOM:
            retVal = reverseCountColumnVisibility(rowOrColumn);
            break;
        case LEFT:
            retVal = countRowVisibility(rowOrColumn);
            break;
        default:
            // Nothing to do
            break;
        }

        return retVal;
    }

    CUDA_DEVICE bool Board::isBuildingPlaceable(size_t row, size_t column, boardFieldT building)
    {
        if (getCell(row, column) != 0)
        {
            return false;
        }
#ifdef ENABLE_MEMOIZATION
        return setRows[row * getSize() + building - 1] == 0 && setColumns[column * getSize() + building - 1] == 0;
#else
        auto rowVec = getRow(row);
        auto columnVec = getColumn(column);
        auto valueElementsInRow = std::count(rowVec.begin(), rowVec.end(), building);
        auto valueElementsInColumn = std::count(columnVec.begin(), columnVec.end(), building);

        ASSERT(valueElementsInRow <= 1 && valueElementsInColumn <= 1);

        return valueElementsInRow == 0 && valueElementsInColumn == 0;
#endif // ENABLE_MEMOIZATION
    }

    CUDA_DEVICE bool Board::isBoardPartiallyValid(size_t row, size_t column)
    {
        const auto rowEdge = whichEdgeRow(row);
        const auto columnEdge = whichEdgeColumn(column);

        const auto leftVisible = getVisibleBuildings(LEFT, row);
        const auto& leftHints = hints[LEFT][row];
        const auto topVisible = getVisibleBuildings(TOP, column);
        const auto& topHints = hints[TOP][column];

        auto retVal = (leftVisible <= leftHints) && (topVisible <= topHints);

        if (columnEdge == RIGHT)
        {
            const auto rightVisible = getVisibleBuildings(RIGHT, row);
            const auto& rightHints = hints[RIGHT][row];
            retVal = retVal && (leftVisible == leftHints) && (rightVisible == rightHints);
        }

        if (rowEdge == BOTTOM)
        {
            const auto bottomVisible = getVisibleBuildings(BOTTOM, column);
            const auto& bottomHints = hints[BOTTOM][column];
            retVal = retVal && (topVisible == topHints) && (bottomVisible == bottomHints);
        }

        return retVal;
    }

    CUDA_DEVICE void Board::setCell(size_t row, size_t column, boardFieldT value)
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
            setRows[row* getSize() + currentValue - 1] = false;
            setColumns[column * getSize() + currentValue - 1] = false;
            SquareMatrix<boardFieldT>::setCell(row, column, value);
        }

        if (value != 0 && value != currentValue)
        {
            setRows[row* getSize() + value - 1] = true;
            setColumns[column* getSize() + value - 1] = true;
        }
    }

    CUDA_DEVICE void Board::clearCell(size_t row, size_t column)
    {
        setCell(row, column, 0);
    }

    CUDA_DEVICE boardFieldT Board::getCell(size_t row, size_t column) const
    {
        return SquareMatrix<boardFieldT>::getCell(row, column);
    }

}
