#include "Board.cuh"

namespace cuda
{
    const SideE Board::validSides[validSidesNumber] =
    {
        TOP,
        RIGHT,
        BOTTOM,
        LEFT
    };

    Board::Board(const boardFieldT boardSize) :
        SquareMatrix<boardFieldT>(boardSize)
    {
        // Alloc and memset setRows
        cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&setRows), boardSize * sizeof(boardFieldT));
        if (err != cudaSuccess)
        {
            CUDA_PRINT_ERROR("Failed allocation setRows", err);
            setRows = nullptr;
        }
        else
        {
            err = cudaMemset(reinterpret_cast<void**>(&setRows), 0, boardSize * sizeof(boardFieldT));
            if (err != cudaSuccess)
            {
                CUDA_PRINT_ERROR("Failed memset setRows", err);
            }
        }

        // Alloc and memset setColumns
        err = cudaMalloc(reinterpret_cast<void**>(&setColumns), boardSize * sizeof(boardFieldT));
        if (err != cudaSuccess)
        {
            CUDA_PRINT_ERROR("Failed allocation setColumns", err);
            setColumns = nullptr;
        }
        else
        {
            err = cudaMemset(reinterpret_cast<void**>(&setColumns), 0, boardSize * sizeof(boardFieldT));
            if (err != cudaSuccess)
            {
                CUDA_PRINT_ERROR("Failed memset setColumns", err);
            }
        }

        // Alloc and memset hints
        for (size_t side = 0; side < hintsSize; side++)
        {
            err = cudaMalloc(reinterpret_cast<void**>(&hints[side]), boardSize * sizeof(boardFieldT));
            if (err != cudaSuccess)
            {
                CUDA_PRINT_ERROR("Failed allocation hints[side]", err);
                hints[side] = nullptr;
            }
            else
            {
                err = cudaMemset(reinterpret_cast<void**>(&hints[side]), 0, boardSize * sizeof(boardFieldT));
                if (err != cudaSuccess)
                {
                    CUDA_PRINT_ERROR("Failed memset hints[side]", err);
                }
            }
        }
    }

    /*void Board::calculateHints()
    {
        // Fill hints for TOP, RIGHT, BOTTOM and LEFT
        for (size_t i = 0; i < getSize(); i++)
        {
            for (size_t side = 0; side < hintsSize; side++)
            {
                const auto validSide = validSides[side];
                hints[side][i] = getVisibleBuildings(validSide, i);
            }
        }
    }*/

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

    /*bool Board::checkIfLatinSquare() const
    {
        for (size_t i = 0; i < getSize(); i++)
        {
            std::vector<bool> rowChecker(getSize(), false);
            std::vector<bool> columnChecker(getSize(), false);
            for (size_t j = 0; j < getSize(); j++)
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
    }*/

    /*bool Board::checkValidityWithHints() const
    {
        if (!checkIfLatinSquare())
        {
            return false;
        }

        for (size_t i = 0; i < getSize(); i++)
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
    }*/

    CUDA_HOST_DEVICE size_t Board::getSize() const
    {
        return SquareMatrix<boardFieldT>::getSize();
    }

    CUDA_DEVICE void Board::fill(const boardFieldT & value)
    {
        SquareMatrix<boardFieldT>::fill(value);
    }

    CUDA_HOST_DEVICE SideE Board::whichEdgeRow(size_t row) const
    {
        return SquareMatrix<boardFieldT>::whichEdgeRow(row);
    }

    CUDA_HOST_DEVICE SideE Board::whichEdgeColumn(size_t column) const
    {
        return SquareMatrix<boardFieldT>::whichEdgeColumn(column);
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

    /*void Board::print() const
    {
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
    }*/

    /*void Board::resize(const boardFieldT boardSize)
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
    }*/

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
            return false;
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

    CUDA_DEVICE boardFieldT Board::getCell(size_t row, size_t column)
    {
        return SquareMatrix<boardFieldT>::getCell(row, column);
    }

    CUDA_DEVICE boardFieldT Board::getCell(size_t row, size_t column) const
    {
        return SquareMatrix<boardFieldT>::getCell(row, column);
    }

}
