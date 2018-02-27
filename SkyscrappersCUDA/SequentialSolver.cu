#include "SequentialSolver.cuh"

namespace cuda
{
    namespace solver
    {
        SequentialSolver::SequentialSolver(const board::Board & board) :
            Solver(board)
        {
            // Nothing to do
        }

        CUDA_DEVICE size_t SequentialSolver::solve(cuda::Board* resultArray, size_t threadIdx)
        {

#ifdef BT_WITH_STACK
            auto retVal = backTrackingWithStack(resultArray, threadIdx);
#else
            auto freeCell = rowAndColumnPairT(0, 0);
            if (board.getCell(0, 0) != 0)
            {
                freeCell = getNextFreeCell(0, 0);
            }

            std::vector<board::Board> retVal;
            backTracking(retVal, 0, freeCell.first, freeCell.second);
#endif // BT_WITH_STACK

            return retVal;
        }

#ifndef BT_WITH_STACK
        void solver::SequentialSolver::backTracking(std::vector<board::Board> & retVal, size_t level, size_t row, size_t column)
        {
            const auto treeRowSize = board.size();

            // Check if it is last cell
            const auto cellPair = getNextFreeCell(row, column);
            if (cellPair == lastCellPair)
            {
                DEBUG_PRINTLN_VERBOSE_INFO("Last cell");
                for (size_t i = 0; i < treeRowSize; i++)
                {
                    const auto consideredBuilding = i + 1;

                    if (board.isBuildingPlaceable(row, column, consideredBuilding))
                    {
                        board.setCell(row, column, consideredBuilding);
                        if (board.isBoardPartiallyValid(row, column))
                        {
                            DEBUG_PRINTLN_VERBOSE_INFO("Found result");
                            DEBUG_CALL(board.print());
                            retVal.emplace_back(board);
                        }
                        board.clearCell(row, column);
                    }
                }
            }
            else
            {
                for (size_t i = 0; i < treeRowSize; i++)
                {
                    const auto consideredBuilding = i + 1;
                    if (board.isBuildingPlaceable(row, column, consideredBuilding))
                    {
                        board.setCell(row, column, consideredBuilding);
                        if (board.isBoardPartiallyValid(row, column))
                        {
                            backTracking(retVal, level + 1, cellPair.first, cellPair.second);
                        }

                        board.clearCell(row, column);
                    }
                }
            }
        }
#else
        CUDA_DEVICE size_t SequentialSolver::backTrackingWithStack(cuda::Board* resultArray, size_t threadIdx)
        {
            CUDA_PRINT("%llu: %s: BEGIN\n",
                       threadIdx,
                       __FUNCTION__);
            const auto boardCellsCount = board.getSize() * board.getSize();
            size_t* stack = reinterpret_cast<size_t*>(malloc(boardCellsCount * sizeof(size_t)));
            size_t* stackRows = reinterpret_cast<size_t*>(malloc(boardCellsCount * sizeof(size_t)));
            size_t* stackColumns = reinterpret_cast<size_t*>(malloc(boardCellsCount * sizeof(size_t)));
            if (stack != nullptr &&
                stackRows != nullptr &&
                stackColumns != nullptr)
            {
                CUDA_PRINT("%llu: %s: Stack arrays allocated successfully\n",
                           threadIdx,
                           __FUNCTION__);
                memset(stack, 0, boardCellsCount * sizeof(size_t));
                memset(stackRows, 0, boardCellsCount * sizeof(size_t));
                memset(stackColumns, 0, boardCellsCount * sizeof(size_t));

                //CUDA_PRINT("%s: Stack frames:\n", __FUNCTION__);
                //for (size_t i = 0; i < boardCellsCount; i++)
                //{
                //    CUDA_PRINT("%u: 0x%08llx\n", i, stack[i]);
                //}
            }
            else
            {
                CUDA_PRINT("%llu: %s: Stack arrays allocation failed. Returning...\n", threadIdx, __FUNCTION__);
                free(stack);
                free(stackRows);
                free(stackColumns);
                stack = nullptr;
                stackRows = nullptr;
                stackColumns = nullptr;
                return 0;
            }

            // Result boards count
            size_t resultsCount = 0;
            // Current valid stack frames
            size_t stackSize = 0;
            // Used for row result from getNextFreeCell()
            size_t rowRef = 0;
            // Used for column result from getNextFreeCell()
            size_t columnRef = 0;

            if (board.getCell(0, 0) != 0)
            {
                getNextFreeCell(0, 0, rowRef, columnRef);
            }

            auto stackEntrySize = board.getSize();
            stackRows[stackSize] = rowRef;
            stackColumns[stackSize++] = columnRef;
            //CUDA_PRINT("%llu: %s: stackSize=%llu\n", threadIdx, __FUNCTION__, stackSize);
            do
            {
                board.print(threadIdx);
                auto & entry = stack[stackSize - 1];
                auto & row = stackRows[stackSize - 1];
                auto & column = stackColumns[stackSize - 1];

                auto idx = BitManipulation::firstZero(entry);
                idx = idx >= board.getSize() ? CUDA_BAD_INDEX : idx; // Make sure index is in range
                //CUDA_PRINT("%llu: %s: First zero on index: %llu stack[%llu]=0x%08llx\n",
                //           threadIdx,
                //           __FUNCTION__,
                //           idx,
                //           stackSize - 1,
                //           entry);
                if (idx != CUDA_BAD_INDEX)
                {
                    BitManipulation::setBit(entry, idx);

                    const auto consideredBuilding = idx + 1;
                    if (board.isBuildingPlaceable(row, column, consideredBuilding))
                    {
                        //CUDA_PRINT("%llu: %s: Building %llu is placeable at (%llu, %llu)\n",
                        //           threadIdx,
                        //           __FUNCTION__,
                        //           consideredBuilding,
                        //           row,
                        //           column);
                        board.setCell(row, column, consideredBuilding);
                        if (board.isBoardPartiallyValid(row, column))
                        {
                            //CUDA_PRINT("%llu: %s: Board partially VALID till (%llu, %llu)\n",
                            //           threadIdx,
                            //           __FUNCTION__,
                            //           row,
                            //           column);
                            getNextFreeCell(row, column, rowRef, columnRef);
                            if (!isCellValid(rowRef, columnRef))
                            {
                                if (resultsCount < maxResultsPerThread)
                                {
                                    CUDA_PRINT("%llu: %s: Found a result, copying to global memory\n",
                                               threadIdx,
                                               __FUNCTION__);
                                    board.copyInto(resultArray[resultsCount++]);
                                }
                                else
                                {
                                    CUDA_PRINT("%llu: %s: Found a result, but it doesn't fit inside array\n",
                                               threadIdx,
                                               __FUNCTION__);
                                }
                                board.clearCell(row, column);
                            }
                            else
                            {
                                stack[stackSize] = 0;
                                stackRows[stackSize] = rowRef;
                                stackColumns[stackSize++] = columnRef;
                                //CUDA_PRINT("%llu: %s: Next valid cell (%llu, %llu), stackSize: %llu\n",
                                //           threadIdx,
                                //           __FUNCTION__,
                                //           rowRef,
                                //           columnRef,
                                //           stackSize);
                            }
                        }
                        else
                        {
                            //CUDA_PRINT("%llu: %s: Board partially INVALID till (%llu, %llu)\n",
                            //           threadIdx,
                            //           __FUNCTION__,
                            //           row,
                            //           column);
                            board.clearCell(row, column);
                        }
                    }
                }
                else
                {
                    //CUDA_PRINT("%llu: %s: Searched through all variants. Popping stack...\n",
                    //           threadIdx,
                    //           __FUNCTION__);
                    board.clearCell(row, column);
                    --stackSize;
                    if (stackSize > 0)
                    {
                        board.clearCell(stackRows[stackSize - 1], stackColumns[stackSize - 1]);
                    }
                }

                //CUDA_PRINT("%llu: %s: stackSize %u\n",
                //           threadIdx,
                //           __FUNCTION__,
                //           stackSize);
            } while (stackSize > 0);

            free(stack);
            free(stackRows);
            free(stackColumns);
            stack = nullptr;
            stackRows = nullptr;
            stackColumns = nullptr;

            CUDA_PRINT("%llu: %s: END\n",
                       threadIdx,
                       __FUNCTION__);
            return resultsCount;
        }
#endif // !BT_WITH_STACK

        CUDA_DEVICE void SequentialSolver::getNextFreeCell(size_t row,
                                                           size_t column,
                                                           size_t & rowOut,
                                                           size_t & columnOut) const
        {
            const auto maxSize = board.getSize();

            // Search till free cell is found
            do
            {
                // Next column
                if (column < maxSize - 1)
                {
                    column++;
                }
                // Next row
                else if (column >= maxSize - 1)
                {
                    column = 0;
                    row++;
                }
            } while (row < maxSize && board.getCell(row, column) != 0);

            // If row is too big return max values
            if (row >= maxSize)
            {
                row = CUDA_SIZE_T_MAX;
                column = CUDA_SIZE_T_MAX;
            }

            rowOut = row;
            columnOut = column;
        }

        CUDA_DEVICE bool SequentialSolver::isCellValid(size_t row, size_t column)
        {
            return row != CUDA_SIZE_T_MAX && column != CUDA_SIZE_T_MAX;
        }
        CUDA_DEVICE const cuda::Board & SequentialSolver::getBoard() const
        {
            return board;
        }
    }
}
