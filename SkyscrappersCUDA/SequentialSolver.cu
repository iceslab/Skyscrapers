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

        CUDA_DEVICE uint32T SequentialSolver::solve(cuda::Board* resultArray, uint32T threadIdx)
        {

            auto retVal = backTrackingBase(resultArray, threadIdx);
            return retVal;
        }

        CUDA_DEVICE uint32T SequentialSolver::backTrackingBase(cuda::Board* resultArray, uint32T threadIdx)
        {
            //CUDA_PRINT("%llu: %s: BEGIN\n",
            //           threadIdx,
            //           __FUNCTION__);
            const auto boardCellsCount = board.getSize() * board.getSize();
            uint32T* stack = reinterpret_cast<uint32T*>(malloc(boardCellsCount * sizeof(uint32T)));
            uint32T* stackRows = reinterpret_cast<uint32T*>(malloc(boardCellsCount * sizeof(uint32T)));
            uint32T* stackColumns = reinterpret_cast<uint32T*>(malloc(boardCellsCount * sizeof(uint32T)));
            if (stack != nullptr &&
                stackRows != nullptr &&
                stackColumns != nullptr)
            {
                //CUDA_PRINT("%llu: %s: Stack arrays allocated successfully\n",
                //           threadIdx,
                //           __FUNCTION__);
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
                //CUDA_PRINT("%llu: %s: Stack arrays allocation failed. Returning...\n", threadIdx, __FUNCTION__);
                free(stack);
                free(stackRows);
                free(stackColumns);
                stack = nullptr;
                stackRows = nullptr;
                stackColumns = nullptr;
                return 0;
            }

            // Result boards count
            uint32T resultsCount = 0;
            // Current valid stack frames
            uint32T stackSize = 0;
            // Used for row result from getNextFreeCell()
            uint32T rowRef = 0;
            // Used for column result from getNextFreeCell()
            uint32T columnRef = 0;

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
                //board.print(threadIdx);
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
                                if (resultsCount < CUDA_MAX_RESULTS_PER_THREAD)
                                {
                                    //CUDA_PRINT("%llu: %s: Found a result, copying to global memory\n",
                                    //           threadIdx,
                                    //           __FUNCTION__);
                                    board.copyInto(resultArray[resultsCount++]);
                                }
                                else
                                {
                                    //CUDA_PRINT("%llu: %s: Found a result, but it doesn't fit inside array\n",
                                    //           threadIdx,
                                    //           __FUNCTION__);
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

            //CUDA_PRINT("%llu: %s: END\n",
            //           threadIdx,
            //           __FUNCTION__);
            return resultsCount;
        }

        CUDA_DEVICE uint32T SequentialSolver::backTrackingAOSStack(cuda::Board * resultArray,
                                                                   uint32T threadIdx,
                                                                   stackAOST * stack)
        {
            return uint32T();
        }

        CUDA_DEVICE uint32T SequentialSolver::backTrackingSOAStack(cuda::Board * resultArray,
                                                                   uint32T threadIdx,
                                                                   stackSOAT * stack)
        {
            return uint32T();
        }

        CUDA_DEVICE void SequentialSolver::getNextFreeCell(uint32T row,
                                                           uint32T column,
                                                           uint32T & rowOut,
                                                           uint32T & columnOut) const
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
                row = CUDA_UINT32_T_MAX;
                column = CUDA_UINT32_T_MAX;
            }

            rowOut = row;
            columnOut = column;
        }

        CUDA_DEVICE bool SequentialSolver::isCellValid(uint32T row, uint32T column)
        {
            return row != CUDA_UINT32_T_MAX && column != CUDA_UINT32_T_MAX;
        }
        CUDA_DEVICE const cuda::Board & SequentialSolver::getBoard() const
        {
            return board;
        }
    }
}
