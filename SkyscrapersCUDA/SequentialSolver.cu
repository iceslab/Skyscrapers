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

        CUDA_HOST_DEVICE SequentialSolver::SequentialSolver(const cuda::Board& board,
                                                            void * constantMemoryPtr,
                                                            void * sharedMemoryPtr) :
            Solver(board, constantMemoryPtr, sharedMemoryPtr)
        {
            // Nothing to do
        }

        CUDA_HOST_DEVICE SequentialSolver::~SequentialSolver()
        {
            // Nothing to do
        }

        CUDA_DEVICE void SequentialSolver::backTrackingBase(cuda::Board* resultArray,
                                                            uint32T* allResultsCount,
                                                            uint32T threadIdx,
                                                            cuda::cudaEventsDeviceT & timers)
        {
            cuda::cudaEventsDeviceT localTimers = { 0 };
            localTimers.initBegin = clock64();
            const auto boardCellsCount = board.getSize() * board.getSize();
            uint32T* stack = reinterpret_cast<uint32T*>(malloc(boardCellsCount * sizeof(uint32T)));
            uint32T* stackRows = reinterpret_cast<uint32T*>(malloc(boardCellsCount * sizeof(uint32T)));
            uint32T* stackColumns = reinterpret_cast<uint32T*>(malloc(boardCellsCount * sizeof(uint32T)));
            if (stack != nullptr &&
                stackRows != nullptr &&
                stackColumns != nullptr)
            {
                memset(stack, 0, boardCellsCount * sizeof(size_t));
                memset(stackRows, 0, boardCellsCount * sizeof(size_t));
                memset(stackColumns, 0, boardCellsCount * sizeof(size_t));
            }
            else
            {
                free(stack);
                free(stackRows);
                free(stackColumns);
                stack = nullptr;
                stackRows = nullptr;
                stackColumns = nullptr;
                return;
            }

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
            localTimers.initEnd = clock64();
            localTimers.loopBegin = clock64();
            do
            {
                int64T firstZeroBegin = clock64();
                //board.print(threadIdx);
                auto & entry = stack[stackSize - 1];
                auto & row = stackRows[stackSize - 1];
                auto & column = stackColumns[stackSize - 1];

                auto idx = BitManipulation::firstZero(entry);
                idx = idx >= board.getSize() ? CUDA_BAD_INDEX : idx; // Make sure index is in range
                localTimers.firstZeroDiff += clock64() - firstZeroBegin;
                if (idx != CUDA_BAD_INDEX)
                {
                    int64T goodIndexBegin = clock64();
                    BitManipulation::setBit(entry, idx);

                    const auto consideredBuilding = idx + 1;
                    int64T placeableFnBegin = clock64();
                    bool placeable = board.isBuildingPlaceable(row, column, consideredBuilding);
                    localTimers.placeableFnDiff += clock64() - placeableFnBegin;
                    if (placeable)
                    {
                        int64T placeableBegin = clock64();
                        board.setCell(row, column, consideredBuilding);
                        int64T boardValidFnBegin = clock64();
                        bool valid = board.isBoardPartiallyValid(row, column);
                        localTimers.boardValidFnDiff = clock64() - boardValidFnBegin;
                        if (valid)
                        {
                            int64T boardValidBegin = clock64();
                            getNextFreeCell(row, column, rowRef, columnRef);
                            if (!isCellValid(rowRef, columnRef))
                            {
                                int64T lastCellBegin = clock64();

                                // Returns 0 when value is equal or above limit
                                uint32T resultIndex = atomicInc(allResultsCount, CUDA_UINT32T_MAX);
                                //CUDA_PRINT("resultIndex: %u\n", resultIndex);
                                if (resultIndex < CUDA_MAX_RESULTS)
                                {
                                    int64T copyResultBegin = clock64();
                                    board.copyInto(resultArray[resultIndex]);
                                    localTimers.copyResultDiff += clock64() - copyResultBegin;
                                }
                                else
                                {
                                    // Ensures addition with saturation
                                    atomicExch(allResultsCount, CUDA_MAX_RESULTS);
                                }
                                board.clearCell(row, column);
                                localTimers.lastCellDiff += clock64() - lastCellBegin;
                            }
                            else
                            {
                                int64T notLastCellBegin = clock64();
                                stack[stackSize] = 0;
                                stackRows[stackSize] = rowRef;
                                stackColumns[stackSize++] = columnRef;
                                localTimers.notLastCellDiff += clock64() - notLastCellBegin;
                            }
                            localTimers.boardValidDiff += clock64() - boardValidBegin;
                        }
                        else
                        {
                            int64T boardInvalidBegin = clock64();
                            board.clearCell(row, column);
                            localTimers.boardInvalidDiff += clock64() - boardInvalidBegin;
                        }
                        localTimers.placeableDiff += clock64() - placeableBegin;
                    }
                    localTimers.goodIndexDiff += clock64() - goodIndexBegin;
                }
                else
                {
                    int64T badIndexBegin = clock64();
                    board.clearCell(row, column);
                    --stackSize;
                    if (stackSize > 0)
                    {
                        board.clearCell(stackRows[stackSize - 1], stackColumns[stackSize - 1]);
                    }
                    localTimers.badIndexDiff += clock64() - badIndexBegin;
                }
            } while (stackSize > 0);
            localTimers.loopEnd = clock64();

            free(stack);
            free(stackRows);
            free(stackColumns);
            stack = nullptr;
            stackRows = nullptr;
            stackColumns = nullptr;

            timers = localTimers;
        }

        CUDA_DEVICE void SequentialSolver::backTrackingIncrementalStack(cuda::Board* resultArray,
                                                                        uint32T* allResultsCount,
                                                                        uint32T threadIdx,
                                                                        cuda::cudaEventsDeviceT & timers)
        {
            cuda::cudaEventsDeviceT localTimers = { 0 };
            localTimers.initBegin = clock64();
            const auto boardCellsCount = board.getSize() * board.getSize();
            uint32T* stack = reinterpret_cast<uint32T*>(malloc(boardCellsCount * sizeof(uint32T)));
            uint32T* stackRows = reinterpret_cast<uint32T*>(malloc(boardCellsCount * sizeof(uint32T)));
            uint32T* stackColumns = reinterpret_cast<uint32T*>(malloc(boardCellsCount * sizeof(uint32T)));
            if (stack != nullptr &&
                stackRows != nullptr &&
                stackColumns != nullptr)
            {
                memset(stack, 0, boardCellsCount * sizeof(size_t));
                memset(stackRows, 0, boardCellsCount * sizeof(size_t));
                memset(stackColumns, 0, boardCellsCount * sizeof(size_t));
            }
            else
            {
                free(stack);
                free(stackRows);
                free(stackColumns);
                stack = nullptr;
                stackRows = nullptr;
                stackColumns = nullptr;
                return;
            }

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
            localTimers.initEnd = clock64();
            localTimers.loopBegin = clock64();
            do
            {
                int64T firstZeroBegin = clock64();
                //board.print(threadIdx);
                auto & entry = stack[stackSize - 1];
                auto & row = stackRows[stackSize - 1];
                auto & column = stackColumns[stackSize - 1];

                localTimers.firstZeroDiff += clock64() - firstZeroBegin;
                if (entry < board.getSize())
                {
                    int64T goodIndexBegin = clock64();
                    // Increment value instead of bit manipulation
                    ++entry;

                    const auto consideredBuilding = entry;
                    int64T placeableFnBegin = clock64();
                    bool placeable = board.isBuildingPlaceable(row, column, consideredBuilding);
                    localTimers.placeableFnDiff += clock64() - placeableFnBegin;
                    if (placeable)
                    {

                        int64T placeableBegin = clock64();
                        board.setCell(row, column, consideredBuilding);
                        int64T boardValidFnBegin = clock64();
                        bool valid = board.isBoardPartiallyValid(row, column);
                        localTimers.boardValidFnDiff = clock64() - boardValidFnBegin;
                        if (valid)
                        {
                            int64T boardValidBegin = clock64();
                            getNextFreeCell(row, column, rowRef, columnRef);
                            if (!isCellValid(rowRef, columnRef))
                            {
                                int64T lastCellBegin = clock64();
                                // Returns 0 when value is equal or above limit
                                uint32T resultIndex = atomicInc(allResultsCount, CUDA_UINT32T_MAX);
                                if (resultIndex < CUDA_MAX_RESULTS)
                                {
                                    int64T copyResultBegin = clock64();
                                    board.copyInto(resultArray[resultIndex]);
                                    localTimers.copyResultDiff += clock64() - copyResultBegin;
                                }
                                else
                                {
                                    // Nothing to do
                                }
                                board.clearCell(row, column);
                                localTimers.lastCellDiff += clock64() - lastCellBegin;
                            }
                            else
                            {
                                int64T notLastCellBegin = clock64();
                                stack[stackSize] = 0;
                                stackRows[stackSize] = rowRef;
                                stackColumns[stackSize++] = columnRef;
                                localTimers.notLastCellDiff += clock64() - notLastCellBegin;
                            }
                            localTimers.boardValidDiff += clock64() - boardValidBegin;
                        }
                        else
                        {
                            int64T boardInvalidBegin = clock64();
                            board.clearCell(row, column);
                            localTimers.boardInvalidDiff += clock64() - boardInvalidBegin;
                        }
                        localTimers.placeableDiff += clock64() - placeableBegin;
                    }
                    localTimers.goodIndexDiff += clock64() - goodIndexBegin;
                }
                else
                {
                    int64T badIndexBegin = clock64();
                    board.clearCell(row, column);
                    --stackSize;
                    if (stackSize > 0)
                    {
                        board.clearCell(stackRows[stackSize - 1], stackColumns[stackSize - 1]);
                    }
                    localTimers.badIndexDiff += clock64() - badIndexBegin;
                }
            } while (stackSize > 0);
            localTimers.loopEnd = clock64();

            free(stack);
            free(stackRows);
            free(stackColumns);
            stack = nullptr;
            stackRows = nullptr;
            stackColumns = nullptr;

            timers = localTimers;
        }

        CUDA_DEVICE void SequentialSolver::backTrackingAOSStack(cuda::Board * resultArray,
                                                                uint32T* allResultsCount,
                                                                stackAOST * stack,
                                                                const uint32T threadIdx,
                                                                const uint32T threadsCount,
                                                                cuda::cudaEventsDeviceT & timers)
        {
            cuda::cudaEventsDeviceT localTimers = { 0 };
            localTimers.initBegin = clock64();
            const auto boardCellsCount = board.getSize() * board.getSize();

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

            // Stack is interwoven between threads, it means that stack is laid like that:
            // [0:0], [0:1], [0:2], ..., [0:n], [1:0], [1:1], [1:2], ..., [1:n], ...
            // where [stackCounter:threadIdx]
            auto stackEntrySize = board.getSize();
            stack[getStackFrameNumber(stackSize, threadIdx, threadsCount)].row = rowRef;
            stack[getStackFrameNumber(stackSize++, threadIdx, threadsCount)].column = columnRef;

            localTimers.initEnd = clock64();
            localTimers.loopBegin = clock64();
            do
            {
                int64T firstZeroBegin = clock64();
                //board.print(threadIdx);
                auto & entry = stack[getStackFrameNumber(stackSize - 1, threadIdx, threadsCount)].entry;
                auto & row = stack[getStackFrameNumber(stackSize - 1, threadIdx, threadsCount)].row;
                auto & column = stack[getStackFrameNumber(stackSize - 1, threadIdx, threadsCount)].column;

                auto idx = BitManipulation::firstZero(entry);
                localTimers.firstZeroDiff += clock64() - firstZeroBegin;
                idx = idx >= board.getSize() ? CUDA_BAD_INDEX : idx; // Make sure index is in range
                if (idx != CUDA_BAD_INDEX)
                {
                    int64T goodIndexBegin = clock64();
                    BitManipulation::setBit(entry, idx);

                    const auto consideredBuilding = idx + 1;
                    int64T placeableFnBegin = clock64();
                    auto placeable = board.isBuildingPlaceable(row, column, consideredBuilding);
                    localTimers.placeableFnDiff += clock64() - placeableFnBegin;
                    if (placeable)
                    {
                        int64T placeableBegin = clock64();
                        board.setCell(row, column, consideredBuilding);
                        int64T boardValidFnBegin = clock64();
                        bool valid = board.isBoardPartiallyValid(row, column);
                        localTimers.boardValidFnDiff = clock64() - boardValidFnBegin;
                        if (valid)
                        {
                            int64T boardValidBegin = clock64();
                            getNextFreeCell(row, column, rowRef, columnRef);
                            if (!isCellValid(rowRef, columnRef))
                            {
                                int64T lastCellBegin = clock64();
                                // Returns 0 when value is equal or above limit
                                uint32T resultIndex = atomicInc(allResultsCount, CUDA_UINT32T_MAX);
                                if (resultIndex < CUDA_MAX_RESULTS)
                                {
                                    int64T copyResultBegin = clock64();
                                    board.copyInto(resultArray[resultIndex]);
                                    localTimers.copyResultDiff += clock64() - copyResultBegin;
                                }
                                else
                                {
                                    // Nothing to do
                                }
                                board.clearCell(row, column);
                                localTimers.lastCellDiff += clock64() - lastCellBegin;
                            }
                            else
                            {
                                int64T notLastCellBegin = clock64();
                                stack[getStackFrameNumber(stackSize, threadIdx, threadsCount)].entry = 0;
                                stack[getStackFrameNumber(stackSize, threadIdx, threadsCount)].row = rowRef;
                                stack[getStackFrameNumber(stackSize++, threadIdx, threadsCount)].column = columnRef;
                                localTimers.notLastCellDiff += clock64() - notLastCellBegin;
                            }
                        }
                        else
                        {
                            int64T boardInvalidBegin = clock64();
                            board.clearCell(row, column);
                            localTimers.boardInvalidDiff += clock64() - boardInvalidBegin;
                        }
                    }
                    localTimers.goodIndexDiff += clock64() - goodIndexBegin;
                }
                else
                {
                    int64T badIndexBegin = clock64();
                    board.clearCell(row, column);
                    --stackSize;
                    if (stackSize > 0)
                    {
                        board.clearCell(stack[getStackFrameNumber(stackSize - 1, threadIdx, threadsCount)].row,
                                        stack[getStackFrameNumber(stackSize - 1, threadIdx, threadsCount)].column);
                    }
                    localTimers.badIndexDiff += clock64() - badIndexBegin;
                }

            } while (stackSize > 0);
            localTimers.loopEnd = clock64();

            timers = localTimers;
        }

        CUDA_DEVICE void SequentialSolver::backTrackingSOAStack(cuda::Board* resultArray,
                                                                uint32T* allResultsCount,
                                                                stackSOAT* stack,
                                                                const uint32T threadIdx,
                                                                const uint32T threadsCount,
                                                                cuda::cudaEventsDeviceT & timers)
        {
            cuda::cudaEventsDeviceT localTimers = { 0 };
            localTimers.initBegin = clock64();
            const auto boardCellsCount = board.getSize() * board.getSize();

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

            // Stack is interwoven between threads, it means that stack is laid like that:
            // [0:0], [0:1], [0:2], ..., [0:n], [1:0], [1:1], [1:2], ..., [1:n], ...
            // where [stackCounter:threadIdx]
            auto stackEntrySize = board.getSize();
            stack->row[getStackFrameNumber(stackSize, threadIdx, threadsCount)] = rowRef;
            stack->column[getStackFrameNumber(stackSize++, threadIdx, threadsCount)] = columnRef;

            localTimers.initEnd = clock64();
            localTimers.loopBegin = clock64();
            do
            {
                //board.print(threadIdx);
                auto & entry = stack->entry[getStackFrameNumber(stackSize - 1, threadIdx, threadsCount)];
                auto & row = stack->row[getStackFrameNumber(stackSize - 1, threadIdx, threadsCount)];
                auto & column = stack->column[getStackFrameNumber(stackSize - 1, threadIdx, threadsCount)];

                auto idx = BitManipulation::firstZero(entry);
                idx = idx >= board.getSize() ? CUDA_BAD_INDEX : idx; // Make sure index is in range
                if (idx != CUDA_BAD_INDEX)
                {
                    int64T goodIndexBegin = clock64();
                    BitManipulation::setBit(entry, idx);

                    const auto consideredBuilding = idx + 1;
                    int64T placeableFnBegin = clock64();
                    auto placeable = board.isBuildingPlaceable(row, column, consideredBuilding);
                    localTimers.placeableFnDiff += clock64() - placeableFnBegin;
                    if (placeable)
                    {
                        int64T placeableBegin = clock64();
                        board.setCell(row, column, consideredBuilding);
                        int64T boardValidFnBegin = clock64();
                        bool valid = board.isBoardPartiallyValid(row, column);
                        localTimers.boardValidFnDiff = clock64() - boardValidFnBegin;
                        if (valid)
                        {
                            getNextFreeCell(row, column, rowRef, columnRef);
                            if (!isCellValid(rowRef, columnRef))
                            {
                                int64T lastCellBegin = clock64();
                                // Returns 0 when value is equal or above limit
                                uint32T resultIndex = atomicInc(allResultsCount, CUDA_UINT32T_MAX);
                                if (resultIndex < CUDA_MAX_RESULTS)
                                {
                                    int64T copyResultBegin = clock64();
                                    board.copyInto(resultArray[resultIndex]);
                                    localTimers.copyResultDiff += clock64() - copyResultBegin;
                                }
                                else
                                {
                                    // Nothing to do
                                }
                                board.clearCell(row, column);
                                localTimers.lastCellDiff += clock64() - lastCellBegin;
                            }
                            else
                            {
                                int64T notLastCellBegin = clock64();
                                stack->entry[getStackFrameNumber(stackSize, threadIdx, threadsCount)] = 0;
                                stack->row[getStackFrameNumber(stackSize, threadIdx, threadsCount)] = rowRef;
                                stack->column[getStackFrameNumber(stackSize++, threadIdx, threadsCount)] = columnRef;
                                localTimers.notLastCellDiff += clock64() - notLastCellBegin;
                            }
                        }
                        else
                        {
                            int64T boardInvalidBegin = clock64();
                            board.clearCell(row, column);
                            localTimers.boardInvalidDiff += clock64() - boardInvalidBegin;
                        }
                    }
                    localTimers.goodIndexDiff += clock64() - goodIndexBegin;
                }
                else
                {
                    int64T badIndexBegin = clock64();
                    board.clearCell(row, column);
                    --stackSize;
                    if (stackSize > 0)
                    {
                        board.clearCell(stack->row[getStackFrameNumber(stackSize - 1, threadIdx, threadsCount)],
                                        stack->column[getStackFrameNumber(stackSize - 1, threadIdx, threadsCount)]);
                    }
                    localTimers.badIndexDiff += clock64() - badIndexBegin;
                }
            } while (stackSize > 0);
            localTimers.loopEnd = clock64();

            timers = localTimers;
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
                row = CUDA_UINT32T_MAX;
                column = CUDA_UINT32T_MAX;
            }

            rowOut = row;
            columnOut = column;
        }

        CUDA_DEVICE bool SequentialSolver::isCellValid(uint32T row, uint32T column)
        {
            return row != CUDA_UINT32T_MAX && column != CUDA_UINT32T_MAX;
        }

        CUDA_DEVICE const cuda::Board & SequentialSolver::getBoard() const
        {
            return board;
        }

        CUDA_HOST_DEVICE uint32T SequentialSolver::getStackFrameNumber(uint32T stackSize,
                                                                       const uint32T threadId,
                                                                       const uint32T threadsCount)
        {
            return stackSize * threadsCount + threadId;
        }
    }
}
