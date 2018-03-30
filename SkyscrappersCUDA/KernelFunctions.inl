#ifndef __INCLUDED_KERNEL_FUNCTIONS_INL__
#define __INCLUDED_KERNEL_FUNCTIONS_INL__

#include "CUDAUtilities.cuh"
#include "ParallelSolver.cuh"
#include "Stack.cuh"
#include "BitManipulation.cuh"

CUDA_CONSTANT cuda::boardFieldT constantMemoryPtr[(16 << 10) >> 2]; // 16 kB

CUDA_GLOBAL void parallelSolvingBase(cuda::solver::kernelInputT d_solvers,
                                     cuda::solver::kernelOutputT d_outputBoards,
                                     cuda::uint32T* d_outputBoardsSize,
                                     cuda::cudaEventsDeviceT* d_timers)
{
    // It denotes thread index and array index
    const auto idx = threadIdx.x;
    d_solvers[idx].backTrackingBase(d_outputBoards,
                                    d_outputBoardsSize,
                                    idx,
                                    d_timers[idx]);
}

CUDA_GLOBAL void parallelSolvingIncrementalStack(cuda::solver::kernelInputT d_solvers,
                                                 cuda::solver::kernelOutputT d_outputBoards,
                                                 cuda::uint32T* d_outputBoardsSize,
                                                 cuda::cudaEventsDeviceT* d_timers)
{
    // It denotes thread index and array index
    const auto idx = threadIdx.x;
    d_solvers[idx].backTrackingIncrementalStack(d_outputBoards,
                                                d_outputBoardsSize,
                                                idx,
                                                d_timers[idx]);
}

CUDA_GLOBAL void parallelSolvingSharedMemory(cuda::solver::kernelInputT d_solvers,
                                             cuda::solver::kernelOutputT d_outputBoards,
                                             cuda::uint32T* d_outputBoardsSize,
                                             cuda::cudaEventsDeviceT* d_timers)
{
    // Shared memory placeholder
    extern CUDA_SHARED char sharedMemoryDecl[];
    char* sharedMemoryPtr = sharedMemoryDecl;

    // It denotes thread index and array index
    const auto idx = threadIdx.x;
    const auto bytesPerBoard = d_solvers[idx].board.getBoardMemoryUsage();
    // Local solver to copy contents to shared memory
    cuda::solver::SequentialSolver d_localSolver(d_solvers[idx].board,
                                                 constantMemoryPtr,
                                                 sharedMemoryPtr + idx * bytesPerBoard);
    d_localSolver.backTrackingBase(d_outputBoards,
                                   d_outputBoardsSize,
                                   idx,
                                   d_timers[idx]);
}

CUDA_GLOBAL void parallelSolvingAOSStack(cuda::solver::kernelInputT d_solvers,
                                         cuda::solver::kernelOutputT d_outputBoards,
                                         cuda::uint32T* d_outputBoardsSize,
                                         cuda::solver::stackAOST* stack,
                                         cuda::cudaEventsDeviceT* d_timers)
{
    // It denotes thread index and array index
    const auto idx = threadIdx.x;
    const auto threads = blockDim.x;
    d_solvers[idx].backTrackingAOSStack(d_outputBoards,
                                        d_outputBoardsSize,
                                        stack,
                                        idx,
                                        threads,
                                        d_timers[idx]);
}

CUDA_GLOBAL void parallelSolvingSOAStack(cuda::solver::kernelInputT d_solvers,
                                         cuda::solver::kernelOutputT d_outputBoards,
                                         cuda::uint32T* d_outputBoardsSize,
                                         cuda::solver::stackSOAT* stack,
                                         cuda::cudaEventsDeviceT* d_timers)
{
    // It denotes thread index and array index
    const auto idx = threadIdx.x;
    const auto threads = blockDim.x;
    d_solvers[idx].backTrackingSOAStack(d_outputBoards,
                                        d_outputBoardsSize,
                                        stack,
                                        idx,
                                        threads,
                                        d_timers[idx]);
}

// Microkernels

CUDA_GLOBAL void mcInit(cuda::solver::threadLocalsT* d_locals,
                        cuda::solver::kernelInputT d_solvers)
{
    // Indices for easier understanding
    const auto idx = threadIdx.x;
    const auto threads = blockDim.x;
    auto & locals = d_locals[idx];
    const auto & solver = d_solvers[idx];
    const auto & board = solver.getBoard();
    const auto boardCellsCount = board.getSize() * board.getSize();
    auto & stack = locals.stack;
    auto & stackRows = locals.stackRows;
    auto & stackColumns = locals.stackColumns;
    auto & stackSize = locals.stackSize;
    auto & rowRef = locals.rowRef;
    auto & columnRef = locals.columnRef;
    auto & stackEntrySize = locals.stackEntrySize;


    stack = reinterpret_cast<cuda::uint32T*>(malloc(boardCellsCount * sizeof(cuda::uint32T)));
    stackRows = reinterpret_cast<cuda::uint32T*>(malloc(boardCellsCount * sizeof(cuda::uint32T)));
    stackColumns = reinterpret_cast<cuda::uint32T*>(malloc(boardCellsCount * sizeof(cuda::uint32T)));
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
    }

    if (board.getCell(0, 0) != 0)
    {
        solver.getNextFreeCell(0, 0, rowRef, columnRef);
    }

    stackEntrySize = board.getSize();
    stackRows[stackSize] = rowRef;
    stackColumns[stackSize++] = columnRef;
}

CUDA_GLOBAL void mcGetStackIndex(cuda::solver::threadLocalsT* d_locals,
                                 cuda::solver::kernelInputT d_solvers)
{
    const auto idx = threadIdx.x;
    auto & locals = d_locals[idx];
    const auto & solver = d_solvers[idx];
    const auto & board = solver.getBoard();

    locals.buildingIdx = cuda::solver::BitManipulation::firstZero(locals.stack[locals.stackSize - 1]);
    // Make sure index is in range
    locals.buildingIdx = locals.buildingIdx >= board.getSize() ? CUDA_BAD_INDEX : locals.buildingIdx;
}

// Pushes valid indices to left using scatter operation
// +-------------------------------+
// | a | b | # | c | # | d | # | # |
// +-------------------------------+
//
// New indices (theoretical, in practice there will be permutation)
// +-------------------------------+
// | 0 | 1 | # | 2 | # | 3 | # | # |
// +-------------------------------+
//
// +-------------------------------+
// | 0 | 1 | 3 | 5 | # | # | # | # |
// +-------------------------------+
//
// After operation:
// +-------------------------------+
// | a | b | c | d | # | # | # | # |
// +-------------------------------+
CUDA_GLOBAL void mcScatterPushValidIndicesToLeft(cuda::solver::threadLocalsT* d_locals,
                                                 cuda::uint32T* threadLogicIdcs,
                                                 cuda::uint32T & validIndices)
{
    const auto idx = threadIdx.x;
    const auto threads = blockDim.x;

    // Shared memory initialization
    CUDA_SHARED cuda::uint32T lockIncrement;
    if (idx == 0)
    {
        lockIncrement = 0;
    }

    // Wait for all threads to make sure shared memory is initialized
    __syncthreads();

    // When index is valid, write current thread index
    // to array at newly acquired (using atomic increment) index
    if (d_locals[idx].buildingIdx != CUDA_BAD_INDEX)
    {
        const auto nextIdx = atomicInc(&lockIncrement, threads);
        threadLogicIdcs[nextIdx] = idx;
    }
}

CUDA_HOST void launchMicrokernels(cuda::solver::kernelInputT d_solvers,
                                  cuda::solver::kernelOutputT d_outputBoards,
                                  cuda::solver::kernelOutputSizesT d_outputBoardsSizes,
                                  cuda::solver::threadLocalsT* d_threadLocals,
                                  cuda::uint32T* d_scatterArray,
                                  cuda::uint32T solversCount)
{

    dim3 numBlocks(1);
    dim3 threadsPerBlock(solversCount);

    // Initialization kernel
    mcInit << < numBlocks, threadsPerBlock >> > (d_threadLocals,
                                                 d_solvers);
    // Algorithm main loop
    do
    {
        mcGetStackIndex << < numBlocks, threadsPerBlock >> > (d_threadLocals,
                                                              d_solvers);
        cuda::uint32T validIndices = 0;
        mcScatterPushValidIndicesToLeft << < numBlocks, threadsPerBlock >> > (d_threadLocals,
                                                                              d_scatterArray,
                                                                              validIndices);
    } while (true);
}

//CUDA_GLOBAL void parallelSolvingSOAStack(cuda::solver::kernelInputT d_solvers,
//                                         cuda::solver::kernelOutputT d_outputBoards,
//                                         cuda::solver::kernelOutputSizesT d_outputBoardsSizes,
//                                         cuda::solver::stackSOAT* stack)
//{
//
//
//    //CUDA_PRINT("%llu: %s: BEGIN\n",
//    //           threadIdx,
//    //           __FUNCTION__);
//
//    /* ========== KERNEL 1 START ========== */
//
//    /* ========== KERNEL 1 STOP  ========== */
//
//    //CUDA_PRINT("%llu: %s: stackSize=%llu\n", threadIdx, __FUNCTION__, stackSize);
//    do
//    {
//        /* ========== KERNEL 2 START ========== */
//        //board.print(threadIdx);
//        auto & entry = stack[stackSize - 1];
//        auto & row = stackRows[stackSize - 1];
//        auto & column = stackColumns[stackSize - 1];
//
//        auto idx = BitManipulation::firstZero(entry);
//        idx = idx >= board.getSize() ? CUDA_BAD_INDEX : idx; // Make sure index is in range
//        /* ========== KERNEL 2 STOP  ========== */
//
//        //CUDA_PRINT("%llu: %s: First zero on index: %llu stack[%llu]=0x%08llx\n",
//        //           threadIdx,
//        //           __FUNCTION__,
//        //           idx,
//        //           stackSize - 1,
//        //           entry);
//        if (idx != CUDA_BAD_INDEX)
//        {
//            BitManipulation::setBit(entry, idx);
//
//            const auto consideredBuilding = idx + 1;
//            if (board.isBuildingPlaceable(row, column, consideredBuilding))
//            {
//                //CUDA_PRINT("%llu: %s: Building %llu is placeable at (%llu, %llu)\n",
//                //           threadIdx,
//                //           __FUNCTION__,
//                //           consideredBuilding,
//                //           row,
//                //           column);
//                board.setCell(row, column, consideredBuilding);
//                if (board.isBoardPartiallyValid(row, column))
//                {
//                    //CUDA_PRINT("%llu: %s: Board partially VALID till (%llu, %llu)\n",
//                    //           threadIdx,
//                    //           __FUNCTION__,
//                    //           row,
//                    //           column);
//                    getNextFreeCell(row, column, rowRef, columnRef);
//                    if (!isCellValid(rowRef, columnRef))
//                    {
//                        if (resultsCount < CUDA_MAX_RESULTS_PER_THREAD)
//                        {
//                            //CUDA_PRINT("%llu: %s: Found a result, copying to global memory\n",
//                            //           threadIdx,
//                            //           __FUNCTION__);
//                            board.copyInto(resultArray[resultsCount++]);
//                        }
//                        else
//                        {
//                            //CUDA_PRINT("%llu: %s: Found a result, but it doesn't fit inside array\n",
//                            //           threadIdx,
//                            //           __FUNCTION__);
//                        }
//                        board.clearCell(row, column);
//                    }
//                    else
//                    {
//                        stack[stackSize] = 0;
//                        stackRows[stackSize] = rowRef;
//                        stackColumns[stackSize++] = columnRef;
//                        //CUDA_PRINT("%llu: %s: Next valid cell (%llu, %llu), stackSize: %llu\n",
//                        //           threadIdx,
//                        //           __FUNCTION__,
//                        //           rowRef,
//                        //           columnRef,
//                        //           stackSize);
//                    }
//                }
//                else
//                {
//                    //CUDA_PRINT("%llu: %s: Board partially INVALID till (%llu, %llu)\n",
//                    //           threadIdx,
//                    //           __FUNCTION__,
//                    //           row,
//                    //           column);
//                    board.clearCell(row, column);
//                }
//            }
//        }
//        else
//        {
//            //CUDA_PRINT("%llu: %s: Searched through all variants. Popping stack...\n",
//            //           threadIdx,
//            //           __FUNCTION__);
//            board.clearCell(row, column);
//            --stackSize;
//            if (stackSize > 0)
//            {
//                board.clearCell(stackRows[stackSize - 1], stackColumns[stackSize - 1]);
//            }
//        }
//
//        //CUDA_PRINT("%llu: %s: stackSize %u\n",
//        //           threadIdx,
//        //           __FUNCTION__,
//        //           stackSize);
//    } while (stackSize > 0);
//
//    free(stack);
//    free(stackRows);
//    free(stackColumns);
//    stack = nullptr;
//    stackRows = nullptr;
//    stackColumns = nullptr;
//
//    //CUDA_PRINT("%llu: %s: END\n",
//    //           threadIdx,
//    //           __FUNCTION__);
//    return resultsCount;
//}


#endif // !__INCLUDED_KERNEL_FUNCTIONS_INL__
