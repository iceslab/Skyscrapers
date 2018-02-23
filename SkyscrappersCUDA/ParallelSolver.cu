#include "ParallelSolver.cuh"

namespace cuda
{
    namespace solver
    {

        //CUDA_HOST std::vector<board::Board> generateBoards(const board::Board & board, const size_t stopLevel)
        //{
        //    std::vector<board::Board> retVal;
        //    generateBoards(board, stopLevel, retVal);
        //    return retVal;
        //}

        //CUDA_HOST void generateBoards(const board::Board & board,
        //                              size_t stopLevel,
        //                              std::vector<board::Board> & retVal,
        //                              size_t level,
        //                              size_t row,
        //                              size_t column)
        //{
        //    ASSERT(stopLevel > 0 && stopLevel <= board.size());
        //    DEBUG_CALL(std::cout << "level: " << level << " row: " << row << " column: " << column << "\n";);
        //    DEBUG_CALL(board.print());
        //    const auto treeRowSize = board.size();

        //    // Check if it is last cell
        //    const auto cellPair = getNextFreeCell(row, column);
        //    if (level == stopLevel || cellPair == lastCellPair)
        //    {
        //        retVal.emplace_back(board);
        //    }
        //    else
        //    {
        //        for (size_t i = 0; i < treeRowSize; i++)
        //        {
        //            const auto consideredBuilding = static_cast<board::boardFieldT>(i + 1);
        //            if (board.isBuildingPlaceable(row, column, consideredBuilding))
        //            {
        //                board.setCell(row, column, consideredBuilding);
        //                if (board.isBoardPartiallyValid(row, column))
        //                {
        //                    generateBoards(stopLevel, retVal, level + 1, cellPair.first, cellPair.second);
        //                }

        //                board.clearCell(row, column);
        //            }
        //        }
        //    }
        //}

        CUDA_HOST kernelInputT prepareSolvers(const std::vector<board::Board> & boards, size_t & count)
        {
            // Create array on host
            std::vector<solver::SequentialSolver> prepareRetVal;
            prepareRetVal.reserve(boards.size());

            //for (size_t i = 0; i < boards.size(); i++)
            //{
            //    auto & el = boards[i];
            //    prepareRetVal.push_back(std::move(el));
            //}

            for (auto & el : boards)
            {
                prepareRetVal.push_back(std::move(el));
            }

            // Create array on device
            kernelInputT d_retVal = nullptr;
            count = boards.size();
            cudaError_t err = cudaMalloc(&d_retVal, count * sizeof(*d_retVal));
            if (err != cudaSuccess)
            {
                CUDA_PRINT_ERROR("Failed allocation", err);
                d_retVal = nullptr;
                count = 0;
            }
            else
            {
                // Copy host array to device
                err = cudaMemcpy(&d_retVal, prepareRetVal.data(), count * sizeof(*d_retVal), cudaMemcpyHostToDevice);
                if (err != cudaSuccess)
                {
                    CUDA_PRINT_ERROR("Failed memcpy", err);
                    cudaFree(d_retVal);
                    d_retVal = nullptr;
                    count = 0;
                }
            }

            return d_retVal;
        }

        CUDA_HOST kernelOutputT prepareResultArray(size_t solversCount)
        {
            kernelOutputT d_retVal = nullptr;
            cudaError_t err = cudaMalloc(&d_retVal, solversCount * maxResultsPerThread * sizeof(*d_retVal));
            if (err != cudaSuccess)
            {
                CUDA_PRINT_ERROR("Failed allocation", err);
                d_retVal = nullptr;
            }

            return d_retVal;
        }

        CUDA_HOST kernelOutputSizesT prepareResultArraySizes(size_t solversCount)
        {
            kernelOutputSizesT d_retVal = nullptr;
            cudaError_t err = cudaMalloc(&d_retVal, solversCount * sizeof(*d_retVal));
            if (err != cudaSuccess)
            {
                CUDA_PRINT_ERROR("Failed allocation", err);
                d_retVal = nullptr;
            }

            return d_retVal;
        }

        CUDA_HOST stackT prepareStack(size_t boardSize, size_t solversCount)
        {
            stackT d_retVal = nullptr;
            cudaError_t err = cudaMalloc(&d_retVal, boardSize * boardSize * solversCount * sizeof(*d_retVal));
            if (err != cudaSuccess)
            {
                CUDA_PRINT_ERROR("Failed allocation", err);
                d_retVal = nullptr;
            }

            return d_retVal;
        }

        CUDA_HOST void freeSolvers(kernelInputT & d_solvers)
        {
            cudaFree(d_solvers);
            d_solvers = nullptr;
        }

        CUDA_HOST void freeResultArray(kernelOutputT & d_outputBoards)
        {
            cudaFree(d_outputBoards);
            d_outputBoards = nullptr;
        }

        CUDA_HOST void freeResultArraySizes(kernelOutputSizesT & d_outputBoardsSizes)
        {
            cudaFree(d_outputBoardsSizes);
            d_outputBoardsSizes = nullptr;
        }

        CUDA_HOST void freeStack(stackT & d_stack)
        {
            cudaFree(d_stack);
            d_stack = nullptr;
        }

        CUDA_HOST bool verifyAllocation(kernelInputT & d_solvers,
                                        kernelOutputT & d_outputBoards,
                                        kernelOutputSizesT & d_outputBoardsSizes,
                                        stackT & d_stack)
        {
            return d_solvers != nullptr &&
                d_outputBoards != nullptr &&
                d_outputBoardsSizes != nullptr &&
                d_stack != nullptr;
        }

        CUDA_GLOBAL void parallelBoardSolving(const kernelInputT & d_solvers,
                                              kernelOutputT & d_outputBoards,
                                              kernelOutputSizesT & d_outputBoardsSizes,
                                              stackT & d_stack)
        {
            // It denotes thread index and array index
            const auto idx = threadIdx.x;
            d_outputBoardsSizes[idx] = d_solvers[idx].solve(d_outputBoards + idx * maxResultsPerThread, d_stack + idx);
        }
    }
}
